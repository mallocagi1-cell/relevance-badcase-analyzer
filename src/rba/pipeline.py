"""End-to-end analysis. Heavy stages are cached; the report is always rewritten."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from rba.analysis.delta_w import analyze_delta
from rba.analysis.grouping import cluster_group, confusion, group_errors, pick_detail
from rba.analysis.hypotheses import aggregate_mechanistic, propose_checks, rank_hypotheses, summarize_group
from rba.analysis.instrument import collect_hidden_batch, instrument_batch
from rba.analysis.probes import probe_attributes
from rba.analysis.token_attr import attribute_batch
from rba.cache import RunCache
from rba.config import Config
from rba.console import info, lines, reset, warn
from rba.data.encode import EncodedCase, collate, encode_cases
from rba.errors import RBAError
from rba.llm.client import chat, extract_json
from rba.llm.prompt import build_group_prompt
from rba.models.loader import load_models
from rba.models.paths import attention_module, describe_model, output_projection, resolve_paths
from rba.repair.evaluate import compare, conclusion, forward_grade_logits, metric_block, predictions_from_logits
from rba.repair.interventions import apply_patch
from rba.report.plots import save_plots
from rba.report.render import METHODOLOGY, write_reports
from rba.schema import load_cases
from rba.util import batched, file_fingerprint, oom_message, stable_hash, to_builtin


def analyze(cfg: Config) -> dict:
    reset()
    bad_cases, good_cases = _validate_and_load(cfg)
    fingerprint = stable_hash(
        {
            "config": cfg.fingerprint_payload(),
            "bad": file_fingerprint(cfg.badcases),
            "good": file_fingerprint(cfg.goodcases) if cfg.goodcases else None,
        }
    )
    output = _prepare_output(cfg, fingerprint)
    cache = RunCache(output, fingerprint, resume=cfg.resume and not cfg.fresh)
    if cfg.fresh or not cfg.resume:
        cache.clear()
    info(f"输出目录：{output}")
    _check_api_key(cfg)

    info("[1/7] 加载模型")
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, cfg.module_paths, loaded.preset)
    for note in resolved.notes:
        warn(note)
    architecture = describe_model(loaded.ft, resolved)
    info(
        f"模块已对齐：层 {resolved.layers_path}，头 {resolved.lm_head_path}，"
        f"视觉塔 {resolved.visual_path or '无'}，最终 norm {resolved.final_norm_path or '无'}。"
    )
    if cfg.force_eager_attn:
        _try_eager(loaded.ft)

    grade_ids = loaded.grade_ids
    grades = cfg.grades
    grade_id_list = [grade_ids[grade] for grade in grades]
    image_token_id = loaded.image_token_id
    if image_token_id is None:
        image_token_id = getattr(getattr(loaded.ft, "config", None), "image_token_id", None)
    pad_id = _pad_id(loaded.tokenizer)

    info("[2/7] 编码样本")
    image_size = cfg.tiny.image_size if loaded.kind == "tiny" else 8
    bad_encoded = encode_cases(bad_cases, loaded.tokenizer, cfg, loaded.processor, image_size=image_size)
    good_encoded = encode_cases(good_cases, loaded.tokenizer, cfg, loaded.processor, image_size=image_size) if good_cases else []
    if cfg.max_cases is not None:
        bad_encoded = bad_encoded[: cfg.max_cases]
        info(f"按 max_cases 只保留前 {len(bad_encoded)} 条坏例。")

    batch_size = _auto_batch(cfg, loaded.device, any(item.pixel_values is not None for item in bad_encoded))
    info(f"batch size {batch_size}。显存不够会自动减半。")

    info("[3/7] 重新推断")
    bad_rows = _cached_infer(cache, "infer_bad", loaded, bad_encoded, grade_id_list, grades, pad_id, batch_size, "推断坏例")
    good_rows = _cached_infer(cache, "infer_good", loaded, good_encoded, grade_id_list, grades, pad_id, batch_size, "推断对照") if good_encoded else []
    by_id = {row["id"]: row for row in bad_rows}
    groups = group_errors(bad_rows)
    table = confusion(bad_rows, grades)
    info(f"坏例 {len(bad_rows)} 条，判错 {sum(len(v) for v in groups.values())} 条，混淆组 {len(groups)} 个。")

    ranked_groups = list(groups.items())[: cfg.max_llm_groups]
    if len(groups) > len(ranked_groups):
        warn(f"只对样本最多的 {len(ranked_groups)} 个混淆组做逐层分析和大模型提示。其余组只出现在混淆表里。")

    detail_ids = {}
    for offset, (key, rows) in enumerate(ranked_groups):
        detail_ids[key] = [row["id"] for row in pick_detail(rows, cfg.max_detail_cases, cfg.seed + offset)]

    info("[4/7] 机制分析（logit lens、归因、注意力）")
    mech_by_id = _cached_mech(cache, loaded, resolved, bad_encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size)

    info("[5/7] Token 归因与探针")
    attr_by_id = {}
    if cfg.analyses.token_attr:
        attr_by_id = _cached_attr(cache, cfg, loaded, resolved, bad_encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size)
    probes = {"skipped": "配置关闭了探针。", "layers": []}
    if cfg.analyses.probes:
        probes = _cached_probes(cache, cfg, loaded, resolved, bad_encoded + good_encoded, by_id, grades, grade_id_list, pad_id, batch_size)

    info("[6/7] 干预复核与 ΔW")
    verifications = {}
    if cfg.analyses.verify and cfg.analyses.dla:
        verifications = _verify(cache, cfg, loaded, resolved, bad_encoded, by_id, groups, detail_ids, mech_by_id, grade_ids, grades, image_token_id, pad_id)
    delta = {"top": [], "lm_head_grades": {"rows": {}}}
    if cfg.analyses.delta_w:
        delta = _cached_delta(cache, cfg, loaded, grade_ids)

    info("[7/7] 假设、提示词和报告")
    group_reports = []
    prompts = []
    encoded_by_id = {item.id: item for item in bad_encoded}
    for key, rows in groups.items():
        detail_rows = [by_id[i] for i in detail_ids.get(key, [])]
        clusters = None
        if cfg.analyses.cluster:
            clusters = cluster_group(rows, grades, cfg.cluster_k, cfg.seed)
        summary = summarize_group(key, rows, detail_rows, grades, clusters)
        mech_rows = []
        for row in detail_rows:
            mech = mech_by_id.get(row["id"])
            if mech is None:
                continue
            mech = dict(mech)
            mech["label"] = row["label"]
            mech["pred"] = row["pred"]
            mech_rows.append(mech)
        mech_summary = aggregate_mechanistic(mech_rows, grades) if mech_rows else {}
        if mech_summary.get("decomposition_gap", 0) > 0.15:
            warn(f"{key} 的残差分解相对差距 {mech_summary['decomposition_gap']:.2f}，组件归因先别当真。")
        hypotheses = []
        if mech_summary:
            hypotheses = rank_hypotheses(summary, mech_summary, delta, verifications.get(key, []), architecture["n_parameters"])
            _annotate_fraction(hypotheses, loaded.ft, resolved, architecture["n_parameters"])
            _mark_recommended(hypotheses)
            _attach_commands(hypotheses, cfg, output, key)
        token_summary = _mean_attr([attr_by_id[i] for i in detail_ids.get(key, []) if i in attr_by_id])
        group_reports.append(
            {
                **summary,
                "hypotheses": hypotheses,
                "components": (mech_summary.get("components") or [])[:12],
                "lens": mech_summary.get("lens"),
                "token_attr": token_summary,
                "verifications": verifications.get(key, []),
                "flip_layer": mech_summary.get("flip_layer"),
                "decomposition_gap": mech_summary.get("decomposition_gap"),
            }
        )
        if cfg.analyses.llm and mech_summary:
            prompts.append(
                build_group_prompt(
                    architecture=architecture,
                    grades=grades,
                    grade_ids=grade_ids,
                    group=summary,
                    mech=mech_summary,
                    hypotheses=hypotheses,
                    delta_top=delta.get("top") or [],
                    template=cfg.prompt_template,
                    budget_tokens=cfg.llm.max_prompt_tokens,
                    chars_per_token=cfg.chars_per_token,
                )
            )

    prompt_dir = output / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        safe = prompt["group"].replace("->", "_to_").replace("/", "_")
        (prompt_dir / f"{safe}.txt").write_text(prompt["text"], encoding="utf-8")
    llm_results = []
    if prompts and not cfg.dry_run and cfg.analyses.llm and cfg.llm.enabled:
        info(f"调用 {cfg.llm.model}，共 {len(prompts)} 组。")
        for prompt in prompts:
            raw = chat(cfg.llm.base_url, cfg.llm.model, cfg.llm.api_key_env, prompt["text"], cfg.llm.temperature, cfg.llm.timeout_s)
            parsed = extract_json(raw)
            if parsed is None:
                warn(f"{prompt['group']} 的回复里没有合法 JSON，原文已保存。")
            safe = prompt["group"].replace("->", "_to_")
            (prompt_dir / f"{safe}.reply.txt").write_text(raw, encoding="utf-8")
            llm_results.append({"group": prompt["group"], "parsed": parsed, "raw": raw[:8000]})
    elif cfg.dry_run and prompts:
        info(f"干跑：写了 {len(prompts)} 份提示词，没有调用接口。")

    _write_patches(output, group_reports, cfg, image_token_id)
    overview = _overview(bad_rows, good_rows)
    banner = None
    if loaded.kind in {"tiny", "qwen3_5_tiny"}:
        plant = loaded.plant or {}
        banner = (
            "这次跑的是内置小模型，用来确认工具链。"
            + (f" 人为改动：{plant.get('what')}。" if plant else " 没有植入错误。")
            + " 报告里的数字不是线上相关性模型的结论。"
        )
    report = {
        "output_dir": str(output),
        "banner": banner,
        "lead": _lead(group_reports),
        "overview": overview,
        "grades": grades,
        "grade_ids": grade_ids,
        "confusion": table,
        "groups": group_reports,
        "delta_w": {key: value for key, value in delta.items() if key != "missing"} | {"n_missing": delta.get("n_missing", 0)},
        "probes": probes,
        "architecture": architecture,
        "resolved_notes": resolved.notes,
        "device": str(loaded.device),
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "dry_run": cfg.dry_run or not cfg.analyses.llm or not cfg.llm.enabled,
        "llm_results": llm_results,
        "prompt_files": [f"prompts/{path.name}" for path in sorted(prompt_dir.glob("*.txt"))],
        "warnings": [line[3:] for line in lines() if line.startswith("注意：")],
        "methodology": METHODOLOGY,
        "plant": loaded.plant,
        "fingerprint": fingerprint,
    }
    report = to_builtin(report)
    plot_names = save_plots(report, output / "plots")
    md_path, html_path = write_reports(report, output, plot_names)
    _dump_used_config(cfg, output)
    (output / "run.log").write_text("\n".join(lines()) + "\n", encoding="utf-8")
    info(f"报告已写入 {html_path}")
    info(f"Markdown：{md_path}")
    _print_lead(group_reports)
    return {"output_dir": str(output), "html": str(html_path), "markdown": str(md_path), "report": report}


def evaluate_patch(cfg: Config, patch: dict) -> dict:
    reset()
    if patch.get("tiny_seed") is not None:
        cfg.seed = int(patch["tiny_seed"])
    if "plant_error" in patch:
        cfg.plant_error = bool(patch["plant_error"])
    bad_cases, good_cases = _validate_and_load(cfg)
    info("加载模型以验收补丁")
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, cfg.module_paths, loaded.preset)
    grades = cfg.grades
    grade_ids = loaded.grade_ids
    grade_id_list = [grade_ids[grade] for grade in grades]
    if patch.get("method") in {"ablate_image", "ablate_tokens"} and not patch.get("token_ids"):
        if loaded.image_token_id is None:
            raise RBAError("图像消融没有 token id。", "在补丁里写 token_ids，或换一个补丁。")
        patch = dict(patch)
        patch["method"] = "ablate_tokens"
        patch["token_ids"] = [int(loaded.image_token_id)]
    pad_id = _pad_id(loaded.tokenizer)
    image_size = cfg.tiny.image_size if loaded.kind == "tiny" else 8
    bad_encoded = encode_cases(bad_cases, loaded.tokenizer, cfg, loaded.processor, image_size=image_size)
    good_encoded = encode_cases(good_cases, loaded.tokenizer, cfg, loaded.processor, image_size=image_size) if good_cases else []
    batch_size = _auto_batch(cfg, loaded.device, any(item.pixel_values is not None for item in bad_encoded))

    def score(items, patched: bool):
        def run():
            rows = []

            def infer(chunk):
                batch = _batch(loaded, chunk, pad_id)
                logits = forward_grade_logits(loaded.ft, batch, grade_id_list, loaded.device, loaded.dtype)
                preds = predictions_from_logits(logits, grades)
                built = []
                for item, pred in zip(chunk, preds):
                    built.append({"id": item.id, "label": item.case.label, "pred": pred})
                return built

            if patched:
                with apply_patch(loaded.ft, resolved, patch, grade_ids):
                    produced, _ = _run_batches(items, batch_size, infer, "验收补丁")
            else:
                produced, _ = _run_batches(items, batch_size, infer, "验收前")
            rows.extend(produced)
            return rows

        return run()

    bad_before = score(bad_encoded, False)
    bad_after = score(bad_encoded, True)
    good_before = score(good_encoded, False) if good_encoded else []
    good_after = score(good_encoded, True) if good_encoded else []
    bad_cmp = compare(bad_before, bad_after)
    good_cmp = compare(good_before, good_after) if good_before else None
    # metric_block inside compare doesn't recompute mae from floats because grades were empty.
    bad_cmp["before"] = metric_block(bad_before, grades)
    bad_cmp["after"] = metric_block(bad_after, grades)
    if good_cmp is not None:
        good_cmp["before"] = metric_block(good_before, grades)
        good_cmp["after"] = metric_block(good_after, grades)
    text = conclusion(bad_cmp, good_cmp)
    result = {"patch": patch, "bad": bad_cmp, "good": good_cmp, "conclusion": text}
    info(text)
    _print_eval(bad_cmp, good_cmp)
    return to_builtin(result)


def _print_eval(bad: dict, good: dict | None) -> None:
    def acc(block, key):
        value = (block.get(key) or {}).get("accuracy")
        return "—" if value is None else f"{value:.1%}"

    info(
        f"坏例：准确率 {acc(bad, 'before')} → {acc(bad, 'after')}，"
        f"修复率 {_pct(bad.get('fix_rate'))}（{bad.get('n_fixed')}/{bad.get('n_wrong_before')}）。"
    )
    if good:
        info(
            f"对照：准确率 {acc(good, 'before')} → {acc(good, 'after')}，"
            f"回退率 {_pct(good.get('regression_rate'))}（{good.get('n_regressed')}/{good.get('n_right_before')}）。"
        )


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _validate_and_load(cfg: Config):
    if (cfg.base_model == "tiny") != (cfg.ft_model == "tiny") and cfg.model_type == "auto":
        raise RBAError(
            "tiny 要基座和微调一起用。",
            "两边都写 tiny，或都写真实目录。试跑：rba demo",
        )
    if cfg.base_model == cfg.ft_model and cfg.base_model not in {"tiny", "qwen3_5_tiny"}:
        warn("基座和微调是同一个目录，ΔW 会接近 0。若这不是有意的，检查 --base 和 --ft。")
    bad_path = cfg.resolve_path(cfg.badcases, from_cli=False)
    # resolve_path uses config dir. CLI paths are already absolute when set by cli.
    cfg.badcases = bad_path or cfg.badcases
    if cfg.goodcases:
        cfg.goodcases = cfg.resolve_path(cfg.goodcases, from_cli=False)
    if cfg.model_type in {"hf", "auto"} and cfg.ft_model not in {"tiny", "qwen3_5_tiny"}:
        cfg.ft_model = cfg.resolve_path(cfg.ft_model) or cfg.ft_model
        cfg.base_model = cfg.resolve_path(cfg.base_model) or cfg.base_model
        ft = Path(cfg.ft_model)
        if not ft.exists():
            raise RBAError(
                f"找不到微调模型 {cfg.ft_model}。",
                "指到含 config.json 的目录。只想看流程：rba analyze --base tiny --ft tiny --badcases examples/badcases.jsonl --dry-run",
            )
    bad = load_cases(cfg.badcases, cfg.grades, what="坏例文件")
    good = []
    if cfg.goodcases:
        good = load_cases(cfg.goodcases, cfg.grades, what="对照集")
    missing = [case.image for case in bad + good if case.image and not Path(case.image).exists()]
    if missing:
        warn(f"{len(missing)} 张图片路径现在还不存在，编码时会跳过并按纯文本继续。例如 {missing[0]}")
    info(f"坏例 {len(bad)} 条" + (f"，对照 {len(good)} 条。" if good else "。"))
    return bad, good


def _check_api_key(cfg: Config) -> None:
    if cfg.dry_run or not cfg.analyses.llm or not cfg.llm.enabled:
        return
    if not os.environ.get(cfg.llm.api_key_env):
        raise RBAError(
            f"环境变量 {cfg.llm.api_key_env} 未设置，还没开始算就停了。",
            "先导出密钥；或加上 --dry-run，把提示词写到输出目录，分析本身照常完成。",
        )


def _prepare_output(cfg: Config, fingerprint: str) -> Path:
    if cfg.output_dir:
        output = Path(cfg.output_dir)
        if not output.is_absolute():
            output = Path.cwd() / output
    else:
        output = Path.cwd() / "runs" / fingerprint[:12]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "cache" / "manifest.json"
    if manifest.exists() and not cfg.fresh:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        found = {item.get("fingerprint") for item in data.values() if isinstance(item, dict)}
        found.discard(None)
        if found and found != {fingerprint}:
            raise RBAError(
                f"{output} 里已经有一次不同配置或不同数据的缓存。",
                "换一个 --output-dir，或加上 --fresh 覆盖缓存。报告会被重写，模型权重不会被改。",
            )
    return output


def _batch(loaded, chunk, pad_id: int) -> dict:
    batch = collate(chunk, pad_id)
    # The stand-in consumes NCHW pixels. A real Qwen tower expects the
    # processor's packed patches; without that, drop pixels and keep token ids.
    if loaded.kind != "tiny" and "pixel_values" in batch:
        batch.pop("pixel_values", None)
    return batch


def _pad_id(tokenizer) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def _auto_batch(cfg: Config, device, has_image: bool) -> int:
    if cfg.batch_size and cfg.batch_size > 0:
        return cfg.batch_size
    if device.type == "cuda":
        return 1 if has_image else 2
    return 2 if has_image else 4


def _try_eager(model) -> None:
    setter = getattr(model, "set_attn_implementation", None)
    if not callable(setter):
        warn("这个模型没有 set_attn_implementation，注意力份额可能测不到。")
        return
    try:
        setter("eager")
        info("注意力实现已切到 eager，以便读取注意力权重。")
    except Exception as exc:
        warn(f"没能切到 eager attention：{exc}")


def _run_batches(items, batch_size, fn, desc: str):
    if not items:
        return [], batch_size
    size = max(1, batch_size)
    while True:
        try:
            outputs = []
            chunks = list(batched(items, size))
            for chunk in tqdm(chunks, desc=desc, leave=False):
                outputs.extend(fn(chunk))
            return outputs, size
        except RuntimeError as exc:
            if oom_message(exc) and size > 1:
                size = max(1, size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                warn(f"显存不足，batch 降为 {size} 后重试这一阶段。")
                continue
            raise


def _infer_rows(loaded, items: list[EncodedCase], grade_id_list, grades, pad_id, batch_size, desc: str):
    def fn(chunk):
        batch = _batch(loaded, chunk, pad_id)
        logits = forward_grade_logits(loaded.ft, batch, grade_id_list, loaded.device, loaded.dtype)
        preds = predictions_from_logits(logits, grades)
        probs = torch.softmax(logits, dim=-1)
        rows = []
        for item, pred, prob, grade_logit in zip(chunk, preds, probs, logits):
            rows.append(
                {
                    "id": item.id,
                    "label": item.case.label,
                    "pred": pred,
                    "probs": {grade: float(prob[index]) for index, grade in enumerate(grades)},
                    "grade_logits": [float(value) for value in grade_logit],
                    "file_prediction": item.case.prediction,
                    "query": item.case.query,
                    "doc": item.case.doc,
                    "has_image": bool(item.case.image and item.pixel_values is not None),
                    "attributes": item.case.attributes,
                    "correct": pred == item.case.label,
                }
            )
        return rows

    rows, _ = _run_batches(items, batch_size, fn, desc)
    return rows


def _cached_infer(cache, stage, loaded, items, grade_id_list, grades, pad_id, batch_size, desc):
    if cache.has(stage):
        info(f"从缓存恢复：{desc}")
        return cache.load_json(stage)
    rows = _infer_rows(loaded, items, grade_id_list, grades, pad_id, batch_size, desc)
    cache.save_json(stage, rows)
    return rows


def _detail_items(encoded, detail_ids) -> list[EncodedCase]:
    wanted = {case_id for ids in detail_ids.values() for case_id in ids}
    return [item for item in encoded if item.id in wanted]


def _cached_mech(cache, loaded, resolved, encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size):
    if cache.has("mech"):
        info("从缓存恢复：机制分析")
        payload = cache.load_json("mech")
        return {row["id"]: row for row in payload}
    items = _detail_items(encoded, detail_ids)
    id_to_pred = {}
    id_to_gold = {}
    for item in items:
        row = by_id[item.id]
        id_to_pred[item.id] = grade_id_list[grades.index(row["pred"])]
        id_to_gold[item.id] = grade_id_list[grades.index(row["label"])]

    def fn(chunk):
        batch = _batch(loaded, chunk, pad_id)
        pred = torch.tensor([id_to_pred[item.id] for item in chunk], dtype=torch.long)
        gold = torch.tensor([id_to_gold[item.id] for item in chunk], dtype=torch.long)
        records = instrument_batch(
            loaded.ft,
            batch,
            resolved,
            grade_id_list,
            pred,
            gold,
            image_token_id,
            loaded.device,
            loaded.dtype,
        )
        cleaned = []
        for record in records:
            record.pop("hidden", None)
            cleaned.append(record)
        return cleaned

    if not items:
        cache.save_json("mech", [])
        return {}
    rows, _ = _run_batches(items, min(batch_size, 2), fn, "机制分析")
    cache.save_json("mech", rows)
    return {row["id"]: row for row in rows}


def _cached_attr(cache, cfg, loaded, resolved, encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size):
    if cache.has("token_attr"):
        info("从缓存恢复：token 归因")
        return {row["id"]: row for row in cache.load_json("token_attr")}
    items = _detail_items(encoded, detail_ids)

    def fn(chunk):
        batch = _batch(loaded, chunk, pad_id)
        pred = torch.tensor([grade_id_list[grades.index(by_id[item.id]["pred"])] for item in chunk])
        gold = torch.tensor([grade_id_list[grades.index(by_id[item.id]["label"])] for item in chunk])
        return attribute_batch(
            loaded.ft,
            batch,
            resolved,
            pred,
            gold,
            image_token_id,
            loaded.tokenizer,
            loaded.device,
            loaded.dtype,
            cfg.token_attr_method,
            cfg.ig_steps,
        )

    rows, _ = _run_batches(items, 1 if cfg.token_attr_method == "integrated_gradients" else min(2, batch_size), fn, "token 归因")
    cache.save_json("token_attr", rows)
    return {row["id"]: row for row in rows}


def _cached_probes(cache, cfg, loaded, resolved, encoded, by_id, grades, grade_id_list, pad_id, batch_size):
    if cache.has("probes"):
        info("从缓存恢复：探针")
        return cache.load_json("probes")
    chosen = encoded[: cfg.probe_max_cases]
    if len(chosen) < 4:
        payload = {"skipped": "样本少于 4 条，跳过探针。", "layers": []}
        cache.save_json("probes", payload)
        return payload

    def fn(chunk):
        batch = _batch(loaded, chunk, pad_id)
        hidden = collect_hidden_batch(loaded.ft, batch, resolved, loaded.device, loaded.dtype)
        return [{"id": item.id, "hidden": tensor} for item, tensor in zip(chunk, hidden)]

    rows, _ = _run_batches(chosen, batch_size, fn, "探针隐状态")
    import numpy as np

    hidden = np.stack([row["hidden"].numpy() for row in rows], axis=0)
    records = []
    for row in rows:
        source = by_id.get(row["id"])
        if source is None:
            item = next(item for item in chosen if item.id == row["id"])
            source = {"attributes": item.case.attributes, "label": item.case.label, "pred": None}
        records.append({"attributes": source.get("attributes") or {}, "label": source.get("label"), "pred": source.get("pred")})
    payload = probe_attributes(records, hidden, cfg.probe_attributes)
    cache.save_json("probes", payload)
    return payload


def _verify(cache, cfg, loaded, resolved, encoded, by_id, groups, detail_ids, mech_by_id, grade_ids, grades, image_token_id, pad_id):
    if cache.has("verify"):
        info("从缓存恢复：干预复核")
        return cache.load_json("verify")
    encoded_by_id = {item.id: item for item in encoded}
    results = {}
    jobs = []
    delta_placeholder = None
    for key, _rows in list(groups.items())[: cfg.max_llm_groups]:
        detail = [by_id[i] for i in detail_ids.get(key, [])]
        mech_rows = []
        for row in detail:
            mech = mech_by_id.get(row["id"])
            if not mech:
                continue
            mech = dict(mech)
            mech["label"] = row["label"]
            mech["pred"] = row["pred"]
            mech_rows.append(mech)
        if not mech_rows:
            continue
        summary = summarize_group(key, detail, detail, grades, None)
        mech_summary = aggregate_mechanistic(mech_rows, grades)
        checks = propose_checks(summary, mech_summary, delta_placeholder, cfg.verify_top_k)
        for check in checks:
            jobs.append((key, check, [encoded_by_id[row["id"]] for row in detail]))
    info(f"干预复核 {len(jobs)} 次。每次只动一个组件。")
    for key, check, items in tqdm(jobs, desc="干预复核", leave=False):
        patch = dict(check["patch"])
        if patch.get("method") == "ablate_image":
            if image_token_id is None:
                continue
            patch = {"method": "ablate_tokens", "token_ids": [int(image_token_id)]}
        deltas = []
        improved = 0
        id_list = [grade_ids[grade] for grade in grades]
        for item in items:
            row = by_id[item.id]
            # Pair the intervention with a single-example forward. A batched
            # baseline can differ by padding even when the tokens match.
            solo = _batch(loaded, [item], pad_id)
            before = forward_grade_logits(loaded.ft, solo, id_list, loaded.device, loaded.dtype)[0]
            before_diff = float(before[grades.index(row["pred"])] - before[grades.index(row["label"])])
            with apply_patch(loaded.ft, resolved, patch, grade_ids):
                after = forward_grade_logits(loaded.ft, solo, id_list, loaded.device, loaded.dtype)[0]
            after_diff = float(after[grades.index(row["pred"])] - after[grades.index(row["label"])])
            delta = after_diff - before_diff
            deltas.append(delta)
            if delta < -1e-5:
                improved += 1
        n = len(deltas)
        results.setdefault(key, []).append(
            {
                "key": check["key"],
                "mean_delta": (sum(deltas) / n) if n else None,
                "n": n,
                "n_improved": improved,
                "agreement": (improved / n) if n else None,
                "patch": patch,
            }
        )
    cache.save_json("verify", results)
    return results


def _cached_delta(cache, cfg, loaded, grade_ids):
    if cache.has("delta"):
        info("从缓存恢复：ΔW")
        return cache.load_json("delta")
    info("逐张量计算 ΔW。大矩阵只对排在前面的做低秩 SVD。")
    weight = loaded.ft.lm_head.weight.detach().float().cpu()
    payload = analyze_delta(
        loaded.base,
        loaded.ft,
        loaded.base_path,
        weight,
        grade_ids,
        loaded.tokenizer,
        cfg.delta_w_top_k,
        cfg.svd_rank,
        cfg.svd_max_dim,
    )
    if payload.get("n_missing"):
        warn(f"有 {payload['n_missing']} 个参数没能和基座对齐。前几个：{payload['missing'][:3]}")
    cache.save_json("delta", payload)
    return payload


def _mean_attr(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    def avg(key):
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "method": rows[0]["method"],
        "text_abs_share": avg("text_abs_share"),
        "image_abs_share": avg("image_abs_share"),
        "text_signed": avg("text_signed"),
        "image_signed": avg("image_signed"),
        "n": len(rows),
    }


def _annotate_fraction(hypotheses, model, resolved, n_parameters: int) -> None:
    if not n_parameters:
        return
    for item in hypotheses:
        patch = item.get("patch") or {}
        if patch.get("method") == "head_scale":
            layer = resolved.layers(model)[int(patch["layer"])]
            found = attention_module(layer)
            if found is None:
                continue
            info_proj = output_projection(found[1])
            if info_proj is None:
                continue
            proj, head_dim, _n = info_proj
            touched = len(patch["heads"]) * head_dim * proj.weight.shape[0]
            item["weight_fraction"] = touched / n_parameters
        elif patch.get("method") == "lm_head_blend":
            item["weight_fraction"] = model.lm_head.weight.shape[1] / n_parameters
        elif patch.get("method") in {"mlp_zero", "ablate_image", "ablate_tokens"}:
            item["weight_fraction"] = 0.0


def _mark_recommended(hypotheses: list[dict]) -> None:
    """Only a couple of hypotheses per group get an eval command.

    Rejected ablations stay in the list so they are not hidden, but they are
    not offered as the next thing to run.
    """
    for item in hypotheses:
        item["recommend"] = False
    chosen = [item for item in hypotheses if item.get("status") == "已验证" and item.get("patch")]
    if not chosen:
        for item in hypotheses:
            if item.get("patch") and item.get("status") not in {"被复核否定", "消融无效果", "证据不足"}:
                chosen.append(item)
                break
    for item in chosen[:2]:
        item["recommend"] = True


def _attach_commands(hypotheses, cfg: Config, output: Path, group_key: str) -> None:
    for index, item in enumerate(hypotheses, start=1):
        if not item.get("patch") or not item.get("recommend"):
            item["command"] = ""
            item["patch_file"] = None
            continue
        safe = group_key.replace("->", "_to_")
        filename = f"patches/{safe}_h{index}.json"
        item["patch_file"] = filename
        ft = cfg.ft_model
        bad = cfg.badcases
        good = f" --goodcases {cfg.goodcases}" if cfg.goodcases else ""
        item["command"] = (
            f"rba eval --ft {ft} --badcases {bad}{good} --patch {output / filename} --device {cfg.device if cfg.device != 'auto' else 'cpu'}"
        )


def _write_patches(output: Path, groups: list[dict], cfg: Config, image_token_id) -> None:
    folder = output / "patches"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for group in groups:
        for item in group.get("hypotheses") or []:
            rel = item.get("patch_file")
            patch = item.get("patch")
            if not rel or not patch:
                continue
            payload = dict(patch)
            if payload.get("method") == "ablate_image" and image_token_id is not None:
                payload["method"] = "ablate_tokens"
                payload["token_ids"] = [int(image_token_id)]
            payload.update(
                {
                    "comment": item["title"],
                    "group": group["key"],
                    "status": item["status"],
                    "tiny_seed": cfg.seed,
                    "plant_error": cfg.plant_error,
                    "grades": cfg.grades,
                }
            )
            path = output / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _overview(bad_rows, good_rows) -> dict:
    bad_metrics = metric_block(bad_rows, [])
    good_metrics = metric_block(good_rows, []) if good_rows else {"n": 0, "accuracy": None, "mae": None, "n_correct": 0}
    return {
        "n_bad": bad_metrics["n"],
        "n_bad_correct": bad_metrics["n_correct"],
        "n_bad_wrong": bad_metrics["n"] - bad_metrics["n_correct"],
        "bad_accuracy": bad_metrics["accuracy"],
        "bad_mae": bad_metrics["mae"],
        "n_good": good_metrics["n"],
        "good_accuracy": good_metrics["accuracy"],
        "good_mae": good_metrics["mae"],
    }


def _lead(groups: list[dict]) -> str:
    analyzed = [group for group in groups if group.get("hypotheses")]
    if not analyzed:
        if not groups:
            return "模型在这批坏例上的等级 argmax 和 label 一致。先确认文件里是不是真的错例，以及模板、等级 token 是否和训练时一致。"
        return "有判错的组，但这轮没有逐层证据（可能关了分析，或组没有被抽到）。打开 dla 和 verify 再跑一次。"
    verified = sum(1 for group in analyzed for item in group["hypotheses"] if item["status"] == "已验证")
    return (
        f"下面按混淆组列出根因假设。{verified} 条在该组抽样上被干预复核支持。"
        "抽样复核不是对照集：每条命令会在全部坏例和对照集上重跑。"
        "修复率不高于回退率的补丁不要留。"
    )


def _print_lead(groups: list[dict]) -> None:
    for group in groups:
        hyps = group.get("hypotheses") or []
        if not hyps:
            continue
        top = hyps[0]
        info(f"{group['key']}（{group['count']}）优先看：[{top['status']}] {top['title']}")


def _dump_used_config(cfg: Config, output: Path) -> None:
    payload = {
        "base_model": cfg.base_model,
        "ft_model": cfg.ft_model,
        "model_type": cfg.model_type,
        "badcases": cfg.badcases,
        "goodcases": cfg.goodcases,
        "grades": cfg.grades,
        "prompt_template": cfg.prompt_template,
        "image_token": cfg.image_token,
        "image_seq_len": cfg.image_seq_len,
        "input_mode": cfg.input_mode,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "seed": cfg.seed,
        "plant_error": cfg.plant_error,
        "module_paths": cfg.module_paths,
        "tiny": cfg.tiny.__dict__,
    }
    (output / "config.used.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
