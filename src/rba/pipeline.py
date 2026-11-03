"""End-to-end analysis. Heavy stages are cached; the report is always rewritten.

Stage order matters: ΔW runs before the mechanism and verification stages so
`propose_checks` and `instrument_batch` see the real weight deltas.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

from rba import __version__
from rba.analysis.delta_w import analyze_delta
from rba.analysis.grouping import cluster_group, confusion, group_errors, pick_detail
from rba.analysis.hypotheses import aggregate_mechanistic, propose_checks, rank_hypotheses, summarize_group
from rba.analysis.instrument import collect_hidden_batch, instrument_batch
from rba.analysis.probes import probe_attributes
from rba.analysis.token_attr import attribute_batch
from rba.cache import RunCache
from rba.config import Config, config_to_dict
from rba.console import info, lines, reset, warn
from rba.data.encode import EncodedCase, ExcludedCase, collate, encode_cases, needs_single_item_batches
from rba.errors import RBAError
from rba.llm.client import chat, extract_json
from rba.llm.prompt import build_group_prompt
from rba.models.loader import check_checkpoint_dir, is_tied_lm_head, load_models
from rba.models.paths import attention_module, describe_model, output_projection, resolve_paths
from rba.repair import evaluate as _evaluate
from rba.repair.evaluate import compare, conclusion, metric_block
from rba.repair.interventions import apply_patch
from rba.report.plots import save_plots
from rba.report.render import METHODOLOGY, write_reports
from rba.report.visual import build_visual_report
from rba.schema import Case, load_cases
from rba.util import batched, decision_index, file_fingerprint, oom_message, stable_hash, to_builtin

INVALID_GRADE = getattr(_evaluate, "INVALID_GRADE", "__invalid__")
VERIFIED_STATUSES = {"消融有效", "已验证"}
NOT_RECOMMENDED_STATUSES = {"被复核否定", "消融无效果", "证据不足", "先处理这个", "未执行"}
CACHE_SCHEMA = 4
_IDENTITY_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.json",
    "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
)
PATCH_MANIFEST = ".rba_patches.json"


# --------------------------------------------------------------------------- analyze


def analyze(cfg: Config) -> dict:
    reset()
    all_bad_cases, good_cases, split = _validate_and_load(cfg)
    holdout_ids = set(split.get("holdout_ids") or [])
    bad_cases = [case for case in all_bad_cases if case.id not in holdout_ids]
    if holdout_ids and not bad_cases:
        raise RBAError(
            f"holdout_fraction={cfg.holdout_fraction} 把全部 {len(all_bad_cases)} 条坏例都留给了验收集，发现集为空。",
            "调小 holdout_fraction，或给更多坏例。",
        )
    excluded: list[ExcludedCase] = []
    good_cases = _drop_goodcase_overlap(good_cases, [case for case in all_bad_cases if case.id in holdout_ids], excluded, split)
    identities = run_identities(cfg, all_bad_cases, good_cases)
    fingerprint = stable_hash({"cache_schema": CACHE_SCHEMA, "config": cfg.fingerprint_payload(), **identities})
    output, stale = _prepare_output(cfg, fingerprint)
    cache = RunCache(output, fingerprint, resume=cfg.resume and not cfg.fresh and not stale)
    if cfg.fresh or not cfg.resume or stale:
        cache.clear()
    info(f"输出目录：{output}")
    llm_mode = _llm_mode(cfg)

    info("[1/8] 加载模型")
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, cfg.module_paths, loaded.preset)
    loaded.tied_lm_head = is_tied_lm_head(loaded.ft, resolved.lm_head(loaded.ft))
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
    image_token_id = _image_token_id(loaded)
    pad_id = _pad_id(loaded.tokenizer)

    info("[2/8] 编码样本")
    if cfg.max_cases is not None:
        bad_cases = bad_cases[: cfg.max_cases]
        info(f"按 max_cases 只保留前 {len(bad_cases)} 条坏例。")
    bad_encoded, good_encoded = _encode_all(cfg, loaded, bad_cases, good_cases, excluded)
    if not bad_encoded:
        raise RBAError("没有可分析的坏例。", "所有坏例都被剔除了（图片读不到）。修好图片路径再跑。")
    batch_size = _effective_batch(cfg, loaded.device, bad_encoded + good_encoded)

    info(f"[3/8] 重新推断（解码协议 {cfg.decode_protocol}）")
    bad_rows_all = _cached_infer(cache, "infer_bad", loaded, bad_encoded, grade_id_list, grades, pad_id, batch_size, cfg.decode_protocol, "推断坏例")
    good_rows_all = (
        _cached_infer(cache, "infer_good", loaded, good_encoded, grade_id_list, grades, pad_id, batch_size, cfg.decode_protocol, "推断对照")
        if good_encoded
        else []
    )
    bad_rows, invalid_rows = _split_invalid(bad_rows_all, loaded.tokenizer)
    good_rows, invalid_good = _split_invalid(good_rows_all, loaded.tokenizer)
    if invalid_rows or invalid_good:
        warn(
            f"全词表 argmax 不是等级词的样本：坏例 {len(invalid_rows)} 条，对照 {len(invalid_good)} 条。"
            "它们单独列在报告里，不进入按等级索引的归因。"
        )
    by_id = {row["id"]: row for row in bad_rows}
    groups = group_errors(bad_rows)
    table = confusion(bad_rows, grades)
    info(f"坏例 {len(bad_rows_all)} 条，判错 {sum(len(v) for v in groups.values())} 条，混淆组 {len(groups)} 个。")

    ranked_groups = list(groups.items())[: cfg.max_llm_groups]
    if len(groups) > len(ranked_groups):
        warn(f"只对样本最多的 {len(ranked_groups)} 个混淆组做逐层分析和大模型提示。其余组只出现在混淆表里。")
    detail_ids: dict[str, list[str]] = {}
    verify_ids: dict[str, list[str]] = {}
    for offset, (key, rows) in enumerate(ranked_groups):
        detail = pick_detail(rows, cfg.max_detail_cases, cfg.seed + offset)
        detail_ids[key] = [row["id"] for row in detail]
        rest = [row for row in rows if row["id"] not in set(detail_ids[key])]
        verify_ids[key] = [row["id"] for row in pick_detail(rest, cfg.max_detail_cases, cfg.seed + 1000 + offset)] if rest else []

    info("[4/8] ΔW（在机制分析之前，供假设和复核使用）")
    delta: dict = {"top": [], "lm_head_grades": {"rows": {}}}
    if cfg.analyses.delta_w:
        delta = _cached_delta(cache, cfg, loaded, grade_ids, resolved)

    info("[5/8] 机制分析（logit lens、归因、注意力）")
    mech_by_id = _cached_mech(cache, loaded, resolved, bad_encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size, delta)

    info("[6/8] Token 归因与探针")
    attr_by_id: dict = {}
    if cfg.analyses.token_attr:
        attr_by_id = _cached_attr(cache, cfg, loaded, resolved, bad_encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size)
    probes = {"skipped": "配置关闭了探针。", "layers": []}
    if cfg.analyses.probes:
        probes = _cached_probes(cache, cfg, loaded, resolved, bad_encoded, good_encoded, {**by_id, **{r["id"]: r for r in good_rows}}, pad_id, batch_size)

    info("[7/8] 干预复核" + ("（目标、随机对照、恒等对照）" if cfg.verify_controls else "（只有目标，没有对照）"))
    verifications: dict = {}
    if cfg.analyses.verify and cfg.analyses.dla:
        verifications = _verify(cache, cfg, loaded, resolved, bad_encoded, by_id, groups, detail_ids, verify_ids, mech_by_id, grade_ids, grades, image_token_id, pad_id, delta)

    info("[8/8] 假设、提示词和报告")
    group_reports = []
    prompts = []
    for key, rows in groups.items():
        detail_rows = [by_id[i] for i in detail_ids.get(key, [])]
        clusters = cluster_group(rows, grades, cfg.cluster_k, cfg.seed) if cfg.analyses.cluster else None
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
        token_summary = _mean_attr([attr_by_id[i] for i in detail_ids.get(key, []) if i in attr_by_id])
        group_verifications = verifications.get(key, [])
        hypotheses = []
        if mech_summary:
            hypotheses = _rank(summary, mech_summary, delta, group_verifications, architecture["n_parameters"], token_summary)
            _annotate_fraction(hypotheses, loaded.ft, resolved, architecture["n_parameters"])
            _mark_recommended(hypotheses, loaded.tied_lm_head)
            _attach_commands(hypotheses, output, key)
        prompt = None
        if llm_mode != "off" and mech_summary:
            compact = {**mech_summary, "components": (mech_summary.get("components") or [])[: cfg.prompt_max_components]}
            try:
                prompt = _build_prompt(cfg, architecture, grades, grade_ids, summary, compact, hypotheses, delta, token_summary, probes, group_verifications)
            except RBAError as exc:
                warn(f"{key} 的提示词没能装进预算：{exc.message} {exc.hint}")
                prompt = None
            if prompt is not None:
                prompts.append(prompt)
        group_reports.append(
            {
                **summary,
                "hypotheses": hypotheses,
                "components": (mech_summary.get("components") or [])[:12],
                "lens": mech_summary.get("lens"),
                "token_attr": token_summary,
                "verifications": group_verifications,
                "flip_layer": mech_summary.get("flip_layer"),
                "decomposition_gap": mech_summary.get("decomposition_gap"),
                "attribution_reliable": mech_summary.get("attribution_reliable"),
                "reconstruction_error": mech_summary.get("reconstruction_error"),
                "evidence_packet": (prompt or {}).get("evidence_packet"),
                "prompt_evidence_ids": (prompt or {}).get("prompt_evidence_ids"),
            }
        )

    prompt_dir = output / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        (prompt_dir / f"{_safe_name(prompt['group'])}.txt").write_text(prompt["text"], encoding="utf-8")
    llm_results, llm_failures = _call_llm(cfg, prompts, prompt_dir, llm_mode)

    _write_patches(output, group_reports, cfg, image_token_id, fingerprint, loaded.tied_lm_head)
    overview = _overview(bad_rows_all, good_rows_all, invalid_rows, invalid_good, excluded)
    report = {
        "rba_version": __version__,
        "output_dir": str(output),
        "banner": _banner(loaded),
        "lead": _lead(group_reports, invalid_rows, len(bad_rows)),
        "overview": overview,
        "grades": grades,
        "grade_ids": grade_ids,
        "grade_mode": cfg.grade_mode,
        "decode_protocol": cfg.decode_protocol,
        "input_mode": cfg.input_mode,
        "tied_lm_head": loaded.tied_lm_head,
        "confusion": table,
        "groups": group_reports,
        "invalid_rows": invalid_rows[:50],
        "excluded": [{"id": item.id, "reason": item.reason} for item in excluded],
        "split": split,
        "delta_w": {key: value for key, value in delta.items() if key != "missing"} | {"n_missing": delta.get("n_missing", 0)},
        "probes": probes,
        "architecture": architecture,
        "resolved_notes": resolved.notes,
        "device": str(loaded.device),
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "llm_mode": llm_mode,
        "dry_run": llm_mode != "remote",
        "llm_results": llm_results,
        "llm_failures": llm_failures,
        "prompt_files": [f"prompts/{path.name}" for path in sorted(prompt_dir.glob("*.txt"))],
        "warnings": [line[3:] for line in lines() if line.startswith("注意：")],
        "methodology": METHODOLOGY,
        "plant": loaded.plant,
        "fingerprint": fingerprint,
    }
    report = to_builtin(report)
    plot_names = save_plots(report, output / "plots")
    md_path, _detail_path = write_reports(report, output, plot_names)
    html_path = build_visual_report(output, cfg.language, {"badcases": str(cfg.badcases) if cfg.badcases else None, "grades": list(cfg.grades), "language": cfg.language})
    _dump_used_config(cfg, output, loaded, split, excluded, fingerprint, identities)
    (output / "run.log").write_text("\n".join(lines()) + "\n", encoding="utf-8")
    info(f"报告已写入 {html_path}")
    info(f"Markdown：{md_path}")
    _print_lead(group_reports)
    if llm_failures:
        warn(f"{len(llm_failures)} 组的大模型调用失败，报告已照常写出；失败详情在 report.json 的 llm_failures。")
    return {"output_dir": str(output), "html": str(html_path), "markdown": str(md_path), "report": report, "llm_failures": llm_failures}


# --------------------------------------------------------------------------- eval


def evaluate_patch(cfg: Config, patch: dict, run_meta: dict | None = None) -> dict:
    reset()
    if patch.get("tiny_seed") is not None:
        cfg.seed = int(patch["tiny_seed"])
    if "plant_error" in patch:
        cfg.plant_error = bool(patch["plant_error"])
    if patch.get("grades") and [str(g) for g in patch["grades"]] != [str(g) for g in cfg.grades]:
        raise RBAError(
            f"补丁是在 grades={patch['grades']} 下生成的，当前配置是 {cfg.grades}。",
            "用 rba eval --run 指向生成补丁的那次分析目录，不要手改等级。",
        )
    bad_cases, good_cases, split = _validate_and_load(cfg)
    info("加载模型以验收补丁")
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, cfg.module_paths, loaded.preset)
    loaded.tied_lm_head = is_tied_lm_head(loaded.ft, resolved.lm_head(loaded.ft))
    grades = cfg.grades
    grade_ids = loaded.grade_ids
    grade_id_list = [grade_ids[grade] for grade in grades]
    _preflight_patch(patch, loaded)
    provenance_notes = []
    if run_meta and run_meta.get("fingerprint") and patch.get("run_fingerprint") and run_meta["fingerprint"] != patch["run_fingerprint"]:
        provenance_notes.append("补丁不是这次分析目录生成的（指纹不同）。")
    if run_meta and run_meta.get("identities"):
        current = run_identities(cfg, bad_cases, good_cases)
        changed = [key for key in ("ft", "base", "bad", "good", "images") if run_meta["identities"].get(key) != current.get(key)]
        if changed:
            provenance_notes.append(f"分析之后这些文件变过：{', '.join(changed)}。这不是同一次实验的回放，验收集的独立性也不再成立。")
    for note in provenance_notes:
        warn(note)
    if patch.get("method") in {"ablate_image", "ablate_tokens"} and not patch.get("token_ids"):
        image_token_id = _image_token_id(loaded)
        if image_token_id is None:
            raise RBAError("图像消融没有 token id。", "在补丁里写 token_ids，或换一个补丁。")
        patch = dict(patch)
        patch["method"] = "ablate_tokens"
        patch["token_ids"] = [int(image_token_id)]
    pad_id = _pad_id(loaded.tokenizer)
    excluded: list[ExcludedCase] = []
    bad_encoded, good_encoded = _encode_all(cfg, loaded, bad_cases, good_cases, excluded)
    if not bad_encoded:
        raise RBAError(
            "没有可验收的坏例（全部被剔除或文件为空）。",
            "检查坏例文件和图片路径；missing_image: exclude 剔除了哪些样本见上面的注意。",
        )
    holdout_ids = set((split or {}).get("holdout_ids") or [])
    holdout_encoded = [item for item in bad_encoded if item.id in holdout_ids]
    holdout_content = {_content_key(item.case) for item in holdout_encoded}
    overlap_good = [item.id for item in good_encoded if _content_key(item.case) in holdout_content]
    if overlap_good:
        good_encoded = [item for item in good_encoded if item.id not in set(overlap_good)]
        for case_id in overlap_good:
            excluded.append(ExcludedCase(id=case_id, reason="对照样本内容和验收集重复，和分析阶段一样剔除。"))
        warn(f"{len(overlap_good)} 条对照样本和验收集内容重复，已剔除（和分析阶段一致）。")
    discovery_encoded = [item for item in bad_encoded if item.id not in holdout_ids]
    batch_size = _effective_batch(cfg, loaded.device, bad_encoded + good_encoded)
    requested_scope = str(patch.get("scope") or "all_positions")
    scopes_used: set[str] = set()

    def score(items, patched: bool, desc: str):
        def infer(chunk):
            batch = _batch(loaded, chunk, pad_id)
            if patched:
                with _scoped_patch(loaded.ft, resolved, patch, grade_ids, requested_scope, batch, loaded.device) as scope:
                    rows = _score_batch(loaded, batch, grade_id_list, grades, cfg.decode_protocol)
                scopes_used.add(scope)
            else:
                rows = _score_batch(loaded, batch, grade_id_list, grades, cfg.decode_protocol)
            return [{**row, "id": item.id, "label": item.case.label, "attributes": item.case.attributes} for item, row in zip(chunk, rows)]

        if not items:
            return []
        produced, _ = _run_batches(items, batch_size, infer, desc)
        return produced

    def paired(items, label: str):
        if not items:
            return None
        before = score(items, False, f"{label}·改前")
        after = score(items, True, f"{label}·改后")
        block = compare(before, after)
        block["before"] = metric_block(before, grades)
        block["after"] = metric_block(after, grades)
        block["n_invalid_before"] = sum(1 for row in before if row["pred"] == INVALID_GRADE)
        block["n_invalid_after"] = sum(1 for row in after if row["pred"] == INVALID_GRADE)
        return block

    if holdout_encoded:
        bad_cmp = paired(holdout_encoded, "验收集")
        bad_source = "holdout"
        discovery_cmp = paired(discovery_encoded, "发现集复评")
    else:
        bad_cmp = paired(bad_encoded, "发现集复评")
        bad_source = "discovery"
        discovery_cmp = None
    good_cmp = paired(good_encoded, "对照") if good_encoded else None
    text = conclusion(bad_cmp, good_cmp)
    if bad_source == "discovery":
        text += " 这些坏例就是用来选补丁的发现集，没有独立验收集，不能据此说会泛化；要独立验收就设 holdout_fraction。"
    else:
        text += f" 验收集是按 id 哈希留出的 {len(holdout_encoded)} 条，没有参与选补丁。"
    if loaded.tied_lm_head and patch.get("method") == "lm_head_blend":
        text += " 注意：lm_head 与 embedding 共享权重，这个补丁同时改了输入表示。"
    actual_scope = scopes_used.pop() if len(scopes_used) == 1 else (sorted(scopes_used) if scopes_used else requested_scope)
    if patch.get("untested"):
        text += " 这个补丁的系数或作用范围和抽样复核时的干预不同，属于未经复核的提议。"
    if provenance_notes:
        text = " ".join(provenance_notes) + " " + text
    result = {
        "patch": patch,
        "bad": bad_cmp,
        "bad_source": bad_source,
        "discovery": discovery_cmp,
        "good": good_cmp,
        "excluded": [{"id": item.id, "reason": item.reason} for item in excluded],
        "tied_lm_head": loaded.tied_lm_head,
        "decode_protocol": cfg.decode_protocol,
        "requested_scope": requested_scope,
        "scope": actual_scope,
        "provenance_notes": provenance_notes,
        "replay_valid": not provenance_notes,
        "conclusion": text,
    }
    info(text)
    _print_eval(bad_cmp, good_cmp, bad_source)
    return to_builtin(result)


def _preflight_patch(patch: dict, loaded) -> None:
    method = patch.get("method")
    if method == "lm_head_blend" and loaded.tied_lm_head:
        raise RBAError(
            "这个模型的 lm_head 和输入 embedding 共享权重，lm_head_blend 会同时改掉该等级 token 的输入表示。",
            "本轮不把它当作「只改读出行」的补丁来验收。换一个作用在注意力头或 MLP 上的补丁；共享权重下的独立读出编辑留待后续版本。",
        )


def _print_eval(bad: dict | None, good: dict | None, source: str) -> None:
    def acc(block, key):
        value = ((block or {}).get(key) or {}).get("accuracy")
        return "—" if value is None else f"{value:.1%}"

    label = "验收集" if source == "holdout" else "发现集复评"
    if bad:
        info(
            f"{label}：准确率 {acc(bad, 'before')} → {acc(bad, 'after')}，"
            f"修复率 {_pct(bad.get('fix_rate'))}（{bad.get('n_fixed')}/{bad.get('n_wrong_before')}），"
            f"净正确数变化 {bad.get('net_correct_change', '—')}。"
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


# --------------------------------------------------------------------------- validation and loading


def _file_identity(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    stat = path.stat()
    if stat.st_size <= 64 * 1024 * 1024:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return {"size": stat.st_size, "sha256": digest.hexdigest()}
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_identity(path: str | None) -> dict | None:
    """Config/tokenizer/processor files hashed, weight files by size and mtime.
    Changing any of them at the same path invalidates the run cache."""
    if path is None:
        return None
    if path in {"tiny", "qwen3_5_tiny"}:
        return {"kind": path}
    folder = Path(path)
    if folder.is_file():
        return _file_identity(folder)
    if not folder.is_dir():
        return {"missing": str(folder)}
    files = {name: _file_identity(folder / name) for name in _IDENTITY_FILES if (folder / name).exists()}
    weights = []
    for weight in sorted(folder.glob("*.safetensors")) + sorted(folder.glob("pytorch_model*.bin")):
        stat = weight.stat()
        weights.append({"name": weight.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"dir": str(folder.resolve()), "files": files, "weights": weights}


def run_identities(cfg: Config, bad_cases: list[Case], good_cases: list[Case]) -> dict:
    images = {}
    for case in list(bad_cases) + list(good_cases):
        if case.image and case.image not in images:
            file = Path(case.image)
            images[case.image] = {"size": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns} if file.exists() else None
    return {
        "ft": model_identity(cfg.ft_model),
        "base": model_identity(cfg.base_model) if cfg.analyses.delta_w else None,
        "bad": file_fingerprint(cfg.badcases),
        "good": file_fingerprint(cfg.goodcases) if cfg.goodcases else None,
        "images": images,
    }


def _content_key(case: Case) -> tuple:
    return (case.query, case.doc, case.image or "")


def _drop_goodcase_overlap(good_cases: list[Case], holdout_cases: list[Case], excluded: list[ExcludedCase], split: dict) -> list[Case]:
    """Control rows whose content equals a holdout row are removed from the analysis
    (and recorded), so holdout content never reaches probes or prompts. If it did,
    the holdout could not be called independent of selection."""
    split["n_good_excluded_overlap"] = 0
    if not good_cases or not holdout_cases:
        return list(good_cases)
    holdout_content = {_content_key(case) for case in holdout_cases}
    kept = []
    dropped = []
    for case in good_cases:
        if _content_key(case) in holdout_content:
            dropped.append(case.id)
            excluded.append(ExcludedCase(id=case.id, reason="对照样本内容和验收集的某条坏例完全相同；为保持验收集独立，这一轮不使用它。"))
        else:
            kept.append(case)
    if dropped:
        split["n_good_excluded_overlap"] = len(dropped)
        split["good_excluded_overlap_ids"] = dropped
        warn(f"{len(dropped)} 条对照样本和验收集的内容完全相同（例如 {dropped[0]}），已从分析和对照里剔除，验收集保持独立。")
    return kept


def _validate_and_load(cfg: Config):
    if (cfg.base_model == "tiny") != (cfg.ft_model == "tiny") and cfg.model_type == "auto":
        raise RBAError(
            "tiny 要基座和微调一起用。",
            "两边都写 tiny，或都写真实目录。试跑：rba demo",
        )
    if cfg.base_model == cfg.ft_model and cfg.base_model not in {"tiny", "qwen3_5_tiny"}:
        warn("基座和微调是同一个目录，ΔW 会接近 0。若这不是有意的，检查 --base 和 --ft。")
    bad_path = cfg.resolve_path(cfg.badcases, from_cli=False)
    cfg.badcases = bad_path or cfg.badcases
    if cfg.goodcases:
        cfg.goodcases = cfg.resolve_path(cfg.goodcases, from_cli=False)
    if cfg.model_type in {"hf", "auto"} and cfg.ft_model not in {"tiny", "qwen3_5_tiny"}:
        cfg.ft_model = cfg.resolve_path(cfg.ft_model) or cfg.ft_model
        cfg.base_model = cfg.resolve_path(cfg.base_model) or cfg.base_model
        check_checkpoint_dir(cfg.ft_model, "微调模型")
        if cfg.analyses.delta_w:
            check_checkpoint_dir(cfg.base_model, "基座")
    bad = load_cases(cfg.badcases, cfg.grades, what="坏例文件")
    good = []
    if cfg.goodcases:
        good = load_cases(cfg.goodcases, cfg.grades, what="对照集")
    all_ids = [case.id for case in bad + good]
    if len(set(all_ids)) != len(all_ids):
        raise RBAError(
            "样本 id 重复（包括坏例和对照之间），会混淆探针标签与归因记录。",
            "给所有样本使用全局唯一 id，例如坏例 b001、对照 g001；检查自动生成的 id。",
        )
    missing = [case.image for case in bad + good if case.image and not Path(case.image).exists()]
    if missing:
        if cfg.missing_image == "exclude":
            warn(f"{len(missing)} 张图片路径不存在，这些样本会被剔除并写进报告。例如 {missing[0]}")
        else:
            raise RBAError(
                f"{len(missing)} 张图片路径不存在，例如 {missing[0]}。",
                "修好路径；或加 --allow-missing-images 剔除这些样本。程序不会把有图样本悄悄当成纯文本。",
            )
    split = _split_info(bad, cfg)
    info(f"坏例 {len(bad)} 条" + (f"，对照 {len(good)} 条。" if good else "。") + (f" 其中 {split['n_holdout']} 条按 id 哈希留作验收集。" if split["n_holdout"] else ""))
    return bad, good, split


def _split_info(bad: list[Case], cfg: Config) -> dict:
    fraction = float(cfg.holdout_fraction or 0.0)
    holdout = []
    if fraction > 0:
        for case in bad:
            digest = hashlib.sha256(f"{cfg.seed}:{case.id}".encode("utf-8")).digest()
            if int.from_bytes(digest[:8], "big") / 2**64 < fraction:
                holdout.append(case.id)
    holdout_ids = set(holdout)
    held_content = {_content_key(case) for case in bad if case.id in holdout_ids}
    if any(_content_key(case) in held_content for case in bad if case.id not in holdout_ids):
        raise RBAError(
            "同一坏例内容以不同 id 同时落入发现集和验收集，验收集不独立。",
            "先按 query、doc、image 去重，再运行分析和验收；不能仅靠更换 id 隔离同一输入。",
        )
    return {
        "holdout_fraction": fraction,
        "n_total": len(bad),
        "n_holdout": len(holdout),
        "n_discovery": len(bad) - len(holdout),
        "holdout_ids": holdout,
        "note": "分析只用发现集；rba eval 优先在验收集上报告。" if holdout else "没有留出验收集：rba eval 的结果是发现集复评，不代表泛化。",
    }


def _llm_mode(cfg: Config) -> str:
    if not cfg.analyses.llm or not cfg.llm.enabled:
        return "off"
    if cfg.dry_run:
        return "dry_run"
    if not os.environ.get(cfg.llm.api_key_env):
        warn(
            f"环境变量 {cfg.llm.api_key_env} 未设置，这次只把提示词写到输出目录，不调用接口（等同 --dry-run）。"
            "分析本身照常完成；设置密钥后再跑同一条命令会接着用缓存。"
        )
        return "dry_run"
    return "remote"


def _prepare_output(cfg: Config, fingerprint: str) -> tuple[Path, bool]:
    """Returns the output folder and whether an older cache schema there must be discarded.

    A cache written by an earlier rba version is not "different data": it is
    cleared with a warning instead of asking the user for --fresh."""
    if cfg.output_dir:
        output = Path(cfg.output_dir)
        if not output.is_absolute():
            output = Path.cwd() / output
    else:
        output = Path.cwd() / "runs" / fingerprint[:12]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "cache" / "manifest.json"
    used_path = output / "config.used.json"
    if manifest.exists() and used_path.exists() and not cfg.fresh:
        try:
            previous_schema = (json.loads(used_path.read_text(encoding="utf-8")).get("runtime") or {}).get("cache_schema")
        except json.JSONDecodeError:
            previous_schema = None
        if previous_schema != CACHE_SCHEMA:
            warn(f"{output} 里的缓存是旧版本 rba 写的（cache_schema {previous_schema} → {CACHE_SCHEMA}），这次重算全部阶段。")
            return output, True
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
    return output, False


def _encode_all(cfg: Config, loaded, bad_cases, good_cases, excluded: list[ExcludedCase]):
    processor = loaded.processor if loaded.kind == "hf" else None
    image_size = cfg.tiny.image_size if loaded.kind == "tiny" else 8
    bad_encoded = encode_cases(bad_cases, loaded.tokenizer, cfg, processor, image_size=image_size, excluded=excluded)
    good_encoded = encode_cases(good_cases, loaded.tokenizer, cfg, processor, image_size=image_size, excluded=excluded) if good_cases else []
    if loaded.kind == "hf" and processor is None and any(case.image for case in list(bad_cases) + list(good_cases)):
        raise RBAError(
            "有图片样本，但没有加载到这个模型的处理器，图片无法编码。",
            "确认模型目录里有 preprocessor_config.json / processor 配置；或先只分析纯文本样本。",
        )
    if loaded.kind == "qwen3_5_tiny":
        # The random real class has no processor; the stand-in's NCHW pixels cannot
        # enter its vision tower. Keep the placeholder tokens, drop the pixels, say so.
        dropped = 0
        for item in bad_encoded + good_encoded:
            if item.pixel_values is not None:
                item.pixel_values = None
                item.modalities = {"text"}
                dropped += 1
        if dropped:
            warn(f"qwen3_5_tiny 没有视觉处理器：{dropped} 条样本的图片不进入模型，只保留占位 token。这只用于检查真实类的钩子接线。")
    return bad_encoded, good_encoded


def _image_token_id(loaded):
    image_token_id = loaded.image_token_id
    if image_token_id is None:
        image_token_id = getattr(getattr(loaded.ft, "config", None), "image_token_id", None)
    return image_token_id


def _batch(loaded, chunk, pad_id: int) -> dict:
    return collate(chunk, pad_id)


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


def _effective_batch(cfg: Config, device, items: list[EncodedCase]) -> int:
    has_image = any(item.has_image for item in items)
    size = _auto_batch(cfg, device, has_image)
    if needs_single_item_batches(items):
        if size != 1:
            info("样本带处理器输出（打包的视觉张量），这一轮统一逐条前向：batch size 1。")
        return 1
    info(f"batch size {size}。显存不够会自动减半。")
    return size


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


# --------------------------------------------------------------------------- batching with OOM handling


def _run_batches(items, batch_size, fn, desc: str):
    if not items:
        return [], batch_size
    size = max(1, batch_size)
    while True:
        try:
            outputs = []
            chunks = list(batched(items, size))
            for chunk in tqdm(chunks, desc=desc, leave=False, disable=not sys.stderr.isatty()):
                outputs.extend(fn(chunk))
            return outputs, size
        except RuntimeError as exc:
            if not oom_message(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if size > 1:
                size = max(1, size // 2)
                warn(f"显存不足，batch 降为 {size} 后重试这一阶段。")
                continue
            raise RBAError(
                f"阶段「{desc}」在 batch size 1 时仍然显存不足。",
                "可按顺序试：把 max_length 调小；把 image_max_pixels 调小（图像 token 变少）；"
                "token 归因用 token_attr_method: gradxinput 或在 analyses 里关掉 token_attr；"
                "关掉 force_eager_attn；最后才考虑 device: cpu（很慢）。已完成的阶段在缓存里，重跑会接着用。",
            ) from exc


# --------------------------------------------------------------------------- scoring


def _score_batch(loaded, batch: dict, grade_id_list, grades, decode_protocol: str) -> list[dict]:
    """Rows with pred / grade_logits / full_top_id / legal / grade_mass. Uses Astra's scorer when present."""
    forward_scores = getattr(_evaluate, "forward_scores", None)
    score_rows = getattr(_evaluate, "score_rows", None)
    if callable(forward_scores) and callable(score_rows):
        scores = forward_scores(loaded.ft, batch, grade_id_list, loaded.device, loaded.dtype)
        return score_rows(scores, grades, decode_protocol)
    # Fallback for an older evaluate module: compute the same fields here.
    from rba.util import call_model, logits_of, move_batch

    moved = move_batch({k: v for k, v in batch.items() if k != "case_ids"}, loaded.device, loaded.dtype)
    with torch.no_grad():
        logits = logits_of(call_model(loaded.ft, moved))
    index = decision_index(moved["attention_mask"])
    taken = logits[torch.arange(logits.size(0), device=logits.device), index].float()
    columns = torch.tensor(grade_id_list, device=taken.device, dtype=torch.long)
    grade_logits = taken.index_select(-1, columns)
    top = taken.argmax(-1)
    mass = (grade_logits - taken.logsumexp(-1, keepdim=True)).exp().sum(-1)
    rows = []
    for i in range(taken.size(0)):
        legal = bool(torch.isin(top[i], columns))
        pred = grades[int(grade_logits[i].argmax())]
        if decode_protocol == "free":
            pred = grades[grade_id_list.index(int(top[i]))] if legal else INVALID_GRADE
        rows.append(
            {
                "pred": pred,
                "grade_logits": grade_logits[i].cpu().tolist(),
                "full_top_id": int(top[i]),
                "full_top_is_grade": legal,
                "legal": legal,
                "grade_mass": float(mass[i]),
                "decode_protocol": decode_protocol,
            }
        )
    return rows


def _infer_rows(loaded, items: list[EncodedCase], grade_id_list, grades, pad_id, batch_size, decode_protocol: str, desc: str):
    def fn(chunk):
        batch = _batch(loaded, chunk, pad_id)
        scored = _score_batch(loaded, batch, grade_id_list, grades, decode_protocol)
        rows = []
        for item, row in zip(chunk, scored):
            grade_logits = torch.tensor(row["grade_logits"], dtype=torch.float32)
            probs = torch.softmax(grade_logits, dim=-1)
            rows.append(
                {
                    **row,
                    "id": item.id,
                    "label": item.case.label,
                    "probs": {grade: float(probs[index]) for index, grade in enumerate(grades)},
                    "file_prediction": item.case.prediction,
                    "query": item.case.query,
                    "doc": item.case.doc,
                    "has_image": item.has_image,
                    "n_image_tokens": item.n_image_tokens,
                    "protocol": item.protocol,
                    "attributes": item.case.attributes,
                    "correct": row["pred"] == item.case.label,
                }
            )
        return rows

    rows, _ = _run_batches(items, batch_size, fn, desc)
    return rows


def _cached_infer(cache, stage, loaded, items, grade_id_list, grades, pad_id, batch_size, decode_protocol, desc):
    if cache.has(stage):
        info(f"从缓存恢复：{desc}")
        return cache.load_json(stage)
    rows = _infer_rows(loaded, items, grade_id_list, grades, pad_id, batch_size, decode_protocol, desc)
    cache.save_json(stage, rows)
    return rows


def _split_invalid(rows: list[dict], tokenizer):
    valid = []
    invalid = []
    for row in rows:
        if row.get("pred") == INVALID_GRADE:
            invalid.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "full_top_id": row.get("full_top_id"),
                    "full_top_token": _display_token(tokenizer, row.get("full_top_id")),
                    "grade_mass": row.get("grade_mass"),
                    "constrained_pred": _constrained_pred(row),
                }
            )
        else:
            valid.append(row)
    return valid, invalid


def _constrained_pred(row: dict):
    logits = row.get("grade_logits")
    probs = row.get("probs") or {}
    if probs:
        return max(probs, key=probs.get)
    if logits:
        return int(max(range(len(logits)), key=lambda i: logits[i]))
    return None


def _display_token(tokenizer, token_id) -> str | None:
    if token_id is None:
        return None
    decode = getattr(tokenizer, "decode", None)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    raw = None
    if callable(convert):
        try:
            raw = str(convert(int(token_id)))
        except Exception:
            raw = None
    text = None
    if callable(decode):
        try:
            text = str(decode([int(token_id)]))
        except Exception:
            text = None
    if text and "�" not in text:
        shown = text
    else:
        shown = raw or ""
    shown = shown.replace("\n", "\\n")
    return f"{shown!r} (id {int(token_id)})"


# --------------------------------------------------------------------------- mechanism stages


def _detail_items(encoded, detail_ids) -> list[EncodedCase]:
    wanted = {case_id for ids in detail_ids.values() for case_id in ids}
    return [item for item in encoded if item.id in wanted]


def _cached_mech(cache, loaded, resolved, encoded, by_id, detail_ids, grade_id_list, grades, image_token_id, pad_id, batch_size, delta):
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
    accepts_delta = _accepts(instrument_batch, "delta")

    def fn(chunk):
        batch = _batch(loaded, chunk, pad_id)
        pred = torch.tensor([id_to_pred[item.id] for item in chunk], dtype=torch.long)
        gold = torch.tensor([id_to_gold[item.id] for item in chunk], dtype=torch.long)
        kwargs = {"delta": delta} if accepts_delta else {}
        records = instrument_batch(loaded.ft, batch, resolved, grade_id_list, pred, gold, image_token_id, loaded.device, loaded.dtype, **kwargs)
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
            loaded.ft, batch, resolved, pred, gold, image_token_id, loaded.tokenizer, loaded.device, loaded.dtype, cfg.token_attr_method, cfg.ig_steps
        )

    size = 1 if cfg.token_attr_method == "integrated_gradients" else min(2, batch_size)
    rows, _ = _run_batches(items, size, fn, "token 归因")
    cache.save_json("token_attr", rows)
    return {row["id"]: row for row in rows}


def _stratified(bad: list[EncodedCase], good: list[EncodedCase], limit: int) -> list[EncodedCase]:
    """Interleave bad and good rows so a small budget still covers both sets."""
    chosen: list[EncodedCase] = []
    i = j = 0
    while len(chosen) < limit and (i < len(bad) or j < len(good)):
        if i < len(bad):
            chosen.append(bad[i])
            i += 1
        if len(chosen) < limit and j < len(good):
            chosen.append(good[j])
            j += 1
    return chosen


def _cached_probes(cache, cfg, loaded, resolved, bad_encoded, good_encoded, rows_by_id, pad_id, batch_size):
    if cache.has("probes"):
        info("从缓存恢复：探针")
        return cache.load_json("probes")
    chosen = _stratified(bad_encoded, good_encoded, cfg.probe_max_cases)
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
        source = rows_by_id.get(row["id"])
        if source is None:
            item = next(item for item in chosen if item.id == row["id"])
            source = {"attributes": item.case.attributes, "label": item.case.label, "pred": None}
        records.append({"attributes": source.get("attributes") or {}, "label": source.get("label"), "pred": source.get("pred")})
    payload = probe_attributes(records, hidden, cfg.probe_attributes)
    payload["n_bad"] = sum(1 for item in chosen if any(item.id == b.id for b in bad_encoded))
    payload["n_good"] = len(chosen) - payload["n_bad"]
    cache.save_json("probes", payload)
    return payload


# --------------------------------------------------------------------------- verification with decision-position scope


def _verify(cache, cfg, loaded, resolved, encoded, by_id, groups, detail_ids, verify_ids, mech_by_id, grade_ids, grades, image_token_id, pad_id, delta):
    if cache.has("verify"):
        info("从缓存恢复：干预复核")
        return cache.load_json("verify")
    encoded_by_id = {item.id: item for item in encoded}
    results: dict = {}
    jobs = []
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
        if _accepts(propose_checks, "controls"):
            checks = propose_checks(summary, mech_summary, delta, cfg.verify_top_k, controls=cfg.verify_controls)
        else:
            checks = propose_checks(summary, mech_summary, delta, cfg.verify_top_k)
        disjoint_ids = verify_ids.get(key) or []
        target_ids = disjoint_ids if disjoint_ids else detail_ids.get(key, [])
        rows = [by_id[i] for i in target_ids]
        for check in checks:
            jobs.append((key, check, [encoded_by_id[row["id"]] for row in rows], bool(disjoint_ids)))
    info(f"干预复核 {len(jobs)} 次（含对照）。每次只动一个组件。")
    id_list = [grade_ids[grade] for grade in grades]
    baseline_cache: dict[str, torch.Tensor] = {}
    for key, check, items, disjoint in tqdm(jobs, desc="干预复核", leave=False, disable=not sys.stderr.isatty()):
        patch = dict(check["patch"])
        if patch.get("method") == "ablate_image":
            if image_token_id is None:
                continue
            patch = {"method": "ablate_tokens", "token_ids": [int(image_token_id)]}
        record = {
            "key": check["key"],
            "target_key": check.get("target_key", check["key"]),
            "control_type": check.get("control_type", "target"),
            "requested_scope": check.get("scope", "decision"),
            "diagnostic_only": bool(check.get("diagnostic_only", False)),
            "selection_disjoint": disjoint,
            "patch": patch,
        }
        if patch.get("method") == "lm_head_blend" and loaded.tied_lm_head:
            record.update({"scope": "readout", "skipped": "lm_head 与 embedding 共享权重，行编辑不是独立的读出干预，本轮不执行。", "mean_delta": None, "n": 0, "n_improved": 0, "agreement": None, "per_case": []})
            results.setdefault(key, []).append(record)
            continue
        deltas = []
        per_case = []
        improved = 0
        actual_scope = None
        for item in items:
            row = by_id[item.id]
            solo = _batch(loaded, [item], pad_id)
            pred_index = grades.index(row["pred"])
            gold_index = grades.index(row["label"])
            if item.id not in baseline_cache:
                baseline_cache[item.id] = _grade_logits(loaded, solo, id_list)
            before = baseline_cache[item.id]
            before_diff = float(before[pred_index] - before[gold_index])
            with _scoped_patch(loaded.ft, resolved, patch, grade_ids, check.get("scope", "decision"), solo, loaded.device) as scope:
                after = _grade_logits(loaded, solo, id_list)
            actual_scope = scope
            after_diff = float(after[pred_index] - after[gold_index])
            change = after_diff - before_diff
            deltas.append(change)
            per_case.append({"id": item.id, "before_diff": before_diff, "after_diff": after_diff, "delta": change})
            if change < -1e-5:
                improved += 1
        n = len(deltas)
        record.update(
            {
                "scope": actual_scope or record["requested_scope"],
                "mean_delta": (sum(deltas) / n) if n else None,
                "n": n,
                "n_improved": improved,
                "agreement": (improved / n) if n else None,
                "per_case": per_case,
            }
        )
        results.setdefault(key, []).append(record)
    cache.save_json("verify", results)
    return results


def _grade_logits(loaded, batch, id_list) -> torch.Tensor:
    forward_scores = getattr(_evaluate, "forward_scores", None)
    if callable(forward_scores):
        return forward_scores(loaded.ft, batch, id_list, loaded.device, loaded.dtype)["grade_logits"][0]
    return _evaluate.forward_grade_logits(loaded.ft, batch, id_list, loaded.device, loaded.dtype)[0]


@contextmanager
def _scoped_patch(model, resolved, patch: dict, grade_ids, requested_scope: str, batch: dict, device):
    """Apply a check. Decision-position scope is implemented here with hooks; anything
    else falls back to apply_patch and reports the scope that was really used."""
    method = patch.get("method")
    if requested_scope == "decision" and method in {"head_scale", "mlp_zero", "steering"}:
        mask = batch["attention_mask"]
        index = decision_index(mask if torch.is_tensor(mask) else torch.tensor(mask)).to(device)
        with _decision_position_patch(model, resolved, patch, grade_ids, index):
            yield "decision"
        return
    if method == "lm_head_blend":
        actual = "readout"
    elif method in {"ablate_tokens", "ablate_image"}:
        actual = "token_positions"
    else:
        actual = "all_positions"
    with apply_patch(model, resolved, patch, grade_ids):
        yield actual


@contextmanager
def _decision_position_patch(model, resolved, patch: dict, grade_ids, index: torch.Tensor):
    method = patch["method"]
    layers = resolved.layers(model)
    layer_list = [int(v) for v in (patch.get("layers") or [patch.get("layer", len(layers) - 1)])]
    for value in layer_list:
        if value < 0 or value >= len(layers):
            raise RBAError(f"层号 {value} 超出范围 0..{len(layers) - 1}。", "补丁里的 layer 改成报告里列出的层。")
    layer_index = layer_list[0]
    layer = layers[layer_index]
    handles = []
    rows = torch.arange(index.numel(), device=index.device)

    def _check_shape(hidden: torch.Tensor, where: str) -> None:
        if hidden.dim() != 3 or hidden.size(0) != index.numel():
            raise RBAError(
                f"决策位置干预遇到 {where} 的形状 {tuple(hidden.shape)}，不是 [batch, seq, hidden]。",
                "这个模块的输入布局不支持按位置干预；用 scope: all_positions 重跑，或反馈模型类名。",
            )
    if method == "head_scale":
        found = attention_module(layer)
        proj_info = output_projection(found[1]) if found is not None else None
        if proj_info is None:
            raise RBAError(f"第 {layer_index} 层的注意力输出没法按头切开。", "这一层不做 head_scale 复核。")
        proj, head_dim, n_heads = proj_info
        # Same semantics as interventions.scale_heads: a head listed twice is scaled once.
        heads = sorted({int(h) for h in patch["heads"]})
        bad = [h for h in heads if h < 0 or h >= n_heads]
        if bad:
            raise RBAError(f"第 {layer_index} 层没有头 {bad}（共 {n_heads} 个头）。", "头的编号从 0 开始。")
        scale = float(patch.get("scale", 0.0))

        def pre_hook(_module, inputs):
            hidden = inputs[0]
            _check_shape(hidden, "注意力输出投影输入")
            edited = hidden.clone()
            idx = index.to(hidden.device)
            for head in heads:
                sl = slice(head * head_dim, (head + 1) * head_dim)
                edited[rows.to(hidden.device), idx, sl] = edited[rows.to(hidden.device), idx, sl] * scale
            return (edited,) + tuple(inputs[1:])

        handles.append(proj.register_forward_pre_hook(pre_hook))
    elif method == "mlp_zero":
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise RBAError(f"第 {layer_index} 层没有 MLP。", "换一条作用在注意力头上的补丁。")

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            _check_shape(hidden, "MLP 输出")
            edited = hidden.clone()
            edited[rows.to(hidden.device), index.to(hidden.device)] = 0
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

        handles.append(mlp.register_forward_hook(hook))
    elif method == "steering":
        from rba.repair.interventions import steering_vector

        if patch.get("vector_source", "lm_head_diff") == "lm_head_diff":
            vector = steering_vector(model, resolved, grade_ids[str(patch["from_grade"])], grade_ids[str(patch["to_grade"])])
        else:
            vector = torch.tensor(patch["vector"], dtype=torch.float32)
        alpha = float(patch.get("alpha", 1.0))

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            _check_shape(hidden, "层输出")
            edited = hidden.clone()
            edited[rows.to(hidden.device), index.to(hidden.device)] += alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

        for value in layer_list:
            handles.append(layers[value].register_forward_hook(hook))
    else:
        raise RBAError(f"决策位置干预不支持 {method}。", "这是程序问题。")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


# --------------------------------------------------------------------------- ΔW


def _cached_delta(cache, cfg, loaded, grade_ids, resolved=None):
    if cache.has("delta"):
        info("从缓存恢复：ΔW")
        return cache.load_json("delta")
    alignment = check_base_grade_alignment(cfg, loaded, grade_ids)
    info("逐张量计算 ΔW。大矩阵只对排在前面的做有界低秩 SVD。")
    head = resolved.lm_head(loaded.ft) if resolved is not None else loaded.ft.lm_head
    weight = head.weight.detach()
    payload = analyze_delta(
        loaded.base, loaded.ft, loaded.base_path, weight, grade_ids, loaded.tokenizer, cfg.delta_w_top_k, cfg.svd_rank, cfg.svd_max_dim
    )
    if payload.get("n_missing"):
        warn(f"有 {payload['n_missing']} 个参数没能和基座对齐。前几个：{payload['missing'][:3]}")
    if payload.get("tied_lm_head") is None:
        payload["tied_lm_head"] = loaded.tied_lm_head
    payload["base_grade_alignment"] = alignment
    if alignment["status"] != "aligned":
        # Fail closed: without a verified base tokenizer the grade rows of the base
        # head cannot be matched to ours, so no row-level evidence is produced.
        grades_block = payload.get("lm_head_grades") or {}
        grades_block["rows"] = {grade: {k: v for k, v in row.items() if k == "ft_norm"} for grade, row in (grades_block.get("rows") or {}).items()}
        grades_block["delta_row_cosine"] = {}
        grades_block["missing_reason"] = alignment["reason"]
        payload["lm_head_grades"] = grades_block
        _sanitize_unaligned_directions(payload, alignment["reason"])
        warn(f"ΔW 的 lm_head 等级行证据已关闭：{alignment['reason']}")
    cache.save_json("delta", payload)
    return payload


def _sanitize_unaligned_directions(payload: dict, reason: str) -> None:
    """Without a verified base vocabulary, a vocab-space ΔW direction (lm_head rows) has
    no token meaning: its grade/top-token readings are removed. Residual-space
    projections only use the fine-tuned head and are kept, but marked."""
    for row in payload.get("top") or []:
        svd = row.get("svd")
        if not isinstance(svd, dict):
            continue
        svd["base_alignment"] = "unknown"
        svd["base_alignment_reason"] = reason
        for direction in svd.get("directions") or []:
            direction["base_alignment"] = "unknown"
            if direction.get("space") == "vocab":
                direction.pop("grades", None)
                direction.pop("top_tokens", None)
                direction["semantics"] = "unavailable: 基座词表对齐未知，按 token 读取这个方向没有意义"
            elif direction.get("space") == "residual":
                direction["semantics"] = "fine-tuned head only: 投影只用了微调侧 lm_head，和基座无关"


def check_base_grade_alignment(cfg: Config, loaded, grade_ids: dict) -> dict:
    """Base tokenizer must reproduce the FT grade encodings. Anything short of a
    verified match disables grade-row evidence instead of silently comparing rows."""
    if loaded.kind != "hf" or not loaded.base_path or loaded.base_path in {"tiny", "qwen3_5_tiny"}:
        return {"status": "aligned", "reason": None, "checked": "builtin"}
    try:
        from transformers import AutoTokenizer

        base_tok = AutoTokenizer.from_pretrained(loaded.base_path)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"基座分词器没能加载（{str(exc)[:200]}），等级行无法对齐。", "checked": None}
    mismatched = {}
    for grade, token_id in grade_ids.items():
        try:
            base_encoded = [int(v) for v in base_tok.encode(grade, add_special_tokens=False)]
            ft_encoded = [int(v) for v in loaded.tokenizer.encode(grade, add_special_tokens=False)]
        except Exception as exc:
            mismatched[grade] = {"error": str(exc)[:120]}
            continue
        if cfg.grade_mode == "single_token":
            ok = base_encoded == ft_encoded == [int(token_id)]
        else:
            ok = bool(base_encoded) and base_encoded[0] == int(token_id) and ft_encoded[:1] == base_encoded[:1]
        if not ok:
            mismatched[grade] = {"base": base_encoded, "ft": ft_encoded, "expected": int(token_id)}
    if mismatched:
        raise RBAError(
            f"基座和微调的分词器对等级 token 的编码不同：{mismatched}。",
            "ΔW 的 lm_head 等级行无法逐行对应。确认 --base 是这次微调真正的起点；或在 analyses 里关掉 delta_w。",
        )
    # Matching five label IDs says nothing about the other vocabulary rows.
    # A full-head SVD mixes all rows, so its token semantics require the entire
    # token -> ID mapping (including added tokens) to match.
    try:
        base_vocab = base_tok.get_vocab()
        ft_vocab = loaded.tokenizer.get_vocab()
    except (AttributeError, TypeError, NotImplementedError):
        return {"status": "unavailable", "reason": "等级编码相同，但无法验证完整词表 ID 对齐，关闭读出 ΔW 语义。", "checked": cfg.grade_mode}
    if not base_vocab or base_vocab != ft_vocab:
        return {"status": "unavailable", "reason": "等级编码相同，但基座和微调完整词表的 token→ID 映射不同，关闭读出 ΔW 语义。", "checked": cfg.grade_mode}
    return {"status": "aligned", "reason": None, "checked": "full_vocabulary"}


# --------------------------------------------------------------------------- hypotheses, prompts, llm


def _accepts(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _rank(summary, mech_summary, delta, verifications, n_parameters, token_summary):
    if _accepts(rank_hypotheses, "token_attr"):
        return rank_hypotheses(summary, mech_summary, delta, verifications, n_parameters, token_attr=token_summary)
    return rank_hypotheses(summary, mech_summary, delta, verifications, n_parameters)


def _build_prompt(cfg, architecture, grades, grade_ids, summary, mech_summary, hypotheses, delta, token_summary, probes, verifications):
    kwargs = dict(
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
    extra = {
        "token_attr": token_summary,
        "probes": probes,
        "verifications": verifications,
        "grade_mode": cfg.grade_mode,
        "decode_protocol": cfg.decode_protocol,
        "delta_report": delta,
        "language": getattr(cfg, "language", "zh"),
    }
    for name, value in extra.items():
        if _accepts(build_group_prompt, name):
            kwargs[name] = value
    return build_group_prompt(**kwargs)


def _call_llm(cfg: Config, prompts: list[dict], prompt_dir: Path, llm_mode: str):
    results = []
    failures = []
    if not prompts:
        return results, failures
    if llm_mode != "remote":
        info(f"写了 {len(prompts)} 份提示词到 {prompt_dir}，没有调用接口。")
        return results, failures
    info(f"调用 {cfg.llm.model}，共 {len(prompts)} 组。")
    for prompt in prompts:
        safe = _safe_name(prompt["group"])
        try:
            raw = chat(cfg.llm.base_url, cfg.llm.model, cfg.llm.api_key_env, prompt["text"], cfg.llm.temperature, cfg.llm.timeout_s, getattr(cfg, "language", "zh"))
        except RBAError as exc:
            warn(f"{prompt['group']} 的大模型调用失败：{exc.message}")
            failures.append({"group": prompt["group"], "error": exc.message, "hint": exc.hint, "prompt_file": f"prompts/{safe}.txt"})
            (prompt_dir / f"{safe}.error.txt").write_text(str(exc), encoding="utf-8")
            continue
        parsed = extract_json(raw)
        if parsed is None:
            warn(f"{prompt['group']} 的回复里没有合法 JSON，原文已保存。")
        (prompt_dir / f"{safe}.reply.txt").write_text(raw, encoding="utf-8")
        results.append({"group": prompt["group"], "parsed": parsed, "raw": raw[:8000], "evidence_ids": prompt.get("prompt_evidence_ids")})
    return results, failures


def _safe_name(group: str) -> str:
    return group.replace("->", "_to_").replace("/", "_").replace(INVALID_GRADE, "invalid")


def _mean_attr(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    def avg(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    out = {"method": rows[0].get("method"), "n": len(rows)}
    for key in ("text_abs_share", "image_abs_share", "text_signed", "image_signed", "completeness_gap"):
        if any(key in row for row in rows):
            out[key] = avg(key)
    if any("reliable" in row or "attribution_reliable" in row for row in rows):
        reliable = all(bool(row.get("attribution_reliable", row.get("reliable", True))) for row in rows)
        out["attribution_reliable"] = reliable
        out["reliable"] = reliable
    return out


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
            touched = len({int(h) for h in patch["heads"]}) * head_dim * proj.weight.shape[0]
            item["weight_fraction"] = touched / n_parameters
        elif patch.get("method") == "lm_head_blend":
            item["weight_fraction"] = resolved.lm_head(model).weight.shape[1] / n_parameters
        elif patch.get("method") in {"mlp_zero", "ablate_image", "ablate_tokens"}:
            item["weight_fraction"] = 0.0


def _mark_recommended(hypotheses: list[dict], tied_lm_head: bool) -> None:
    """Only hypotheses whose intervention worked on the sample get an eval command.

    If nothing reached 「消融有效」, the single best-ranked component that still
    carries a patch is offered as an *exploratory* check, labeled as such.
    Diagnostic-only readout edits (lm_head row mixing) are never offered, and
    neither are rejected or protocol-level hypotheses.
    """
    for item in hypotheses:
        item["recommend"] = False
        item["exploratory"] = False

    def diagnostic(item) -> bool:
        patch = item.get("patch") or {}
        return bool(item.get("diagnostic_only")) or str(item.get("key", "")).startswith("lm_head:") or patch.get("method") == "lm_head_blend"

    def has_patch(item) -> bool:
        patch = item.get("patch")
        if not patch or diagnostic(item):
            if patch and tied_lm_head and patch.get("method") == "lm_head_blend":
                item["tied_note"] = "lm_head 与 embedding 共享权重，这个补丁会同时改输入表示，不作为下一步。"
            return False
        return True

    effective = [item for item in hypotheses if item.get("status") in VERIFIED_STATUSES and has_patch(item)]
    for item in effective[:2]:
        item["recommend"] = True
    if effective:
        return
    for item in hypotheses:
        if has_patch(item) and item.get("status") not in {"被复核否定", "先处理这个"}:
            item["recommend"] = True
            item["exploratory"] = True
            item["exploratory_note"] = (
                "这条干预还没有被抽样复核支持（状态 "
                + str(item.get("status"))
                + "）。下面的命令只是探索性验收，结果不能当作机制证据。"
            )
            break


def _attach_commands(hypotheses, output: Path, group_key: str) -> None:
    for index, item in enumerate(hypotheses, start=1):
        if not item.get("patch") or not item.get("recommend"):
            item["command"] = ""
            item["patch_file"] = None
            continue
        filename = f"patches/{_safe_name(group_key)}_h{index}.json"
        item["patch_file"] = filename
        item["command"] = f"rba eval --run {shlex.quote(str(output))} --patch {shlex.quote(str(output / filename))}"


def _write_patches(output: Path, groups: list[dict], cfg: Config, image_token_id, fingerprint: str, tied: bool) -> None:
    """Patch files carry the intervention that was actually verified (method, coefficient,
    scope). A hypothesis proposal that differs is kept alongside as `proposal_patch` and
    the file is marked `untested`. Only files listed in our own manifest are replaced;
    user-authored files in patches/ are never touched."""
    folder = output / "patches"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / PATCH_MANIFEST
    previous = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
        except (json.JSONDecodeError, AttributeError):
            previous = []
    for name in previous:
        candidate = folder / name
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
    written = []
    for group in groups:
        verified = {
            v.get("target_key", v.get("key")): v
            for v in (group.get("verifications") or [])
            if v.get("control_type", "target") == "target" and not v.get("skipped") and v.get("mean_delta") is not None
        }
        for item in group.get("hypotheses") or []:
            rel = item.get("patch_file")
            proposal = item.get("patch")
            if not rel or not proposal:
                continue
            record = verified.get(item.get("key"))
            if record is not None and record.get("patch"):
                payload = dict(record["patch"])
                payload["scope"] = record.get("scope", "all_positions")
                payload["verified_on_sample"] = True
                payload["untested"] = False
                if _normalized(record["patch"]) != _normalized(proposal):
                    payload["proposal_patch"] = dict(proposal)
                    payload["proposal_note"] = "假设里给的是更温和的提议（例如 scale 0.5）；这里验收的是抽样复核时真正用过的干预。"
            else:
                payload = dict(proposal)
                default_scope = "all_positions" if payload.get("method") in {"head_scale", "mlp_zero", "steering"} else ("readout" if payload.get("method") == "lm_head_blend" else "token_positions")
                payload.setdefault("scope", default_scope)
                payload["verified_on_sample"] = False
                payload["untested"] = True
            if payload.get("method") == "ablate_image" and image_token_id is not None:
                payload["method"] = "ablate_tokens"
                payload["token_ids"] = [int(image_token_id)]
            payload.update(
                {
                    "comment": item["title"],
                    "group": group["key"],
                    "status": item["status"],
                    "exploratory": bool(item.get("exploratory")),
                    "tiny_seed": cfg.seed,
                    "plant_error": cfg.plant_error,
                    "grades": cfg.grades,
                    "run_fingerprint": fingerprint,
                    "tied_lm_head": tied,
                }
            )
            path = output / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(path.name)
    manifest_path.write_text(json.dumps({"files": written}, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalized(patch: dict) -> dict:
    out = {}
    for key, value in patch.items():
        if key in {"scope", "verified_on_sample", "untested", "proposal_patch", "proposal_note"}:
            continue
        if isinstance(value, float):
            value = round(value, 6)
        if isinstance(value, list):
            value = [int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for v in value]
        out[key] = value
    return out


# --------------------------------------------------------------------------- report helpers


def _overview(bad_rows, good_rows, invalid_bad, invalid_good, excluded) -> dict:
    bad_metrics = metric_block(bad_rows, [])
    good_metrics = metric_block(good_rows, []) if good_rows else {"n": 0, "accuracy": None, "mae": None, "n_correct": 0}
    return {
        "n_bad": bad_metrics["n"],
        "n_bad_correct": bad_metrics["n_correct"],
        "n_bad_wrong": bad_metrics["n"] - bad_metrics["n_correct"],
        "bad_accuracy": bad_metrics["accuracy"],
        "bad_mae": bad_metrics["mae"],
        "bad_legal_rate": bad_metrics.get("legal_output_rate"),
        "n_good": good_metrics["n"],
        "good_accuracy": good_metrics["accuracy"],
        "good_mae": good_metrics["mae"],
        "good_legal_rate": good_metrics.get("legal_output_rate"),
        "n_invalid_bad": len(invalid_bad),
        "n_invalid_good": len(invalid_good),
        "n_excluded": len(excluded),
    }


def _banner(loaded) -> str | None:
    if loaded.kind not in {"tiny", "qwen3_5_tiny"}:
        return None
    plant = loaded.plant or {}
    return (
        "这次跑的是内置小模型，用来确认工具链。"
        + (f" 人为改动：{plant.get('what')}。" if plant else " 没有植入错误。")
        + " 随机初始化的小模型几乎不把概率放在等级词上，所以每组都会先出现「先处理这个：下一个 token 的概率大多不在等级词上」。"
        "这是检查在正确工作，不是植入的错误；植入的组件在它后面的假设里。报告里的数字不是线上相关性模型的结论。"
    )


def _lead(groups: list[dict], invalid_rows: list[dict], n_valid: int | None = None) -> str:
    analyzed = [group for group in groups if group.get("hypotheses")]
    prefix = ""
    if invalid_rows:
        prefix = f"有 {len(invalid_rows)} 条坏例的全词表 argmax 不是等级词（见「非法输出」），它们没有进入按等级索引的归因。"
    if invalid_rows and n_valid == 0:
        return (
            prefix
            + "没有任何坏例落在等级词上，这一轮没有按等级的分析可做。先核对模板、等级 token 协议和解码协议是否和训练、线上一致。"
        )
    if not analyzed:
        if not groups:
            return prefix + "落在等级词上的坏例里，等级 argmax 都和 label 一致。先确认文件里是不是真的错例，以及模板、等级 token 是否和训练时一致。"
        return prefix + "有判错的组，但这轮没有逐层证据（可能关了分析，或组没有被抽到）。打开 dla 和 verify 再跑一次。"
    effective = sum(1 for group in analyzed for item in group["hypotheses"] if item["status"] in VERIFIED_STATUSES)
    return (
        prefix
        + f"下面按混淆组列出假设。{effective} 条在该组抽样上的干预让错误 logit 下降（「消融有效」）。"
        "这不是根因证明，也不是对照集：每条命令会在验收集（或发现集）和对照集上重跑，修复率和回退率是两个条件率，不能直接相减当净收益。"
    )


def _print_lead(groups: list[dict]) -> None:
    for group in groups:
        hyps = group.get("hypotheses") or []
        if not hyps:
            continue
        top = hyps[0]
        info(f"{group['key']}（{group['count']}）优先看：[{top['status']}] {top['title']}")


def _dump_used_config(cfg: Config, output: Path, loaded, split: dict, excluded: list[ExcludedCase], fingerprint: str, identities: dict) -> None:
    payload = config_to_dict(cfg)
    payload["runtime"] = {
        "rba_version": __version__,
        "cache_schema": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "identities": identities,
        "device_resolved": str(loaded.device),
        "dtype_resolved": str(loaded.dtype).replace("torch.", ""),
        "identity": loaded.identity,
        "ft_fingerprint": file_fingerprint(cfg.ft_model),
        "base_fingerprint": file_fingerprint(cfg.base_model),
        "grade_ids": loaded.grade_ids,
        "image_token_id": _image_token_id(loaded),
        "tied_lm_head": loaded.tied_lm_head,
        "split": split,
        "excluded": [{"id": item.id, "reason": item.reason} for item in excluded],
    }
    (output / "config.used.json").write_text(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")
