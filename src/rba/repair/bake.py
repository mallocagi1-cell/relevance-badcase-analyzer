"""Bake validated patches into weights and export a standard HF checkpoint.

Every patch method is mapped to an exact weight edit or refused:

- head_scale   -> scale the input columns of o_proj (full attention) / out_proj
                  (GatedDeltaNet) that belong to those heads. Exact for all positions.
- mlp_zero     -> zero mlp.down_proj (weight and bias). Exact for all positions.
- lm_head_blend-> rewrite one lm_head row. With tied embeddings this would also change
                  the input embedding, so it is refused unless --untie-embeddings.
- steering     -> a constant added to a layer's output equals a bias on the last module
                  that writes the residual stream (mlp.down_proj). Only possible when that
                  module already has a bias; otherwise refused (adding a bias tensor the
                  model class does not know would be dropped on load).
- ablate_*     -> depends on which tokens are in the input; no weight edit does that.

A weight edit changes every position. A patch validated with scope "decision" (only the
last position) is therefore not exactly bakeable; export refuses it unless
--allow-all-positions, and then the equivalence check compares against the hook applied
at all positions, not against what was validated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from rba import __version__
from rba.console import info, reset, warn
from rba.errors import RBAError
from rba.models.paths import attention_module, output_projection
from rba.repair.common import TOLERANCE, check_output_dir, free, require_hf, save_checkpoint, score_items
from rba.repair.evaluate import INVALID_GRADE, compare, metric_block
from rba.repair.interventions import _layer, steering_vector
from rba.util import file_fingerprint, to_builtin

BAKEABLE = {"head_scale", "mlp_zero", "lm_head_blend", "steering"}
NOT_BAKEABLE = {"ablate_tokens", "ablate_image"}
REJECTED_STATUSES = {"被复核否定", "消融无效果"}
POST_MLP_NORMS = ("post_feedforward_layernorm", "post_mlp_layernorm")


def patch_sha(patch: dict) -> str:
    return hashlib.sha256(json.dumps(patch, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def check_patch(name: str, patch: dict, *, force: bool, allow_all_positions: bool, tied: bool, untie: bool) -> list[str]:
    """Raise RBAError for patches that cannot or should not be baked. Returns notes."""
    method = patch.get("method")
    notes: list[str] = []
    if method in NOT_BAKEABLE:
        raise RBAError(
            f"{name}：{method} 不能烘焙进权重。它把输入里特定 token（如图像 token）的表示清零，"
            "取决于每条输入的内容，没有一个固定的权重修改与它等价。",
            "这类补丁只能在推理时用钩子施加；或用 rba finetune 做针对性训练。",
        )
    if method not in BAKEABLE:
        raise RBAError(f"{name}：不认识的补丁方法 {method!r}。", "可烘焙的只有 head_scale、mlp_zero、lm_head_blend、steering。")
    unverified = bool(patch.get("untested")) or patch.get("verified_on_sample") is False
    if unverified and not force:
        raise RBAError(
            f"{name}：这个补丁没有按原样复核过（untested / verified_on_sample=false）。",
            "先 rba eval 看修复率和回退率；确认要导出再加 --force。",
        )
    if unverified:
        notes.append(f"{name}：未经复核，用 --force 强制导出。")
    if patch.get("status") in REJECTED_STATUSES and not force:
        raise RBAError(
            f"{name}：分析结论是「{patch.get('status')}」，不该导出。",
            "换一条状态为「消融有效」的补丁；确实要导出就加 --force。",
        )
    scope = str(patch.get("scope") or "all_positions")
    if method in {"head_scale", "mlp_zero", "steering"} and scope != "all_positions":
        if not allow_all_positions:
            raise RBAError(
                f"{name}：补丁的作用范围是 {scope}（只改决策位置），烘焙进权重后会作用在所有位置，行为不同。",
                "先用 scope: all_positions 重新 rba eval 这个补丁；若结果仍可接受，再加 --allow-all-positions 导出。",
            )
        notes.append(f"{name}：复核时作用范围是 {scope}，烘焙后改成所有位置。等价性检查对比的是所有位置的钩子版本，回退率以本次导出后的数字为准。")
        warn(notes[-1])
    if method == "lm_head_blend" and tied and not untie:
        raise RBAError(
            f"{name}：lm_head 和输入 embedding 共享权重，改 lm_head 行会同时改该 token 的输入表示。",
            "加 --untie-embeddings：导出时把 lm_head 拆成独立权重（config 里 tie_word_embeddings 置为 false，checkpoint 会多一份 lm_head）。",
        )
    return notes


def untie_lm_head(model, resolved) -> None:
    head = resolved.lm_head(model)
    head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is not None and hasattr(config, "tie_word_embeddings"):
            config.tie_word_embeddings = False
    info("已拆开 lm_head 和 embedding 的共享权重（tie_word_embeddings=false）。")


def bake_patch(model, resolved, patch: dict, grade_ids: dict[str, int]) -> dict:
    """Edit weights in place so the plain forward equals apply_patch at all positions."""
    method = patch["method"]
    with torch.no_grad():
        if method == "head_scale":
            index = int(patch["layer"])
            found = attention_module(_layer(model, resolved, index))
            proj_info = output_projection(found[1]) if found is not None else None
            if proj_info is None:
                raise RBAError(f"第 {index} 层的注意力输出没法按头切开，不能烘焙 head_scale。", "换一个补丁。")
            proj, head_dim, n_heads = proj_info
            heads = [int(h) for h in patch["heads"]]
            bad = [h for h in heads if h < 0 or h >= n_heads]
            if bad:
                raise RBAError(f"第 {index} 层没有头 {bad}（共 {n_heads} 个头）。", "头的编号从 0 开始。")
            scale = float(patch.get("scale", 0.0))
            for head in heads:
                proj.weight[:, head * head_dim:(head + 1) * head_dim] *= scale
            return {"edited": f"layer{index}.{found[0]}.{'o_proj' if hasattr(found[1], 'o_proj') else 'out_proj'}", "columns": [[h * head_dim, (h + 1) * head_dim] for h in heads], "scale": scale}
        if method == "mlp_zero":
            index = int(patch["layer"])
            mlp = getattr(_layer(model, resolved, index), "mlp", None)
            down = getattr(mlp, "down_proj", None)
            if not isinstance(down, nn.Linear):
                raise RBAError(f"第 {index} 层的 MLP 没有 down_proj，不能烘焙 mlp_zero。", "换一个补丁。")
            down.weight.zero_()
            if down.bias is not None:
                down.bias.zero_()
            return {"edited": f"layer{index}.mlp.down_proj", "zeroed": True}
        if method == "lm_head_blend":
            head = resolved.lm_head(model)
            source = grade_ids[str(patch["grade"])]
            toward = grade_ids[str(patch["toward"])]
            alpha = float(patch.get("alpha", 0.5))
            if source != toward:
                row = (1.0 - alpha) * head.weight[source].clone() + alpha * head.weight[toward].clone()
                head.weight[source].copy_(row.to(head.weight.dtype))
            return {"edited": "lm_head", "row": source, "toward_row": toward, "alpha": alpha}
        if method == "steering":
            layers = [int(v) for v in (patch.get("layers") or [patch.get("layer", len(resolved.layers(model)) - 1)])]
            if patch.get("vector_source", "lm_head_diff") == "lm_head_diff":
                vector = steering_vector(model, resolved, grade_ids[str(patch["from_grade"])], grade_ids[str(patch["to_grade"])])
            else:
                vector = torch.tensor(patch["vector"], dtype=torch.float32)
            alpha = float(patch.get("alpha", 1.0))
            edited = []
            targets = []
            for index in layers:
                layer = _layer(model, resolved, index)
                down = getattr(getattr(layer, "mlp", None), "down_proj", None)
                if any(hasattr(layer, name) for name in POST_MLP_NORMS):
                    raise RBAError(
                        f"第 {index} 层在 MLP 后面还有一个归一化，常数偏置不等于在层输出上加向量，steering 不能精确烘焙。",
                        "这个补丁只能用钩子施加；或用 rba finetune。",
                    )
                if not isinstance(down, nn.Linear) or down.bias is None:
                    raise RBAError(
                        f"第 {index} 层没有能放常数的偏置（mlp.down_proj 不带 bias），steering 不能精确烘焙。"
                        "Qwen3.5 的线性层都不带 bias；硬加一个 bias 张量，标准类加载时会丢掉它，导出的模型就不等价了。",
                        "这个补丁只能在推理时用钩子施加；或用 rba finetune 把这个方向训进权重。",
                    )
                targets.append(down)
                edited.append(f"layer{index}.mlp.down_proj.bias")
            for down in targets:
                down.bias.add_((alpha * vector).to(device=down.bias.device, dtype=down.bias.dtype))
            return {"edited": edited, "alpha": alpha}
    raise RBAError(f"不认识的补丁方法 {method!r}。", "可烘焙的只有 head_scale、mlp_zero、lm_head_blend、steering。")


def equivalence(hooked: list[dict], exported: list[dict], dtype: torch.dtype, tolerance: float | None = None) -> dict:
    tol = float(tolerance if tolerance is not None else TOLERANCE.get(dtype, 0.05))
    max_diff = 0.0
    mismatched = []
    near_ties = []
    for a, b in zip(hooked, exported):
        la = torch.tensor(a["grade_logits"])
        lb = torch.tensor(b["grade_logits"])
        max_diff = max(max_diff, float((la - lb).abs().max()))
        if a["pred"] != b["pred"]:
            top2 = la.topk(min(2, la.numel())).values
            margin = float(top2[0] - top2[-1])
            (near_ties if margin <= 2 * tol else mismatched).append({"id": a["id"], "hooked": a["pred"], "exported": b["pred"], "margin": margin})
    return {
        "n": len(hooked),
        "dtype": str(dtype).replace("torch.", ""),
        "tolerance": tol,
        "max_abs_logit_diff": max_diff,
        "n_pred_mismatch": len(mismatched),
        "pred_mismatch": mismatched,
        "near_ties": near_ties,
        "identical": not mismatched and max_diff <= tol,
    }


def export_patches(cfg, patches: list[tuple[str, dict]], out: Path, *, force=False, allow_all_positions=False,
                   untie=False, overwrite=False, tolerance=None, dry_run=False) -> dict:
    from rba.models.loader import is_tied_lm_head, load_models
    from rba.models.paths import resolve_paths
    from rba.pipeline import _effective_batch, _encode_all, _validate_and_load

    reset()
    if not patches:
        raise RBAError("没有给补丁。", "rba export --run 分析目录 --patch 补丁1.json [补丁2.json ...] --out 导出目录")
    for name, patch in patches:
        if patch.get("grades") and [str(g) for g in patch["grades"]] != [str(g) for g in cfg.grades]:
            raise RBAError(f"{name} 是在 grades={patch['grades']} 下生成的，当前是 {cfg.grades}。", "用 --run 指向生成补丁的那次分析目录。")
    notes: list[str] = []
    for name, patch in patches:
        notes += check_patch(name, patch, force=force, allow_all_positions=allow_all_positions, tied=False, untie=untie)
    out = Path(out).resolve()
    check_output_dir(out, overwrite)
    cfg.analyses.delta_w = False
    bad_cases, good_cases, split = _validate_and_load(cfg)
    loaded = load_models(cfg)
    require_hf(loaded, "导出")
    resolved = resolve_paths(loaded.ft, cfg.module_paths, loaded.preset)
    tied = is_tied_lm_head(loaded.ft, resolved.lm_head(loaded.ft))
    if tied:
        for name, patch in patches:
            check_patch(name, patch, force=True, allow_all_positions=True, tied=tied, untie=untie)
    plain = [{key: value for key, value in patch.items() if key != "scope"} for _name, patch in patches]
    if dry_run:
        info("检查通过：这些补丁都能精确烘焙。去掉 --check-only 才会真的写出模型。")
        return {"checked": [name for name, _ in patches], "notes": notes}
    grades = cfg.grades
    grade_ids = loaded.grade_ids
    grade_id_list = [grade_ids[g] for g in grades]
    excluded = []
    bad_encoded, good_encoded = _encode_all(cfg, loaded, bad_cases, good_cases, excluded)
    items = bad_encoded + good_encoded
    if not items:
        raise RBAError("没有可用来核对的样本。", "检查 --badcases / --goodcases。")
    batch_size = _effective_batch(cfg, loaded.device, items)
    untied = False
    if tied and untie and any(p.get("method") == "lm_head_blend" for _, p in patches):
        untie_lm_head(loaded.ft, resolved)
        untied = True
        notes.append("lm_head 已与 embedding 拆开；导出目录的 config 里 tie_word_embeddings=false。")
    before = score_items(loaded, items, grade_id_list, grades, cfg.decode_protocol, batch_size, "导出·改前")
    hooked = score_items(loaded, items, grade_id_list, grades, cfg.decode_protocol, batch_size, "导出·钩子补丁",
                         patches=plain, resolved=resolved, grade_ids=grade_ids, scope="all_positions")
    records = []
    for (name, patch), clean in zip(patches, plain):
        record = bake_patch(loaded.ft, resolved, clean, grade_ids)
        records.append({"file": name, "method": patch["method"], "sha": patch_sha(patch), "scope_validated": patch.get("scope"), "status": patch.get("status"), "bake": record, "patch": patch})
        info(f"已烘焙 {name}（{patch['method']}）。")
    source_dir = loaded.ft_path
    copied = save_checkpoint(loaded.ft, source_dir, out)
    dtype = loaded.dtype
    free(loaded.ft)
    loaded.ft = None
    info("重新从导出目录加载（不带任何 rba 钩子）做核对")
    import copy

    cfg2 = copy.deepcopy(cfg)
    cfg2.ft_model = str(out)
    reloaded = load_models(cfg2)
    exported = score_items(reloaded, items, grade_id_list, grades, cfg.decode_protocol, batch_size, "导出·新模型")
    eq = equivalence(hooked, exported, dtype, tolerance)
    n_bad = len(bad_encoded)
    holdout_ids = set((split or {}).get("holdout_ids") or [])

    def block(rows_before, rows_after, ids):
        b = [r for r in rows_before if r["id"] in ids]
        a = [r for r in rows_after if r["id"] in ids]
        if not b:
            return None
        cmp = compare(b, a)
        cmp["before"] = metric_block(b, grades)
        cmp["after"] = metric_block(a, grades)
        cmp["n_invalid_after"] = sum(1 for r in a if r["pred"] == INVALID_GRADE)
        return cmp

    bad_ids = {item.id for item in bad_encoded}
    eval_numbers = {
        "bad_holdout": block(before, exported, bad_ids & holdout_ids),
        "bad_discovery": block(before, exported, bad_ids - holdout_ids),
        "good": block(before, exported, {item.id for item in good_encoded}),
    }
    manifest = {
        "rba_version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_model": str(source_dir),
        "source_fingerprint": file_fingerprint(source_dir),
        "exported_model": str(out),
        "exported_fingerprint": file_fingerprint(out),
        "dtype": str(dtype).replace("torch.", ""),
        "untied_lm_head": untied,
        "patches": records,
        "copied_files": copied,
        "equivalence": eq,
        "eval": eval_numbers,
        "n_bad": n_bad,
        "n_good": len(good_encoded),
        "excluded": [{"id": item.id, "reason": item.reason} for item in excluded],
        "notes": notes,
    }
    (out / "rba_export_manifest.json").write_text(json.dumps(to_builtin(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"等价性核对：{eq['n']} 条样本，最大等级 logit 差 {eq['max_abs_logit_diff']:.3g}（{eq['dtype']} 容差 {eq['tolerance']:g}），"
         f"预测不一致 {eq['n_pred_mismatch']} 条" + (f"，另有 {len(eq['near_ties'])} 条是容差内的并列" if eq["near_ties"] else "") + "。")
    for key, label in (("bad_holdout", "验收集"), ("bad_discovery", "发现集"), ("good", "对照")):
        cmp = eval_numbers[key]
        if cmp:
            fix = "—" if cmp["fix_rate"] is None else f"{cmp['fix_rate']:.1%}"
            reg = "—" if cmp["regression_rate"] is None else f"{cmp['regression_rate']:.1%}"
            info(f"导出模型 vs 原模型 · {label} {cmp['n']} 条：修复率 {fix}，回退率 {reg}，净正确数 {cmp['net_correct_change']:+d}。")
    info(f"清单：{out / 'rba_export_manifest.json'}")
    if not eq["identical"]:
        raise RBAError(
            f"导出模型和钩子补丁的结果不一致（最大 logit 差 {eq['max_abs_logit_diff']:.3g}，预测不一致 {eq['n_pred_mismatch']} 条）。导出目录保留着供检查，不要上线。",
            "看清单里的 equivalence.pred_mismatch。若是低精度舍入，可以用 --dtype float32 再导出一次对比；否则请带着清单反馈。",
        )
    info("导出模型和钩子补丁的预测一致。这个目录可以直接用 transformers 加载，不需要 rba。")
    return to_builtin(manifest)
