"""Descriptive hypotheses; ablation effectiveness is not root-cause proof."""
from __future__ import annotations

import math

STATUS_RULES = {
    "消融有效": "至少 4 条复核样本；平均分差变化 < -1e-4 且至少 60% 同向；仅表示干预效果，不证明根因。",
    "仅相关": "未有有效复核，或仅有权重几何/代数性行混合证据。",
    "被复核否定": "至少 4 条样本；平均分差变化 > 1e-4 且改善比例 <= 40%。",
    "消融无效果": "至少 4 条样本；平均分差变化绝对值 <= 1e-4。",
    "复核不稳定": "平均变化与改善比例未满足一致效果阈值。",
    "证据不足": "复核样本少于 4 条、identity 逐例非零，或最终读出重建不可靠。",
    "先处理这个": "等级词概率质量不足，先核对输入与解码协议。",
    "未执行": "复核记录含 skipped，未产生可用干预测量。",
}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def summarize_group(group_key: str, cases: list[dict], detail: list[dict], grades: list[str], clusters) -> dict:
    label, pred = group_key.split("->", 1)
    return {
        "key": group_key,
        "label": label,
        "pred": pred,
        "count": len(cases),
        "detail_ids": [row["id"] for row in detail],
        "n_detail": len(detail),
        "grades": grades,
        "mean_probs": {grade: _mean([float(row["probs"].get(grade, 0.0)) for row in cases]) for grade in grades},
        "grade_mass": _mean([float(row.get("grade_mass", 1.0)) for row in detail]) if detail else None,
        "full_top_grade_rate": _mean([1.0 if row.get("full_top_is_grade", True) else 0.0 for row in detail]) if detail else None,
        "has_image_fraction": _mean([1.0 if row.get("has_image") else 0.0 for row in cases]),
        "clusters": clusters,
        "examples": [
            {
                "id": row["id"],
                "query": (row.get("query") or "")[:160],
                "doc": (row.get("doc") or "")[:160],
                "has_image": bool(row.get("has_image")),
            }
            for row in cases[:3]
        ],
    }


def aggregate_mechanistic(detail_rows: list[dict], grades: list[str]) -> dict:
    if not detail_rows:
        return {}
    # Preserve original proxy fields on disk; aggregate the norm-aware readout.
    detail_rows = [dict(row) for row in detail_rows]
    for row in detail_rows:
        for kind in ("head", "attn", "mlp"):
            if f"norm_{kind}_attr" in row:
                row[f"{kind}_attr"] = row[f"norm_{kind}_attr"]
    n_layers = len(detail_rows[0]["attn_attr"])
    components = []
    for layer in range(n_layers):
        kind = detail_rows[0]["kinds"][layer]
        head_rows = [row["head_attr"][layer] for row in detail_rows if row["head_attr"] and row["head_attr"][layer] is not None]
        if head_rows:
            n_heads = len(head_rows[0])
            for head in range(n_heads):
                values = [float(row[head]) for row in head_rows]
                components.append(
                    {
                        "key": f"L{layer}.{kind}.h{head}",
                        "kind": "attn_head",
                        "layer": layer,
                        "head": head,
                        "attn_name": kind,
                        "mean_attr": _mean(values),
                        "std_attr": _std(values),
                    }
                )
        mlp_values = [float(row["mlp_attr"][layer]) for row in detail_rows if row["mlp_attr"][layer] is not None]
        if mlp_values:
            components.append(
                {
                    "key": f"L{layer}.mlp",
                    "kind": "mlp",
                    "layer": layer,
                    "head": None,
                    "attn_name": kind,
                    "mean_attr": _mean(mlp_values),
                    "std_attr": _std(mlp_values),
                }
            )
    total = sum(abs(item["mean_attr"]) for item in components) + 1e-12
    for item in components:
        item["share"] = abs(item["mean_attr"]) / total
    components.sort(key=lambda item: abs(item["mean_attr"]), reverse=True)
    lens_pred = []
    label_index = grades.index(detail_rows[0]["label"]) if detail_rows[0]["label"] in grades else 0
    pred_index = grades.index(detail_rows[0]["pred"]) if detail_rows[0]["pred"] in grades else 0
    # lens entries: embed + each layer. Use numeric layers only.
    numeric_layers = []
    for layer_pos in range(len(detail_rows[0]["lens"])):
        layer_name = detail_rows[0]["lens"][layer_pos]["layer"]
        if not isinstance(layer_name, int):
            continue
        pred_prob = _mean([row["lens"][layer_pos]["probs"][pred_index] for row in detail_rows])
        gold_prob = _mean([row["lens"][layer_pos]["probs"][label_index] for row in detail_rows])
        diffs = [row["lens"][layer_pos]["logits"][pred_index] - row["lens"][layer_pos]["logits"][label_index]
                 for row in detail_rows if "logits" in row["lens"][layer_pos]]
        numeric_layers.append({"layer": layer_name, "p_pred": pred_prob, "p_gold": gold_prob,
                               "logit_diff": _mean(diffs) if diffs else None})
        lens_pred.append(pred_prob)
    flip = None
    for item in numeric_layers:
        if item["p_pred"] >= item["p_gold"] and item["p_pred"] == max(item["p_pred"], item["p_gold"]):
            flip = item["layer"]
            break
    # Prefer the first layer where pred prob exceeds gold prob.
    flip = None
    for item in numeric_layers:
        if (item["logit_diff"] > 0 if item.get("logit_diff") is not None else item["p_pred"] > item["p_gold"]):
            flip = item["layer"]
            break
    gaps = [abs(row["component_sum"] - row["prenorm_dot"]) / (abs(row["prenorm_dot"]) + 1e-6) for row in detail_rows]
    return {
        "components": components,
        "case_evidence": [{"id": row.get("id", f"detail_{i}"), "label": row.get("label"),
                           "pred": row.get("pred"), "grade_mass": row.get("grade_mass"),
                           "full_top_id": row.get("full_top_id"), "logit_diff": row.get("logit_diff"),
                           "lens": row.get("lens"), "head_attr": row.get("head_attr"),
                           "mlp_attr": row.get("mlp_attr"), "attribution_reliable": row.get("attribution_reliable", False),
                           "reconstruction_error": row.get("reconstruction_error")}
                          for i, row in enumerate(detail_rows)],
        "attribution_reliable": all(row.get("attribution_reliable", False) for row in detail_rows),
        "readout_kind": "final_norm_dla" if all("norm_attn_attr" in row for row in detail_rows) else "pre_norm_proxy",
        "reconstruction_error": max((row.get("reconstruction_error") or 0.) for row in detail_rows),
        "delta_contributions": [{"id": row["id"], "contributions": row.get("delta_contributions", [])} for row in detail_rows if "id" in row],
        "lens": numeric_layers,
        "flip_layer": flip,
        "n_layers": n_layers,
        "logit_diff_mean": _mean([float(row["logit_diff"]) for row in detail_rows]),
        "decomposition_gap": _mean(gaps),
        "image_token_count_fraction": _mean([float(row["image_token_fraction"]) for row in detail_rows]),
        "image_token_fraction": _mean([float(row["image_token_fraction"]) for row in detail_rows]),
        "attn_image_share": _mean([float(row["attn_image_share"]) for row in detail_rows if row["attn_image_share"] is not None])
        if any(row["attn_image_share"] is not None for row in detail_rows)
        else None,
        "attn_layers_used": max(int(row["attn_layers_used"]) for row in detail_rows),
        "grade_mass": _mean([float(row["grade_mass"]) for row in detail_rows]),
        "full_top_grade_rate": _mean([1.0 if row["full_top_is_grade"] else 0.0 for row in detail_rows]),
    }


def propose_checks(summary: dict, mech: dict, delta: dict | None, verify_top_k: int,
                   controls: bool = True) -> list[dict]:
    import hashlib
    import random
    rng = random.Random(int(hashlib.sha256(summary["key"].encode()).hexdigest()[:16], 16))
    components = mech.get("components") or []
    target_keys = {c["key"] for c in components[:verify_top_k]}
    checks = []
    for component in components[:verify_top_k]:
        key = component["key"]
        patch = _action_patch(summary, component, False)
        checks.append({"key": key, "target_key": key, "scope": "decision", "control_type": "target", "patch": patch})
        if not controls:
            continue
        # Heads can be matched in the same layer; an MLP has no same-layer peer.
        # Other candidate targets are not "random": a control drawn from them would
        # carry its own effect and make the specificity comparison meaningless.
        peers = [c for c in components if c["kind"] == component["kind"] and c["key"] not in target_keys
                 and (c["layer"] == component["layer"] if c["kind"] == "attn_head" else True)]
        if not peers:
            peers = [c for c in components if c["kind"] == component["kind"] and c["key"] != key
                     and (c["layer"] == component["layer"] if c["kind"] == "attn_head" else True)]
        if peers:
            peer = rng.choice(peers)
            checks.append({"key": key + ":random", "target_key": key, "scope": "decision", "control_type": "random",
                           "matched_component": peer["key"], "same_layer": peer["layer"] == component["layer"],
                           "patch": _action_patch(summary, peer, False)})
        if component["kind"] == "attn_head":
            checks.append({"key": key + ":identity", "target_key": key, "scope": "decision", "control_type": "identity",
                           "patch": {**patch, "scale": 1.0}})
    if _lm_head_interesting(delta, summary) and not (delta or {}).get("tied_lm_head", False):
        key = f"lm_head:{summary['pred']}->{summary['label']}"
        checks.append({"key": key, "target_key": key, "scope": "readout", "control_type": "target", "diagnostic_only": True,
                       "patch": {"method": "lm_head_blend", "grade": summary["pred"], "toward": summary["label"], "alpha": .5}})
    # Token count is exposure, not attribution. A diagnostic image ablation is
    # permitted with attention evidence, but does not by itself nominate a cause.
    if (mech.get("attn_image_share") or 0) >= .35:
        checks.append({"key": "image_tokens", "target_key": "image_tokens", "scope": "image_positions",
                       "control_type": "target", "patch": {"method": "ablate_image"}})
    return checks


def _status(verification):
    if not verification:
        return "仅相关", None, None
    if verification.get("skipped"):
        return "未执行", None, None
    effect, agreement = verification.get("mean_delta"), verification.get("agreement")
    if effect is None:
        return "仅相关", None, agreement
    if verification.get("diagnostic_only") or str(verification.get("key", "")).startswith("lm_head:"):
        return "仅相关", effect, agreement
    if verification.get("n", 0) < 4:
        return "证据不足", effect, agreement
    if abs(effect) <= 1e-4:
        return "消融无效果", effect, agreement
    if effect < -1e-4 and (agreement or 0) >= .6:
        return "消融有效", effect, agreement
    if effect > 1e-4 and (agreement or 0) <= .4:
        return "被复核否定", effect, agreement
    return "复核不稳定", effect, agreement


def rank_hypotheses(summary, mech, delta, verifications, n_parameters, token_attr=None):
    by_key = {v["key"]: v for v in verifications if v.get("control_type", "target") == "target"}
    hypotheses = []
    reliable = mech.get("attribution_reliable", mech.get("decomposition_gap", 0) <= .15)

    def add(key, title, evidence, patch, score, diagnostic_only=False):
        verification = by_key.get(key)
        status, effect, agreement = _status(verification)
        insufficient_reasons = []
        if status == "证据不足" and verification and verification.get("n", 0) < 4:
            insufficient_reasons.append(f"复核样本 {verification.get('n', 0)} 条 < 4")
        controls = [v for v in verifications if v.get("target_key") == key and v.get("control_type") in {"identity", "random"}]
        def identity_failed_case(control):
            if control.get("control_type") != "identity" or control.get("skipped"):
                return False
            changes = [case.get("delta") for case in control.get("per_case", [])]
            if not changes:
                changes = [control.get("mean_delta")]
            return any(value is not None and (not math.isfinite(value) or abs(value) > 1e-4) for value in changes)
        identity_failed = any(identity_failed_case(v) for v in controls)
        if identity_failed:
            changes = [abs(case["delta"]) for v in controls if v.get("control_type") == "identity"
                       for case in v.get("per_case", []) if case.get("delta") is not None]
            changes += [abs(v["mean_delta"]) for v in controls if v.get("control_type") == "identity" and v.get("mean_delta") is not None]
            insufficient_reasons.append(f"identity 对照非零或非有限（最大绝对变化 {max(changes, default=float('nan')):.4g}）")
        if not reliable and key.startswith("L"):
            insufficient_reasons.append("最终读出重建不可靠")
        if identity_failed or (not reliable and key.startswith("L")):
            status = "证据不足"
            patch = None
        if diagnostic_only and key.startswith("lm_head:") and status != "未执行":
            status = "仅相关"
        if effect is not None:
            evidence += f" 干预错误 logit 差均值变化 {effect:.4g}；作用范围 {verification.get('scope', '未记录')}。"
        if status == "消融有效":
            score += 100
            evidence += " 仅说明该干预降低分差，不证明故障机制；未控制分布外破坏。"
        elif status == "被复核否定":
            score = 1
        if verification and verification.get("skipped"):
            status = "未执行"
            evidence += f" 未执行原因：{verification['skipped']}。"
            patch = None
        if verification and not verification.get("selection_disjoint", False):
            evidence += " 选择与验证样本未证明独立。"
        executed_controls = [v for v in controls if not v.get("skipped") and v.get("mean_delta") is not None]
        if not executed_controls and verification:
            evidence += " 缺少已执行的随机/identity 控制。"
        random_effects = [v["mean_delta"] for v in executed_controls if v.get("control_type") == "random"]
        if effect is not None and any(value <= effect for value in random_effects):
            evidence += " 随机对照下降不少于目标，尚无组件特异性证据。"
        if insufficient_reasons:
            evidence += " 证据不足原因：" + "；".join(insufficient_reasons) + "。"
        evidence_ids = [f"component:{key}"] if key.startswith("L") else ["token_attribution" if key == "image_tokens" else "delta"]
        if verification:
            evidence_ids.append(f"verification:{key}")
        evidence_ids.extend(f"verification:{control['key']}" for control in controls)
        diagnostic_patch = None
        if verification and not verification.get("skipped"):
            diagnostic_patch = dict(verification.get("patch") or {}) or None
            if diagnostic_patch is not None:
                diagnostic_patch["scope"] = verification.get("scope", "all_positions")
        proposal_tested = patch is not None and diagnostic_patch == patch
        if patch is not None and not proposal_tested:
            evidence += " 此修复提案的系数/范围未按原样复核，须单独验收。"
        hypotheses.append(dict(key=key, title=title, status=status, score=score, effect=effect,
                               agreement=agreement, confidence=None, evidence_strength=status, diagnostic_only=diagnostic_only,
                               insufficient_reason="；".join(insufficient_reasons) if insufficient_reasons else None,
                               evidence=evidence, evidence_ids=evidence_ids, patch=patch,
                               diagnostic_patch=diagnostic_patch, proposal_tested=proposal_tested, weight_fraction=None,
                               would_refute="独立样本不能复现效果、identity 控制非零、随机控制同样有效，或回退超过预设上限。"))

    for c in (mech.get("components") or [])[:5]:
        title = f"{c['key']} 的最终读出贡献（正值支持原预测，负值支持金标）"
        add(c["key"], title,
            f"归因均值 {c['mean_attr']:.4g}，标准差 {c['std_attr']:.4g}，绝对均值占比 {c['share']:.1%}。不是总因果效应。",
            _action_patch(summary, c, True), 10 + 20*c["share"], diagnostic_only=c["kind"] == "mlp")

    lm_key = f"lm_head:{summary['pred']}->{summary['label']}"
    if _lm_head_interesting(delta, summary) or lm_key in by_key:
        tied = (delta or {}).get("tied_lm_head", False)
        patch = None if tied else {"method": "lm_head_blend", "grade": summary["pred"], "toward": summary["label"], "alpha": .5}
        add(lm_key, "等级读出行相对基座发生变化",
            "权重差是几何证据。行混合代数性缩小固定 hidden 的分差，不能验证根因。"
            + ("该 lm_head 与输入 embedding 共享权重，行编辑也改变输入 embedding；不自动推荐。" if tied else ""), patch, 25, True)

    token_attr = token_attr or {}
    signed = token_attr.get("image_signed")
    share = token_attr.get("image_abs_share")
    if summary.get("has_image_fraction", 0) > 0 and signed is not None and signed > 0 and share is not None and share >= .2 and token_attr.get("attribution_reliable", token_attr.get("reliable", True)):
        add("image_tokens", "图像 token 对当前错误分差有正向局部归因",
            f"图像 signed 归因 {signed:.4g}，绝对归因占比 {share:.2f}；token 数量比例 {mech.get('image_token_count_fraction', mech.get('image_token_fraction'))}，注意力份额 {mech.get('attn_image_share')}，三者不可互换。",
            {"method": "ablate_image", "scope": "token_positions"}, 30, diagnostic_only=True)
    mass = mech.get("grade_mass")
    if mass is not None and mass < .5:
        hypotheses.append(dict(key="grade_mass", title="下一个 token 的概率大多不在等级词上", status="先处理这个", score=300,
                               effect=None, agreement=None, confidence=None, evidence_strength="协议检查",
                               evidence=f"等级质量 {mass:.3g}；全词表 top 合法率 {mech.get('full_top_grade_rate')}。先核对模板和解码协议；随机 demo 也保留此检查。",
                               evidence_ids=["decision"], patch=None, diagnostic_patch=None, weight_fraction=None,
                               would_refute="训练输入和解码协议复现后，合法输出率与线上一致。"))
    if not reliable:
        hypotheses.append(dict(key="decomposition_gap", title="最终读出分解不可靠", status="证据不足", score=290,
                               effect=None, agreement=None, confidence=None, evidence_strength="校验失败", insufficient_reason="最终读出重建不可靠",
                               evidence=f"重建误差 {mech.get('reconstruction_error')}；pre-norm gap {mech.get('decomposition_gap')}。停止据此推荐组件补丁。",
                               evidence_ids=["decision"], patch=None, diagnostic_patch=None, weight_fraction=None,
                               would_refute="精确读出重建在 dtype 容差内通过。"))
    hypotheses.sort(key=lambda h: h["score"], reverse=True)
    for rank, h in enumerate(hypotheses, 1):
        h["rank"] = rank
    return hypotheses


def _action_patch(summary, component, milder):
    if component["kind"] == "mlp":
        return {"method": "mlp_zero", "layer": component["layer"], "scope": "decision"}
    return {"method": "head_scale", "layer": component["layer"], "heads": [component["head"]], "scale": .5 if milder else 0., "scope": "decision"}


def _lm_head_interesting(delta, summary):
    if ((delta or {}).get("base_grade_alignment") or {}).get("status", "aligned") != "aligned":
        return False
    row = ((delta or {}).get("lm_head_grades") or {}).get("rows", {}).get(summary["pred"], {})
    return row.get("cosine_to_base", 1.) < .98 or row.get("delta_norm", 0.) > .5
