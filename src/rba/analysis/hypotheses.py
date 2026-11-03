"""Rank root-cause hypotheses from measurements, not from the LLM.

A component is "已验证" only when a targeted intervention on the detail
sample reduced the error logit (pred minus gold) on a majority of those
rows. That check is not a regression test: the report tells the user to run
`rba eval` on the full badcase file and the goodcase file before keeping it.

Correlation (direct logit attribution, ΔW, logit lens) never outranks a
negative ablation. Ablations that increase the error are kept in the list
and marked as rejected, so a reviewer can see we did not hide them.
"""

from __future__ import annotations


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
        numeric_layers.append({"layer": layer_name, "p_pred": pred_prob, "p_gold": gold_prob})
        lens_pred.append(pred_prob)
    flip = None
    for item in numeric_layers:
        if item["p_pred"] >= item["p_gold"] and item["p_pred"] == max(item["p_pred"], item["p_gold"]):
            flip = item["layer"]
            break
    # Prefer the first layer where pred prob exceeds gold prob.
    flip = None
    for item in numeric_layers:
        if item["p_pred"] > item["p_gold"]:
            flip = item["layer"]
            break
    gaps = [abs(row["component_sum"] - row["prenorm_dot"]) / (abs(row["prenorm_dot"]) + 1e-6) for row in detail_rows]
    return {
        "components": components,
        "lens": numeric_layers,
        "flip_layer": flip,
        "n_layers": n_layers,
        "logit_diff_mean": _mean([float(row["logit_diff"]) for row in detail_rows]),
        "decomposition_gap": _mean(gaps),
        "image_token_fraction": _mean([float(row["image_token_fraction"]) for row in detail_rows]),
        "attn_image_share": _mean([float(row["attn_image_share"]) for row in detail_rows if row["attn_image_share"] is not None])
        if any(row["attn_image_share"] is not None for row in detail_rows)
        else None,
        "attn_layers_used": max(int(row["attn_layers_used"]) for row in detail_rows),
        "grade_mass": _mean([float(row["grade_mass"]) for row in detail_rows]),
        "full_top_grade_rate": _mean([1.0 if row["full_top_is_grade"] else 0.0 for row in detail_rows]),
    }


def propose_checks(summary: dict, mech: dict, delta: dict | None, verify_top_k: int) -> list[dict]:
    checks = []
    for component in (mech.get("components") or [])[:verify_top_k]:
        if component["kind"] == "attn_head":
            checks.append(
                {
                    "key": component["key"],
                    "patch": {
                        "method": "head_scale",
                        "layer": component["layer"],
                        "heads": [component["head"]],
                        "scale": 0.0,
                    },
                }
            )
        elif component["kind"] == "mlp":
            checks.append(
                {
                    "key": component["key"],
                    "patch": {
                        "method": "steering",
                        "layers": [component["layer"]],
                        "vector_source": "lm_head_diff",
                        "from_grade": summary["pred"],
                        "to_grade": summary["label"],
                        "alpha": 0.0,
                    },
                }
            )
    # MLP zeroing is a real ablation. Steering with alpha 0 does nothing.
    # Replace the placeholder with a head-style note: we ablate by steering? 
    # A zero MLP is implemented as scale of the whole module via a dedicated patch
    # encoded as head_scale is wrong. Use method mlp_zero, handled below if present.
    checks = [check for check in checks if check["patch"]["method"] != "steering"]
    for component in (mech.get("components") or [])[:verify_top_k]:
        if component["kind"] == "mlp":
            checks.append({"key": component["key"], "patch": {"method": "mlp_zero", "layer": component["layer"]}})
    lm = (delta or {}).get("lm_head_grades", {}).get("rows", {})
    pred_row = lm.get(summary["pred"], {})
    if pred_row.get("cosine_to_base", 1.0) < 0.98 or pred_row.get("delta_norm", 0.0) > 0.5:
        checks.append(
            {
                "key": f"lm_head:{summary['pred']}->{summary['label']}",
                "patch": {
                    "method": "lm_head_blend",
                    "grade": summary["pred"],
                    "toward": summary["label"],
                    "alpha": 0.5,
                },
            }
        )
    image_share = mech.get("attn_image_share")
    token_share = mech.get("image_token_fraction") or 0.0
    if (image_share is not None and image_share >= 0.35) or token_share >= 0.25:
        checks.append({"key": "image_tokens", "patch": {"method": "ablate_image"}})
    # Unique by key, cap.
    dedup = []
    seen = set()
    for check in checks:
        if check["key"] in seen:
            continue
        seen.add(check["key"])
        dedup.append(check)
    return dedup[:6]


def rank_hypotheses(summary: dict, mech: dict, delta: dict | None, verifications: list[dict], n_parameters: int) -> list[dict]:
    by_key = {item["key"]: item for item in verifications}
    components = {item["key"]: item for item in mech.get("components") or []}
    delta_names = [row["name"] for row in ((delta or {}).get("top") or [])[:8]]
    hypotheses = []

    def overlap(layer: int | None, kind: str | None) -> str | None:
        if layer is None:
            return None
        needles = []
        if kind == "mlp":
            needles.append(f"layers.{layer}.mlp")
        elif kind:
            needles.append(f"layers.{layer}.{kind}")
            needles.append(f"layers.{layer}.self_attn")
            needles.append(f"layers.{layer}.linear_attn")
        for name in delta_names:
            if any(needle in name for needle in needles):
                return name
        return None

    top_component_keys = set(list(components)[:5])
    for key, component in components.items():
        if key not in top_component_keys and key not in by_key:
            continue
        if key not in by_key and abs(component["mean_attr"]) <= 0:
            continue
        verification = by_key.get(key)
        status, effect, agreement = _status(verification)
        score = abs(component["mean_attr"])
        if status == "已验证":
            # Cap the ablation term so a huge logit on an untrained net cannot
            # outrank "the grade tokens carry no probability".
            score = 120 + min(15.0, abs(effect or 0.0)) + score
        elif status == "被复核否定":
            score = 1 + score * 0.01
        else:
            score = 10 + component["share"] * 20
        touched = overlap(component["layer"], "mlp" if component["kind"] == "mlp" else component.get("attn_name"))
        if touched and status != "被复核否定":
            score += 15
        fraction = _fraction_for(component, n_parameters)
        confidence = _confidence(status, effect, agreement, component["share"])
        n_check = (verification or {}).get("n") or 0
        if status == "已验证" and n_check < 4:
            confidence = min(confidence, 0.7)
        hypotheses.append(
            _hypothesis(
                key=key,
                title=_component_title(component),
                status=status,
                score=score,
                effect=effect,
                agreement=agreement,
                confidence=confidence,
                evidence=_component_evidence(component, verification, touched),
                patch=_action_patch(summary, component, milder=True),
                diagnostic_patch=_action_patch(summary, component, milder=False),
                weight_fraction=fraction,
                would_refute="在这组抽样上把该组件清零后，logit(预测)-logit(金标) 没有下降；或在全部坏例上修复率不超过对照集回退率。",
            )
        )

    lm_key = f"lm_head:{summary['pred']}->{summary['label']}"
    if lm_key in by_key or _lm_head_interesting(delta, summary):
        verification = by_key.get(lm_key)
        status, effect, agreement = _status(verification)
        row = ((delta or {}).get("lm_head_grades") or {}).get("rows", {}).get(summary["pred"], {})
        score = 20 + float(row.get("delta_norm") or 0.0)
        if status == "已验证":
            score += 120 + min(15.0, abs(effect or 0.0))
        elif status == "被复核否定":
            score = 1.0
        fraction = (1.0 / n_parameters) if n_parameters else None
        # one row has `hidden` parameters; caller passes total. fraction computed outside if hidden known.
        hypotheses.append(
            _hypothesis(
                key=lm_key,
                title=f"lm_head 上等级 {summary['pred']} 的行被写偏，读出来更像 {summary['pred']} 而不是 {summary['label']}",
                status=status if verification else "仅相关",
                score=score,
                effect=effect,
                agreement=agreement,
                confidence=_confidence(status if verification else "仅相关", effect, agreement, 0.4),
                evidence=(
                    f"该行相对基座的余弦是 {row.get('cosine_to_base', '未知')}，差的范数是 {row.get('delta_norm', '未知')}。"
                    + (f" 抽样上把这一行向 {summary['label']} 混合一半后，错误 logit 平均变化 {effect:.3f}。" if effect is not None else " 还没有做行混合复核。")
                ),
                patch={
                    "method": "lm_head_blend",
                    "grade": summary["pred"],
                    "toward": summary["label"],
                    "alpha": 0.5,
                },
                diagnostic_patch={
                    "method": "lm_head_blend",
                    "grade": summary["pred"],
                    "toward": summary["label"],
                    "alpha": 1.0,
                },
                weight_fraction=fraction,
                would_refute="混合之后这组的错误 logit 不降，或对照集里原本预测对的样本被改错。这一行是全局的，比关一个头更容易伤及其它等级。",
            )
        )

    if summary.get("has_image_fraction", 0) > 0 and (
        (mech.get("image_token_fraction") or 0) >= 0.2 or (mech.get("attn_image_share") or 0) >= 0.3
    ):
        verification = by_key.get("image_tokens")
        status, effect, agreement = _status(verification)
        share = mech.get("attn_image_share")
        token_share = mech.get("image_token_fraction")
        score = 30 + 40 * float(token_share or 0)
        if status == "已验证":
            score += 120 + min(15.0, abs(effect or 0.0))
        elif status == "被复核否定":
            score = 2
        hypotheses.append(
            _hypothesis(
                key="image_tokens",
                title="错误主要跟图像 token 走，而不是正文",
                status=status if verification else "仅相关",
                score=score,
                effect=effect,
                agreement=agreement,
                confidence=_confidence(status if verification else "仅相关", effect, agreement, float(token_share or 0)),
                evidence=(
                    f"决策位置上图像 token 的绝对归因占比约 {token_share:.2f}。"
                    + (f" 全注意力层里，决策位置分给图像 token 的注意力约 {share:.2f}。" if share is not None else " 当前注意力实现没有返回权重，注意力份额没有测。")
                    + (f" 把图像 token 在语言层入口置零后，错误 logit 平均变化 {effect:.3f}。" if effect is not None else "")
                ),
                patch={"method": "ablate_image"},
                diagnostic_patch={"method": "ablate_image"},
                weight_fraction=0.0,
                would_refute="去掉图像 token 后错误 logit 几乎不动。那时应回到文本 token 和等级头，而不是怪视觉塔。",
            )
        )

    mass = mech.get("grade_mass")
    if mass is not None and mass < 0.5:
        hypotheses.append(
            _hypothesis(
                key="grade_mass",
                title="下一个 token 的概率大多不在五个等级词上",
                status="先处理这个",
                score=300,
                effect=None,
                agreement=None,
                confidence=0.7,
                evidence=(
                    f"五个等级词的概率质量平均只有 {mass:.2f}，全词表 argmax 落在等级词上的比例是 {mech.get('full_top_grade_rate'):.2f}。"
                    "下面的头和 MLP 只解释「如果强迫在五个等级里选」时的 logit 差，对不上线上直接解码的那个 token。"
                ),
                patch=None,
                diagnostic_patch=None,
                weight_fraction=None,
                would_refute="换上与训练完全一致的模板后，等级词质量回到接近 1。这通常是模板或 grade token 切分问题，不是某一层的机制。",
            )
        )

    if mech.get("decomposition_gap", 0) > 0.15:
        hypotheses.append(
            _hypothesis(
                key="decomposition_gap",
                title="残差分解对不上，组件归因不可靠",
                status="证据不足",
                score=5,
                effect=None,
                agreement=None,
                confidence=0.2,
                evidence=f"各组件点积之和与最终范数前残差的点积相对差距约 {mech['decomposition_gap']:.2f}。有残差支路没有被钩住。",
                patch=None,
                diagnostic_patch=None,
                weight_fraction=None,
                would_refute="差距回到接近 0 之后再读组件排序。",
            )
        )

    hypotheses.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(hypotheses, start=1):
        item["rank"] = rank
    return hypotheses


def _status(verification: dict | None):
    if not verification:
        return "仅相关", None, None
    effect = verification.get("mean_delta")
    agreement = verification.get("agreement")
    if effect is None:
        return "仅相关", None, agreement
    if effect < -1e-4 and (agreement or 0) >= 0.6:
        return "已验证", effect, agreement
    if effect > 1e-4 and (agreement or 0) <= 0.4:
        return "被复核否定", effect, agreement
    if abs(effect) <= 1e-4:
        return "消融无效果", effect, agreement
    return "复核不稳定", effect, agreement


def _confidence(status: str, effect, agreement, share: float) -> float:
    if status == "已验证":
        base = 0.55 + 0.25 * min(1.0, abs(effect or 0) / 2.0) + 0.1 * (agreement or 0)
        return round(min(0.9, base), 2)
    if status == "被复核否定":
        return 0.15
    if status == "证据不足":
        return 0.2
    return round(min(0.55, 0.2 + 0.3 * min(1.0, share)), 2)


def _component_title(component: dict) -> str:
    if component["kind"] == "mlp":
        return f"第 {component['layer']} 层 MLP 把残差推向错误等级"
    return f"第 {component['layer']} 层 {component['attn_name']} 的第 {component['head']} 个头把残差推向错误等级"


def _component_evidence(component, verification, touched) -> str:
    text = (
        f"直接 logit 归因均值 {component['mean_attr']:.4f}（标准差 {component['std_attr']:.4f}），"
        f"占本组组件绝对值的 {component['share']:.0%}。"
    )
    if touched:
        text += f" 同一层的权重差也排在前面：{touched}。"
    if verification and verification.get("mean_delta") is not None:
        text += (
            f" 抽样消融后 logit(预测)-logit(金标) 平均变化 {verification['mean_delta']:.4f}，"
            f"{verification.get('n_improved', 0)}/{verification.get('n', 0)} 条误差变小。"
        )
    else:
        text += " 这条还没有做消融，只能当相关。"
    return text


def _action_patch(summary, component, milder: bool) -> dict:
    if component["kind"] == "mlp":
        return {"method": "mlp_zero", "layer": component["layer"]}
    scale = 0.5 if milder else 0.0
    return {
        "method": "head_scale",
        "layer": component["layer"],
        "heads": [component["head"]],
        "scale": scale,
    }


def _fraction_for(component, n_parameters: int) -> float | None:
    if not n_parameters:
        return None
    # Filled properly by the caller when head_dim is known. Placeholder fraction
    # stays None here; pipeline overwrites with the real element count.
    return None


def _lm_head_interesting(delta, summary) -> bool:
    rows = ((delta or {}).get("lm_head_grades") or {}).get("rows", {})
    row = rows.get(summary["pred"])
    if not row:
        return False
    return row.get("cosine_to_base", 1.0) < 0.98 or row.get("delta_norm", 0.0) > 0.5


def _hypothesis(**kwargs) -> dict:
    return kwargs
