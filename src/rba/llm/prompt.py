"""Auditable evidence packets, separated observations and proposed interpretations."""
from __future__ import annotations

import json
from collections import Counter
from rba.errors import RBAError


LANGUAGE_INSTRUCTIONS = {
    "zh": "请全部用中文回答：分析过程和 JSON 里的文字字段（mechanism、verification、would_refute 等）都用中文；JSON 键名保持英文；logit lens、DLA、ΔW、SVD、lm_head、head_scale 等技术术语和 evidence_id 保留原文。",
    "en": "Answer in English. Keep JSON keys and evidence_id values unchanged.",
}


def language_instruction(language: str = "zh") -> str:
    return LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["zh"])


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / max(chars_per_token, .5)))


def fit_budget(parts: list[dict], budget_tokens: int, chars_per_token: float) -> tuple[str, list[str]]:
    # Reserve ALL required material first, including the trailing output schema.
    required = [p for p in parts if not p.get("droppable", False)]
    text = "\n\n".join(p["text"].strip() for p in required)
    if estimate_tokens(text, chars_per_token) > budget_tokens:
        raise RBAError("提示词必要约束超过预算。", "增大 llm.max_prompt_tokens；不会截断输出协议。")
    kept = {i for i,p in enumerate(parts) if not p.get("droppable", False)}
    dropped = []
    for i, part in enumerate(parts):
        if i in kept:
            continue
        available = {name for j,p in enumerate(parts) if j in kept for name in [p["name"], *p.get("provides", [])]}
        if not set(part.get("requires", [])).issubset(available):
            dropped.append(part["name"])
            continue
        candidate = "\n\n".join(p["text"].strip() for j,p in enumerate(parts) if j in kept or j == i)
        if estimate_tokens(candidate, chars_per_token) <= budget_tokens:
            kept.add(i)
        else:
            dropped.append(part["name"])
    return "\n\n".join(p["text"].strip() for i,p in enumerate(parts) if i in kept), dropped


def _compact_delta(rows, alignment_known=True):
    result = []
    for row in rows:
        item = {k:v for k,v in row.items() if k != "svd"}
        svd = row.get("svd") or {}
        item["svd"] = {k:v for k,v in svd.items() if k != "directions"}
        item["svd"]["directions"] = [{k:v for k,v in d.items() if k not in {"left_vector", "right_vector"}}
                                      for d in svd.get("directions", [])]
        if not alignment_known:
            for direction in item["svd"]["directions"]:
                if direction.get("space") == "vocab":
                    direction.pop("grades", None)
                    direction.pop("top_tokens", None)
                    direction["semantic_projection_unavailable"] = "基座词表对齐未确认"
        result.append(item)
    return result


def _compact_architecture(architecture):
    compact = {k: v for k, v in architecture.items() if k != "layers"}
    compact["layer_kind_counts"] = dict(Counter(str(layer.get("kind", "unknown")) for layer in architecture.get("layers", [])))
    return compact


def _compact_case(case, grades, limit=8):
    result = {k:v for k,v in case.items() if k not in {"head_attr", "mlp_attr", "lens"}}
    writes = []
    for layer, heads in enumerate(case.get("head_attr") or []):
        for head, value in enumerate(heads or []):
            if value is not None:
                writes.append({"layer": layer, "head": head, "kind": "head", "attribution": value})
    for layer, value in enumerate(case.get("mlp_attr") or []):
        if value is not None:
            writes.append({"layer": layer, "kind": "mlp", "attribution": value})
    result["top_readout_components"] = sorted(writes, key=lambda c: abs(c["attribution"]), reverse=True)[:limit]
    result["omitted_component_count"] = max(0, len(writes)-limit)
    lens = case.get("lens") or []
    result["final_grade_logits"] = lens[-1].get("logits") if lens else None
    pred, gold = case.get("pred"), case.get("label")
    if pred in grades and gold in grades:
        pi, gi = grades.index(pred), grades.index(gold)
        result["lens_logit_diff"] = [{"layer": row["layer"], "value": row["logits"][pi]-row["logits"][gi]}
                                     for row in lens if "logits" in row]
    return result


def build_group_prompt(*, architecture, grades, grade_ids, group, mech, hypotheses,
                       delta_top, template, budget_tokens, chars_per_token,
                       token_attr=None, probes=None, verifications=None,
                       grade_mode="single_token", decode_protocol="constrained", delta_report=None,
                       language="zh"):
    observations = []
    def observe(eid, kind, data):
        observations.append({"evidence_id": eid, "kind": kind, "data": data})
    observe("decision", "measurement", {"group": group["key"], "n": group["count"],
            "n_detail": group["n_detail"], "detail_ids": group.get("detail_ids", []),
            "mean_probs": group.get("mean_probs"), "grade_mass": mech.get("grade_mass"),
            "full_top_grade_rate": mech.get("full_top_grade_rate"), "logit_diff_mean": mech.get("logit_diff_mean"),
            "readout_kind": mech.get("readout_kind"), "reconstruction_error": mech.get("reconstruction_error"),
            "attribution_reliable": mech.get("attribution_reliable"), "decomposition_gap": mech.get("decomposition_gap")})
    for result in verifications or []:
        observe("verification:"+result["key"], "intervention", result)
    observe("controls_and_limits", "limitations", {
        "causal_restoration_tested": False,
        "selection_disjoint": bool(verifications) and all(v.get("selection_disjoint", False) for v in verifications),
        "controls_present": any(v.get("control_type") in {"identity", "random"} and not v.get("skipped") and v.get("mean_delta") is not None for v in verifications or []),
        "grade_mode": grade_mode, "decode_protocol": decode_protocol,
        "first_token_note": "首 token 分类不等于完整等级生成或序列似然；本轮未测后续 token。" if grade_mode == "first_token" else None,
        "baseline_note": "权重基座不是正确行为的保证；未运行基座行为对照。",
        "base_grade_alignment": (delta_report or {}).get("base_grade_alignment")})
    for case in mech.get("case_evidence") or []:
        observe("case:" + str(case["id"]), "per_case_measurement", _compact_case(case, grades))
    for c in mech.get("components") or []:
        observe("component:"+c["key"], "readout_attribution", c)
    observe("token_attribution", "local_gradient", token_attr or {"missing": True})
    observe("lens", "readout_measurement", mech.get("lens") or [])
    observe("delta", "weight_geometry", _compact_delta(delta_top, ((delta_report or {}).get("base_grade_alignment") or {}).get("status", "aligned") == "aligned"))
    if delta_report is not None:
        grade_report = delta_report.get("lm_head_grades") or {}
        observe("lm_head_grades", "weight_geometry", {
            "tied_lm_head": delta_report.get("tied_lm_head"),
            "rows": {grade: {k:v for k,v in row.items() if k != "delta_vector"}
                     for grade, row in grade_report.get("rows", {}).items()},
            "delta_row_cosine": grade_report.get("delta_row_cosine"),
            "base_source": grade_report.get("base_source"), "missing_reason": grade_report.get("missing_reason")})
    observe("delta_activations", "local_weight_write", mech.get("delta_contributions") or [])
    observe("probes", "descriptive_probe", probes or {"missing": True})
    observe("examples", "input_excerpt", group.get("examples") or [])
    interpretations = [{"key": h["key"], "status": h["status"], "claim": h["title"],
                        "evidence_ids": h.get("evidence_ids", []), "evidence": h.get("evidence"),
                        "evidence_strength": h.get("evidence_strength", h["status"]),
                        "proposal": h.get("patch"), "proposal_tested": h.get("proposal_tested", False),
                        "diagnostic_only": h.get("diagnostic_only", False), "diagnostic_patch": h.get("diagnostic_patch")} for h in hypotheses]
    predictions = [{"key": h["key"], "would_refute": h.get("would_refute")} for h in hypotheses]
    packet = {"schema_version": 1, "group": group["key"],
              "protocol": {"grades": grades, "grade_ids": grade_ids, "grade_mode": grade_mode,
                           "decode_protocol": decode_protocol, "template": template},
              "architecture": architecture, "observations": observations,
              "interpretations": interpretations, "predictions": predictions}
    header = f"""任务。分析相关性模型混淆组 {group['key']}，引用 evidence_id 提出可证伪的机制解释与最小补丁。
协议：grade_mode={grade_mode}，decode_protocol={decode_protocol}；等级 ID={json.dumps(grade_ids, ensure_ascii=False)}。
输入模板：{template}
架构：{json.dumps(_compact_architecture(architecture), ensure_ascii=False)}
最终 norm DLA 是固定观测 RMS 分母的读出分解，不是总因果效应。pre-norm proxy 不得当真实 logit 贡献。
消融有效不等于根因已验证；检查作用位置、随机/identity 对照和选择/验证是否独立。lm_head 行混合会代数性压缩分差，不证明根因；共享行编辑还会改变输入 embedding。
数量比例、注意力份额、梯度归因不可互换。SVD 几何方向缺少 v dot x 时不能解释某个样本的效应。探针只说明可读出性。
修复率和回退率是不同条件率，不可直接比较净收益；需独立验收与预设回退上限。不要建议梯度训练，不要通过全局缩放或修改 final RMSNorm 调整分布。不得声称未经校准的概率置信度。"""
    ask = """请先说明支持与反证，再给 JSON：
{"group":"金标->预测","mechanism":"待验证解释","evidence_ids":[],"components":[],"minimal_fix":null,"verification":"独立验收与对照方案","would_refute":"可证伪观察","evidence_strength":"不足|描述性|干预有效","insufficient_evidence":true}
每条解释必须引用可见 evidence_id；缺失对照时说明限制。minimal_fix 可用 head_scale、mlp_zero（定位用途）、steering、lm_head_blend；未知共享权重语义时不建议行编辑。不要把缺失或失败的实验写成通过。"""
    ask += "\n" + language_instruction(language)
    # Controls/refutations precede optional explanatory detail. Required material
    # remains valid JSON; budget trimming never slices an observation in half.
    required_observations = [o for o in observations if o["evidence_id"] in {"decision", "controls_and_limits"}]
    parts = [{"name":"header", "text":header, "droppable":False},
             {"name":"core", "text":json.dumps(required_observations, ensure_ascii=False), "droppable":False,
              "provides":[o["evidence_id"] for o in required_observations]}]
    def priority(observation):
        if observation["kind"] == "intervention":
            return 0
        if observation["evidence_id"] in {"token_attribution", "lens", "delta", "lm_head_grades", "delta_activations", "probes"}:
            return 1
        if observation["kind"] == "readout_attribution":
            return 2
        return 4 if observation["evidence_id"] == "examples" else 3
    ordered = sorted([o for o in observations if o not in required_observations], key=priority)
    for o in ordered:
        parts.append({"name":o["evidence_id"], "text":json.dumps(o, ensure_ascii=False), "droppable":True})
    for interpretation, prediction in zip(interpretations, predictions):
        parts.append({"name":"interpretation:"+interpretation["key"],
                      "text":json.dumps({"interpretation":interpretation,"prediction":prediction},ensure_ascii=False),
                      "droppable":True, "requires":interpretation["evidence_ids"]})
    parts.append({"name":"ask", "text":ask, "droppable":False})
    text, dropped = fit_budget(parts, budget_tokens, chars_per_token)
    visible = [o["evidence_id"] for o in observations if o["evidence_id"] not in dropped]
    return {"group":group["key"], "text":text, "dropped":dropped,
            "estimated_tokens":estimate_tokens(text,chars_per_token), "evidence_packet":packet,
            "prompt_evidence_ids":visible}
