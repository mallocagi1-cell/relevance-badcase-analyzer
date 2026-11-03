"""Protocol-aware scoring and descriptive paired repair statistics."""
from __future__ import annotations

import math
import torch

from rba.errors import RBAError
from rba.util import call_model, decision_index, logits_of, move_batch

INVALID_GRADE = "__invalid__"


def forward_scores(model, batch, grade_ids: list[int], device, dtype) -> dict:
    model.eval()
    sentinel = object()
    previous = getattr(model, "_rba_input_ids", sentinel)
    try:
        moved = move_batch({k: v for k, v in batch.items() if k != "case_ids"}, device, dtype)
        model._rba_input_ids = moved.get("input_ids")
        with torch.no_grad():
            logits = logits_of(call_model(model, moved))
        index = decision_index(moved["attention_mask"])
        taken = logits[torch.arange(logits.size(0), device=logits.device), index].float()
        columns = torch.tensor(grade_ids, device=taken.device, dtype=torch.long)
        grade_logits = taken.index_select(-1, columns)
        top = taken.argmax(-1)
        return {
            "grade_logits": grade_logits.cpu(),
            "full_top_id": top.cpu().tolist(),
            "full_top_is_grade": torch.isin(top, columns).cpu().tolist(),
            "grade_mass": (grade_logits - taken.logsumexp(-1, keepdim=True)).exp().sum(-1).cpu().tolist(),
            "grade_ids": list(grade_ids),
        }
    finally:
        if previous is sentinel:
            if hasattr(model, "_rba_input_ids"):
                delattr(model, "_rba_input_ids")
        else:
            model._rba_input_ids = previous


def forward_grade_logits(model, batch, grade_ids: list[int], device, dtype) -> torch.Tensor:
    return forward_scores(model, batch, grade_ids, device, dtype)["grade_logits"]


def predictions_from_logits(grade_logits: torch.Tensor, grades: list[str]) -> list[str]:
    return [grades[int(i)] for i in grade_logits.argmax(-1).tolist()]


def predictions_from_scores(scores: dict, grades: list[str], decode_protocol="constrained") -> list[str]:
    if decode_protocol not in {"constrained", "free"}:
        raise RBAError(f"未知 decode_protocol={decode_protocol}。", "使用 constrained 或 free。")
    preds = predictions_from_logits(scores["grade_logits"], grades)
    if decode_protocol == "free":
        # Full-vocab argmax is authoritative, including tie-breaking by token ID.
        lookup = dict(zip(scores.get("grade_ids", []), grades))
        preds = [lookup.get(top, pred) if legal else INVALID_GRADE
                 for top, legal, pred in zip(scores["full_top_id"], scores["full_top_is_grade"], preds)]
    return preds


def score_rows(scores: dict, grades: list[str], decode_protocol="constrained") -> list[dict]:
    preds = predictions_from_scores(scores, grades, decode_protocol)
    return [{"pred": pred, "grade_logits": scores["grade_logits"][i].tolist(),
             "full_top_id": int(scores["full_top_id"][i]),
             "full_top_is_grade": bool(scores["full_top_is_grade"][i]),
             "legal": bool(scores["full_top_is_grade"][i]),
             "grade_mass": float(scores["grade_mass"][i]), "decode_protocol": decode_protocol}
            for i, pred in enumerate(preds)]


def error_logit(grade_logits, label, pred_grade, grades) -> float:
    return float(grade_logits[grades.index(pred_grade)] - grade_logits[grades.index(label)])


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def metric_block(rows: list[dict], grades: list[str]) -> dict:
    correct = sum(row["label"] == row["pred"] for row in rows)
    mae_terms = []
    for row in rows:
        try:
            mae_terms.append(abs(float(row["label"]) - float(row["pred"])))
        except (ValueError, TypeError):
            mae_terms = []
            break
    legal = [bool(row.get("legal", row.get("full_top_is_grade"))) for row in rows
             if "legal" in row or "full_top_is_grade" in row]
    return {"n": len(rows), "accuracy": correct/len(rows) if rows else None,
            "mae": sum(mae_terms)/len(mae_terms) if mae_terms else None, "n_correct": correct,
            "legal_output_rate": sum(legal)/len(legal) if legal else None,
            "n_legal_measured": len(legal)}


def _paired_counts(before, after):
    wrong = [i for i, row in enumerate(before) if row["label"] != row["pred"]]
    right = [i for i, row in enumerate(before) if row["label"] == row["pred"]]
    fixed = sum(after[i]["pred"] == before[i]["label"] for i in wrong)
    regressed = sum(after[i]["pred"] != before[i]["label"] for i in right)
    n = len(before)
    return {"n": n, "n_wrong_before": len(wrong), "n_right_before": len(right),
            "n_fixed": fixed, "n_regressed": regressed,
            "fix_rate": fixed/len(wrong) if wrong else None,
            "regression_rate": regressed/len(right) if right else None,
            "fix_rate_ci": wilson_interval(fixed, len(wrong)),
            "regression_rate_ci": wilson_interval(regressed, len(right)),
            "net_correct_change": fixed-regressed, "accuracy_delta": (fixed-regressed)/n if n else None}


def compare(before: list[dict], after: list[dict]) -> dict:
    before_ids = [row["id"] for row in before]
    after_ids = [row["id"] for row in after]
    if len(set(before_ids)) != len(before_ids) or len(set(after_ids)) != len(after_ids) or set(before_ids) != set(after_ids):
        raise RBAError("配对评估的样本 ID 重复或不一致。", "修改前后必须包含相同的唯一 ID。")
    by_id = {row["id"]: row for row in after}
    paired = [by_id[row["id"]] for row in before]
    if any(a["label"] != b["label"] for a,b in zip(before, paired)):
        raise RBAError("配对评估金标发生变化。", "保持修改前后的标签一致。")
    result = _paired_counts(before, paired)
    result.update(before=metric_block(before, []), after=metric_block(paired, []))
    result["n_new_illegal"] = sum(bool(a.get("legal", a.get("full_top_is_grade", False))) and
                                 not bool(b.get("legal", b.get("full_top_is_grade", True)))
                                 for a,b in zip(before, paired))
    result["per_grade"] = {}
    result["per_attribute"] = {}
    for label in sorted({r["label"] for r in before}):
        indexes = [i for i,r in enumerate(before) if r["label"] == label]
        result["per_grade"][label] = _paired_counts([before[i] for i in indexes], [paired[i] for i in indexes])
    for key in sorted({k for r in before for k in (r.get("attributes") or {})}):
        values = sorted({str(r["attributes"][key]) for r in before if key in (r.get("attributes") or {})})
        result["per_attribute"][key] = {}
        for value in values:
            indexes = [i for i,r in enumerate(before) if key in (r.get("attributes") or {}) and str(r["attributes"][key]) == value]
            result["per_attribute"][key][value] = _paired_counts([before[i] for i in indexes], [paired[i] for i in indexes])
    result["interval_note"] = "95% Wilson 条件率区间，假设样本独立；未校正多次选择或同 query 聚类。"
    return result


def conclusion(bad: dict, good: dict | None) -> str:
    fix = bad.get("fix_rate")
    text = "坏例修复率无定义（修改前没有错误样本）。" if fix is None else f"坏例修复率 {fix:.1%}。"
    if good is None:
        text += "没有对照集，回退风险未知。"
    elif good.get("regression_rate") is None:
        text += "对照集里没有原本判对的样本，回退率无定义。"
    else:
        text += f"对照回退率 {good['regression_rate']:.1%}。"
    if bad.get("net_correct_change") is not None:
        text += f"当前坏例集净正确数变化 {bad['net_correct_change']:+d}。"
    if good is not None and good.get("net_correct_change") is not None:
        text += f"当前对照集净正确数变化 {good['net_correct_change']:+d}。"
    new_illegal = bad.get("n_new_illegal", 0) + (good or {}).get("n_new_illegal", 0)
    if new_illegal:
        text += f"补丁新增 {new_illegal} 条非法输出，不要保留未经复核的补丁。"
    text += "两项条件率不能直接比较净收益；净收益取决于线上错误占比，应使用独立验收集和预设回退上限。"
    return text
