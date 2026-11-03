"""Score a patch on the badcase file and the optional control set."""

from __future__ import annotations

import torch

from rba.util import call_model, decision_index, logits_of, move_batch


def forward_grade_logits(model, batch, grade_ids: list[int], device, dtype) -> torch.Tensor:
    model._rba_input_ids = batch.get("input_ids")
    try:
        moved = move_batch({key: value for key, value in batch.items() if key != "case_ids"}, device, dtype)
        if model._rba_input_ids is not None:
            model._rba_input_ids = moved["input_ids"]
        with torch.no_grad():
            logits = logits_of(call_model(model, moved))
        index = decision_index(moved["attention_mask"])
        taken = logits[torch.arange(logits.size(0), device=logits.device), index]
        columns = torch.tensor(grade_ids, device=taken.device, dtype=torch.long)
        return taken.index_select(-1, columns).float().cpu()
    finally:
        if hasattr(model, "_rba_input_ids"):
            delattr(model, "_rba_input_ids")


def predictions_from_logits(grade_logits: torch.Tensor, grades: list[str]) -> list[str]:
    indexes = grade_logits.argmax(dim=-1).tolist()
    return [grades[int(index)] for index in indexes]


def error_logit(grade_logits: torch.Tensor, label: str, pred_grade: str, grades: list[str]) -> float:
    """logit(reference pred) - logit(label). Caller passes the original prediction."""
    pred_index = grades.index(pred_grade)
    gold_index = grades.index(label)
    return float(grade_logits[pred_index] - grade_logits[gold_index])


def metric_block(rows: list[dict], grades: list[str]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": None, "mae": None, "n_correct": 0}
    correct = [row["label"] == row["pred"] for row in rows]
    mae_terms = []
    for row in rows:
        try:
            mae_terms.append(abs(float(row["label"]) - float(row["pred"])))
        except ValueError:
            mae_terms = []
            break
    return {
        "n": len(rows),
        "accuracy": sum(correct) / len(rows),
        "n_correct": int(sum(correct)),
        "mae": (sum(mae_terms) / len(mae_terms)) if mae_terms else None,
    }


def compare(before: list[dict], after: list[dict]) -> dict:
    by_id = {row["id"]: row for row in after}
    wrong = [row for row in before if row["label"] != row["pred"]]
    right = [row for row in before if row["label"] == row["pred"]]
    fixed = 0
    for row in wrong:
        if by_id[row["id"]]["pred"] == row["label"]:
            fixed += 1
    regressed = 0
    for row in right:
        if by_id[row["id"]]["pred"] != row["label"]:
            regressed += 1
    return {
        "n": len(before),
        "n_wrong_before": len(wrong),
        "n_right_before": len(right),
        "n_fixed": fixed,
        "fix_rate": (fixed / len(wrong)) if wrong else None,
        "n_regressed": regressed,
        "regression_rate": (regressed / len(right)) if right else None,
        "before": metric_block(before, []),
        "after": metric_block(list(by_id[row["id"]] for row in before), []),
    }


def conclusion(bad: dict, good: dict | None) -> str:
    fix = bad.get("fix_rate")
    reg = None if not good else good.get("regression_rate")
    if fix is None:
        return "坏例在修改前已经全部预测正确，修复率没有定义。先确认坏例文件是不是真的判错了。"
    if reg is None:
        return f"坏例修复率 {fix:.1%}。没有对照集，看不出有没有伤及原本判对的样本。加上 --goodcases 再评一次。"
    if fix > reg:
        return f"坏例修复率 {fix:.1%}，对照回退率 {reg:.1%}。修复多于回退。仍建议在更大的对照集上再确认一次，抽样复核不能代替这一步。"
    return f"坏例修复率 {fix:.1%}，对照回退率 {reg:.1%}。回退不少于修复，不要保留这个补丁。"
