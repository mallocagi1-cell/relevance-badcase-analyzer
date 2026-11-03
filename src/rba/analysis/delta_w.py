"""ΔW = W_ft - W_base, one tensor at a time.

Magnitude is not causality. Singular vectors of a residual-writing matrix
(o_proj, down_proj) are projected through lm_head onto the grade tokens and a
short, filtered slice of the vocabulary. Vectors that do not live in the
residual stream are reported as magnitudes only.
"""

from __future__ import annotations

import torch

from rba.models.loader import iter_checkpoint_tensors
from rba.util import cosine


def _iter_pairs(base_model, ft_model, base_path: str | None):
    ft_named = dict(ft_model.named_parameters())
    if base_model is not None:
        for name, base_tensor in base_model.named_parameters():
            if name in ft_named:
                yield name, base_tensor.detach().float().cpu(), ft_named[name].detach().float().cpu()
        return
    if not base_path or base_path in {"tiny", "qwen3_5_tiny"}:
        return
    for name, base_tensor in iter_checkpoint_tensors(base_path):
        if name not in ft_named or not torch.is_tensor(base_tensor):
            continue
        yield name, base_tensor.detach().float().cpu(), ft_named[name].detach().float().cpu()


def _residual_out(name: str, out_features: int, hidden: int) -> bool:
    if out_features != hidden:
        return False
    leaf = name.split(".")[-2] if name.endswith(".weight") else name
    return leaf in {"o_proj", "out_proj", "down_proj"} or name.endswith("lm_head.weight")


def analyze_delta(
    base_model,
    ft_model,
    base_path: str | None,
    lm_head_weight: torch.Tensor,
    grade_ids: dict[str, int],
    tokenizer,
    top_k: int,
    svd_rank: int,
    svd_max_dim: int,
) -> dict:
    hidden = int(lm_head_weight.shape[1])
    stats = []
    missing = []
    ft_names = {name for name, _ in ft_model.named_parameters()}
    seen = set()
    total = 0
    for name, base_tensor, ft_tensor in _iter_pairs(base_model, ft_model, base_path):
        seen.add(name)
        total += ft_tensor.numel()
        if base_tensor.shape != ft_tensor.shape:
            missing.append({"name": name, "reason": "形状不一致", "base": list(base_tensor.shape), "ft": list(ft_tensor.shape)})
            continue
        delta = ft_tensor - base_tensor
        base_norm = float(torch.linalg.vector_norm(base_tensor))
        delta_norm = float(torch.linalg.vector_norm(delta))
        stats.append(
            {
                "name": name,
                "shape": list(ft_tensor.shape),
                "numel": int(ft_tensor.numel()),
                "delta_fro": delta_norm,
                "base_fro": base_norm,
                "rel": delta_norm / (base_norm + 1e-12),
                "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                "global_knob": name.endswith("norm.weight"),
            }
        )
        del delta
    for name in sorted(ft_names - seen):
        missing.append({"name": name, "reason": "基座里没有这个参数"})
    stats.sort(key=lambda row: row["rel"], reverse=True)
    top = stats[:top_k]
    # Second pass only for the matrices we will factor.
    wanted = {row["name"] for row in top if len(row["shape"]) == 2 and not row["global_knob"]}
    factors = {row["name"]: None for row in top}
    if wanted:
        for name, base_tensor, ft_tensor in _iter_pairs(base_model, ft_model, base_path):
            if name not in wanted or base_tensor.shape != ft_tensor.shape:
                continue
            delta = ft_tensor - base_tensor
            factors[name] = _svd_project(
                name,
                delta,
                lm_head_weight.detach().float().cpu(),
                grade_ids,
                tokenizer,
                svd_rank,
                svd_max_dim,
                hidden,
            )
            del delta
    for row in top:
        row["svd"] = factors.get(row["name"])
    grade_report = _grade_row_report(base_model, ft_model, base_path, grade_ids)
    changed = sum(row["numel"] for row in stats if row["rel"] > 1e-8)
    return {
        "n_tensors": len(stats),
        "n_parameters_compared": int(total),
        "n_parameters_changed": int(changed),
        "changed_fraction": (changed / total) if total else 0.0,
        "top": top,
        "missing": missing[:20],
        "n_missing": len(missing),
        "lm_head_grades": grade_report,
    }


def _grade_row_report(base_model, ft_model, base_path, grade_ids: dict[str, int]) -> dict:
    ft_head = ft_model.lm_head.weight.detach().float().cpu()
    base_head = None
    if base_model is not None and hasattr(base_model, "lm_head"):
        base_head = base_model.lm_head.weight.detach().float().cpu()
    elif base_path and base_path not in {"tiny", "qwen3_5_tiny"}:
        for name, tensor in iter_checkpoint_tensors(base_path):
            if name == "lm_head.weight":
                base_head = tensor.detach().float().cpu()
                break
    rows = {}
    ft_rows = {grade: ft_head[index] for grade, index in grade_ids.items()}
    for grade, index in grade_ids.items():
        item = {"ft_norm": float(ft_rows[grade].norm())}
        if base_head is not None:
            base_row = base_head[index]
            delta = ft_rows[grade] - base_row
            item["cosine_to_base"] = cosine(ft_rows[grade], base_row)
            item["delta_norm"] = float(delta.norm())
        rows[grade] = item
    delta_cosine = {}
    if base_head is not None:
        deltas = {grade: ft_rows[grade] - base_head[grade_ids[grade]] for grade in grade_ids}
        for left in grade_ids:
            delta_cosine[left] = {right: cosine(deltas[left], deltas[right]) for right in grade_ids}
    return {"rows": rows, "delta_row_cosine": delta_cosine}


def _svd_project(name, delta, lm_head, grade_ids, tokenizer, rank, max_dim, hidden) -> dict:
    out_features, in_features = delta.shape
    rank = int(min(rank, out_features, in_features))
    if rank < 1:
        return {"skipped": "矩阵太小"}
    if max(out_features, in_features) > max_dim:
        try:
            left, values, _right = torch.svd_lowrank(delta, q=rank, niter=2)
        except RuntimeError as exc:
            return {"skipped": f"低秩 SVD 失败：{exc}"}
    else:
        left, values, _vh = torch.linalg.svd(delta, full_matrices=False)
        left = left[:, :rank]
        values = values[:rank]
    directions = []
    residual = out_features == hidden and (
        name.endswith("o_proj.weight")
        or name.endswith("out_proj.weight")
        or name.endswith("down_proj.weight")
        or name.endswith("lm_head.weight") is False
        and out_features == hidden
        and any(name.endswith(suffix) for suffix in ("o_proj.weight", "out_proj.weight", "down_proj.weight"))
    )
    is_lm_head = name.endswith("lm_head.weight")
    # Residual writers we know how to read: o_proj, out_proj, down_proj.
    residual = (not is_lm_head) and out_features == hidden and any(
        token in name for token in ("o_proj.weight", "out_proj.weight", "down_proj.weight")
    )
    for index in range(rank):
        vector = left[:, index]
        vector = vector / (vector.norm() + 1e-12)
        item = {"singular_value": float(values[index]), "which": index}
        if is_lm_head:
            item["space"] = "vocab"
            item["grades"] = {grade: float(vector[token_id]) for grade, token_id in grade_ids.items()}
            item["top_tokens"] = _topk_vocab_vector(vector, tokenizer, grade_ids)
        elif residual:
            item["space"] = "residual"
            scores = torch.mv(lm_head, vector)
            item["grades"] = {grade: float(scores[token_id]) for grade, token_id in grade_ids.items()}
            item["top_tokens"] = _topk_scores(scores, tokenizer)
        else:
            item["space"] = "not_residual"
        directions.append(item)
    return {"directions": directions, "residual_stream": residual or is_lm_head}


def _safe_token(tokenizer, index: int) -> str:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return f"#{index}"
    try:
        token = str(convert(int(index)))
    except Exception:
        return f"#{index}"
    if not token or any(ord(char) < 32 for char in token):
        return f"#{index}"
    if len(token) > 16:
        return f"#{index}"
    return token


def _topk_scores(scores: torch.Tensor, tokenizer, limit: int = 6) -> list[dict]:
    # Chunked top-k so a 248k vocabulary never becomes a full Python list.
    values, indexes = torch.topk(scores.abs(), k=min(limit * 4, scores.numel()))
    rows = []
    for value, index in zip(values.tolist(), indexes.tolist()):
        token = _safe_token(tokenizer, int(index))
        if token.startswith("#") and not str(index) in token:
            # Keep ids when the surface form is long or non-printable.
            pass
        rows.append({"id": int(index), "token": token, "score": float(scores[index])})
        if len(rows) >= limit:
            break
    return rows


def _topk_vocab_vector(vector: torch.Tensor, tokenizer, grade_ids: dict[str, int]) -> list[dict]:
    values, indexes = torch.topk(vector.abs(), k=min(6, vector.numel()))
    rows = []
    grade_by_id = {index: grade for grade, index in grade_ids.items()}
    for index in indexes.tolist():
        label = grade_by_id.get(int(index))
        token = label if label is not None else _safe_token(tokenizer, int(index))
        rows.append({"id": int(index), "token": token, "score": float(vector[index])})
    return rows
