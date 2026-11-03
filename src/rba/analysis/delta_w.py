"""ΔW = W_ft - W_base, one tensor at a time.

Magnitude is not causality. Singular vectors of a residual-writing matrix
(o_proj, down_proj) are projected through lm_head onto the grade tokens and a
short, filtered slice of the vocabulary. Vectors that do not live in the
residual stream are reported as magnitudes only.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rba.errors import RBAError
from rba.models.loader import iter_checkpoint_tensors
from rba.util import cosine


def _parameter_key(parameter):
    return (parameter.device, parameter.data_ptr(), tuple(parameter.shape), tuple(parameter.stride()))


def _readout_info(model, weight=None):
    # The caller's resolved head tensor is authoritative, including custom paths.
    if weight is not None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and _parameter_key(module.weight) == _parameter_key(weight):
                return f"{name}.weight" if name else "weight", module.weight
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(model, "lm_head", None)
    if head is not None and hasattr(head, "weight"):
        for name, module in model.named_modules():
            if module is head:
                return f"{name}.weight", head.weight
    raise RBAError("无法定位用于 ΔW 的读出矩阵。", "传入 resolved.lm_head(model).weight，检查自定义模块路径。")


def _tied_head(model, weight=None):
    _, head = _readout_info(model, weight)
    return any(isinstance(module, torch.nn.Embedding) and _parameter_key(module.weight) == _parameter_key(head)
               for module in model.modules())


def _iter_pairs(base_model, ft_model, base_path: str | None, head_weight=None, sources=None):
    # Preserve aliases while comparing each physical parameter only once.
    head_name, head_weight = _readout_info(ft_model, head_weight)
    named = dict(ft_model.named_parameters(remove_duplicate=False))
    canonical = {}
    for name, parameter in named.items():
        canonical.setdefault(_parameter_key(parameter), name)
    if _tied_head(ft_model, head_weight):
        canonical[_parameter_key(head_weight)] = head_name
    ft_names = {canonical[_parameter_key(parameter)]: parameter for parameter in named.values()}
    alias = {name: canonical[_parameter_key(parameter)] for name, parameter in named.items()}
    if base_model is not None:
        base_named = dict(base_model.named_parameters(remove_duplicate=False))
        for key, parameter in ft_names.items():
            tensor = base_named.get(key)
            if tensor is not None:
                if sources is not None:
                    sources[key] = key
                yield key, tensor.detach(), parameter.detach()
        return
    if not base_path or base_path in {"tiny", "qwen3_5_tiny"}:
        return
    config_file = Path(base_path) / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    base_tied = config.get("tie_word_embeddings", (config.get("text_config") or {}).get("tie_word_embeddings", False))
    seen = set()
    for name, tensor in iter_checkpoint_tensors(base_path):
        key = alias.get(name)
        if key is None or key in seen or key not in ft_names:
            continue
        if key == head_name and name != key and not base_tied:
            continue  # Never substitute an untied base embedding for its head.
        seen.add(key)
        if sources is not None:
            sources[key] = name
        yield key, tensor.detach(), ft_names[key].detach()


def _norm_stats(base, ft, chunk_size=1_048_576):
    # Only one bounded fp32 chunk of each operand/delta is resident on CPU.
    a, b = base.reshape(-1), ft.reshape(-1)
    base_sq = delta_sq = maximum = 0.0
    changed = 0
    for start in range(0, a.numel(), chunk_size):
        aa = a[start:start+chunk_size].float().cpu()
        dd = b[start:start+chunk_size].float().cpu() - aa
        base_sq += float(aa.double().square().sum())
        delta_sq += float(dd.double().square().sum())
        maximum = max(maximum, float(dd.abs().max()))
        changed += int(torch.count_nonzero(dd))
    return base_sq**.5, delta_sq**.5, maximum, changed


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
    head_name, head_weight = _readout_info(ft_model, lm_head_weight)
    tied = _tied_head(ft_model, head_weight)
    hidden = int(lm_head_weight.shape[1])
    stats = []
    missing = []
    ft_names = {name for name, _ in ft_model.named_parameters()}
    if tied:
        ft_names = {name for name, param in ft_model.named_parameters() if _parameter_key(param) != _parameter_key(head_weight)} | {head_name}
    seen = set()
    total = 0
    sources = {}
    base_grade_rows = None
    for name, base_tensor, ft_tensor in _iter_pairs(base_model, ft_model, base_path, head_weight, sources):
        seen.add(name)
        if base_tensor.shape != ft_tensor.shape:
            missing.append({"name": name, "reason": "形状不一致", "base": list(base_tensor.shape), "ft": list(ft_tensor.shape)})
            continue
        if name == head_name:
            base_grade_rows = base_tensor[list(grade_ids.values())].float().cpu()
        total += ft_tensor.numel()
        base_norm, delta_norm, maximum, changed_elements = _norm_stats(base_tensor, ft_tensor)
        stats.append(
            {
                "name": name,
                "shape": list(ft_tensor.shape),
                "numel": int(ft_tensor.numel()),
                "delta_fro": delta_norm,
                "base_fro": base_norm,
                "rel": delta_norm / (base_norm + 1e-12),
                "max_abs": maximum,
                "changed_elements": changed_elements,
                "global_knob": name.endswith("norm.weight"),
            }
        )
    for name in sorted(ft_names - seen):
        missing.append({"name": name, "reason": "基座里没有这个参数"})
    stats.sort(key=lambda row: row["rel"], reverse=True)
    top = stats[:top_k]
    # Second pass only for the matrices we will factor.
    wanted = {row["name"] for row in top if len(row["shape"]) == 2 and not row["global_knob"]}
    factors = {row["name"]: None for row in top}
    if wanted:
        for name, base_tensor, ft_tensor in _iter_pairs(base_model, ft_model, base_path, head_weight):
            if name not in wanted or base_tensor.shape != ft_tensor.shape:
                continue
            # Bounded working set: large embedding/head matrices still get
            # chunked norm and grade-row reports, but not an unbounded SVD.
            if ft_tensor.numel() > 16_777_216:
                factors[name] = {"skipped": "SVD 工作集超过 16M 元素上限；保留分块统计与等级行"}
                continue
            delta = ft_tensor.float().cpu() - base_tensor.float().cpu()
            factors[name] = _svd_project(
                "lm_head.weight" if name == head_name else name,
                delta,
                lm_head_weight.detach(),
                grade_ids,
                tokenizer,
                svd_rank,
                svd_max_dim,
                hidden,
            )
            del delta
    for row in top:
        row["svd"] = factors.get(row["name"])
    grade_report = _grade_row_report(head_weight, base_grade_rows, grade_ids, sources.get(head_name))
    changed = sum(row["changed_elements"] for row in stats)
    return {
        "tied_lm_head": tied,
        "readout_name": head_name,
        "tied_edit_note": "共享 lm_head 行编辑也改变输入 embedding" if tied else None,
        "n_tensors": len(stats),
        "n_parameters_compared": int(total),
        "n_parameters_changed": int(changed),
        "changed_fraction": (changed / total) if total else 0.0,
        "top": top,
        "missing": missing[:20],
        "n_missing": len(missing),
        "lm_head_grades": grade_report,
    }


def _grade_row_report(head_weight, base_rows, grade_ids, source_name):
    # Base grade rows are retained during pass one (only G×hidden), so no third
    # checkpoint scan and no full-vocabulary fp32 copy are needed.
    ft_rows = head_weight.detach()[list(grade_ids.values())].float().cpu()
    rows, deltas = {}, {}
    for i, grade in enumerate(grade_ids):
        row = {"ft_norm": float(ft_rows[i].norm())}
        if base_rows is not None:
            d = ft_rows[i] - base_rows[i]
            deltas[grade] = d
            row.update(cosine_to_base=cosine(ft_rows[i], base_rows[i]), delta_norm=float(d.norm()),
                       delta_vector=d.tolist(), sign_convention="W_ft - W_base (hidden coordinates; not a logit effect)")
        rows[grade] = row
    return {"rows": rows, "base_source": source_name,
            "missing_reason": "未找到形状匹配的基座读出行" if base_rows is None else None,
            "delta_row_cosine": {a:{b:cosine(x,y) for b,y in deltas.items()} for a,x in deltas.items()}}


def _svd_project(name, delta, lm_head, grade_ids, tokenizer, rank, max_dim, hidden) -> dict:
    out_features, in_features = delta.shape
    rank = int(min(rank, out_features, in_features))
    if rank < 1:
        return {"skipped": "矩阵太小"}
    if max(out_features, in_features) > max_dim:
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                left, values, right = torch.svd_lowrank(delta, q=min(rank + 4, min(delta.shape)), niter=2)
            left, values, right = left[:, :rank], values[:rank], right[:, :rank]
        except RuntimeError as exc:
            return {"skipped": f"低秩 SVD 失败：{exc}"}
    else:
        left, values, vh = torch.linalg.svd(delta, full_matrices=False)
        right = vh[:rank].T
        left = left[:, :rank]
        values = values[:rank]
    energy = float(delta.double().square().sum())
    directions = []
    is_lm_head = name.endswith("lm_head.weight")
    # Residual writers we know how to read: o_proj, out_proj, down_proj.
    residual = (not is_lm_head) and out_features == hidden and any(
        token in name for token in ("o_proj.weight", "out_proj.weight", "down_proj.weight")
    )
    for index in range(rank):
        vector = left[:, index]
        # Deterministic display sign; paired vectors flip together, preserving ΔW.
        sign = -1.0 if vector[vector.abs().argmax()] < 0 else 1.0
        vector = vector * sign
        right_vector = right[:, index] * sign
        item = {"singular_value": float(values[index]), "which": index,
                "left_vector": vector.tolist(), "right_vector": right_vector.tolist(),
                "explained_energy": float(values[index].double().square()) / energy if energy else 0.,
                "sign_convention": "largest-absolute left coordinate positive; effect requires sigma*u*(v dot x)"}
        if is_lm_head:
            item["space"] = "vocab"
            item["grades"] = {grade: float(vector[token_id]) for grade, token_id in grade_ids.items()}
            item["top_tokens"] = _topk_vocab_vector(vector, tokenizer, grade_ids)
        elif residual:
            item["space"] = "residual"
            scores = torch.empty(lm_head.shape[0], dtype=torch.float32)
            for start in range(0, lm_head.shape[0], 2048):
                scores[start:start+2048] = torch.mv(lm_head[start:start+2048].float().cpu(), vector.float().cpu())
            item["grades"] = {grade: float(scores[token_id]) for grade, token_id in grade_ids.items()}
            item["top_tokens"] = _topk_scores(scores, tokenizer)
        else:
            item["space"] = "not_residual"
        directions.append(item)
    explained = sum(d["explained_energy"] for d in directions)
    return {"directions": directions, "residual_stream": residual or is_lm_head,
            "explained_energy": explained, "relative_residual_fro": max(0., 1.-explained)**.5,
            "note": "几何投影不是因果效应；只对实际捕获的模块输入计算 v dot x。"}


def _token_fields(tokenizer, index):
    raw = f"#{index}"
    try:
        raw = str(tokenizer.convert_ids_to_tokens(int(index)))
    except (AttributeError, TypeError, ValueError):
        pass
    display = raw
    try:
        decoded = tokenizer.decode([int(index)], skip_special_tokens=False)
        if decoded and "\ufffd" not in decoded:
            display = decoded
    except (AttributeError, TypeError, ValueError):
        pass
    display = display.replace("\n", "\\n").replace("\r", "\\r")
    return {"token": display, "display": display, "raw_token": raw}


def _safe_token(tokenizer, index):
    return _token_fields(tokenizer, index)["display"]


def _topk_scores(scores: torch.Tensor, tokenizer, limit: int = 6) -> list[dict]:
    # Chunked top-k so a 248k vocabulary never becomes a full Python list.
    values, indexes = torch.topk(scores.abs(), k=min(limit * 4, scores.numel()))
    rows = []
    for value, index in zip(values.tolist(), indexes.tolist()):
        token = _safe_token(tokenizer, int(index))
        if token.startswith("#") and not str(index) in token:
            # Keep ids when the surface form is long or non-printable.
            pass
        rows.append({"id": int(index), **_token_fields(tokenizer, int(index)), "score": float(scores[index])})
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
        rows.append({"id": int(index), **_token_fields(tokenizer, int(index)), "grade": label, "score": float(vector[index])})
    return rows
