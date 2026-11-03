"""Small numeric and hashing helpers shared by the analyses."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from rba.errors import RBAError


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().float().cpu().tolist()
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (bytes, str)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(to_builtin(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    text = str(path)
    if text in {"tiny", "qwen3_5_tiny"}:
        return {"kind": text}
    p = Path(text)
    if not p.exists():
        return {"missing": text}
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        stat = p.stat()
        return {"file": str(p), "size": stat.st_size, "sha256": h.hexdigest()}
    cfg = p / "config.json"
    weight_files = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin"))
    parts = []
    for wf in weight_files:
        stat = wf.stat()
        parts.append({"name": wf.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    cfg_hash = None
    if cfg.exists():
        cfg_hash = hashlib.sha256(cfg.read_bytes()).hexdigest()
    return {"dir": str(p.resolve()), "config_sha256": cfg_hash, "weights": parts}


def decision_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-padding token. Works for left or right padding."""
    if attention_mask.dtype != torch.bool:
        valid = attention_mask != 0
    else:
        valid = attention_mask
    flipped = valid.flip(-1)
    # argmax of a bool tensor returns the first True, i.e. the last valid token.
    offset = flipped.long().argmax(-1)
    last = valid.size(-1) - 1 - offset
    empty = ~valid.any(-1)
    if empty.any():
        raise RBAError(
            "有一条样本的 attention_mask 全是填充，没有可打分的位置。",
            "检查 max_length 是否把整段输入都截没了，或模板是否为空。",
        )
    return last


def call_model(model: torch.nn.Module, kwargs: dict) -> Any:
    forward = model.forward
    try:
        sig = inspect.signature(forward)
    except (TypeError, ValueError):
        sig = None
    payload = {k: v for k, v in kwargs.items() if v is not None}
    payload.setdefault("use_cache", False)
    if sig is not None and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        allowed = set(sig.parameters)
        payload = {k: v for k, v in payload.items() if k in allowed}
    return model(**payload)


def logits_of(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    raise RBAError(
        "模型 forward 没有返回 logits。",
        "确认加载的是带 lm_head 的条件生成模型，而不是裸 backbone。",
    )


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype | None = None) -> dict:
    out = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.is_floating_point():
            out[key] = value.to(device=device, dtype=dtype or value.dtype)
        else:
            out[key] = value.to(device)
    return out


def batched(items: list, batch_size: int) -> Iterable[list]:
    step = max(1, batch_size)
    for start in range(0, len(items), step):
        yield items[start : start + step]


def oom_message(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda" in text and "memory" in text


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm()) + 1e-12
    return float((a * b).sum() / denom)


def grade_rows(lm_head: torch.nn.Module, grade_ids: dict[str, int]) -> dict[str, torch.Tensor]:
    weight = lm_head.weight.detach().float().cpu()
    return {grade: weight[index].clone() for grade, index in grade_ids.items()}
