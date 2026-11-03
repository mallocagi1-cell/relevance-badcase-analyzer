"""Shared pieces for `rba export` and `rba finetune`: loading, scoring, saving."""

from __future__ import annotations

import shutil
from contextlib import ExitStack
from pathlib import Path

import torch

from rba.console import info
from rba.errors import RBAError

# Non-weight files a standard HF checkpoint needs next to the weights. config.json and
# generation_config.json are written by save_pretrained itself.
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.json", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
)

# Logit tolerance between hook-applied patch and baked weights, by dtype. Baking changes
# the order of floating point operations (scale-then-matmul vs matmul-then-scale), so
# low-precision dtypes differ by rounding, not by meaning.
TOLERANCE = {torch.float32: 1e-3, torch.float64: 1e-6, torch.bfloat16: 0.25, torch.float16: 0.05}


def score_items(loaded, items, grade_id_list, grades, decode_protocol, batch_size, desc, patches=None, resolved=None, grade_ids=None, scope="all_positions"):
    """Rows with id/label/pred/grade_logits. `patches` are applied as stacked hooks."""
    from rba.pipeline import _batch, _pad_id, _run_batches, _score_batch, _scoped_patch

    pad_id = _pad_id(loaded.tokenizer)

    def infer(chunk):
        batch = _batch(loaded, chunk, pad_id)
        with ExitStack() as stack:
            for patch in patches or []:
                stack.enter_context(_scoped_patch(loaded.ft, resolved, patch, grade_ids, scope, batch, loaded.device))
            rows = _score_batch(loaded, batch, grade_id_list, grades, decode_protocol)
        return [{**row, "id": item.id, "label": item.case.label, "attributes": item.case.attributes} for item, row in zip(chunk, rows)]

    if not items:
        return []
    rows, _ = _run_batches(items, batch_size, infer, desc)
    return rows


def require_hf(loaded, what: str) -> None:
    if loaded.kind != "hf" or not hasattr(loaded.ft, "save_pretrained"):
        raise RBAError(
            f"{what}要一个真实的 Hugging Face 模型目录，内置小模型（{loaded.kind}）没法存成标准 checkpoint。",
            "把 --ft 指到微调模型目录。只想看流程：rba demo。",
        )


def save_checkpoint(model, source_dir: str | Path, out: Path) -> list[str]:
    """save_pretrained into `out` and copy tokenizer / processor files from the source."""
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    copied = []
    source = Path(source_dir)
    for name in TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, out / name)
            copied.append(name)
    if not any(name.startswith("tokenizer") or name in {"vocab.json", "merges.txt"} for name in copied):
        raise RBAError(
            f"源目录 {source} 里没有分词器文件，导出的目录没法单独加载。",
            "确认 --ft 指的是带 tokenizer.json / tokenizer_config.json 的完整模型目录。",
        )
    info(f"已写出 {out}（权重 + {len(copied)} 个分词器/处理器文件）。")
    return copied


def check_output_dir(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise RBAError(
                f"输出目录 {out} 已经有文件，没有覆盖。",
                "换一个 --out，或加 --overwrite（会删掉这个目录里原有的内容）。",
            )
        shutil.rmtree(out)


def free(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
