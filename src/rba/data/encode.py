"""Turn cases into the same token ids the fine-tune saw."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rba.config import Config
from rba.console import warn
from rba.data.images import load_rgb
from rba.errors import RBAError
from rba.schema import Case


@dataclass
class EncodedCase:
    id: str
    case: Case
    input_ids: list[int] | None
    attention_mask: list[int] | None
    pixel_values: torch.Tensor | None = None
    model_inputs: dict | None = None
    text: str = ""
    image_token_positions: list[int] = field(default_factory=list)
    truncated: bool = False


def render_text(case: Case, cfg: Config, image_token_count: int | None) -> str:
    # {image} is only the placeholder tokens the fine-tune saw, or empty.
    # Put any caption such as "图像：" in the template yourself.
    if case.image and image_token_count and "{image}" in cfg.prompt_template:
        image_block = cfg.image_token * image_token_count + "\n"
    else:
        image_block = ""
    try:
        return cfg.prompt_template.format(query=case.query, doc=case.doc, image=image_block)
    except KeyError as exc:
        raise RBAError(
            f"模板里有无法填的占位符 {exc}。",
            "只使用 {query}、{doc}、{image}。{image} 在没有图片时会变成空字符串。",
        ) from exc


def encode_cases(cases: list[Case], tokenizer, cfg: Config, processor=None, image_size: int = 8) -> list[EncodedCase]:
    encoded = []
    truncated = 0
    missing_images = 0
    for case in cases:
        image_ok = False
        pixels = None
        if case.image:
            try:
                pixels = load_rgb(case.image, size=image_size)
                image_ok = True
            except RBAError as exc:
                missing_images += 1
                warn(f"{case.id} 的图片跳过：{exc.message} {exc.hint}")
                pixels = None
        use_processor = processor is not None and cfg.input_mode in {"chat", "auto"} and image_ok
        if use_processor:
            item = _encode_with_processor(case, cfg, processor, pixels)
        else:
            count = cfg.image_seq_len if image_ok else 0
            text = render_text(case, cfg, count)
            ids = list(_encode_text(tokenizer, text))
            if len(ids) > cfg.max_length:
                ids = ids[-cfg.max_length :]
                truncated += 1
                was_truncated = True
            else:
                was_truncated = False
            item = EncodedCase(
                id=case.id,
                case=case,
                input_ids=ids,
                attention_mask=[1] * len(ids),
                pixel_values=pixels if image_ok else None,
                text=text,
                truncated=was_truncated,
            )
        if not item.input_ids and item.model_inputs is None:
            raise RBAError(
                f"样本 {case.id} 编码后是空的。",
                "检查模板和 query 是否为空，以及 max_length 是否过小。",
            )
        encoded.append(item)
    if truncated:
        warn(f"{truncated} 条样本超过 max_length={cfg.max_length}，已保留末尾。决策 token 在末尾，截断的是开头。")
    if missing_images:
        warn(f"{missing_images} 条样本的图片没有读入，已按纯文本继续。")
    if any(case.image for case in cases) and "{image}" not in cfg.prompt_template and cfg.input_mode == "template":
        warn("有图片样本，但模板里没有 {image}，图片 token 不会写进输入。")
    return encoded


def _encode_text(tokenizer, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if encode is None and hasattr(tokenizer, "tokenizer"):
        encode = tokenizer.tokenizer.encode
    try:
        return list(encode(text, add_special_tokens=False))
    except TypeError:
        return list(encode(text))


def _encode_with_processor(case: Case, cfg: Config, processor, pixels: torch.Tensor) -> EncodedCase:
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RBAError(
            "input_mode=chat 需要 Pillow 才能把图片交给处理器。",
            "pip install pillow，或把 input_mode 设为 template，并用 {image} 占位。",
        ) from exc
    array = (pixels.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    image = Image.fromarray(array)
    text_only = render_text(case, cfg, image_token_count=0)
    try:
        if cfg.input_mode == "chat" and hasattr(processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_only},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=[text_only], images=[image], return_tensors="pt")
    except Exception as exc:
        raise RBAError(
            f"样本 {case.id} 用处理器编码失败：{exc}",
            "微调若没用对话模板，把 input_mode 设为 template，并让 prompt_template 与训练完全一致。图像占位用 {image}。",
        ) from exc
    input_ids = inputs["input_ids"][0].tolist() if "input_ids" in inputs else None
    mask = inputs["attention_mask"][0].tolist() if "attention_mask" in inputs else None
    single = {key: value for key, value in inputs.items()}
    return EncodedCase(
        id=case.id,
        case=case,
        input_ids=input_ids,
        attention_mask=mask,
        model_inputs=single,
        text=text_only,
    )


def collate(items: list[EncodedCase], pad_id: int) -> dict:
    if any(item.model_inputs is not None for item in items):
        if len(items) != 1:
            raise RBAError(
                "带处理器输出的样本只能逐条前向。",
                "把 batch_size 设为 1，或改用 input_mode: template。",
            )
        batch = {key: value for key, value in items[0].model_inputs.items()}
        batch["case_ids"] = [items[0].id]
        return batch
    width = max(len(item.input_ids) for item in items)
    ids = []
    mask = []
    pixels = []
    for item in items:
        pad = width - len(item.input_ids)
        ids.append([pad_id] * pad + item.input_ids)
        mask.append([0] * pad + [1] * len(item.input_ids))
        if item.pixel_values is not None:
            pixels.append(item.pixel_values)
    batch = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "case_ids": [item.id for item in items],
    }
    if pixels:
        if len(pixels) != len(items):
            # Keep a side list; the model scatters by image-token order, which
            # matches the order of examples that actually have pixels.
            batch["pixel_values"] = torch.stack(pixels, dim=0)
        else:
            batch["pixel_values"] = torch.stack(pixels, dim=0)
    return batch
