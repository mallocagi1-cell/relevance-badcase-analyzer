"""Turn cases into the same model inputs the fine-tune saw.

Two paths:

* **Real model (processor available).** Every row, with or without an image,
  goes through the model's own processor so text and image rows share one
  protocol. `template` mode feeds `prompt_template` with `{image}` replaced by
  `image_placeholder`; the processor expands the placeholder to the right number
  of image tokens from the real picture at native resolution. `chat` mode uses
  the processor's chat template for every row. Nothing here resizes images or
  hand-computes token counts.
* **Tiny stand-in (no processor).** `{image}` becomes `image_token` repeated
  `image_seq_len` times and the picture is a small tensor scattered onto those
  positions by the stand-in itself.

A missing or unreadable image never degrades a row to text-only: that would
change the protocol silently. It is an error, or with `missing_image: exclude`
the row is dropped and listed in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rba.config import Config
from rba.console import warn
from rba.data.images import load_image, load_rgb
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
    modalities: set = field(default_factory=lambda: {"text"})
    n_image_tokens: int = 0
    image_grid_thw: list | None = None
    protocol: str = "template"

    @property
    def has_image(self) -> bool:
        return "image" in self.modalities


@dataclass
class ExcludedCase:
    id: str
    reason: str


def render_text(case: Case, cfg: Config, image_token_count: int | None, placeholder: str | None = None) -> str:
    """Fill the template. `{image}` is only the placeholder the fine-tune saw, or empty.

    Tiny path: `image_token` × count. Real path: `placeholder`, which the
    processor expands. Any caption such as "图像：" belongs in the template.
    """
    if case.image and "{image}" in cfg.prompt_template:
        if placeholder is not None:
            image_block = placeholder
        elif image_token_count:
            image_block = cfg.image_token * image_token_count
        else:
            image_block = ""
    else:
        image_block = ""
    try:
        return cfg.prompt_template.format(query=case.query, doc=case.doc, image=image_block)
    except (KeyError, IndexError) as exc:
        raise RBAError(
            f"模板里有无法填的占位符 {exc}。",
            "只使用 {query}、{doc}、{image}。{image} 在没有图片时会变成空字符串。",
        ) from exc


def encode_cases(
    cases: list[Case],
    tokenizer,
    cfg: Config,
    processor=None,
    image_size: int = 8,
    excluded: list[ExcludedCase] | None = None,
) -> list[EncodedCase]:
    """Encode every case. Rows excluded under `missing_image: exclude` go to `excluded`."""
    use_processor = processor is not None
    if use_processor and cfg.input_mode == "template" and any(case.image for case in cases) and "{image}" not in cfg.prompt_template:
        raise RBAError(
            "有图片样本，但 prompt_template 里没有 {image}，图片无法进入模型。",
            "在模板里和训练时一样的位置写 {image}；或把 input_mode 设为 chat 走对话模板。",
        )
    if use_processor and cfg.input_mode == "chat" and getattr(processor, "chat_template", None) is None:
        raise RBAError(
            "input_mode=chat，但这个处理器没有 chat_template。",
            "改用 input_mode: template，并让 prompt_template 与训练输入一致。",
        )
    encoded: list[EncodedCase] = []
    truncated = 0
    for case in cases:
        image = None
        if case.image:
            try:
                image = load_image(case.image) if use_processor else load_rgb(case.image, size=image_size)
            except RBAError as exc:
                if cfg.missing_image == "exclude" and excluded is not None:
                    excluded.append(ExcludedCase(id=case.id, reason=f"{exc.message} {exc.hint}".strip()))
                    warn(f"{case.id} 已剔除：{exc.message}")
                    continue
                raise RBAError(
                    f"样本 {case.id} 的图片读不到：{exc.message}",
                    "修好路径；或加 --allow-missing-images（missing_image: exclude）把这条样本剔除。程序不会把有图样本悄悄当成纯文本。",
                ) from exc
        if use_processor:
            item = _encode_with_processor(case, cfg, processor, image)
        else:
            item = _encode_with_tokenizer(case, cfg, tokenizer, image)
        if item.truncated:
            truncated += 1
        if not item.input_ids:
            raise RBAError(
                f"样本 {case.id} 编码后是空的。",
                "检查模板和 query 是否为空，以及 max_length 是否过小。",
            )
        encoded.append(item)
    if truncated:
        warn(f"{truncated} 条样本超过 max_length={cfg.max_length}，已截掉开头的文本 token。决策 token 在末尾，图像 token 没有被切。")
    if not use_processor and any(case.image for case in cases) and "{image}" not in cfg.prompt_template:
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


def _encode_with_tokenizer(case: Case, cfg: Config, tokenizer, pixels: torch.Tensor | None) -> EncodedCase:
    count = cfg.image_seq_len if pixels is not None else 0
    text = render_text(case, cfg, count)
    ids = list(_encode_text(tokenizer, text))
    image_token_id = getattr(tokenizer, "image_token_id", None)
    was_truncated = False
    if len(ids) > cfg.max_length:
        cut = len(ids) - cfg.max_length
        if image_token_id is not None and any(token == image_token_id for token in ids[:cut]):
            raise RBAError(
                f"样本 {case.id} 有 {len(ids)} 个 token，按 max_length={cfg.max_length} 截断会切掉图像 token。",
                "把 max_length 调大，或缩短模板/文档。程序不会把图像块切成一半。",
            )
        ids = ids[cut:]
        was_truncated = True
    n_image = sum(1 for token in ids if image_token_id is not None and token == image_token_id)
    return EncodedCase(
        id=case.id,
        case=case,
        input_ids=ids,
        attention_mask=[1] * len(ids),
        pixel_values=pixels,
        text=text,
        truncated=was_truncated,
        modalities={"text", "image"} if pixels is not None else {"text"},
        n_image_tokens=n_image,
        protocol="template",
    )


def _image_kwargs(cfg: Config) -> dict:
    if cfg.image_max_pixels is None:
        return {}
    return {"max_pixels": int(cfg.image_max_pixels)}


def _call_processor(processor, cfg: Config, text: str, image):
    images = [image] if image is not None else None
    kwargs = _image_kwargs(cfg) if image is not None else {}
    try:
        return processor(text=[text], images=images, return_tensors="pt", **kwargs)
    except TypeError:
        if not kwargs:
            raise
        # Older processors do not take image kwargs in __call__; set them on the image processor.
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            for key, value in kwargs.items():
                if hasattr(image_processor, key):
                    setattr(image_processor, key, value)
            size = getattr(image_processor, "size", None)
            if isinstance(size, dict) and "longest_edge" in size and "max_pixels" in kwargs:
                size["longest_edge"] = kwargs["max_pixels"]
        return processor(text=[text], images=images, return_tensors="pt")


def _chat_inputs(processor, cfg: Config, text: str, image):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    kwargs = _image_kwargs(cfg) if image is not None else {}
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )


def _encode_with_processor(case: Case, cfg: Config, processor, image) -> EncodedCase:
    mode = cfg.input_mode
    if mode == "auto":
        mode = "chat" if getattr(processor, "chat_template", None) else "template"
    if mode == "chat":
        text = render_text(case, cfg, image_token_count=0)
        text = text.replace("{image}", "")
    else:
        text = render_text(case, cfg, image_token_count=None, placeholder=cfg.image_placeholder if image is not None else "")
    try:
        inputs = _chat_inputs(processor, cfg, text, image) if mode == "chat" else _call_processor(processor, cfg, text, image)
    except RBAError:
        raise
    except Exception as exc:
        raise RBAError(
            f"样本 {case.id} 用处理器编码失败：{exc}",
            "template 模式要求 prompt_template 与训练完全一致，图像占位用 {image}；chat 模式要求处理器带 chat_template。"
            " 图片太大时把 image_max_pixels 调小。",
        ) from exc
    if "input_ids" not in inputs:
        raise RBAError(f"处理器没有返回 input_ids（样本 {case.id}）。", "确认 AutoProcessor 加载的是这个模型的处理器。")
    tensors = {key: value for key, value in inputs.items() if torch.is_tensor(value)}
    input_ids = tensors["input_ids"]
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise RBAError(f"处理器为样本 {case.id} 返回了 {tuple(input_ids.shape)} 形状的 input_ids。", "每条样本应单独编码为 [1, 长度]。")
    if "attention_mask" not in tensors:
        tensors["attention_mask"] = torch.ones_like(input_ids)
    image_token_id = _processor_image_token_id(processor)
    n_image = int((input_ids[0] == image_token_id).sum()) if image_token_id is not None else 0
    if image is not None:
        if "pixel_values" not in tensors:
            raise RBAError(
                f"样本 {case.id} 有图片，但处理器没有返回 pixel_values。",
                "确认处理器是这个视觉模型的 AutoProcessor，并且 Pillow 已安装。",
            )
        if n_image == 0:
            raise RBAError(
                f"样本 {case.id} 有图片，但编码后没有任何图像 token。",
                f"template 模式下 {{image}} 会换成 image_placeholder（当前 {cfg.image_placeholder!r}），它必须是处理器认识的图像占位；"
                "chat 模式交给处理器自己放。检查 image_placeholder 和 image_token 是否对得上这个模型。",
            )
    if image is not None and grid_expected(processor, tensors.get("image_grid_thw")) not in (None, n_image):
        raise RBAError(
            f"样本 {case.id} 的图像 token 数 {n_image} 和处理器网格推出的 {grid_expected(processor, tensors.get('image_grid_thw'))} 不一致。",
            "image_placeholder 和处理器的展开规则对不上，或模板里有多余的图像占位。检查 image_placeholder / image_token。",
        )
    length = input_ids.size(1)
    was_truncated = False
    if length > cfg.max_length:
        cut = length - cfg.max_length
        protected = _vision_block_start(input_ids[0], image_token_id, processor)
        if protected is not None and cut > protected:
            raise RBAError(
                f"样本 {case.id} 编码后有 {length} 个 token，按 max_length={cfg.max_length} 截断会切进视觉块（从位置 {protected} 开始）。",
                "把 max_length 调大，把 image_max_pixels 调小，或缩短文档。程序不会切开 vision_start / image_pad / vision_end。",
            )
        for key in list(tensors):
            value = tensors[key]
            if value.dim() == 2 and value.size(0) == 1 and value.size(1) == length:
                tensors[key] = value[:, cut:]
        input_ids = tensors["input_ids"]
        was_truncated = True
    grid = tensors.get("image_grid_thw")
    return EncodedCase(
        id=case.id,
        case=case,
        input_ids=input_ids[0].tolist(),
        attention_mask=tensors["attention_mask"][0].tolist(),
        model_inputs=tensors,
        text=text,
        truncated=was_truncated,
        modalities={"text", "image"} if image is not None else {"text"},
        n_image_tokens=n_image,
        image_grid_thw=grid.tolist() if grid is not None else None,
        protocol=mode,
    )


def _special_id(processor, attr: str, default_token: str) -> int | None:
    token = getattr(processor, attr, None) or default_token
    tokenizer = getattr(processor, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        value = convert(token)
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if value is None or (unk is not None and value == unk):
        return None
    return int(value)


def _vision_block_start(ids: torch.Tensor, image_token_id: int | None, processor) -> int | None:
    """First position that belongs to a vision block: the vision_start delimiter when
    the tokenizer has one, otherwise the first image token. Returns None without images."""
    if image_token_id is None:
        return None
    image_positions = torch.nonzero(ids == image_token_id, as_tuple=False).flatten()
    if image_positions.numel() == 0:
        return None
    start = int(image_positions[0])
    start_id = _special_id(processor, "vision_start_token", "<|vision_start|>")
    if start_id is not None:
        starts = torch.nonzero(ids == start_id, as_tuple=False).flatten()
        if starts.numel():
            start = min(start, int(starts[0]))
    return start


def grid_expected(processor, grid) -> int | None:
    """Image tokens the processor should have inserted for this grid, or None if unknown."""
    if grid is None:
        return None
    image_processor = getattr(processor, "image_processor", None)
    merge = getattr(image_processor, "merge_size", None)
    if not isinstance(merge, int) or merge <= 0:
        return None
    try:
        rows = grid.tolist() if torch.is_tensor(grid) else list(grid)
        total = 0
        for row in rows:
            t, h, w = (int(v) for v in row)
            total += (t * h * w) // (merge * merge)
        return total
    except (TypeError, ValueError):
        return None


def _processor_image_token_id(processor) -> int | None:
    value = getattr(processor, "image_token_id", None)
    if isinstance(value, int):
        return value
    tokenizer = getattr(processor, "tokenizer", None)
    token = getattr(processor, "image_token", None) or "<|image_pad|>"
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            result = convert(token)
            return int(result) if result is not None else None
        except Exception:
            return None
    return None


def needs_single_item_batches(items: list[EncodedCase]) -> bool:
    """Processor outputs carry packed vision tensors; this round they go one row per forward."""
    return any(item.model_inputs is not None for item in items)


def collate(items: list[EncodedCase], pad_id: int) -> dict:
    if any(item.model_inputs is not None for item in items):
        if len(items) != 1:
            raise RBAError(
                "带处理器输出的样本只能逐条前向。",
                "这是程序内部错误：pipeline 应当已把 batch 降为 1。请保留命令行并反馈。",
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
        # The stand-in scatters by image-token order, which matches the order
        # of the examples that actually carry pixels.
        batch["pixel_values"] = torch.stack(pixels, dim=0)
    return batch
