"""Image loading.

Two consumers with different needs:

* The built-in tiny stand-in wants a small NCHW tensor (`load_rgb`).
* A real Qwen model must receive the picture at native resolution through its
  own processor (`load_image`). Downscaling before the processor would change
  the input the fine-tune saw, so nothing here resizes for the real path.

PPM (P3/P6/P2) is parsed without Pillow so the demo has no image dependency.
"""

from __future__ import annotations

from pathlib import Path

import torch

from rba.errors import RBAError


def load_image(path: str):
    """Native-resolution RGB PIL image for a real processor."""
    file = Path(path)
    if not file.exists():
        raise RBAError(
            f"找不到图片 {path}。",
            "改成存在的本地路径，或删掉这条样本的 image 字段。URL 不会自动下载。",
        )
    try:
        from PIL import Image
    except ImportError as exc:
        raise RBAError(
            "读取图片需要 Pillow。",
            "pip install pillow（已列为依赖，重新 pip install -e . 即可）。",
        ) from exc
    if file.suffix.lower() == ".ppm":
        array = _read_ppm(file)
        return Image.fromarray(array, mode="RGB")
    try:
        with Image.open(file) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise RBAError(f"读不了图片 {path}：{exc}", "确认文件是完整的 PNG/JPEG/WebP，或转成 PNG。") from exc


def load_rgb(path: str, size: int = 8) -> torch.Tensor:
    """Small float tensor [3, size, size] for the tiny stand-in only."""
    file = Path(path)
    if not file.exists():
        raise RBAError(
            f"找不到图片 {path}。",
            "改成存在的本地路径，或删掉这条样本的 image 字段。URL 不会自动下载。",
        )
    suffix = file.suffix.lower()
    if suffix == ".ppm":
        array = _read_ppm(file)
    else:
        array = _read_pil(file)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return _resize(tensor, size)


def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2] == size and image.shape[-1] == size:
        return image
    image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    return image.squeeze(0)


def _read_ppm(path: Path):
    import numpy as np

    raw = path.read_bytes()
    if raw.startswith(b"P6"):
        text_tokens: list[str] = []
        i = 0
        while len(text_tokens) < 4 and i < len(raw):
            while i < len(raw) and raw[i] in b" \t\r\n":
                i += 1
            if i < len(raw) and raw[i] == ord("#"):
                while i < len(raw) and raw[i] not in b"\n":
                    i += 1
                continue
            j = i
            while j < len(raw) and raw[j] not in b" \t\r\n":
                j += 1
            text_tokens.append(raw[i:j].decode("ascii", errors="replace"))
            i = j
        if len(text_tokens) < 4:
            raise RBAError(f"{path} 的 PPM 头不完整。", "重新导出图片。")
        data_start = i + 1 if i < len(raw) else i
        magic, width, height, maxv = text_tokens
        width, height = int(width), int(height)
        expected = width * height * 3
        blob = raw[data_start : data_start + expected]
        if len(blob) < expected:
            raise RBAError(f"{path} 的像素数据不完整。", "重新导出图片。")
        return np.frombuffer(blob, dtype=np.uint8).reshape(height, width, 3).copy()
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(line.split())
    if len(tokens) < 4 or tokens[0] not in {"P3", "P2"}:
        raise RBAError(
            f"读不了图片 {path}。",
            "支持 P2/P3/P6 PPM，以及安装了 Pillow 时的 PNG/JPEG。",
        )
    magic = tokens[0]
    width, height = int(tokens[1]), int(tokens[2])
    values = [int(v) for v in tokens[4:]]
    if magic == "P3":
        if len(values) < width * height * 3:
            raise RBAError(f"{path} 的像素不够。", "检查 PPM 是否写全。")
        return np.asarray(values[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)
    if len(values) < width * height:
        raise RBAError(f"{path} 的像素不够。", "检查 PPM 是否写全。")
    gray = np.asarray(values[: width * height], dtype=np.uint8).reshape(height, width)
    return np.stack([gray, gray, gray], axis=-1)


def _read_pil(path: Path):
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RBAError(
            f"要读取 {path.suffix} 需要安装 Pillow。",
            "pip install pillow，或把图片转成 PPM。内置示例是 examples/images/sample.ppm。",
        ) from exc
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb)
