"""Load a small RGB tensor. PPM is built in; PNG/JPEG uses Pillow when present."""

from __future__ import annotations

from pathlib import Path

import torch

from rba.errors import RBAError


def load_rgb(path: str, size: int = 8) -> torch.Tensor:
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
        header, _, rest = raw.partition(b"\n")
        # Allow comments by a simple re-parse.
        tokens = []
        body = raw
        # split header tokens
        text_tokens = []
        data_start = 0
        i = 0
        while len(text_tokens) < 4:
            # read next token in ascii header
            while i < len(raw) and raw[i] in b" \t\r\n":
                i += 1
            if i < len(raw) and raw[i] == ord("#"):
                while i < len(raw) and raw[i] not in b"\n":
                    i += 1
                continue
            j = i
            while j < len(raw) and raw[j] not in b" \t\r\n":
                j += 1
            text_tokens.append(raw[i:j].decode("ascii"))
            i = j
        data_start = i
        while data_start < len(raw) and raw[data_start] in b" \t\r\n":
            data_start += 1
        magic, width, height, maxv = text_tokens
        if magic != "P6":
            raise RBAError(f"{path} 不是 P6 PPM。", "另存为 PNG，或用 examples/images/sample.ppm 这种纯文本 PPM。")
        width, height, maxv = int(width), int(height), int(maxv)
        expected = width * height * 3
        blob = raw[data_start : data_start + expected]
        if len(blob) < expected:
            raise RBAError(f"{path} 的像素数据不完整。", "重新导出图片。")
        return np.frombuffer(blob, dtype=np.uint8).reshape(height, width, 3).copy()
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(line.split())
    if len(tokens) < 4 or tokens[0] not in {"P3", "P2"}:
        raise RBAError(
            f"读不了图片 {path}。",
            "支持 P3/P6 PPM，以及安装了 Pillow 时的 PNG/JPEG。",
        )
    magic = tokens[0]
    width, height, maxv = int(tokens[1]), int(tokens[2]), int(tokens[3])
    values = [int(v) for v in tokens[4:]]
    import numpy as np

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
