"""Load the fine-tuned model and, when needed, stream the base checkpoint.

Activation analyses keep one model on the chosen device. Delta-W reads the
base tensors one at a time so a 2B bf16 pair is not required to sit in GPU
memory together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from rba.config import Config
from rba.console import info, warn
from rba.errors import RBAError
from rba.models.paths import CLASS_CANDIDATES, preset_for_architecture
from rba.models.qwen_tiny import build_qwen35_tiny_pair
from rba.models.tiny import TinyTokenizer, build_tiny_pair


@dataclass
class LoadedModels:
    ft: torch.nn.Module
    base: torch.nn.Module | None
    tokenizer: object
    processor: object | None
    kind: str
    device: torch.device
    dtype: torch.dtype
    preset: str
    plant: dict | None
    image_token_id: int | None
    grade_ids: dict
    base_path: str | None
    ft_path: str | None


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RBAError(
            "配置要求 CUDA，但这台机器没有可用的 GPU。",
            "把 device 改成 cpu，或在有 GPU 的机器上跑。2B bf16 大约需要 5GB 显存再加激活。",
        )
    return torch.device(name)


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        name = "bfloat16" if device.type == "cuda" else "float32"
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in table:
        raise RBAError(
            f"无法识别的 dtype：{name}。",
            "可选 auto、float32、float16、bfloat16。CPU 上请用 float32。",
        )
    dtype = table[name]
    if device.type == "cpu" and dtype is not torch.float32:
        warn(f"CPU 上把 dtype 从 {name} 改成了 float32。半精度在 CPU 上又慢又容易溢出。")
        return torch.float32
    return dtype


def _read_config_json(path: Path) -> dict:
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        raise RBAError(
            f"{path} 下没有 config.json，无法识别模型。",
            "把路径指到 Hugging Face 模型目录。只想跑通流程时用 --base tiny --ft tiny。",
        )
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RBAError(f"{cfg_path} 不是合法 JSON。", "重新导出模型，或检查文件是否下完整。") from exc


def detect_directory(path: Path) -> dict:
    raw = _read_config_json(path)
    architectures = raw.get("architectures") or []
    architecture = architectures[0] if architectures else None
    model_type = raw.get("model_type")
    text = raw.get("text_config") or raw
    return {
        "architecture": architecture,
        "model_type": model_type,
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "num_attention_heads": text.get("num_attention_heads"),
        "vocab_size": text.get("vocab_size"),
        "image_token_id": raw.get("image_token_id"),
        "torch_dtype": raw.get("torch_dtype"),
        "layer_types": text.get("layer_types"),
        "preset": preset_for_architecture(architecture, model_type),
        "path": str(path),
    }


def load_models(cfg: Config) -> LoadedModels:
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype, device)
    kind = _kind(cfg)
    info(f"设备 {device}，dtype {str(dtype).replace('torch.', '')}，模型类型 {kind}。")
    if kind == "tiny":
        base, ft, tokenizer, meta = build_tiny_pair(cfg)
        ft.to(device=device, dtype=dtype)
        if base is not None:
            base.to(device="cpu", dtype=torch.float32)
        return LoadedModels(
            ft=ft,
            base=base,
            tokenizer=tokenizer,
            processor=None,
            kind="tiny",
            device=device,
            dtype=dtype,
            preset="qwen3_5",
            plant=meta.get("plant"),
            image_token_id=meta.get("image_token_id"),
            grade_ids=resolve_grade_ids(tokenizer, cfg.grades, cfg.grade_token_ids),
            base_path="tiny",
            ft_path="tiny",
        )
    if kind == "qwen3_5_tiny":
        base, ft, tokenizer, meta = build_qwen35_tiny_pair(cfg)
        ft.to(device=device, dtype=dtype)
        base.to(device="cpu", dtype=torch.float32)
        return LoadedModels(
            ft=ft,
            base=base,
            tokenizer=tokenizer,
            processor=None,
            kind="qwen3_5_tiny",
            device=device,
            dtype=dtype,
            preset="qwen3_5",
            plant=meta.get("plant"),
            image_token_id=meta.get("image_token_id"),
            grade_ids=resolve_grade_ids(tokenizer, cfg.grades, cfg.grade_token_ids),
            base_path="qwen3_5_tiny",
            ft_path="qwen3_5_tiny",
        )
    ft_path = Path(cfg.ft_model)
    base_path = Path(cfg.base_model)
    detected = detect_directory(ft_path)
    info(
        f"识别到微调模型架构 {detected['architecture'] or detected['model_type']}，"
        f"{detected['num_hidden_layers']} 层，hidden {detected['hidden_size']}。"
    )
    tokenizer, processor = _load_tokenizer(ft_path)
    grade_ids = resolve_grade_ids(tokenizer, cfg.grades, cfg.grade_token_ids)
    info(f"等级 token：{grade_ids}。每个等级都是单个子词，可以当作下一个 token 来打分。")
    ft = _load_hf(ft_path, cfg.model_class or detected["architecture"], dtype)
    try:
        ft.to(device)
    except RuntimeError as exc:
        raise RBAError(
            f"微调模型没能放到 {device}：{exc}",
            "2B 模型请用 dtype bfloat16。若仍不够，关掉其它进程，或把 device 改成 cpu（会很慢）。",
        ) from exc
    ft.eval()
    for param in ft.parameters():
        param.requires_grad_(False)
    # Base stays on disk until delta-W. Keeping it out of the GPU is the point.
    return LoadedModels(
        ft=ft,
        base=None,
        tokenizer=tokenizer,
        processor=processor,
        kind="hf",
        device=device,
        dtype=dtype,
        preset=detected["preset"],
        plant=None,
        image_token_id=detected.get("image_token_id") if cfg.image_token_id is None else cfg.image_token_id,
        grade_ids=grade_ids,
        base_path=str(base_path),
        ft_path=str(ft_path),
    )


def _kind(cfg: Config) -> str:
    if cfg.model_type == "tiny" or (cfg.model_type == "auto" and cfg.base_model == "tiny" and cfg.ft_model == "tiny"):
        return "tiny"
    if cfg.model_type == "qwen3_5_tiny" or cfg.base_model == "qwen3_5_tiny" or cfg.ft_model == "qwen3_5_tiny":
        if cfg.base_model not in {"qwen3_5_tiny", cfg.ft_model} and cfg.model_type != "qwen3_5_tiny":
            raise RBAError(
                "qwen3_5_tiny 必须基座和微调都用这个内置随机模型。",
                "命令：rba analyze --base qwen3_5_tiny --ft qwen3_5_tiny --badcases 你的文件 --dry-run",
            )
        return "qwen3_5_tiny"
    if cfg.model_type == "hf" or cfg.model_type == "auto":
        return "hf"
    raise RBAError(f"无法决定怎么加载 model_type={cfg.model_type}。", "使用 auto、tiny、qwen3_5_tiny 或 hf。")


def _load_tokenizer(path: Path):
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RBAError(
            "没有安装 transformers，无法加载真实模型。",
            "pip install -e . 后重试。只跑流程可以用 --base tiny --ft tiny。",
        ) from exc
    processor = None
    errors = []
    try:
        processor = AutoProcessor.from_pretrained(path)
    except Exception as exc:
        errors.append(f"AutoProcessor: {exc}")
        processor = None
    tokenizer = processor
    if processor is not None and hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception as exc:
            raise RBAError(
                f"分词器加载失败：{exc}",
                "确认目录里有 tokenizer 文件。前面的处理器错误：" + " | ".join(errors[:2]),
            ) from exc
    return tokenizer, processor


def _load_hf(path: Path, preferred: str | None, dtype: torch.dtype):
    try:
        import transformers
    except ImportError as exc:
        raise RBAError("没有安装 transformers。", "pip install -e .") from exc
    names: list[str] = []
    for name in [preferred, *CLASS_CANDIDATES]:
        if name and name not in names:
            names.append(name)
    errors = []
    for name in names:
        cls = getattr(transformers, name, None)
        if cls is None:
            errors.append(f"{name} 不在 transformers {transformers.__version__} 的导出里")
            continue
        try:
            model = cls.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
            info(f"已用 {name} 加载微调模型。")
            return model
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for auto_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        auto = getattr(transformers, auto_name, None)
        if auto is None:
            continue
        try:
            model = auto.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
            info(f"已用 {auto_name} 加载微调模型。")
            return model
        except Exception as exc:
            errors.append(f"{auto_name}: {exc}")
    detail = "；".join(errors[:6])
    raise RBAError(
        "微调模型加载失败。",
        "确认目录完整，并升级 transformers。Qwen3.5-VL 在 transformers 5.22 中的类名是 "
        f"Qwen3_5ForConditionalGeneration，不是 Qwen3_5VLForConditionalGeneration。详情：{detail}",
    )


def iter_checkpoint_tensors(path: str | Path):
    """Yield (name, cpu tensor) from a HF directory without keeping them."""
    path = Path(path)
    if not path.exists():
        raise RBAError(
            f"找不到基座目录 {path}。",
            "delta_w 需要基座权重。若只想看激活，在 analyses 里设 delta_w: false。",
        )
    safe_index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    bin_index = path / "pytorch_model.bin.index.json"
    single_bin = path / "pytorch_model.bin"
    if safe_index.exists() or single.exists():
        yield from _iter_safetensors(path, safe_index if safe_index.exists() else None)
        return
    if bin_index.exists() or single_bin.exists():
        yield from _iter_bins(path, bin_index if bin_index.exists() else None)
        return
    raise RBAError(
        f"{path} 里没有 model.safetensors 或 pytorch_model.bin。",
        "用 safetensors 保存基座后再跑。激活分析不依赖基座，可先设 analyses.delta_w 为 false。",
    )


def _iter_safetensors(path: Path, index_path: Path | None):
    from safetensors import safe_open

    if index_path is not None:
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = ["model.safetensors"]
    for filename in files:
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _iter_bins(path: Path, index_path: Path | None):
    if index_path is None:
        blob = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        for key, value in blob.items():
            if torch.is_tensor(value):
                yield key, value
        return
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    files = sorted(set(weight_map.values()))
    for filename in files:
        blob = torch.load(path / filename, map_location="cpu", weights_only=True)
        for key, value in blob.items():
            if torch.is_tensor(value):
                yield key, value
        del blob


def resolve_grade_ids(tokenizer, grades: list[str], explicit: dict | None) -> dict[str, int]:
    if explicit:
        missing = [grade for grade in grades if str(grade) not in {str(k) for k in explicit}]
        if missing:
            raise RBAError(
                f"grade_token_ids 缺少 {', '.join(missing)}。",
                "每个等级都要给一个整数 token id。",
            )
        return {str(grade): int(explicit[grade] if grade in explicit else explicit[str(grade)]) for grade in grades}
    ids = {}
    for grade in grades:
        encoded = _encode_grade(tokenizer, grade)
        if len(encoded) != 1:
            pieces = []
            convert = getattr(tokenizer, "convert_ids_to_tokens", None)
            if convert is not None:
                try:
                    pieces = [convert(int(i)) for i in encoded]
                except Exception:
                    pieces = [str(i) for i in encoded]
            raise RBAError(
                f"等级 {grade!r} 被分成了 {len(encoded)} 个子词 {pieces or encoded}，不能当作一个解码 token。",
                "在配置里写 grade_token_ids，把每个等级映射到词表中的单个 id；或改微调，让等级本身是单 token。",
            )
        ids[grade] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise RBAError(
            f"两个等级落到了同一个 token id：{ids}。",
            "检查 grades 的写法，或用 grade_token_ids 手动分开。",
        )
    return ids


def _encode_grade(tokenizer, grade: str) -> list[int]:
    if isinstance(tokenizer, TinyTokenizer):
        return tokenizer.encode(grade, add_special_tokens=False)
    encode = getattr(tokenizer, "encode", None)
    if encode is None and hasattr(tokenizer, "tokenizer"):
        encode = tokenizer.tokenizer.encode
    if encode is None:
        raise RBAError("分词器没有 encode 方法。", "检查 AutoTokenizer 是否加载成功。")
    try:
        return list(encode(grade, add_special_tokens=False))
    except TypeError:
        return list(encode(grade))
