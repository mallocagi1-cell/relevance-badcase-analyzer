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
    tied_lm_head: bool = False
    identity: dict | None = None


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


def _grade_ids(tokenizer, cfg: Config) -> dict[str, int]:
    import inspect

    try:
        accepts_mode = "grade_mode" in inspect.signature(resolve_grade_ids).parameters
    except (TypeError, ValueError):
        accepts_mode = False
    if accepts_mode:
        return resolve_grade_ids(tokenizer, cfg.grades, cfg.grade_token_ids, grade_mode=cfg.grade_mode)
    if cfg.grade_mode != "single_token":
        raise RBAError("当前 loader 不支持 grade_mode=first_token。", "升级 rba，或把 grade_mode 改回 single_token。")
    return resolve_grade_ids(tokenizer, cfg.grades, cfg.grade_token_ids)


def is_tied_lm_head(model, head=None) -> bool:
    """True when lm_head shares storage with the input embedding (editing one edits both)."""
    if head is None:
        head = getattr(model, "lm_head", None)
    getter = getattr(model, "get_input_embeddings", None)
    if head is None or not callable(getter):
        return False
    try:
        embed = getter()
    except Exception:
        return False
    if embed is None or not hasattr(embed, "weight") or not hasattr(head, "weight"):
        return False
    if head.weight is embed.weight:
        return True
    try:
        return head.weight.data_ptr() == embed.weight.data_ptr() and head.weight.shape == embed.weight.shape
    except Exception:
        return False


def check_checkpoint_dir(path: str | Path, what: str) -> dict:
    """Front-loaded check so a bad --base does not fail after the GPU stages."""
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise RBAError(
            f"找不到{what}目录 {path}。",
            "指到含 config.json 和权重文件的 Hugging Face 模型目录。只看流程：rba demo。",
        )
    raw = _read_config_json(path)
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    if not weights:
        raise RBAError(
            f"{what}目录 {path} 里没有 *.safetensors 或 pytorch_model*.bin。",
            "补全权重文件。若只想看激活分析，在 analyses 里把 delta_w 设为 false（基座只用于 ΔW）。",
        )
    return {
        "path": str(path.resolve()),
        "architecture": (raw.get("architectures") or [None])[0],
        "model_type": raw.get("model_type"),
        "tie_word_embeddings": bool(raw.get("tie_word_embeddings", (raw.get("text_config") or {}).get("tie_word_embeddings", False))),
        "n_weight_files": len(weights),
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
            grade_ids=_grade_ids(tokenizer, cfg),
            base_path="tiny",
            ft_path="tiny",
            tied_lm_head=is_tied_lm_head(ft),
            identity={"kind": "tiny", "seed": cfg.seed, "plant_error": cfg.plant_error},
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
            grade_ids=_grade_ids(tokenizer, cfg),
            base_path="qwen3_5_tiny",
            ft_path="qwen3_5_tiny",
            tied_lm_head=is_tied_lm_head(ft),
            identity={"kind": "qwen3_5_tiny", "seed": cfg.seed, "plant_error": cfg.plant_error},
        )
    ft_path = Path(cfg.ft_model)
    base_path = Path(cfg.base_model)
    detected = detect_directory(ft_path)
    info(
        f"识别到微调模型架构 {detected['architecture'] or detected['model_type']}，"
        f"{detected['num_hidden_layers']} 层，hidden {detected['hidden_size']}。"
    )
    tokenizer, processor = _load_tokenizer(ft_path)
    grade_ids = _grade_ids(tokenizer, cfg)
    if cfg.grade_mode == "first_token":
        info(f"等级 token（first_token 模式，只看每个等级的首 token）：{grade_ids}。")
    else:
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
    tied = is_tied_lm_head(ft)
    if tied:
        warn("lm_head 和输入 embedding 共享权重。任何 lm_head 行编辑同时改动该 token 的输入表示，报告和 eval 会标出来。")
    if processor is None:
        warn("没有加载到 AutoProcessor，带图样本无法编码；纯文本样本改用分词器。")
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
        tied_lm_head=tied,
        identity={
            "kind": "hf",
            "model_class": type(ft).__name__,
            "architecture": detected.get("architecture"),
            "tokenizer_class": type(tokenizer).__name__,
            "processor_class": None if processor is None else type(processor).__name__,
            "transformers_version": _transformers_version(),
        },
    )


def _transformers_version() -> str | None:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return None


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
            model = _from_pretrained(cls, path, dtype)
            info(f"已用 {name} 加载微调模型。")
            return model
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for auto_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        auto = getattr(transformers, auto_name, None)
        if auto is None:
            continue
        try:
            model = _from_pretrained(auto, path, dtype)
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


def _from_pretrained(cls, path: Path, dtype: torch.dtype):
    """transformers 5.x takes `dtype`; older releases only know `torch_dtype`."""
    try:
        return cls.from_pretrained(path, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return cls.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)


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


def resolve_grade_ids(tokenizer, grades: list[str], explicit: dict | None,
                      grade_mode: str = "single_token") -> dict[str, int]:
    """Validate the training label protocol; explicit IDs are not an escape hatch.

    first_token is a declared *prefix classifier*, not sequence likelihood or
    a guarantee that greedy generation completes the label string.
    """
    if grade_mode not in {"single_token", "first_token"}:
        raise RBAError(f"未知 grade_mode={grade_mode}。", "使用 single_token 或 first_token。")
    labels = [str(g) for g in grades]
    if not labels or len(set(labels)) != len(labels):
        raise RBAError("等级为空或重复。", "grades 必须是不同的标签字符串。")
    mapping = {}
    for key, value in (explicit or {}).items():
        key = str(key)
        if key in mapping:
            raise RBAError(f"grade_token_ids 规范化后重复键 {key}。", "每个等级只声明一次。")
        if isinstance(value, bool) or not isinstance(value, int):
            raise RBAError(f"等级 {key} 的 token id 不是整数。", "使用词表中的整数 ID。")
        mapping[key] = value
    if explicit is not None and set(mapping) != set(labels):
        raise RBAError("grade_token_ids 的键必须与 grades 完全一致。", f"需要 {labels}。")
    try:
        vocab_size = len(tokenizer)
    except (TypeError, AttributeError):
        vocab_size = getattr(tokenizer, "vocab_size", None)
    ids = {}
    for grade in labels:
        encoded = _encode_grade(tokenizer, grade)
        if not encoded or (grade_mode == "single_token" and len(encoded) != 1):
            raise RBAError(
                f"等级 {grade!r} 被分成 {len(encoded)} 个 token：{encoded}。",
                "single_token 要求训练标签本身为单 token；仅在训练协议明确使用独立首 token 分类时声明 grade_mode: first_token。",
            )
        token_id = mapping.get(grade, int(encoded[0]))
        if token_id < 0 or (vocab_size is not None and token_id >= vocab_size):
            raise RBAError(f"等级 {grade} 的 ID {token_id} 超出词表。", f"候选编码为 {encoded}。")
        if token_id != encoded[0]:
            raise RBAError(f"等级 {grade} 的 ID {token_id} 与实际编码 {encoded} 不符。", "显式 ID 必须匹配训练标签首 token，不支持任意别名。")
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            try:
                decoded = decode(encoded, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except TypeError:
                decoded = decode(encoded)
            if decoded != grade:
                raise RBAError(f"等级 {grade!r} 编码后解码为 {decoded!r}。", "检查 tokenizer 与训练标签的往返一致性。")
        ids[grade] = int(token_id)
    if len(set(ids.values())) != len(ids):
        raise RBAError(f"等级首 token 不唯一：{ids}。", "首 token 无法区分等级；需要完整序列评分，本工具暂不支持。")
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
