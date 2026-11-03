"""Find decoder layers, lm_head, vision tower and final norm.

Qwen3.5-VL in transformers 5.22 is `Qwen3_5ForConditionalGeneration`
(there is no `Qwen3_5VLForConditionalGeneration` in that release).
The language stack lives at `model.language_model`. The same layout is used
by Qwen2-VL, Qwen2.5-VL and Qwen3-VL in that release. Older checkpoints may
still hang `visual` next to `model.layers`; discovery walks the module tree
instead of trusting the preset when the preset path is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rba.errors import RBAError

# Tried in this order when the checkpoint does not name a class we can import.
CLASS_CANDIDATES = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
]

PRESET_PATHS = {
    "qwen3_5": {
        "layers": "model.language_model.layers",
        "lm_head": "lm_head",
        "visual": "model.visual",
        "final_norm": "model.language_model.norm",
        "embed": "model.language_model.embed_tokens",
    },
    "qwen2_vl": {
        "layers": "model.language_model.layers",
        "lm_head": "lm_head",
        "visual": "model.visual",
        "final_norm": "model.language_model.norm",
        "embed": "model.language_model.embed_tokens",
    },
}


def preset_for_architecture(architecture: str | None, model_type: str | None) -> str:
    blob = f"{architecture or ''} {model_type or ''}".lower()
    if "qwen3_5" in blob or "qwen3.5" in blob:
        return "qwen3_5"
    if "qwen3_vl" in blob or "qwen2" in blob:
        return "qwen2_vl"
    return "qwen3_5"


def get_module(model: nn.Module, path: str | None) -> nn.Module | None:
    if not path:
        return None
    current: nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            if not isinstance(current, (nn.ModuleList, nn.Sequential)):
                return None
            index = int(part)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def discover_layers(model: nn.Module) -> list[str]:
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        child = module[0]
        has_mlp = hasattr(child, "mlp")
        has_attn = any(hasattr(child, attr) for attr in ("self_attn", "linear_attn", "attn", "attention"))
        if has_mlp and has_attn:
            found.append(name)
    found.sort(key=lambda name: (0 if "language_model" in name else 1, len(name), name))
    return found


def discover_lm_head(model: nn.Module) -> str | None:
    named = dict(model.named_modules())
    if "lm_head" in named and isinstance(named["lm_head"], nn.Linear):
        return "lm_head"
    best = None
    best_out = -1
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features > module.in_features and module.out_features > best_out:
            best = name
            best_out = module.out_features
    return best


def discover_visual(model: nn.Module) -> str | None:
    for candidate in ("model.visual", "visual", "model.vision_tower", "vision_tower"):
        if get_module(model, candidate) is not None:
            return candidate
    for name, module in model.named_modules():
        if name.endswith("visual") or name.endswith("vision_tower"):
            if any(True for _ in module.parameters()):
                return name
    return None


def discover_embed(model: nn.Module, layers_path: str) -> str | None:
    parent = layers_path.rsplit(".", 1)[0] if "." in layers_path else ""
    for suffix in ("embed_tokens", "tok_embeddings", "wte"):
        path = f"{parent}.{suffix}" if parent else suffix
        module = get_module(model, path)
        if isinstance(module, nn.Embedding):
            return path
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embed = getter()
        for name, module in model.named_modules():
            if module is embed:
                return name
    return None


def discover_final_norm(model: nn.Module, layers_path: str) -> str | None:
    parent = layers_path.rsplit(".", 1)[0] if "." in layers_path else ""
    for suffix in ("norm", "final_layernorm", "ln_f"):
        path = f"{parent}.{suffix}" if parent else suffix
        if get_module(model, path) is not None and path != layers_path:
            module = get_module(model, path)
            if not isinstance(module, nn.ModuleList):
                return path
    return None


@dataclass
class ResolvedModel:
    layers_path: str
    lm_head_path: str
    visual_path: str | None
    final_norm_path: str | None
    embed_path: str | None
    notes: list[str]

    def layers(self, model: nn.Module) -> nn.ModuleList:
        module = get_module(model, self.layers_path)
        if not isinstance(module, nn.ModuleList):
            raise RBAError(
                f"层路径 {self.layers_path} 不是 ModuleList。",
                "把 module_paths.layers 改成语言解码器的 ModuleList，或留空让程序自己找。",
            )
        return module

    def lm_head(self, model: nn.Module) -> nn.Linear:
        module = get_module(model, self.lm_head_path)
        if not isinstance(module, nn.Linear):
            raise RBAError(
                f"lm_head 路径 {self.lm_head_path} 不是 Linear。",
                "Qwen3.5-VL 的分类头在顶层 lm_head。若你改过结构，在 module_paths.lm_head 写明。",
            )
        return module

    def final_norm(self, model: nn.Module) -> nn.Module | None:
        if not self.final_norm_path:
            return None
        return get_module(model, self.final_norm_path)


def resolve_paths(model: nn.Module, user_paths: dict | None, preset: str | None = None) -> ResolvedModel:
    user_paths = user_paths or {}
    notes: list[str] = []
    preset_paths = PRESET_PATHS.get(preset or "", PRESET_PATHS["qwen3_5"])

    def pick(key: str, discovered: list[str] | str | None, required: bool) -> str | None:
        requested = user_paths.get(key)
        if requested:
            if get_module(model, requested) is None:
                hint_pool = discovered if isinstance(discovered, list) else ([discovered] if discovered else [])
                extra = f" 自动发现的候选：{', '.join(hint_pool)}。" if hint_pool else ""
                raise RBAError(
                    f"module_paths.{key}={requested} 在这个模型里不存在。",
                    f"删掉该项以自动发现，或改成真实路径。{extra}",
                )
            return requested
        if isinstance(discovered, list):
            if not discovered:
                found = None
            else:
                preferred = preset_paths.get(key)
                found = preferred if preferred in discovered else discovered[0]
                if preferred and preferred not in discovered:
                    notes.append(f"预设路径 {preferred} 不存在，{key} 改用 {found}。")
        else:
            found = discovered
            preferred = preset_paths.get(key)
            if found and preferred and found != preferred:
                notes.append(f"预设路径 {preferred} 不存在，{key} 改用 {found}。")
        if required and not found:
            raise RBAError(
                f"没有找到{key}。",
                "在配置 module_paths 里写上路径。Qwen3.5-VL（transformers 5.22）的语言层是 "
                "model.language_model.layers，头是 lm_head，视觉塔是 model.visual。",
            )
        return found

    layers = pick("layers", discover_layers(model), required=True)
    lm_head = pick("lm_head", discover_lm_head(model), required=True)
    visual = pick("visual", discover_visual(model), required=False)
    final_norm = pick("final_norm", discover_final_norm(model, layers), required=False)
    embed = pick("embed", discover_embed(model, layers), required=False)
    return ResolvedModel(
        layers_path=layers,
        lm_head_path=lm_head,
        visual_path=visual,
        final_norm_path=final_norm,
        embed_path=embed,
        notes=notes,
    )


def layer_kind(layer: nn.Module) -> str:
    if hasattr(layer, "self_attn"):
        return "full_attention"
    if hasattr(layer, "linear_attn"):
        return "linear_attention"
    if hasattr(layer, "attn") or hasattr(layer, "attention"):
        return "attention"
    return "unknown"


def attention_module(layer: nn.Module) -> tuple[str, nn.Module] | None:
    for name in ("self_attn", "linear_attn", "attn", "attention"):
        if hasattr(layer, name):
            return name, getattr(layer, name)
    return None


def output_projection(attn: nn.Module) -> tuple[nn.Linear, int, int] | None:
    """Return (proj, head_dim, n_heads) when the write can be split into heads.

    Full attention uses o_proj. Gated DeltaNet uses out_proj over value heads.
    The projection's input is already the per-head activations concatenated,
    which is the right place to split a residual write.
    """
    if hasattr(attn, "o_proj") and isinstance(attn.o_proj, nn.Linear):
        proj = attn.o_proj
        head_dim = int(getattr(attn, "head_dim", 0) or 0)
        if head_dim <= 0 or proj.in_features % head_dim != 0:
            return None
        return proj, head_dim, proj.in_features // head_dim
    if hasattr(attn, "out_proj") and isinstance(attn.out_proj, nn.Linear):
        proj = attn.out_proj
        head_dim = int(getattr(attn, "head_v_dim", 0) or getattr(attn, "head_dim", 0) or 0)
        if head_dim <= 0 or proj.in_features % head_dim != 0:
            return None
        n_heads = proj.in_features // head_dim
        return proj, head_dim, n_heads
    return None


def describe_model(model: nn.Module, resolved: ResolvedModel) -> dict:
    layers = resolved.layers(model)
    desc_layers = []
    for index, layer in enumerate(layers):
        kind = layer_kind(layer)
        found = attention_module(layer)
        heads = None
        head_dim = None
        if found is not None:
            proj = output_projection(found[1])
            if proj is not None:
                _, head_dim, heads = proj
        hidden = None
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            hidden = int(mlp.down_proj.out_features)
        desc_layers.append(
            {
                "index": index,
                "kind": kind,
                "attn_name": None if found is None else found[0],
                "heads": heads,
                "head_dim": head_dim,
                "hidden": hidden,
            }
        )
    lm_head = resolved.lm_head(model)
    return {
        "class_name": type(model).__name__,
        "n_layers": len(layers),
        "hidden_size": int(lm_head.in_features),
        "vocab_size": int(lm_head.out_features),
        "layers": desc_layers,
        "paths": {
            "layers": resolved.layers_path,
            "lm_head": resolved.lm_head_path,
            "visual": resolved.visual_path,
            "final_norm": resolved.final_norm_path,
            "embed": resolved.embed_path,
        },
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
