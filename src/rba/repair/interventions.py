"""Context managers for head scaling, residual steering, lm_head row edits and token ablation.

None of these save a new checkpoint. `rba eval` measures them and leaves the
loaded weights as it found them.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from rba.errors import RBAError
from rba.models.paths import ResolvedModel, attention_module, output_projection


def _layer(model, resolved: ResolvedModel, index: int) -> nn.Module:
    layers = resolved.layers(model)
    if index < 0 or index >= len(layers):
        raise RBAError(
            f"层号 {index} 超出范围 0..{len(layers) - 1}。",
            "补丁里的 layer 改成报告里列出的层。",
        )
    return layers[index]


@contextmanager
def scale_heads(model, resolved: ResolvedModel, layer: int, heads: list[int], scale: float):
    module = _layer(model, resolved, layer)
    found = attention_module(module)
    if found is None:
        raise RBAError(f"第 {layer} 层没有注意力模块。", "换一层。线性注意力层用的是 out_proj，这里同样支持。")
    _name, attn = found
    proj_info = output_projection(attn)
    if proj_info is None:
        raise RBAError(
            f"第 {layer} 层的注意力输出没法按头切开。",
            "这一层不会做 head scale。可以改试 steering，或检查 head_dim 是否整除。",
        )
    proj, head_dim, n_heads = proj_info
    unknown = [head for head in heads if head < 0 or head >= n_heads]
    if unknown:
        raise RBAError(
            f"第 {layer} 层没有头 {unknown}（共 {n_heads} 个头）。",
            "头的编号从 0 开始。",
        )
    original = proj.forward
    head_set = sorted(set(heads))

    def forward(hidden):
        # Scale the heads' input columns, then call whatever forward was there before.
        # Linear in the input, so this equals scaling those heads' writes, and two
        # head_scale patches on the same layer compose instead of overwriting each other.
        edited = hidden.clone()
        for head in head_set:
            sl = slice(head * head_dim, (head + 1) * head_dim)
            edited[..., sl] = edited[..., sl] * scale
        return original(edited)

    proj.forward = forward
    try:
        yield
    finally:
        proj.forward = original


@contextmanager
def steer(model, resolved: ResolvedModel, layers: list[int], vector: torch.Tensor, alpha: float):
    hooks = []

    def make_hook():
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            update = alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + update
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    for index in layers:
        hooks.append(_layer(model, resolved, index).register_forward_hook(make_hook()))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def blend_lm_head(model, resolved: ResolvedModel, source_id: int, toward_id: int, alpha: float):
    head = resolved.lm_head(model)
    if source_id == toward_id:
        yield
        return
    weight = head.weight
    backup = weight.detach().clone()
    with torch.no_grad():
        mixed = (1.0 - alpha) * backup[source_id] + alpha * backup[toward_id]
        weight[source_id].copy_(mixed.to(dtype=weight.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            weight.copy_(backup)


@contextmanager
def zero_mlp(model, resolved: ResolvedModel, layer: int):
    module = _layer(model, resolved, layer)
    mlp = getattr(module, "mlp", None)
    if mlp is None:
        raise RBAError(f"第 {layer} 层没有 MLP。", "换一条作用在注意力头上的补丁。")
    original = mlp.forward

    def forward(hidden):
        out_features = mlp.down_proj.out_features if hasattr(mlp, "down_proj") else hidden.shape[-1]
        return torch.zeros(*hidden.shape[:-1], out_features, device=hidden.device, dtype=hidden.dtype)

    mlp.forward = forward
    try:
        yield
    finally:
        mlp.forward = original


@contextmanager
def ablate_token_ids(model, resolved: ResolvedModel, token_ids: list[int]):
    layer0 = resolved.layers(model)[0]
    id_set = set(int(token) for token in token_ids)

    def pre_hook(_module, inputs):
        hidden = inputs[0]
        # The caller stashes input_ids on the module for this forward.
        input_ids = getattr(model, "_rba_input_ids", None)
        if input_ids is None:
            return None
        mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        for token in id_set:
            mask |= input_ids.to(hidden.device) == token
        if mask.any():
            hidden = hidden.clone()
            hidden[mask] = 0
        return (hidden,) + tuple(inputs[1:])

    handle = layer0.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def steering_vector(model, resolved: ResolvedModel, from_id: int, to_id: int) -> torch.Tensor:
    weight = resolved.lm_head(model).weight.detach().float()
    vector = weight[to_id] - weight[from_id]
    norm = float(vector.norm())
    if norm < 1e-8:
        raise RBAError(
            "两个等级的 lm_head 行几乎一样，做不了 steering。",
            "换一对等级，或先看 lm_head 行的余弦。",
        )
    return vector / norm


@contextmanager
def apply_patch(model, resolved: ResolvedModel, patch: dict, grade_ids: dict[str, int]):
    method = patch.get("method")
    if method == "head_scale":
        with scale_heads(model, resolved, int(patch["layer"]), [int(h) for h in patch["heads"]], float(patch.get("scale", 0.0))):
            yield
        return
    if method == "steering":
        layers = [int(layer) for layer in patch.get("layers") or [patch.get("layer", resolved.layers(model).__len__() - 1)]]
        if patch.get("vector_source", "lm_head_diff") == "lm_head_diff":
            from_grade = str(patch["from_grade"])
            to_grade = str(patch["to_grade"])
            vector = steering_vector(model, resolved, grade_ids[from_grade], grade_ids[to_grade])
        else:
            vector = torch.tensor(patch["vector"], dtype=torch.float32)
        with steer(model, resolved, layers, vector, float(patch.get("alpha", 1.0))):
            yield
        return
    if method == "lm_head_blend":
        source = grade_ids[str(patch["grade"])]
        toward = grade_ids[str(patch["toward"])]
        with blend_lm_head(model, resolved, source, toward, float(patch.get("alpha", 0.5))):
            yield
        return
    if method == "ablate_tokens" or method == "ablate_image":
        tokens = patch.get("token_ids") or []
        if not tokens:
            raise RBAError(
                "ablate_tokens 补丁没有 token_ids。",
                "图像消融要带上模型的 image_token_id。报告里的补丁会自动填好。",
            )
        with ablate_token_ids(model, resolved, [int(token) for token in tokens]):
            yield
        return
    if method == "mlp_zero":
        with zero_mlp(model, resolved, int(patch["layer"])):
            yield
        return
    raise RBAError(
        f"不认识的补丁方法 {method!r}。",
        "method 只能是 head_scale、steering、lm_head_blend、ablate_tokens。",
    )
