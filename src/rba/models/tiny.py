"""A small stand-in with the same hook points as Qwen3.5-VL.

Module layout matches transformers 5.22's Qwen3_5ForConditionalGeneration:
`model.language_model.layers`, `self_attn.o_proj` or `linear_attn.out_proj`,
`mlp.{gate,up,down}_proj`, `model.language_model.norm`, `model.visual`, `lm_head`.

It is not a numerical approximation of Qwen. It exists so the analyses, the
CLI and CI can run on CPU without downloading weights. The real class is
exercised separately (see models/qwen_tiny.py and tests/test_real_qwen.py).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from rba.config import Config, TinySpec
from rba.errors import RBAError


class TinyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class TinyAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        if hidden % n_heads != 0:
            raise RBAError(
                f"hidden_size={hidden} 不能被 num_heads={n_heads} 整除。",
                "修改 tiny.hidden_size 或 tiny.num_heads。",
            )
        self.head_dim = hidden // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        batch, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            key_keep = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_keep, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, seq, -1)
        self._last_attn_weights = weights
        return self.o_proj(out), weights


class TinyGatedDelta(nn.Module):
    """Stand-in for Qwen3_5GatedDeltaNet: a per-head value write through out_proj.

    The real module is a recurrence. Head attribution only needs the output
    projection that adds into the residual stream, so the stand-in keeps
    `out_proj`, `num_v_heads` and `head_v_dim` and skips the conv kernel.
    """

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.num_v_heads = n_heads
        self.head_v_dim = hidden // n_heads
        self.in_proj_qkv = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        del attention_mask, kwargs
        values = torch.nn.functional.silu(self.in_proj_qkv(hidden_states))
        return self.out_proj(values)


class TinyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int, kind: str):
        super().__init__()
        self.block_type = kind
        if kind == "linear_attention":
            self.linear_attn = TinyGatedDelta(hidden, n_heads)
        else:
            self.self_attn = TinyAttention(hidden, n_heads)
        self.mlp = TinyMLP(hidden, intermediate)
        self.input_layernorm = TinyRMSNorm(hidden)
        self.post_attention_layernorm = TinyRMSNorm(hidden)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            mixed = self.linear_attn(normed, attention_mask=attention_mask)
        else:
            mixed, _ = self.self_attn(normed, attention_mask=attention_mask)
        hidden_states = residual + mixed
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class TinyVision(nn.Module):
    def __init__(self, hidden: int, image_seq_len: int, image_size: int):
        super().__init__()
        self.spatial_merge_size = 1
        self.image_seq_len = image_seq_len
        self.proj = nn.Linear(3 * image_size * image_size, hidden * image_seq_len)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        flat = pixel_values.flatten(1)
        hidden = self.proj(flat.to(dtype=self.proj.weight.dtype))
        return hidden.view(pixel_values.size(0), self.image_seq_len, -1)


class TinyTextModel(nn.Module):
    def __init__(self, vocab: int, hidden: int, n_layers: int, n_heads: int, intermediate: int, layer_types: list[str]):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [TinyDecoderLayer(hidden, n_heads, intermediate, layer_types[i]) for i in range(n_layers)]
        )
        self.norm = TinyRMSNorm(hidden)

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        del kwargs
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
        return self.norm(hidden)


@dataclass
class TinyOutput:
    logits: torch.Tensor


class TinyQwenVL(nn.Module):
    def __init__(self, spec: TinySpec, image_token_id: int, pad_id: int = 0):
        super().__init__()
        layer_types = list(spec.layer_types or ["full_attention"] * spec.num_layers)
        if len(layer_types) != spec.num_layers:
            raise RBAError(
                "tiny.layer_types 的长度必须等于 tiny.num_layers。",
                "每一项是 full_attention 或 linear_attention。",
            )
        self.image_token_id = image_token_id
        self.pad_id = pad_id
        self.spec = spec
        text = TinyTextModel(
            spec.vocab_size,
            spec.hidden_size,
            spec.num_layers,
            spec.num_heads,
            spec.intermediate_size,
            layer_types,
        )
        visual = TinyVision(spec.hidden_size, spec.image_seq_len, spec.image_size)
        self.model = nn.Module()
        self.model.language_model = text
        self.model.visual = visual
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)
        self.config = type(
            "TinyConfig",
            (),
            {
                "image_token_id": image_token_id,
                "hidden_size": spec.hidden_size,
                "vocab_size": spec.vocab_size,
                "model_type": "tiny_qwen_vl",
            },
        )()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        del image_grid_thw, mm_token_type_ids, use_cache, output_attentions, output_hidden_states, kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise RBAError("forward 缺少 input_ids。", "这是程序错误。")
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None and input_ids is not None:
            feats = self.model.visual(pixel_values)
            flat = feats.reshape(-1, feats.size(-1))
            mask = input_ids == self.image_token_id
            if int(mask.sum()) != flat.size(0):
                raise RBAError(
                    f"图像 token 数（{int(mask.sum())}）和视觉特征数（{flat.size(0)}）不一致。",
                    "模板里的图像占位个数要等于 image_seq_len，并且每张图都成功读入。",
                )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = flat.to(dtype=inputs_embeds.dtype)
        hidden = self.model.language_model(inputs_embeds, attention_mask=attention_mask)
        return TinyOutput(logits=self.lm_head(hidden))


class TinyTokenizer:
    def __init__(self, grades: list[str], image_token: str, vocab_size: int):
        self.grades = list(grades)
        self.image_token = image_token
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        next_id = 2
        for grade in grades:
            if grade in self.token_to_id:
                raise RBAError(
                    f"等级 {grade!r} 和内置特殊符号冲突。",
                    "换一个等级字符串，或减小冲突。",
                )
            self.token_to_id[grade] = next_id
            next_id += 1
        if image_token in self.token_to_id:
            self.image_token_id = self.token_to_id[image_token]
        else:
            self.token_to_id[image_token] = next_id
            self.image_token_id = next_id
            next_id += 1
        if next_id >= vocab_size:
            raise RBAError(
                "tiny.vocab_size 太小，放不下等级和图像 token。",
                "把 tiny.vocab_size 调到大于 等级数+4。",
            )
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self._reserved = next_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.token_to_id:
            return [self.token_to_id[text]]
        return self._scan(text)

    def convert_ids_to_tokens(self, index: int) -> str:
        return self.id_to_token.get(int(index), f"#{int(index)}")

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self.convert_ids_to_tokens(int(i)) for i in ids)

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        span = self.vocab_size - self._reserved
        return self._reserved + (int(digest[:8], 16) % span)

    def _scan(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        image = self.image_token
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            if image and text.startswith(image, i):
                ids.append(self.image_token_id)
                i += len(image)
                continue
            matched_grade = False
            for grade in sorted(self.grades, key=len, reverse=True):
                if text.startswith(grade, i) and self._grade_boundary(text, i, i + len(grade)):
                    ids.append(self.token_to_id[grade])
                    i += len(grade)
                    matched_grade = True
                    break
            if matched_grade:
                continue
            if ord(text[i]) > 127:
                ids.append(self._hash(text[i]))
                i += 1
                continue
            j = i
            while j < len(text):
                if image and text.startswith(image, j):
                    break
                if text[j].isspace() or ord(text[j]) > 127:
                    break
                j += 1
            word = text[i:j]
            ids.append(self.token_to_id[word] if word in self.token_to_id else self._hash(word))
            i = j
        return ids

    @staticmethod
    def _grade_boundary(text: str, start: int, end: int) -> bool:
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        if before.isdigit() or after.isdigit():
            return False
        return True


def plant_error(model: nn.Module, grade_ids: dict[str, int]) -> dict:
    """Push grade "0" and one attention head so the demo has a known delta.

    The edit is intentionally obvious: a uniform shift of the grade-0 unembedding
    row, plus a rank-one write of that direction through layer-0 head 0. It is a
    fixture for the tool, not a claim about real relevance errors.
    """
    if "0" not in grade_ids:
        target = next(iter(grade_ids))
    else:
        target = "0"
    index = grade_ids[target]
    with torch.no_grad():
        model.lm_head.weight[index].add_(2.5)
        direction = model.lm_head.weight[index].detach().clone()
        direction = direction / (direction.norm() + 1e-6)
        layer0 = model.model.language_model.layers[0]
        proj = layer0.self_attn.o_proj if hasattr(layer0, "self_attn") else layer0.linear_attn.out_proj
        head_dim = getattr(getattr(layer0, "self_attn", None) or layer0.linear_attn, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(layer0.linear_attn, "head_v_dim")
        proj.weight[:, : int(head_dim)].add_(direction.unsqueeze(1) * 1.5)
    return {
        "grade": target,
        "layer": 0,
        "head": 0,
        "what": "lm_head 该等级行整体加 2.5；第 0 层第 0 个头的输出投影加上同一方向",
    }


def _init_small(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, TinyRMSNorm):
            nn.init.zeros_(module.weight)


def build_tiny_pair(cfg: Config) -> tuple[TinyQwenVL, TinyQwenVL, TinyTokenizer, dict]:
    torch.manual_seed(cfg.seed)
    spec = cfg.tiny
    tokenizer = TinyTokenizer(cfg.grades, cfg.image_token, spec.vocab_size)
    base = TinyQwenVL(spec, image_token_id=tokenizer.image_token_id)
    _init_small(base)
    ft = copy.deepcopy(base)
    plant_info = None
    if cfg.plant_error:
        grade_ids = {grade: tokenizer.token_to_id[grade] for grade in cfg.grades}
        plant_info = plant_error(ft, grade_ids)
    base.eval()
    ft.eval()
    for model in (base, ft):
        for param in model.parameters():
            param.requires_grad_(False)
    return base, ft, tokenizer, {"plant": plant_info, "image_token_id": tokenizer.image_token_id}
