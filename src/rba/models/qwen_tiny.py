"""Randomly initialized Qwen3_5ForConditionalGeneration small enough for CPU tests.

This is the real class, not the stand-in. Defaults follow the constraints
checked against transformers 5.22: mixed full attention and GatedDeltaNet,
o_proj / out_proj head splits, and a vision tower that is constructed but
not required for the text forward used in tests.
"""

from __future__ import annotations

import copy

import torch

from rba.errors import RBAError
from rba.models.tiny import TinyTokenizer, plant_error


def tiny_qwen35_config(vocab_size: int = 128, image_token_id: int = 10):
    try:
        from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
    except ImportError as exc:
        raise RBAError(
            "当前环境的 transformers 没有 Qwen3_5Config。",
            "升级 transformers（本仓库在 5.22 上核对过 Qwen3_5ForConditionalGeneration），或改用 model_type: tiny。",
        ) from exc
    text = Qwen3_5TextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=vocab_size,
        layer_types=["full_attention", "linear_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        max_position_embeddings=256,
    )
    vision = Qwen3_5VisionConfig(
        hidden_size=32,
        depth=1,
        num_heads=4,
        intermediate_size=64,
        out_hidden_size=64,
        patch_size=16,
        spatial_merge_size=1,
        num_position_embeddings=64,
    )
    return Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=image_token_id,
        video_token_id=image_token_id + 1,
        vision_start_token_id=image_token_id + 2,
        vision_end_token_id=image_token_id + 3,
    )


def build_random_qwen35(seed: int = 0, vocab_size: int = 128, image_token_id: int = 10):
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RBAError(
            "当前环境的 transformers 没有 Qwen3_5ForConditionalGeneration。",
            "升级 transformers。第三方文档里的 Qwen3_5VLForConditionalGeneration 在 5.22 中不存在，类名是 Qwen3_5ForConditionalGeneration。",
        ) from exc
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(tiny_qwen35_config(vocab_size, image_token_id))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_qwen35_tiny_pair(cfg):
    tokenizer = TinyTokenizer(cfg.grades, cfg.image_token, vocab_size=128)
    if tokenizer.image_token_id >= 128 or any(tokenizer.token_to_id[g] >= 128 for g in cfg.grades):
        raise RBAError(
            "qwen3_5_tiny 的词表只有 128，等级 token 放不下。",
            "减少 grades，或改用 model_type: tiny。",
        )
    base = build_random_qwen35(cfg.seed, image_token_id=tokenizer.image_token_id)
    ft = copy.deepcopy(base)
    plant_info = None
    if cfg.plant_error:
        grade_ids = {grade: tokenizer.token_to_id[grade] for grade in cfg.grades}
        plant_info = plant_error(ft, grade_ids)
    return base, ft, tokenizer, {"plant": plant_info, "image_token_id": tokenizer.image_token_id}
