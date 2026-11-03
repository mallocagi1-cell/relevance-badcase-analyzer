"""A real Qwen3VLProcessor over a synthetic tokenizer, paired with the random Qwen3.5.

No weights are downloaded. The tokenizer is char-level so the vocabulary stays
tiny; the image processor uses the same patch geometry as the random model's
vision tower, so processor output really flows through the vision tower.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("PIL")

SPECIALS = ["<pad>", "<unk>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
GRADES = ["-1", "0", "1", "2", "3"]
CHARS = "判断文档与查询的相关性只输出一个等级不要其它文字含义冲突或答非所问相弱高度：\n红烧肉怎么做五花焯水后加冰糖炒色再放酱油和八角炖四十分钟天气北京今预报市象台阴转小雨风三这道菜偏肥热量欧冠决赛在中进行 -qwertyuiopasdfghjklzxcvbnm,.?！，。"
PATCH = 16


def build_tokenizer():
    from tokenizers import Regex, Tokenizer, models
    from tokenizers import pre_tokenizers as pt
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {}
    for token in SPECIALS + GRADES + sorted(set(CHARS)):
        vocab.setdefault(token, len(vocab))
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pt.Split(Regex(r"<\|[a-z_]+\|>|-1|."), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|endoftext|>",
        additional_special_tokens=SPECIALS[2:],
    )


def build_processor(tokenizer, max_pixels: int = PATCH * PATCH * 16):
    import transformers.processing_utils as pu
    from transformers import Qwen2VLImageProcessorPil, Qwen3VLProcessor

    # Synthetic components: the class-registry check needs torchvision's video
    # processor, so it is bypassed only while this one processor is constructed.
    original = pu.ProcessorMixin.check_argument_for_proper_class
    pu.ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    try:
        image_processor = Qwen2VLImageProcessorPil(
            patch_size=PATCH, merge_size=1, temporal_patch_size=2, min_pixels=PATCH * PATCH * 4, max_pixels=max_pixels
        )
        return Qwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer)
    finally:
        pu.ProcessorMixin.check_argument_for_proper_class = original


def build_model(tokenizer, seed: int = 0):
    from rba.models.qwen_tiny import build_random_qwen35

    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model = build_random_qwen35(seed=seed, vocab_size=max(128, len(tokenizer) + 8), image_token_id=image_token_id)
    model.config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    model.config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    return model


def random_image(path, width: int = 60, height: int = 40, seed: int = 0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8")).save(path)
    return str(path)
