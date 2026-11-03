"""Real processor + real Qwen3.5 class: the CLI encoding must equal a direct processor call."""

from pathlib import Path

import pytest
import torch

from rba.config import Config
from rba.data.encode import ExcludedCase, collate, encode_cases, needs_single_item_batches
from rba.errors import RBAError
from rba.schema import Case
from tests.vl_fixture import build_model, build_processor, build_tokenizer, random_image

TEMPLATE = "查询：{query}\n文档：{doc}\n{image}相关性等级："


@pytest.fixture(scope="module")
def vl():
    tokenizer = build_tokenizer()
    processor = build_processor(tokenizer)
    model = build_model(tokenizer)
    return tokenizer, processor, model


def _cfg(**over):
    cfg = Config(base_model="x", ft_model="x", model_type="hf", prompt_template=TEMPLATE, max_length=256, image_max_pixels=None, goodcases=None)
    for key, value in over.items():
        setattr(cfg, key, value)
    cfg.validate_static()
    return cfg


def test_template_mode_matches_direct_processor_call(vl, tmp_path: Path):
    tokenizer, processor, model = vl
    image_path = random_image(tmp_path / "a.png")
    case = Case(id="a", query="红烧肉", doc="五花肉", image=image_path, label="3")
    encoded = encode_cases([case], tokenizer, _cfg(), processor=processor)
    item = encoded[0]
    assert item.has_image and item.n_image_tokens > 0 and item.image_grid_thw is not None
    assert needs_single_item_batches(encoded)
    batch = collate(encoded, tokenizer.pad_token_id)
    assert "pixel_values" in batch and "image_grid_thw" in batch
    from PIL import Image

    direct = processor(
        text=["查询：红烧肉\n文档：五花肉\n<|vision_start|><|image_pad|><|vision_end|>相关性等级："],
        images=[Image.open(image_path).convert("RGB")],
        return_tensors="pt",
    )
    assert direct["input_ids"].tolist() == batch["input_ids"].tolist()
    with torch.no_grad():
        ours = model(**{k: v for k, v in batch.items() if k != "case_ids"}, use_cache=False).logits[0, -1]
        theirs = model(**dict(direct), use_cache=False).logits[0, -1]
        without_pixels = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[0, -1]
    assert torch.allclose(ours, theirs, atol=1e-5)
    assert not torch.allclose(ours, without_pixels, atol=1e-3), "pixels must reach the vision tower"


def test_text_rows_use_the_same_processor(vl):
    tokenizer, processor, _model = vl
    case = Case(id="t", query="天气", doc="北京今天", image=None, label="0")
    item = encode_cases([case], tokenizer, _cfg(), processor=processor)[0]
    assert item.model_inputs is not None and not item.has_image and item.n_image_tokens == 0
    assert item.protocol == "template"


def test_missing_image_is_error_or_exclusion(vl):
    tokenizer, processor, _model = vl
    case = Case(id="m", query="q", doc="d", image="/nonexistent/x.png", label="0")
    with pytest.raises(RBAError) as caught:
        encode_cases([case], tokenizer, _cfg(), processor=processor)
    assert "allow-missing-images" in caught.value.hint
    excluded: list[ExcludedCase] = []
    rows = encode_cases([case], tokenizer, _cfg(missing_image="exclude"), processor=processor, excluded=excluded)
    assert rows == [] and excluded[0].id == "m"


def test_truncation_never_cuts_image_block(vl, tmp_path: Path):
    tokenizer, processor, _model = vl
    image_path = random_image(tmp_path / "b.png")
    case = Case(id="long", query="红烧肉" * 30, doc="五花肉", image=image_path, label="3")
    with pytest.raises(RBAError) as caught:
        encode_cases([case], tokenizer, _cfg(max_length=12), processor=processor)
    assert "视觉块" in caught.value.message
    # Text-only truncation keeps the tail.
    text_case = Case(id="long2", query="天气" * 100, doc="北京", image=None, label="0")
    item = encode_cases([text_case], tokenizer, _cfg(max_length=16), processor=processor)[0]
    assert len(item.input_ids) == 16 and item.truncated


def test_template_without_image_placeholder_is_rejected(vl, tmp_path: Path):
    tokenizer, processor, _model = vl
    image_path = random_image(tmp_path / "c.png")
    case = Case(id="c", query="q", doc="d", image=image_path, label="3")
    with pytest.raises(RBAError) as caught:
        encode_cases([case], tokenizer, _cfg(prompt_template="查询：{query}\n文档：{doc}\n等级："), processor=processor)
    assert "{image}" in caught.value.message


def test_image_max_pixels_caps_token_count(vl, tmp_path: Path):
    tokenizer, processor, _model = vl
    image_path = random_image(tmp_path / "d.png", width=160, height=160)
    case = Case(id="d", query="q", doc="d", image=image_path, label="3")
    big = encode_cases([case], tokenizer, _cfg(), processor=processor)[0]
    small = encode_cases([case], tokenizer, _cfg(image_max_pixels=1024), processor=processor)[0]
    assert small.n_image_tokens < big.n_image_tokens
