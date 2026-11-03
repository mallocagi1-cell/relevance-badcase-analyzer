"""Hooks against the real Qwen3_5ForConditionalGeneration, randomly initialized."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5")


def test_class_name_and_module_tree():
    assert hasattr(transformers, "Qwen3_5ForConditionalGeneration")
    assert not hasattr(transformers, "Qwen3_5VLForConditionalGeneration")
    from rba.models.qwen_tiny import build_random_qwen35
    from rba.models.paths import resolve_paths

    model = build_random_qwen35(seed=0)
    resolved = resolve_paths(model, {}, preset="qwen3_5")
    assert resolved.layers_path == "model.language_model.layers"
    assert resolved.lm_head_path == "lm_head"
    assert resolved.visual_path == "model.visual"
    assert resolved.final_norm_path == "model.language_model.norm"
    kinds = []
    for layer in resolved.layers(model):
        kinds.append("self_attn" if hasattr(layer, "self_attn") else "linear_attn")
    assert kinds == ["self_attn", "linear_attn"]


def test_real_model_dla_identity():
    from rba.analysis.instrument import instrument_batch
    from rba.models.qwen_tiny import build_random_qwen35
    from rba.models.paths import resolve_paths

    model = build_random_qwen35(seed=1)
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter("eager")
    resolved = resolve_paths(model, {}, preset="qwen3_5")
    ids = torch.tensor([[1, 4, 5, 6, 7]])
    mask = torch.ones_like(ids)
    batch = {"input_ids": ids, "attention_mask": mask, "case_ids": ["r0"]}
    # Grade rows: use a few vocab ids inside the tiny vocab.
    grade_ids = [2, 3, 4, 5, 6]
    pred = torch.tensor([4])
    gold = torch.tensor([2])
    rows = instrument_batch(
        model, batch, resolved, grade_ids, pred, gold, image_token_id=10,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    row = rows[0]
    assert abs(row["component_sum"] - row["prenorm_dot"]) < 1e-3
    for got, exp in zip(row["lens"][-1]["probs"], row["cond_probs"]):
        assert abs(got - exp) < 1e-4
    # Layer 0 is full attention and must split into heads.
    assert row["head_attr"][0] is not None
    assert abs(sum(row["head_attr"][0]) - row["attn_attr"][0]) < 1e-3
    # Layer 1 is GatedDeltaNet; out_proj still splits into value heads.
    assert row["head_attr"][1] is not None
    assert abs(sum(row["head_attr"][1]) - row["attn_attr"][1]) < 1e-3
    assert row["attn_layers_used"] >= 1
