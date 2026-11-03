import torch

from rba.analysis.grouping import cluster_group, kmeans
from rba.analysis.hypotheses import aggregate_mechanistic, rank_hypotheses
from rba.analysis.instrument import instrument_batch
from rba.config import Config
from rba.data.encode import encode_cases
from rba.models.loader import load_models
from rba.models.paths import resolve_paths
from rba.models.tiny import build_tiny_pair
from rba.schema import Case
import numpy as np


def _loaded():
    cfg = Config(device="cpu", dtype="float32", goodcases=None, plant_error=True, seed=0)
    cfg.badcases = "examples/badcases.jsonl"
    return cfg, load_models(cfg)


def test_decomposition_and_lens_match_logits():
    cfg, loaded = _loaded()
    resolved = resolve_paths(loaded.ft, {}, "qwen3_5")
    cases = [
        Case(id="a", query="红烧肉", doc="五花肉的做法", image=None, label="3"),
        Case(id="b", query="天气", doc="欧冠决赛", image="examples/images/sample.ppm", label="0"),
    ]
    encoded = encode_cases(cases, loaded.tokenizer, cfg, image_size=cfg.tiny.image_size)
    from rba.data.encode import collate

    batch = collate(encoded, loaded.tokenizer.pad_token_id)
    grades = cfg.grades
    grade_ids = [loaded.grade_ids[g] for g in grades]
    pred = torch.tensor([loaded.grade_ids["0"], loaded.grade_ids["0"]])
    gold = torch.tensor([loaded.grade_ids["3"], loaded.grade_ids["0"]])
    rows = instrument_batch(
        loaded.ft, batch, resolved, grade_ids, pred, gold, loaded.image_token_id, loaded.device, loaded.dtype
    )
    for row in rows:
        assert abs(row["component_sum"] - row["prenorm_dot"]) < 1e-3
        last = row["lens"][-1]["probs"]
        for got, exp in zip(last, row["cond_probs"]):
            assert abs(got - exp) < 1e-4
        # Head slices reconstruct the attention write on every full-attention layer.
        for layer, heads in enumerate(row["head_attr"]):
            if heads is None or row["attn_attr"][layer] is None:
                continue
            assert abs(sum(heads) - row["attn_attr"][layer]) < 1e-3
    image_row = rows[1]
    assert image_row["image_token_fraction"] > 0
    assert image_row["attn_image_share"] is not None


def test_delta_finds_planted_modules():
    from rba.analysis.delta_w import analyze_delta

    cfg = Config(seed=0, plant_error=True)
    base, ft, tokenizer, _meta = build_tiny_pair(cfg)
    grade_ids = {g: tokenizer.token_to_id[g] for g in cfg.grades}
    report = analyze_delta(base, ft, None, ft.lm_head.weight.detach().cpu(), grade_ids, tokenizer, top_k=8, svd_rank=2, svd_max_dim=256)
    names = [row["name"] for row in report["top"][:4]]
    assert any(name.endswith("lm_head.weight") for name in names)
    assert any("layers.0" in name and "o_proj" in name for name in names)
    assert report["lm_head_grades"]["rows"]["0"]["delta_norm"] > 1.0


def test_token_attr_splits_image_tokens():
    from rba.analysis.token_attr import attribute_batch
    from rba.data.encode import collate

    cfg, loaded = _loaded()
    resolved = resolve_paths(loaded.ft, {}, "qwen3_5")
    cases = [Case(id="img", query="红烧肉", doc="做法", image="examples/images/sample.ppm", label="3")]
    encoded = encode_cases(cases, loaded.tokenizer, cfg, image_size=8)
    batch = collate(encoded, 0)
    pred = torch.tensor([loaded.grade_ids["0"]])
    gold = torch.tensor([loaded.grade_ids["3"]])
    rows = attribute_batch(
        loaded.ft, batch, resolved, pred, gold, loaded.image_token_id, loaded.tokenizer,
        loaded.device, loaded.dtype, "gradxinput", 2,
    )
    assert rows[0]["n_image"] == cfg.image_seq_len
    assert rows[0]["n_text"] > 0
    assert abs(rows[0]["text_abs_share"] + rows[0]["image_abs_share"] - 1) < 1e-4
    ig = attribute_batch(
        loaded.ft, batch, resolved, pred, gold, loaded.image_token_id, loaded.tokenizer,
        loaded.device, loaded.dtype, "integrated_gradients", 2,
    )
    assert ig[0]["n_image"] == cfg.image_seq_len


def test_rank_prefers_verified_over_correlation():
    summary = {"key": "3->0", "label": "3", "pred": "0", "count": 8, "has_image_fraction": 0.0}
    mech = {
        "components": [
            {"key": "L0.self_attn.h0", "kind": "attn_head", "layer": 0, "head": 0, "attn_name": "self_attn", "mean_attr": 4.0, "std_attr": 0.1, "share": 0.8},
            {"key": "L1.mlp", "kind": "mlp", "layer": 1, "head": None, "attn_name": "self_attn", "mean_attr": 0.2, "std_attr": 0.2, "share": 0.05},
        ],
        "grade_mass": 0.95,
        "full_top_grade_rate": 1.0,
        "decomposition_gap": 0.0,
        "image_token_fraction": 0.0,
        "attn_image_share": 0.0,
    }
    verifications = [
        {"key": "L0.self_attn.h0", "mean_delta": -1.5, "n": 4, "n_improved": 4, "agreement": 1.0},
        {"key": "L1.mlp", "mean_delta": 0.8, "n": 4, "n_improved": 0, "agreement": 0.0},
    ]
    ranked = rank_hypotheses(summary, mech, {"lm_head_grades": {"rows": {}}, "top": []}, verifications, n_parameters=1000)
    assert ranked[0]["status"] == "消融有效"
    assert ranked[0]["key"] == "L0.self_attn.h0"
    rejected = next(item for item in ranked if item["key"] == "L1.mlp")
    assert rejected["status"] == "被复核否定"
    assert ranked[0]["rank"] == 1
    low_mass = dict(mech)
    low_mass["grade_mass"] = 0.03
    low_mass["full_top_grade_rate"] = 0.0
    reranked = rank_hypotheses(summary, low_mass, {"lm_head_grades": {"rows": {}}, "top": []}, verifications, n_parameters=1000)
    assert reranked[0]["key"] == "grade_mass"
    assert reranked[0]["status"] == "先处理这个"


def test_kmeans_and_cluster_note():
    labels = kmeans(np.array([[0.0], [0.1], [5.0], [5.2]]), k=2, seed=0)
    assert set(labels.tolist()) == {0, 1}
    records = []
    for index in range(6):
        records.append({"probs": {"0": 1.0, "1": 0.0}, "query": "a" * (index + 1), "doc": "b", "has_image": index > 3, "label": "1", "pred": "0"})
    summary = cluster_group(records, ["0", "1"], k=2, seed=0, min_size=6)
    assert summary and sum(item["size"] for item in summary) == 6


def test_aggregate_flip_layer():
    detail = [{
        "label": "1",
        "pred": "0",
        "lens": [
            {"layer": 0, "probs": [0.2, 0.8]},
            {"layer": 1, "probs": [0.7, 0.3]},
        ],
        "attn_attr": [0.1, 0.2],
        "mlp_attr": [0.0, 0.0],
        "head_attr": [[0.1], [0.2]],
        "kinds": ["self_attn", "self_attn"],
        "component_sum": 0.3,
        "prenorm_dot": 0.3,
        "logit_diff": 0.4,
        "image_token_fraction": 0.0,
        "attn_image_share": None,
        "attn_layers_used": 1,
        "grade_mass": 1.0,
        "full_top_is_grade": True,
    }]
    summary = aggregate_mechanistic(detail, ["0", "1"])
    assert summary["flip_layer"] == 1
