"""Round-5 follow-ups (Fable): holdout/control overlap, proposal scope, unaligned SVD, custom head."""

import json
from pathlib import Path

import pytest
import torch

import rba.pipeline as pipeline
from rba.cache import RunCache
from rba.cli import main
from rba.config import Config
from rba.models.loader import load_models
from rba.models.paths import resolve_paths
from rba.schema import Case, load_cases

ROOT = Path(__file__).resolve().parents[1]
BAD = str(ROOT / "examples" / "badcases.jsonl")


def test_control_rows_matching_holdout_content_are_excluded(tmp_path: Path):
    cfg = Config(holdout_fraction=0.5, seed=0, goodcases=None)
    bad = load_cases(BAD, cfg.grades)
    split = pipeline._split_info(bad, cfg)
    holdout = next(case for case in bad if case.id in split["holdout_ids"])
    good_rows = [
        {"id": "dup", "query": holdout.query, "doc": holdout.doc, "image": holdout.image, "label": holdout.label},
        {"id": "fresh1", "query": "全新查询", "doc": "全新文档", "label": 0},
        {"id": "fresh2", "query": "另一个查询", "doc": "另一个文档", "label": 1},
    ]
    good = tmp_path / "good.jsonl"
    good.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in good_rows) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--goodcases", str(good), "--holdout", "0.5", "--dry-run", "--fresh", "--output-dir", str(out), "--device", "cpu"])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["split"]["n_good_excluded_overlap"] == 1 and report["split"]["good_excluded_overlap_ids"] == ["dup"]
    assert any(row["id"] == "dup" and "验收集" in row["reason"] for row in report["excluded"])
    assert report["overview"]["n_good"] == 2
    good_cache = json.loads((out / "cache" / "infer_good.json").read_text(encoding="utf-8"))
    assert {row["id"] for row in good_cache} == {"fresh1", "fresh2"}
    patch = next(iter((out / "patches").glob("*.json")))
    main(["eval", "--run", str(out), "--patch", str(patch)])
    result = json.loads((out / "eval" / "eval.json").read_text(encoding="utf-8"))
    assert result["good"]["n"] == 2 and any(row["id"] == "dup" for row in result["excluded"])


def test_untested_proposal_keeps_its_declared_scope(tmp_path: Path):
    groups = [{
        "key": "3->0",
        "verifications": [],
        "hypotheses": [
            {"key": "L0.self_attn.h0", "title": "t", "status": "仅相关", "patch_file": "patches/a.json", "patch": {"method": "head_scale", "layer": 0, "heads": [0], "scale": 0.5, "scope": "decision"}},
            {"key": "L1.self_attn.h0", "title": "t", "status": "仅相关", "patch_file": "patches/b.json", "patch": {"method": "head_scale", "layer": 1, "heads": [0], "scale": 0.5}},
        ],
    }]
    cfg = Config(goodcases=None)
    pipeline._write_patches(tmp_path, groups, cfg, None, "fp", False)
    declared = json.loads((tmp_path / "patches" / "a.json").read_text(encoding="utf-8"))
    defaulted = json.loads((tmp_path / "patches" / "b.json").read_text(encoding="utf-8"))
    assert declared["scope"] == "decision" and declared["untested"] is True and declared["verified_on_sample"] is False
    assert defaulted["scope"] == "all_positions" and defaulted["untested"] is True


def test_unaligned_base_strips_vocab_direction_semantics(tmp_path: Path, monkeypatch):
    loaded = load_models(Config(device="cpu", dtype="float32", goodcases=None))
    resolved = resolve_paths(loaded.ft, {}, "qwen3_5")
    monkeypatch.setattr(pipeline, "check_base_grade_alignment", lambda cfg, loaded, ids: {"status": "unavailable", "reason": "测试：基座分词器不可用", "checked": None})
    payload = pipeline._cached_delta(RunCache(tmp_path / "c", "fp", resume=False), Config(goodcases=None), loaded, loaded.grade_ids, resolved)
    spaces = set()
    for row in payload["top"]:
        svd = row.get("svd")
        if not isinstance(svd, dict) or not svd.get("directions"):
            continue
        assert svd["base_alignment"] == "unknown"
        for direction in svd["directions"]:
            spaces.add(direction.get("space"))
            assert direction["base_alignment"] == "unknown"
            if direction.get("space") == "vocab":
                assert "grades" not in direction and "top_tokens" not in direction and "unavailable" in direction["semantics"]
            if direction.get("space") == "residual":
                assert "fine-tuned head only" in direction["semantics"]
    assert "vocab" in spaces  # the planted lm_head delta is in the top list


def test_cached_delta_uses_resolved_custom_head(tmp_path: Path):
    loaded = load_models(Config(device="cpu", dtype="float32", goodcases=None))
    custom = torch.nn.Linear(loaded.ft.lm_head.in_features, loaded.ft.lm_head.out_features, bias=False)
    with torch.no_grad():
        custom.weight.copy_(loaded.ft.lm_head.weight * 3.0)
    loaded.ft.custom_head = custom

    class Resolved:
        def lm_head(self, model):
            return model.custom_head

        def layers(self, model):
            return model.model.language_model.layers

    payload = pipeline._cached_delta(RunCache(tmp_path / "c", "fp", resume=False), Config(goodcases=None), loaded, loaded.grade_ids, Resolved())
    expected = float(custom.weight.detach()[loaded.grade_ids["0"]].norm())
    assert payload["lm_head_grades"]["rows"]["0"]["ft_norm"] == pytest.approx(expected, rel=1e-5)
    hypotheses = [{"patch": {"method": "lm_head_blend", "grade": "0", "toward": "3"}}]
    pipeline._annotate_fraction(hypotheses, loaded.ft, Resolved(), 1000)
    assert hypotheses[0]["weight_fraction"] == custom.weight.shape[1] / 1000
