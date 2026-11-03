import json
from pathlib import Path

import torch

from rba.cli import main
from rba.config import Config
from rba.data.encode import collate, encode_cases
from rba.errors import RBAError
from rba.models.loader import load_models
from rba.models.paths import resolve_paths
from rba.repair.evaluate import compare, conclusion
from rba.repair.interventions import apply_patch, blend_lm_head, scale_heads
from rba.schema import Case
from rba.util import call_model, logits_of


def _score_pair():
    before = [
        {"id": "a", "label": "3", "pred": "0"},
        {"id": "b", "label": "0", "pred": "0"},
    ]
    after = [
        {"id": "a", "label": "3", "pred": "3"},
        {"id": "b", "label": "0", "pred": "1"},
    ]
    bad = compare(before, after)
    assert bad["fix_rate"] == 1.0
    assert bad["regression_rate"] == 1.0
    assert "不要保留" in conclusion(bad, bad)


def test_metrics():
    _score_pair()


def test_lm_head_blend_moves_gold_up():
    cfg = Config(device="cpu", dtype="float32", plant_error=False, seed=1, goodcases=None)
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, {}, "qwen3_5")
    head = loaded.ft.lm_head.weight
    gold = loaded.grade_ids["3"]
    wrong = loaded.grade_ids["0"]
    with torch.no_grad():
        head.zero_()
        head[wrong].fill_(3.0)
        head[gold].fill_(0.1)
    case = Case(id="x", query="查询", doc="文档", image=None, label="3")
    encoded = encode_cases([case], loaded.tokenizer, cfg)
    batch = collate(encoded, 0)
    with torch.no_grad():
        before = logits_of(call_model(loaded.ft, batch)).detach()
    with blend_lm_head(loaded.ft, resolved, wrong, gold, alpha=1.0):
        with torch.no_grad():
            after = logits_of(call_model(loaded.ft, batch)).detach()
    decision = before.size(1) - 1
    # Blending the wrong row onto the gold row does not move the gold logit.
    # It collapses the two rows, so the gap between them has to shrink.
    gap_before = abs(float(before[0, decision, wrong] - before[0, decision, gold]))
    gap_after = abs(float(after[0, decision, wrong] - after[0, decision, gold]))
    assert gap_after < gap_before
    assert gap_after < 1e-4
    assert torch.allclose(loaded.ft.lm_head.weight[wrong], torch.full_like(loaded.ft.lm_head.weight[wrong], 3.0))


def test_head_scale_changes_logits_and_restores():
    cfg = Config(device="cpu", dtype="float32", plant_error=False, seed=2, goodcases=None)
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, {}, "qwen3_5")
    case = Case(id="x", query="查询一下", doc="一段文档", image=None, label="2")
    encoded = encode_cases([case], loaded.tokenizer, cfg)
    batch = collate(encoded, 0)
    with torch.no_grad():
        base = logits_of(call_model(loaded.ft, batch)).detach().clone()
    with scale_heads(loaded.ft, resolved, layer=0, heads=[0], scale=0.0):
        with torch.no_grad():
            ablated = logits_of(call_model(loaded.ft, batch)).detach()
    with torch.no_grad():
        again = logits_of(call_model(loaded.ft, batch)).detach()
    assert not torch.allclose(base, ablated)
    assert torch.allclose(base, again)


def test_wrong_module_path_is_chinese():
    cfg = Config(device="cpu", dtype="float32", plant_error=False, goodcases=None)
    loaded = load_models(cfg)
    try:
        resolve_paths(loaded.ft, {"layers": "no.such.layers"}, "qwen3_5")
    except RBAError as exc:
        assert "建议" in str(exc)
        assert "no.such.layers" in exc.message
    else:
        raise AssertionError("应当拒绝不存在的层路径")


def test_cli_dry_run_and_resume(tmp_path: Path, capsys):
    out = tmp_path / "run"
    code = main([
        "analyze", "--config", "example.yaml", "--dry-run", "--fresh",
        "--output-dir", str(out), "--device", "cpu",
    ])
    assert code is None
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "先看这里" in html
    assert "根因" in html or "假设" in html
    assert (out / "report.md").exists()
    assert (out / "report.json").exists()
    prompts = list((out / "prompts").glob("*.txt"))
    assert prompts
    assert "任务。" in prompts[0].read_text(encoding="utf-8")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["groups"]
    assert any(group.get("hypotheses") for group in report["groups"])
    # second run resumes cached stages
    main(["analyze", "--config", "example.yaml", "--dry-run", "--output-dir", str(out), "--device", "cpu"])
    captured = capsys.readouterr()
    assert "从缓存恢复" in captured.out


def test_cli_zero_config_and_init_and_missing(tmp_path: Path, capsys):
    out = tmp_path / "zero"
    main([
        "analyze", "--base", "tiny", "--ft", "tiny",
        "--badcases", "examples/badcases.jsonl",
        "--goodcases", "examples/goodcases.jsonl",
        "--dry-run", "--output-dir", str(out), "--device", "cpu", "--fresh",
    ])
    assert (out / "report.html").exists()
    init_path = tmp_path / "rba.yaml"
    main(["init", "--base", "tiny", "--ft", "tiny", "--badcases", "examples/badcases.jsonl", "--out", str(init_path)])
    text = init_path.read_text(encoding="utf-8")
    assert "model.language_model.layers" in text
    Config.load(init_path)
    try:
        main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", str(tmp_path / "missing.jsonl")])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("缺文件应该退出 2")
    captured = capsys.readouterr()
    assert "建议" in captured.err


def test_eval_patch_on_planted_head(tmp_path: Path):
    """A scale-0 of every head is not the assertion; lm_head blend toward the label is."""
    from rba.pipeline import evaluate_patch

    cfg = Config(
        base_model="tiny",
        ft_model="tiny",
        model_type="tiny",
        badcases="examples/badcases.jsonl",
        goodcases="examples/goodcases.jsonl",
        device="cpu",
        dtype="float32",
        seed=0,
        plant_error=True,
        output_dir=str(tmp_path / "eval"),
        config_path=str(Path("example.yaml").resolve()),
    )
    patch = {
        "method": "lm_head_blend",
        "grade": "0",
        "toward": "3",
        "alpha": 1.0,
        "tiny_seed": 0,
        "plant_error": True,
    }
    result = evaluate_patch(cfg, patch)
    assert "fix_rate" in result["bad"]
    assert result["bad"]["n"] > 0
    assert result["good"]["n"] > 0
    assert "修复率" in result["conclusion"] or "回退" in result["conclusion"]
