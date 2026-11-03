"""Round-3 CLI / flow / report tests (Fable): replay, error handling, protocols, parity."""

import json
import os
import re
from pathlib import Path

import pytest

import rba.cli as cli
import rba.pipeline as pipeline
from rba.cli import main
from rba.config import Config, config_from_used, config_to_dict
from rba.data.encode import EncodedCase, needs_single_item_batches
from rba.errors import RBAError
from rba.schema import Case

ROOT = Path(__file__).resolve().parents[1]
BAD = str(ROOT / "examples" / "badcases.jsonl")
GOOD = str(ROOT / "examples" / "goodcases.jsonl")


def _analyze(out: Path, *extra: str) -> dict:
    main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--goodcases", GOOD, "--dry-run", "--fresh", "--output-dir", str(out), "--device", "cpu", *extra])
    return json.loads((out / "report.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- config replay


def test_config_roundtrip_keeps_every_field():
    cfg = Config(grade_mode="first_token", decode_protocol="free", holdout_fraction=0.3, max_length=777, module_paths={"layers": "a.b"}, goodcases=None)
    cfg.llm.model = "x-model"
    restored = config_from_used(config_to_dict(cfg))
    assert restored.grade_mode == "first_token" and restored.decode_protocol == "free"
    assert restored.holdout_fraction == 0.3 and restored.max_length == 777
    assert restored.module_paths == {"layers": "a.b"} and restored.llm.model == "x-model"
    with pytest.raises(RBAError):
        config_from_used({**config_to_dict(cfg), "schema_version": 99})


def test_int_yaml_grade_keys_are_normalized(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("base_model: tiny\nft_model: tiny\ngrade_token_ids: {-1: 2, 0: 3, 1: 4, 2: 5, 3: 6}\n", encoding="utf-8")
    cfg = Config.load(path)
    assert list(cfg.grade_token_ids) == ["-1", "0", "1", "2", "3"]
    path.write_text('base_model: tiny\nft_model: tiny\ngrade_token_ids: {-1: 2, "-1": 3, 0: 3, 1: 4, 2: 5, 3: 6}\n', encoding="utf-8")
    with pytest.raises(RBAError) as caught:
        Config.load(path)
    assert "重复" in caught.value.message


def test_eval_run_restores_full_config_and_labels_discovery(tmp_path: Path, capsys):
    out = tmp_path / "run"
    report = _analyze(out)
    used = json.loads((out / "config.used.json").read_text(encoding="utf-8"))
    assert used["schema_version"] == 2
    assert used["runtime"]["device_resolved"] == "cpu" and "grade_ids" in used["runtime"]
    assert used["runtime"]["split"]["n_holdout"] == 0
    # The next-step command uses --run, never a hard-coded device.
    commands = [h["command"] for g in report["groups"] for h in g["hypotheses"] if h.get("command")]
    assert commands and all("--run" in c and "--device" not in c for c in commands)
    patch = next(iter((out / "patches").glob("*.json")))
    main(["eval", "--run", str(out), "--patch", str(patch)])
    result = json.loads((out / "eval" / "eval.json").read_text(encoding="utf-8"))
    assert result["bad_source"] == "discovery"
    assert "发现集" in result["conclusion"]
    assert result["good"] is not None and "fix_rate_ci" in result["bad"]
    assert result["patch"]["run_fingerprint"] == report["fingerprint"]
    captured = capsys.readouterr()
    assert "发现集复评" in captured.out


def test_eval_rejects_run_plus_config(tmp_path: Path):
    out = tmp_path / "run"
    _analyze(out)
    patch = next(iter((out / "patches").glob("*.json")))
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--run", str(out), "--config", str(ROOT / "example.yaml"), "--patch", str(patch)])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- holdout split


def test_holdout_split_is_deterministic_and_eval_uses_it(tmp_path: Path):
    out = tmp_path / "run"
    report = _analyze(out, "--holdout", "0.5")
    split = report["split"]
    assert split["n_holdout"] > 0 and split["n_holdout"] + split["n_discovery"] == split["n_total"]
    again = _analyze(tmp_path / "run2", "--holdout", "0.5")["split"]
    assert again["holdout_ids"] == split["holdout_ids"]
    patch = next(iter((out / "patches").glob("*.json")))
    main(["eval", "--run", str(out), "--patch", str(patch)])
    result = json.loads((out / "eval" / "eval.json").read_text(encoding="utf-8"))
    assert result["bad_source"] == "holdout" and result["bad"]["n"] == split["n_holdout"]
    assert result["discovery"] is not None and "验收集" in result["conclusion"]


# --------------------------------------------------------------------------- protocols


def test_free_protocol_separates_invalid_rows(tmp_path: Path):
    out = tmp_path / "free"
    report = _analyze(out, "--decode-protocol", "free")
    assert report["decode_protocol"] == "free"
    # The random tiny model rarely puts its argmax on a grade token.
    assert report["overview"]["n_invalid_bad"] > 0
    grade_rows = sum(g["count"] for g in report["groups"])
    assert grade_rows + report["overview"]["n_invalid_bad"] <= report["overview"]["n_bad"]
    assert all("__invalid__" not in g["key"] for g in report["groups"])
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## 非法输出" in md and "全词表 top token" in md
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "非法输出" in html


def test_missing_image_policy_through_cli(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    rows = [
        {"id": "ok", "query": "红烧肉", "doc": "五花肉", "label": 3},
        {"id": "gone", "query": "天气", "doc": "北京", "image": "nope/missing.png", "label": 0},
        {"id": "ok2", "query": "欧冠", "doc": "决赛", "label": 1},
        {"id": "ok3", "query": "热量", "doc": "偏肥", "label": 2},
    ]
    bad.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", str(bad), "--dry-run", "--output-dir", str(tmp_path / "a"), "--device", "cpu"])
    assert exc.value.code == 2
    main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", str(bad), "--dry-run", "--allow-missing-images", "--output-dir", str(tmp_path / "b"), "--device", "cpu"])
    report = json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))
    assert report["excluded"][0]["id"] == "gone" and report["overview"]["n_excluded"] == 1
    assert report["overview"]["n_bad"] == 3
    assert "剔除的样本" in (tmp_path / "b" / "report.md").read_text(encoding="utf-8")


def test_processor_items_force_single_batches():
    case = Case(id="a", query="q", doc="d", image=None, label="0")
    plain = EncodedCase(id="a", case=case, input_ids=[1, 2], attention_mask=[1, 1])
    packed = EncodedCase(id="b", case=case, input_ids=[1, 2], attention_mask=[1, 1], model_inputs={"input_ids": None})
    assert not needs_single_item_batches([plain])
    assert needs_single_item_batches([plain, packed])
    cfg = Config(batch_size=8, goodcases=None)
    import torch

    assert pipeline._effective_batch(cfg, torch.device("cpu"), [plain, packed]) == 1
    assert pipeline._effective_batch(cfg, torch.device("cpu"), [plain]) == 8


# --------------------------------------------------------------------------- error handling


def test_unexpected_exception_writes_error_log(tmp_path: Path, monkeypatch, capsys):
    def boom(cfg):
        raise KeyError("synthetic")

    monkeypatch.setattr(cli, "analyze", boom)
    out = tmp_path / "x"
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--dry-run", "--output-dir", str(out)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "程序内部错误" in err and "error.log" in err
    assert "KeyError" in (out / "error.log").read_text(encoding="utf-8")


def test_oom_at_batch_one_becomes_rba_error():
    calls = {"n": 0}

    def fn(chunk):
        calls["n"] += 1
        raise RuntimeError("CUDA out of memory. Tried to allocate")

    with pytest.raises(RBAError) as caught:
        pipeline._run_batches([1, 2, 3, 4], 4, fn, "测试阶段")
    assert "max_length" in caught.value.hint and calls["n"] == 3  # 4 -> 2 -> 1


def test_llm_failure_still_writes_report(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def failing_chat(*args, **kwargs):
        raise RBAError("连不上接口", "检查 base_url")

    monkeypatch.setattr(pipeline, "chat", failing_chat)
    out = tmp_path / "llm"
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--fresh", "--output-dir", str(out), "--device", "cpu"])
    assert exc.value.code == 3
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["llm_mode"] == "remote" and report["llm_failures"]
    assert (out / "report.html").exists()
    assert "调用失败" in (out / "report.md").read_text(encoding="utf-8")


def test_missing_api_key_degrades_to_dry_run(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "nokey"
    main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--fresh", "--output-dir", str(out), "--device", "cpu"])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["llm_mode"] == "dry_run"
    assert any("未设置" in w for w in report["warnings"])
    assert list((out / "prompts").glob("*.txt"))


def test_tied_lm_head_patch_is_rejected():
    class Loaded:
        tied_lm_head = True

    with pytest.raises(RBAError) as caught:
        pipeline._preflight_patch({"method": "lm_head_blend", "grade": "0", "toward": "3"}, Loaded())
    assert "共享权重" in caught.value.message
    pipeline._preflight_patch({"method": "head_scale", "layer": 0, "heads": [0]}, Loaded())


# --------------------------------------------------------------------------- report parity


def test_html_matches_markdown_sections(tmp_path: Path):
    out = tmp_path / "parity"
    _analyze(out)
    md = (out / "report.md").read_text(encoding="utf-8")
    html = (out / "report.html").read_text(encoding="utf-8")
    md_h2 = re.findall(r"^## (.+)$", md, flags=re.M)
    html_h2 = re.findall(r"<h2>(.*?)</h2>", html)
    assert md_h2 == html_h2
    assert len(re.findall(r"^\| ---", md, flags=re.M)) == html.count("<table>")
    assert html.count("<img ") == len(re.findall(r"^!\[", md, flags=re.M))
    for status in ("先处理这个", "消融有效", "证据不足"):
        assert status in md
    assert "置信度" not in md  # no fake calibrated confidence
    assert "状态含义" in md and "消融有效" in html
    assert "<h2>证据</h2>" in html and "干预复核" in html


def test_demo_both_archs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(ROOT)
    main(["demo", "--output-dir", str(tmp_path / "d1")])
    main(["demo", "--arch", "qwen3_5_tiny", "--output-dir", str(tmp_path / "d2")])
    for folder in ("d1", "d2"):
        report = json.loads((tmp_path / folder / "report.json").read_text(encoding="utf-8"))
        assert "先处理这个" in report["banner"] or "等级词" in report["banner"]
        assert report["groups"]


def test_init_leaves_module_paths_commented(tmp_path: Path):
    path = tmp_path / "rba.yaml"
    main(["init", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--out", str(path)])
    text = path.read_text(encoding="utf-8")
    assert "# module_paths:" in text and "grade_mode" in text and "decode_protocol" in text
    cfg = Config.load(path)
    assert cfg.module_paths == {}
