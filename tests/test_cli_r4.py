"""Round-4 cross-review fixes (Fable): holdout isolation, cache identity, scope, patches, guards."""

import json
import shlex
import shutil
from pathlib import Path

import pytest
import torch

import rba.pipeline as pipeline
from rba.cli import main
from rba.config import Config
from rba.data.encode import encode_cases
from rba.errors import RBAError
from rba.schema import Case

ROOT = Path(__file__).resolve().parents[1]
BAD = str(ROOT / "examples" / "badcases.jsonl")
GOOD = str(ROOT / "examples" / "goodcases.jsonl")


def _analyze(out: Path, *extra: str, bad: str = BAD, good: str | None = GOOD) -> dict:
    args = ["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", bad, "--dry-run", "--output-dir", str(out), "--device", "cpu", *extra]
    if good:
        args += ["--goodcases", good]
    main(args)
    return json.loads((out / "report.json").read_text(encoding="utf-8"))


def _copy_examples(tmp_path: Path) -> tuple[str, str]:
    folder = tmp_path / "data"
    shutil.copytree(ROOT / "examples", folder)
    return str(folder / "badcases.jsonl"), str(folder / "goodcases.jsonl")


# --------------------------------------------------------------------------- P0-1 holdout isolation


def test_holdout_rows_never_enter_analysis(tmp_path: Path):
    out = tmp_path / "run"
    report = _analyze(out, "--holdout", "0.5", "--fresh")
    holdout = set(report["split"]["holdout_ids"])
    assert holdout
    infer_ids = {row["id"] for row in json.loads((out / "cache" / "infer_bad.json").read_text(encoding="utf-8"))}
    assert not (infer_ids & holdout)
    assert report["overview"]["n_bad"] == report["split"]["n_discovery"]
    for group in report["groups"]:
        assert not (set(group["detail_ids"]) & holdout)
        for verification in group["verifications"]:
            assert not ({case["id"] for case in verification.get("per_case", [])} & holdout)
        packet = group.get("evidence_packet") or {}
        case_ids = {obs["evidence_id"].split(":", 1)[1] for obs in packet.get("observations", []) if obs["evidence_id"].startswith("case:")}
        assert not (case_ids & holdout)
    used = json.loads((out / "config.used.json").read_text(encoding="utf-8"))
    assert set(used["runtime"]["split"]["holdout_ids"]) == holdout
    assert "identities" in used["runtime"]


def test_holdout_everything_is_an_error(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--base", "tiny", "--ft", "tiny", "--badcases", BAD, "--dry-run", "--holdout", "0.999999", "--output-dir", str(tmp_path / "x"), "--device", "cpu"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- P0-2 cache identity


def test_image_change_invalidates_cache(tmp_path: Path, capsys):
    bad, good = _copy_examples(tmp_path)
    out = tmp_path / "run"
    _analyze(out, "--fresh", bad=bad, good=good)
    _analyze(out, bad=bad, good=good)  # same inputs resume
    assert "从缓存恢复" in capsys.readouterr().out
    image = Path(bad).parent / "images" / "sample.ppm"
    image.write_text(image.read_text(encoding="utf-8").replace("220 40 40", "10 10 10", 1), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _analyze(out, bad=bad, good=good)
    assert exc.value.code == 2
    assert "不同配置或不同数据" in capsys.readouterr().err


def test_model_identity_tracks_config_tokenizer_and_weights(tmp_path: Path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (folder / "tokenizer_config.json").write_text('{"chat_template": "x"}', encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"\x00" * 16)
    first = pipeline.model_identity(str(folder))
    (folder / "tokenizer_config.json").write_text('{"chat_template": "y"}', encoding="utf-8")
    second = pipeline.model_identity(str(folder))
    assert first != second and first["files"]["config.json"] == second["files"]["config.json"]
    (folder / "model.safetensors").write_bytes(b"\x00" * 32)
    assert pipeline.model_identity(str(folder))["weights"] != second["weights"]
    assert pipeline.model_identity("tiny") == {"kind": "tiny"}


def test_eval_warns_when_inputs_changed_after_analysis(tmp_path: Path):
    bad, good = _copy_examples(tmp_path)
    out = tmp_path / "run"
    _analyze(out, "--fresh", bad=bad, good=good)
    patch = next(iter(p for p in (out / "patches").glob("*.json")))
    image = Path(bad).parent / "images" / "sample.ppm"
    image.write_text(image.read_text(encoding="utf-8").replace("220 40 40", "10 10 10", 1), encoding="utf-8")
    main(["eval", "--run", str(out), "--patch", str(patch)])
    result = json.loads((out / "eval" / "eval.json").read_text(encoding="utf-8"))
    assert result["replay_valid"] is False and any("images" in note for note in result["provenance_notes"])
    assert "不是同一次实验的回放" in result["conclusion"]


# --------------------------------------------------------------------------- P1-3 controls flag


def test_verify_controls_false_runs_only_targets(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"base_model: tiny\nft_model: tiny\nbadcases: {BAD}\ngoodcases: {GOOD}\nverify_controls: false\n"
        f"output_dir: {tmp_path / 'run'}\ndevice: cpu\ndtype: float32\n",
        encoding="utf-8",
    )
    main(["analyze", "--config", str(cfg_path), "--dry-run", "--fresh"])
    report = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    kinds = {v["control_type"] for g in report["groups"] for v in g["verifications"]}
    assert kinds == {"target"}


# --------------------------------------------------------------------------- P1-4 scope


def test_patch_files_carry_verified_intervention_and_eval_honors_scope(tmp_path: Path):
    out = tmp_path / "run"
    report = _analyze(out, "--fresh")
    files = [p for p in (out / "patches").glob("*.json")]
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["scope"] == "decision" and payload["verified_on_sample"] is True
    assert payload["method"] == "head_scale" and payload["scale"] == 0.0
    assert payload["proposal_patch"]["scale"] == 0.5  # the milder proposal is kept, not silently evaluated
    main(["eval", "--run", str(out), "--patch", str(files[0])])
    result = json.loads((out / "eval" / "eval.json").read_text(encoding="utf-8"))
    assert result["requested_scope"] == "decision" and result["scope"] == "decision"
    # No recommended command points at a diagnostic lm_head row edit.
    for group in report["groups"]:
        for item in group["hypotheses"]:
            if item.get("command"):
                assert item["patch"]["method"] != "lm_head_blend"
                assert item["status"] in pipeline.VERIFIED_STATUSES or item.get("exploratory")


def test_decision_scope_multi_layer_steering_and_shape_guard():
    from rba.models.loader import load_models
    from rba.models.paths import resolve_paths
    from rba.data.encode import collate
    from rba.util import call_model, logits_of

    cfg = Config(device="cpu", dtype="float32", plant_error=False, goodcases=None)
    loaded = load_models(cfg)
    resolved = resolve_paths(loaded.ft, {}, "qwen3_5")
    case = Case(id="x", query="查询一下", doc="一段文档", image=None, label="2")
    batch = collate(encode_cases([case], loaded.tokenizer, cfg), 0)
    index = torch.tensor([batch["input_ids"].size(1) - 1])

    def logits():
        with torch.no_grad():
            return logits_of(call_model(loaded.ft, {k: v for k, v in batch.items() if k != "case_ids"})).detach().clone()

    base = logits()
    one = {"method": "steering", "layers": [0], "from_grade": "0", "to_grade": "2", "alpha": 3.0}
    two = {"method": "steering", "layers": [0, 1], "from_grade": "0", "to_grade": "2", "alpha": 3.0}
    with pipeline._decision_position_patch(loaded.ft, resolved, one, loaded.grade_ids, index):
        after_one = logits()
    with pipeline._decision_position_patch(loaded.ft, resolved, two, loaded.grade_ids, index):
        after_two = logits()
    assert not torch.allclose(base, after_one) and not torch.allclose(after_one, after_two)
    # Earlier positions are untouched by a decision-position intervention.
    assert torch.allclose(base[0, :-1], after_two[0, :-1])
    with pytest.raises(RBAError):
        wrong_index = torch.tensor([0, 0])  # batch of 2 against a batch-1 forward
        with pipeline._decision_position_patch(loaded.ft, resolved, one, loaded.grade_ids, wrong_index):
            logits()


# --------------------------------------------------------------------------- P1-5 vision block boundary


def test_truncation_cannot_drop_vision_start_but_keep_image_tokens(tmp_path: Path):
    pytest.importorskip("transformers.models.qwen3_5")
    from tests.vl_fixture import build_processor, build_tokenizer, random_image

    tokenizer = build_tokenizer()
    processor = build_processor(tokenizer)
    image_path = random_image(tmp_path / "a.png")
    cfg = Config(base_model="x", ft_model="x", model_type="hf", prompt_template="查询：{query}\n文档：{doc}\n{image}相关性等级：", max_length=4096, image_max_pixels=None, goodcases=None)
    case = Case(id="a", query="红烧肉" * 5, doc="五花肉", image=image_path, label="3")
    item = encode_cases([case], tokenizer, cfg, processor=processor)[0]
    start = item.input_ids.index(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    # A cut that removes exactly the delimiter and nothing after it.
    cfg.max_length = len(item.input_ids) - (start + 1)
    with pytest.raises(RBAError) as caught:
        encode_cases([case], tokenizer, cfg, processor=processor)
    assert "视觉块" in caught.value.message
    # A cut that stops right before the delimiter is fine.
    cfg.max_length = len(item.input_ids) - start
    kept = encode_cases([case], tokenizer, cfg, processor=processor)[0]
    assert kept.truncated and kept.n_image_tokens == item.n_image_tokens


# --------------------------------------------------------------------------- P1-6 base alignment fail-closed


def test_base_alignment_unavailable_disables_grade_row_evidence(tmp_path: Path, monkeypatch):
    class Loaded:
        kind = "hf"
        base_path = str(tmp_path)  # no tokenizer files here
        tokenizer = None

    cfg = Config(goodcases=None)
    status = pipeline.check_base_grade_alignment(cfg, Loaded(), {"0": 1})
    assert status["status"] == "unavailable"

    from rba.cache import RunCache
    from rba.models.loader import load_models

    loaded = load_models(Config(device="cpu", dtype="float32", goodcases=None))
    monkeypatch.setattr(pipeline, "check_base_grade_alignment", lambda cfg, loaded, ids: {"status": "unavailable", "reason": "测试：基座分词器不可用", "checked": None})
    cache = RunCache(tmp_path / "c", "fp", resume=False)
    payload = pipeline._cached_delta(cache, Config(goodcases=None), loaded, loaded.grade_ids)
    rows = payload["lm_head_grades"]["rows"]
    assert rows and all(set(row) == {"ft_norm"} for row in rows.values())
    assert payload["lm_head_grades"]["missing_reason"].startswith("测试")
    assert payload["base_grade_alignment"]["status"] == "unavailable"


# --------------------------------------------------------------------------- P1-7 patches dir and quoting


def test_user_patch_files_survive_and_commands_quote_spaces(tmp_path: Path):
    out = tmp_path / "run with space"
    report = _analyze(out, "--fresh")
    mine = out / "patches" / "mine.json"
    mine.write_text('{"method": "head_scale", "layer": 0, "heads": [0], "scale": 0.0}', encoding="utf-8")
    ours = [p.name for p in (out / "patches").glob("*_h*.json")]
    assert ours
    _analyze(out, "--fresh")
    assert mine.exists()
    assert sorted(p.name for p in (out / "patches").glob("*_h*.json")) == sorted(ours)
    command = next(h["command"] for g in report["groups"] for h in g["hypotheses"] if h.get("command"))
    argv = shlex.split(command)
    assert argv[0] == "rba" and Path(argv[argv.index("--patch") + 1]).exists()
    main(argv[1:])
    assert (out / "eval" / "eval.json").exists()


# --------------------------------------------------------------------------- P1-8 empty / all-invalid


def test_lead_all_invalid_and_eval_without_rows(tmp_path: Path):
    text = pipeline._lead([], [{"id": "a"}], n_valid=0)
    assert "没有任何坏例落在等级词上" in text and "一致" not in text.split("。")[0]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "g", "query": "q", "doc": "d", "image": "missing/x.png", "label": 0}) + "\n", encoding="utf-8")
    patch = tmp_path / "p.json"
    patch.write_text('{"method": "head_scale", "layer": 0, "heads": [0], "scale": 0.0}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--ft", "tiny", "--base", "tiny", "--badcases", str(bad), "--patch", str(patch), "--allow-missing-images", "--output-dir", str(tmp_path / "e")])
    assert exc.value.code == 2
