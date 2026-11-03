"""rba export (baking patches into weights) and rba finetune on tiny models, CPU only."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")

from rba.config import Config  # noqa: E402
from rba.errors import RBAError  # noqa: E402
from rba.repair.bake import export_patches  # noqa: E402
from tests.vl_fixture import build_model, build_tokenizer  # noqa: E402

BAD = [("红烧肉怎么做", "五花焯水后加冰糖炒色再放酱油和八角炖四十分钟", "3"),
       ("北京今天天气", "北京市气象台预报今天阴转小雨", "3"),
       ("欧冠决赛", "红烧肉怎么做", "0"),
       ("红烧肉热量", "这道菜偏肥热量高", "2"),
       ("天气预报", "欧冠决赛在中进行", "0"),
       ("八角炖肉", "五花焯水后加冰糖", "1")]
GOOD = [("北京天气", "今天阴转小雨北风三", "3"),
        ("红烧肉", "这道菜偏肥", "1"),
        ("欧冠", "决赛在中进行", "2"),
        ("预报", "红烧肉怎么做", "0"),
        ("热量", "五花焯水", "1"),
        ("气象台", "北京市气象台", "3")]


def _write(path: Path, rows, prefix):
    with path.open("w", encoding="utf-8") as fh:
        for i, (q, d, label) in enumerate(rows):
            fh.write(json.dumps({"id": f"{prefix}{i}", "query": q, "doc": d, "label": int(label)}, ensure_ascii=False) + "\n")
    return str(path)


def _save_model(path: Path, tied: bool = False, seed: int = 0) -> str:
    tok = build_tokenizer()
    model = build_model(tok, seed=seed)
    if tied:
        model.config.tie_word_embeddings = True
        model.config.text_config.tie_word_embeddings = True
        model.lm_head.weight = model.get_input_embeddings().weight
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    return {
        "model": _save_model(root / "ft"),
        "tied": _save_model(root / "tied", tied=True),
        "bad": _write(root / "bad.jsonl", BAD, "b"),
        "good": _write(root / "good.jsonl", GOOD, "g"),
        "root": root,
    }


def _cfg(data, model_key="model") -> Config:
    cfg = Config()
    cfg.base_model = cfg.ft_model = data[model_key]
    cfg.badcases = data["bad"]
    cfg.goodcases = data["good"]
    cfg.device = "cpu"
    cfg.dtype = "float32"
    cfg.holdout_fraction = 0.0
    # The saved char-level tokenizer splits "-1"; the grade protocol here is 0..3.
    cfg.grades = ["0", "1", "2", "3"]
    cfg.analyses.delta_w = False
    return cfg


def _verified(**patch):
    return {"scope": "all_positions", "verified_on_sample": True, "untested": False, "status": "消融有效", **patch}


BAKEABLE = {
    "head_scale_full_attn": [_verified(method="head_scale", layer=0, heads=[1, 2], scale=0.25)],
    "head_scale_linear_attn": [_verified(method="head_scale", layer=1, heads=[0], scale=0.0)],
    "mlp_zero": [_verified(method="mlp_zero", layer=1)],
    "lm_head_blend": [_verified(method="lm_head_blend", grade="2", toward="3", alpha=0.6, scope="readout")],
    "stacked": [_verified(method="lm_head_blend", grade="0", toward="1", alpha=0.5, scope="readout"),
                _verified(method="head_scale", layer=0, heads=[3], scale=2.0),
                _verified(method="mlp_zero", layer=0),
                # Two head_scale patches on the same projection must compose, not overwrite.
                _verified(method="head_scale", layer=0, heads=[0, 3], scale=0.5)],
}


@pytest.mark.parametrize("name", list(BAKEABLE))
def test_export_matches_hooks_and_loads_without_rba(data, tmp_path, name):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    patches = [(f"p{i}.json", p) for i, p in enumerate(BAKEABLE[name])]
    out = tmp_path / "out"
    manifest = export_patches(_cfg(data), patches, out)
    eq = manifest["equivalence"]
    assert eq["identical"] and eq["n_pred_mismatch"] == 0 and eq["max_abs_logit_diff"] <= 1e-3
    saved = json.loads((out / "rba_export_manifest.json").read_text(encoding="utf-8"))
    assert [p["method"] for p in saved["patches"]] == [p["method"] for _, p in patches]
    assert saved["source_fingerprint"] and saved["exported_fingerprint"]
    assert saved["eval"]["bad_discovery"]["n"] == len(BAD) and saved["eval"]["good"]["n"] == len(GOOD)
    # Plain transformers, no rba: the weights really changed.
    plain = Qwen3_5ForConditionalGeneration.from_pretrained(out)
    orig = Qwen3_5ForConditionalGeneration.from_pretrained(data["model"])
    AutoTokenizer.from_pretrained(out)
    changed = [n for n, p in plain.state_dict().items() if not torch.equal(p, orig.state_dict()[n])]
    assert changed


def test_steering_refused_without_bias(data, tmp_path):
    patch = _verified(method="steering", layers=[1], from_grade="0", to_grade="3", alpha=2.0)
    with pytest.raises(RBAError, match="bias"):
        export_patches(_cfg(data), [("s.json", patch)], tmp_path / "out")


def test_steering_bakes_into_existing_bias_exactly():
    from rba.models.paths import resolve_paths
    from rba.repair.bake import bake_patch
    from rba.repair.interventions import apply_patch

    tok = build_tokenizer()
    model = build_model(tok).float()
    resolved = resolve_paths(model, {}, preset="qwen3_5")
    layer = resolved.layers(model)[1]
    old = layer.mlp.down_proj
    new = torch.nn.Linear(old.in_features, old.out_features, bias=True)
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.bias.normal_()
    layer.mlp.down_proj = new
    grade_ids = {g: tok.convert_tokens_to_ids(g) for g in ["0", "1", "2", "3"]}
    patch = {"method": "steering", "layers": [1], "from_grade": "0", "to_grade": "3", "alpha": 3.0}
    ids = torch.tensor([[20, 21, 22, 23, 24]])
    with torch.no_grad(), apply_patch(model, resolved, patch, grade_ids):
        hooked = model(input_ids=ids).logits
    bake_patch(model, resolved, patch, grade_ids)
    with torch.no_grad():
        baked = model(input_ids=ids).logits
    assert torch.allclose(hooked, baked, atol=1e-4)


@pytest.mark.parametrize("patch,match", [
    (_verified(method="ablate_tokens", token_ids=[3]), "不能烘焙"),
    (_verified(method="ablate_image"), "不能烘焙"),
    ({**_verified(method="head_scale", layer=0, heads=[0], scale=0.0), "untested": True, "verified_on_sample": False}, "没有按原样复核"),
    ({**_verified(method="head_scale", layer=0, heads=[0], scale=0.0), "status": "被复核否定"}, "不该导出"),
    ({**_verified(method="head_scale", layer=0, heads=[0], scale=0.0), "scope": "decision"}, "所有位置"),
])
def test_refusals(data, tmp_path, patch, match):
    with pytest.raises(RBAError, match=match):
        export_patches(_cfg(data), [("p.json", patch)], tmp_path / "out")
    assert not (tmp_path / "out" / "model.safetensors").exists()


def test_force_and_allow_all_positions(data, tmp_path):
    patch = {**_verified(method="head_scale", layer=0, heads=[0], scale=0.0), "scope": "decision", "untested": True}
    manifest = export_patches(_cfg(data), [("p.json", patch)], tmp_path / "out", force=True, allow_all_positions=True)
    assert manifest["equivalence"]["identical"]
    assert any("所有位置" in note for note in manifest["notes"])


def test_tied_lm_head_refused_then_untied(data, tmp_path):
    from transformers import Qwen3_5ForConditionalGeneration

    patch = _verified(method="lm_head_blend", grade="2", toward="3", alpha=0.5, scope="readout")
    with pytest.raises(RBAError, match="untie"):
        export_patches(_cfg(data, "tied"), [("p.json", patch)], tmp_path / "a")
    manifest = export_patches(_cfg(data, "tied"), [("p.json", patch)], tmp_path / "b", untie=True)
    assert manifest["untied_lm_head"] and manifest["equivalence"]["identical"]
    plain = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path / "b")
    assert plain.lm_head.weight.data_ptr() != plain.get_input_embeddings().weight.data_ptr()
    assert not torch.equal(plain.lm_head.weight, plain.get_input_embeddings().weight)


def test_existing_output_dir_not_overwritten(data, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(RBAError, match="overwrite"):
        export_patches(_cfg(data), [("p.json", BAKEABLE["mlp_zero"][0])], out)
    assert (out / "keep.txt").exists()


def test_cli_export_refusal_exit_code(data, tmp_path, capsys):
    from rba.cli import main

    path = tmp_path / "p.json"
    path.write_text(json.dumps(_verified(method="ablate_tokens", token_ids=[3])), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["export", "--ft", data["model"], "--badcases", data["bad"], "--goodcases", data["good"],
              "--device", "cpu", "--dtype", "float32", "--patch", str(path), "--out", str(tmp_path / "o")])
    assert exc.value.code == 2
    assert "不能烘焙" in capsys.readouterr().err


# --------------------------------------------------------------------------- finetune


def _grade_logits(model_dir, data):
    from rba.models.loader import load_models
    from rba.pipeline import _encode_all, _validate_and_load
    from rba.repair.common import score_items

    cfg = _cfg(data)
    cfg.ft_model = str(model_dir)
    bad, good, _ = _validate_and_load(cfg)
    loaded = load_models(cfg)
    b, g = _encode_all(cfg, loaded, bad, good, [])
    ids = [loaded.grade_ids[x] for x in cfg.grades]
    rows = score_items(loaded, b + g, ids, cfg.grades, "constrained", 4, "t")
    return torch.tensor([r["grade_logits"] for r in rows])


def test_finetune_lora_changes_logits_and_saves_merged(data, tmp_path):
    pytest.importorskip("peft")
    from transformers import Qwen3_5ForConditionalGeneration

    from rba.repair.finetune import FinetuneOptions, finetune

    opts = FinetuneOptions(out=str(tmp_path / "ft"), layers=[0, 1], steps=4, lr=5e-2, train_batch_size=4, lora_r=4)
    result = finetune(_cfg(data), opts)
    assert result["mode"] == "lora" and result["n_trainable"] > 0
    assert any("o_proj" in t for t in result["targets"]) and any("out_proj" in t for t in result["targets"])
    assert len(result["history"]) == 4
    assert result["bad_holdout"] is not None and result["good_holdout"] is not None
    model_dir = Path(result["model_dir"])
    plain = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir)
    assert not any("lora" in n for n in plain.state_dict())
    before = _grade_logits(data["model"], data)
    after = _grade_logits(model_dir, data)
    assert (before - after).abs().max() > 1e-3
    saved = json.loads((tmp_path / "ft" / "finetune.json").read_text(encoding="utf-8"))
    assert saved["train"]["n"] == result["n_train_bad"]


def test_finetune_full_on_localized_modules_only(data, tmp_path):
    from transformers import Qwen3_5ForConditionalGeneration

    from rba.repair.finetune import FinetuneOptions, finetune

    report = {"groups": [{"hypotheses": [{"status": "消融有效", "patch": {"method": "mlp_zero", "layer": 1}}]}]}
    opts = FinetuneOptions(out=str(tmp_path / "ft"), mode="full", steps=3, lr=1e-2, report=report)
    result = finetune(_cfg(data), opts)
    assert result["layers"] == [1]
    new = Qwen3_5ForConditionalGeneration.from_pretrained(result["model_dir"]).state_dict()
    old = Qwen3_5ForConditionalGeneration.from_pretrained(data["model"]).state_dict()
    changed = {n for n in new if not torch.equal(new[n], old[n])}
    assert changed and all("layers.1." in n for n in changed)


def test_finetune_builtin_tiny_cpu_path(tmp_path):
    from rba.repair.finetune import FinetuneOptions, finetune

    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(root / "example.yaml")
    cfg.device, cfg.dtype = "cpu", "float32"
    result = finetune(cfg, FinetuneOptions(out=str(tmp_path / "ft"), mode="full", layers=[1], steps=2, lr=1e-2))
    assert (Path(result["model_dir"]) / "model_state.pt").exists()


def test_finetune_without_peft_says_so(data, tmp_path, monkeypatch):
    from rba.repair.finetune import FinetuneOptions, finetune

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("no peft")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RBAError, match="peft"):
        finetune(_cfg(data), FinetuneOptions(out=str(tmp_path / "ft"), steps=1))
