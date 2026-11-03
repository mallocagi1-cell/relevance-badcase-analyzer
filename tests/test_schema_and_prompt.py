import json
from pathlib import Path

import pytest

from rba.config import Config
from rba.errors import RBAError
from rba.llm.prompt import build_group_prompt, estimate_tokens, fit_budget
from rba.schema import load_cases


def test_jsonl_and_csv_roundtrip(tmp_path: Path):
    cases = load_cases("examples/badcases.jsonl", ["-1", "0", "1", "2", "3"])
    assert cases[0].label == "3"
    assert cases[0].attributes["domain"] == "food"
    assert cases[0].image and cases[0].image.endswith("sample.ppm")
    csv_cases = load_cases("examples/badcases.csv", ["-1", "0", "1", "2", "3"])
    assert csv_cases[1].label == "0"
    assert csv_cases[0].attributes["domain"] == "food"


def test_bad_json_mentions_fix(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RBAError) as caught:
        load_cases(path, ["0", "1"])
    assert "建议" in str(caught.value)
    assert "JSON" in caught.value.message


def test_missing_query(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"doc": "只有文档", "label": "0"}) + "\n", encoding="utf-8")
    with pytest.raises(RBAError) as caught:
        load_cases(path, ["0", "1"])
    assert "query" in caught.value.message
    assert caught.value.hint


def test_unknown_config_key(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("base_model: tiny\nft_model: tiny\nbadcases: x.jsonl\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(RBAError) as caught:
        Config.load(path)
    assert "not_a_field" in caught.value.message
    assert "建议" in str(caught.value)


def test_prompt_budget_drops_examples_before_header():
    parts = [
        {"name": "header", "text": "任务。必须保留。", "droppable": False},
        {"name": "examples", "text": "样例" + "很长" * 500, "droppable": True},
        {"name": "ask", "text": "请给 JSON。", "droppable": False},
    ]
    text, dropped = fit_budget(parts, budget_tokens=40, chars_per_token=1)
    assert "必须保留" in text
    assert "请给 JSON" in text
    assert dropped
    assert estimate_tokens(text, 1) <= 80


def test_group_prompt_contains_task_and_json_schema():
    prompt = build_group_prompt(
        architecture={"class_name": "TinyQwenVL", "n_layers": 2, "hidden_size": 32, "vocab_size": 64, "n_parameters": 100, "layers": [], "paths": {}},
        grades=["0", "1"],
        grade_ids={"0": 2, "1": 3},
        group={"key": "1->0", "count": 3, "n_detail": 2, "mean_probs": {"0": 0.8, "1": 0.2}, "has_image_fraction": 0.0, "label": "1", "pred": "0", "examples": []},
        mech={"grade_mass": 0.9, "full_top_grade_rate": 1, "logit_diff_mean": 1.2, "decomposition_gap": 0.01, "flip_layer": 1, "image_token_fraction": 0, "attn_image_share": 0, "attn_layers_used": 1, "components": [], "lens": []},
        hypotheses=[],
        delta_top=[],
        template="查询：{query}\n文档：{doc}\n等级：",
        budget_tokens=4000,
        chars_per_token=1.6,
    )
    assert "任务。" in prompt["text"]
    assert "insufficient_evidence" in prompt["text"]
    assert "不要建议梯度训练" in prompt["text"] or "不许建议再训练" in prompt["text"]
