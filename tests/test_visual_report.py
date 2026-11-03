"""Visual report (report.html) built from a demo run."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from rba.cli import main
from rba.report.visual import build_visual_report, write_eval_report


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("vis") / "demo"
    main(["demo", "--output-dir", str(out)])
    return out


class _Checker(HTMLParser):
    VOID = {"meta", "img", "br", "input", "hr", "link", "circle", "rect", "line", "polyline", "path"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()


def _parse(text: str) -> _Checker:
    checker = _Checker()
    checker.feed(text)
    checker.close()
    return checker


def test_demo_writes_visual_and_detail(demo_run: Path):
    assert (demo_run / "report.html").exists()
    assert (demo_run / "report_detail.html").exists()
    assert (demo_run / "report.md").exists()


def test_visual_html_is_valid_and_has_sections(demo_run: Path):
    text = (demo_run / "report.html").read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    checker = _parse(text)
    assert not checker.errors, checker.errors[:5]
    assert not checker.stack, checker.stack
    for heading in ("1. 总览", "2. 逐条坏例", "3. 组件定位", "4. 微调改了什么", "5. 补丁"):
        assert heading in text
    assert text.count("怎么看") >= 8
    assert "结论" in text and "建议" in text
    # confusion heatmap, lens line chart, DLA heatmap, token chips, text-vs-image bar
    assert text.count("<svg") >= 5 and "<polyline" in text
    assert "class='chip'" in text and "class='ti'" in text
    # filter/sort controls and one card per badcase
    report = json.loads((demo_run / "report.json").read_text(encoding="utf-8"))
    assert "id='fgroup'" in text and "id='fsort'" in text and "id='fsearch'" in text
    assert len(re.findall(r"class='card' data-id=", text)) == report["overview"]["n_bad"]
    # demo b001 has an image: thumbnail is inlined
    assert "data:image/jpeg;base64," in text


def test_visual_html_is_offline(demo_run: Path):
    text = (demo_run / "report.html").read_text(encoding="utf-8")
    assert not re.search(r"https?://", text)
    checker = _parse(text)
    for tag, attrs in checker.tags:
        for key in ("src", "href"):
            value = attrs.get(key)
            if value:
                assert value.startswith(("data:", "#")), (tag, key, value[:60])
    assert "<script src" not in text and "<link" not in text


def test_report_command_regenerates(demo_run: Path):
    (demo_run / "report.html").unlink()
    main(["report", str(demo_run)])
    assert "坏例诊断可视化报告" in (demo_run / "report.html").read_text(encoding="utf-8")
    build_visual_report(demo_run, "en")
    text = (demo_run / "report.html").read_text(encoding="utf-8")
    assert "Overview" in text and "lang='en'" in text
    build_visual_report(demo_run)


def test_report_includes_eval_results(demo_run: Path):
    patch = json.loads((demo_run / "patches" / "0_to_2_h2.json").read_text(encoding="utf-8"))
    result = {"patch": patch, "bad": {"n": 8, "fix_rate": 0.5, "before": {"accuracy": 0.1}, "after": {"accuracy": 0.6}},
              "good": {"n": 4, "fix_rate": 0.0, "regression_rate": 0.25}}
    (demo_run / "eval.json").write_text(json.dumps(result), encoding="utf-8")
    try:
        text = build_visual_report(demo_run).read_text(encoding="utf-8")
        assert "50.0%" in text and "25.0%" in text
    finally:
        (demo_run / "eval.json").unlink()
    page = write_eval_report(result, demo_run / "patches").read_text(encoding="utf-8")
    assert "50.0%" in page and not re.search(r"https?://", page)
    (demo_run / "patches" / "eval_report.html").unlink()


def test_report_missing_run_dir_is_friendly(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(["report", str(tmp_path)])
    assert exc.value.code == 2


def test_method_section_explains_signals_and_links_cases(demo_run: Path):
    text = (demo_run / "report.html").read_text(encoding="utf-8")
    assert "<section id='s7'>" in text and "7. 方法原理" in text and "href='#s7'" in text
    section = text.split("<section id='s7'>", 1)[1].split("</section>", 1)[0]
    assert section.count("<svg class='flow'") == 3
    for word in ("logit lens", "DLA", "token 归因", "干预复核", "ΔW / SVD", "W_U[pred] − W_U[gold]"):
        assert word in section
    # every in-page link in the method section resolves to an element in the page
    ids = set(re.findall(r"id='([^']+)'", text))
    targets = re.findall(r"href='#([^']+)'", section)
    assert targets and all(target in ids for target in targets)
    assert any(target.startswith("case-") for target in targets)
    assert not re.search(r"https?://", section)


def test_case_anchor_is_safe():
    from rba.report.visual import _case_anchor

    anchors = [_case_anchor(c) for c in ("b 01/x'y", "样本一", "样本二", "a/b", "a b", "a_b", 7)]
    assert len(set(anchors)) == len(anchors)
    assert all(re.fullmatch(r"case-[0-9a-f]+", a) for a in anchors)
    assert _case_anchor("a_b") == _case_anchor("a_b")


def test_method_section_reports_run_state_honestly():
    from rba.report.visual import _T, _method_section

    t = _T["zh"]
    empty_dw = {"delta_w": {"top": [], "lm_head_grades": {"rows": {}}}}
    groups = [{"key": "3→1", "attribution_reliable": False, "reconstruction_error": 0.3, "decomposition_gap": 0.01}]
    tok = {"c1": {"method": "integrated_gradients"}}
    verify = {"3→1": [{"control_type": "target", "mean_delta": -1.0}, {"control_type": "target", "skipped": "x"}]}
    html = _method_section(empty_dw, {}, groups, {}, tok, verify, t, True)
    assert "有 ΔW 数据" not in html and "无可用 ΔW 数据" in html
    assert "integrated_gradients" in html and "None" not in html
    assert "不可靠" in html and "跳过 1" in html
