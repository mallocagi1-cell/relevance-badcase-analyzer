"""Visual, self-contained HTML report (report.html).

Built only from what `rba analyze` already wrote to the run directory:
report.json, cache/{infer_bad,mech,token_attr,verify}.json, patches/*.json and
any eval*.json. Nothing is recomputed and no model is loaded. All charts are
hand-written inline SVG; the small script for sorting/filtering is inline too,
so the file works offline and contains no external URLs.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

# --------------------------------------------------------------------------- text

_T = {
    "zh": {
        "title": "坏例诊断可视化报告",
        "s1": "1. 总览", "s2": "2. 逐条坏例", "s3": "3. 组件定位", "s4": "4. 微调改了什么", "s5": "5. 补丁",
        "s6": "6. 先看这里（文字结论）",
        "how": "怎么看", "concl": "结论",
        "n_bad": "坏例数", "n_wrong": "当前仍判错", "acc": "坏例准确率", "mae": "平均绝对误差", "n_good": "对照样本",
        "invalid": "非法输出",
        "conf_how": "行是金标、列是模型预测，格子越深条数越多；对角线外的深色格就是主要错误类型。",
        "card_how": "每张卡片是一条坏例：上面是文本和图，概率条看模型最终把概率给了谁，折线看每一层“如果此刻就输出”会选哪个等级。",
        "lens_how": "横轴是层（embed 表示还没进任何层），每条线是一个等级的概率；竖虚线标出预测等级开始稳定领先的层。",
        "tok_how": "红色 token 把答案推向错误预测，蓝色 token 把答案拉回金标，颜色越深作用越大。",
        "ti_how": "条形显示文本 token 与图像 token 在归因绝对值里各占多少。",
        "heat_how": "每个格子是一层的一个注意力头对“错误预测 − 金标”logit 差的平均直接贡献；红色越深越在推错，蓝色在往回拉。",
        "mlp_how": "每层 MLP 对同一个 logit 差的平均直接贡献；朝右的红条是在推错。",
        "topk_how": "把全部坏例的直接贡献平均后排序，最靠前的组件最可能参与了错误，但这仍是相关性，需要干预复核。",
        "dw_how": "条长是该张量 ‖W_ft − W_base‖/‖W_base‖ 的相对变化；只列 ΔW 最大的前几个张量。",
        "svd_how": "每个张量 ΔW 的前几个奇异值；第一根远高于其它说明改动集中在一个方向（低秩）。",
        "lm_how": "左边条是微调后每个等级行变化的大小 ‖Δ行‖，右边数字是新旧行的余弦；余弦明显小于 1 说明这个等级的读出方向被改了。",
        "patch_how": "抽样翻转率：干预后金标 logit 超过原预测的样本比例；修复率 / 回退率来自 rba eval（在验收集与对照集上重跑），没跑过 eval 就显示“未评估”。",
        "filter": "筛选", "all": "全部", "sort": "排序", "search": "搜索 id / 文本",
        "query": "查询", "doc": "文档", "gold": "金标", "pred": "预测", "group": "错误类型",
        "noimg": "无图", "img_missing": "有图但文件找不到",
        "text": "文本", "image": "图像", "flip": "稳定于",
        "no_mech": "这条样本没有逐层明细（只对每组抽样的样本做逐层分析）。",
        "no_tok": "这条样本没有 token 归因。",
        "layer": "层", "head": "头", "comp": "组件", "contrib": "平均贡献", "share": "占比",
        "tensor": "张量", "rel": "相对变化", "grade": "等级", "cos": "余弦",
        "patch": "补丁", "status": "状态", "flip_rate": "抽样翻转率", "fix": "修复率", "reg": "回退率",
        "not_eval": "未评估", "recommend": "建议",
        "detail": "完整文字版（含全部证据表与方法说明）见同目录 report_detail.html / report.md。",
        "lead_hdr": "先看这里",
        "s7": "7. 方法原理",
    },
    "en": {
        "title": "Badcase diagnosis — visual report",
        "s1": "1. Overview", "s2": "2. Badcases", "s3": "3. Component localisation", "s4": "4. What fine-tuning changed",
        "s5": "5. Patches", "s6": "6. Text conclusions",
        "how": "How to read", "concl": "Conclusion",
        "n_bad": "Badcases", "n_wrong": "Still wrong", "acc": "Accuracy on badcases", "mae": "MAE", "n_good": "Controls",
        "invalid": "Invalid outputs",
        "conf_how": "Rows are gold, columns are predictions; darker off-diagonal cells are the dominant error types.",
        "card_how": "One card per badcase: text and image, final grade probabilities, and per-layer logit lens.",
        "lens_how": "x = layer, one line per grade probability; the dashed line marks where the prediction becomes stable.",
        "tok_how": "Red tokens push towards the wrong prediction, blue tokens towards gold; darker = stronger.",
        "ti_how": "Share of absolute attribution on text tokens vs image tokens.",
        "heat_how": "Mean direct contribution of each head to logit(pred) − logit(gold); red pushes the error.",
        "mlp_how": "Mean direct contribution of each MLP; red bars push the error.",
        "topk_how": "Components ranked by mean contribution across badcases; correlational until verified by intervention.",
        "dw_how": "Relative change ‖ΔW‖/‖W_base‖ for the top changed tensors.",
        "svd_how": "Top singular values of ΔW; one dominant value means a low-rank change.",
        "lm_how": "‖Δrow‖ per grade row of lm_head and cosine between new and old row.",
        "patch_how": "Sample flip rate: share of sampled cases where gold overtakes the prediction after the intervention; fix / regression rates come from rba eval.",
        "filter": "Filter", "all": "All", "sort": "Sort", "search": "Search id / text",
        "query": "Query", "doc": "Doc", "gold": "Gold", "pred": "Pred", "group": "Error type",
        "noimg": "no image", "img_missing": "image file not found",
        "text": "text", "image": "image", "flip": "stable from",
        "no_mech": "No per-layer details for this case (only sampled cases per group).",
        "no_tok": "No token attribution for this case.",
        "layer": "Layer", "head": "Head", "comp": "Component", "contrib": "Mean contrib.", "share": "Share",
        "tensor": "Tensor", "rel": "Rel. change", "grade": "Grade", "cos": "Cosine",
        "patch": "Patch", "status": "Status", "flip_rate": "Sample flip rate", "fix": "Fix rate", "reg": "Regression rate",
        "not_eval": "not evaluated", "recommend": "Recommendation",
        "detail": "Full text version: report_detail.html / report.md in the same folder.",
        "lead_hdr": "Read this first",
        "s7": "7. How each signal is computed",
    },
}

PALETTE = ["#7c3aed", "#6b7280", "#0ea5e9", "#16a34a", "#ea580c", "#db2777", "#ca8a04"]


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _load(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- svg helpers


def _diverging(value: float, vmax: float) -> str:
    """Red for positive (pushes the error), blue for negative."""
    t = 0.0 if vmax <= 0 else max(-1.0, min(1.0, value / vmax))
    if t >= 0:
        r, g, b = 255, int(255 - 200 * t), int(255 - 200 * t)
    else:
        r, g, b = int(255 + 200 * t), int(255 + 140 * t), 255
    return f"rgb({r},{g},{b})"


def _seq(value: float, vmax: float) -> str:
    t = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    return f"rgb({int(255 - 180 * t)},{int(255 - 150 * t)},{int(255 - 60 * t)})"


def svg_heatmap(rows: list[str], cols: list[str], values: list[list[float | None]], *, diverging: bool,
                row_title: str = "", col_title: str = "", cell: int = 34, fmt=lambda v: f"{v:.2g}") -> str:
    flat = [abs(v) for r in values for v in r if v is not None]
    vmax = max(flat) if flat else 1.0
    left, top = 70, 34
    w = left + cell * max(1, len(cols)) + 10
    h = top + cell * max(1, len(rows)) + 26
    out = [f"<svg class='chart' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img'>"]
    for j, c in enumerate(cols):
        out.append(f"<text x='{left + j * cell + cell / 2}' y='{top - 8}' class='ax' text-anchor='middle'>{_e(c)}</text>")
    for i, r in enumerate(rows):
        out.append(f"<text x='{left - 6}' y='{top + i * cell + cell / 2 + 4}' class='ax' text-anchor='end'>{_e(r)}</text>")
        for j, _ in enumerate(cols):
            v = values[i][j] if j < len(values[i]) else None
            x, y = left + j * cell, top + i * cell
            if v is None:
                out.append(f"<rect x='{x}' y='{y}' width='{cell - 2}' height='{cell - 2}' fill='#eee'/>")
                continue
            color = _diverging(v, vmax) if diverging else _seq(v, vmax)
            out.append(f"<rect x='{x}' y='{y}' width='{cell - 2}' height='{cell - 2}' fill='{color}'><title>{_e(r)} / {_e(c)}: {v:.4g}</title></rect>")
            if cell >= 30:
                dark = (abs(v) / vmax if vmax else 0) > 0.6
                out.append(f"<text x='{x + cell / 2 - 1}' y='{y + cell / 2 + 4}' class='cv{' inv' if dark else ''}' text-anchor='middle'>{_e(fmt(v))}</text>")
    if row_title:
        out.append(f"<text x='4' y='14' class='ax'>{_e(row_title)} ↓ / {_e(col_title)} →</text>")
    out.append("</svg>")
    return "".join(out)


def svg_hbars(items: list[tuple[str, float]], *, signed: bool, width: int = 520, fmt=lambda v: f"{v:.3g}",
              vmax: float | None = None) -> str:
    if not items:
        return "<p class='muted'>—</p>"
    label_w, bar_h = 190, 20
    vmax = vmax or max((abs(v) for _, v in items), default=1.0) or 1.0
    h = bar_h * len(items) + 8
    plot = width - label_w - 60
    zero = label_w + (plot / 2 if signed else 0)
    scale = (plot / 2 if signed else plot) / vmax
    out = [f"<svg class='chart' viewBox='0 0 {width} {h}' width='{width}' height='{h}'>"]
    if signed:
        out.append(f"<line x1='{zero}' x2='{zero}' y1='0' y2='{h}' stroke='#999'/>")
    for i, (label, v) in enumerate(items):
        y = 4 + i * bar_h
        length = abs(v) * scale
        x = zero if v >= 0 or not signed else zero - length
        color = ("#dc2626" if v >= 0 else "#2563eb") if signed else "#0d9488"
        out.append(f"<text x='{label_w - 6}' y='{y + 14}' class='ax' text-anchor='end'>{_e(label[:28])}</text>")
        out.append(f"<rect x='{x:.1f}' y='{y + 2}' width='{max(length, 1):.1f}' height='{bar_h - 6}' fill='{color}'><title>{_e(label)}: {v:.4g}</title></rect>")
        tx = (x + length + 4) if (v >= 0 or not signed) else (x - 4)
        anchor = "start" if (v >= 0 or not signed) else "end"
        out.append(f"<text x='{tx:.1f}' y='{y + 14}' class='ax' text-anchor='{anchor}'>{_e(fmt(v))}</text>")
    out.append("</svg>")
    return "".join(out)


def svg_lines(xlabels: list[str], series: list[tuple[str, list[float], str]], *, mark: int | None = None,
              width: int = 420, height: int = 190, mark_label: str = "") -> str:
    left, right, top, bottom = 34, 10, 10, 26
    pw, ph = width - left - right, height - top - bottom
    n = max(1, len(xlabels) - 1)
    out = [f"<svg class='chart' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>"]
    for k in range(0, 5):
        y = top + ph * (1 - k / 4)
        out.append(f"<line x1='{left}' x2='{left + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='#eee'/>")
        out.append(f"<text x='{left - 4}' y='{y + 4:.1f}' class='ax' text-anchor='end'>{k / 4:.2g}</text>")
    for i, lab in enumerate(xlabels):
        x = left + pw * i / n
        out.append(f"<text x='{x:.1f}' y='{height - 8}' class='ax' text-anchor='middle'>{_e(lab)}</text>")
    if mark is not None and 0 <= mark < len(xlabels):
        x = left + pw * mark / n
        out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{top}' y2='{top + ph}' stroke='#dc2626' stroke-dasharray='4 3'/>")
        out.append(f"<text x='{x + 3:.1f}' y='{top + 10}' class='ax' fill='#dc2626'>{_e(mark_label)}</text>")
    for name, ys, color in series:
        pts = " ".join(f"{left + pw * i / n:.1f},{top + ph * (1 - max(0.0, min(1.0, y))):.1f}" for i, y in enumerate(ys))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'><title>{_e(name)}</title></polyline>")
        for i, y in enumerate(ys):
            out.append(f"<circle cx='{left + pw * i / n:.1f}' cy='{top + ph * (1 - max(0.0, min(1.0, y))):.1f}' r='2.5' fill='{color}'><title>{_e(name)} @ {_e(xlabels[i])}: {y:.3f}</title></circle>")
    out.append("</svg>")
    return "".join(out)


def _bar_html(frac: float | None, color: str, label: str = "") -> str:
    if frac is None:
        return f"<span class='muted'>{_e(label)}</span>"
    w = max(0.0, min(1.0, frac)) * 100
    return (f"<div class='pbar'><div style='width:{w:.1f}%;background:{color}'></div>"
            f"<span>{_e(label or _pct(frac))}</span></div>")


# --------------------------------------------------------------------------- data helpers


def _flip_layer(lens: list[dict], pred_index: int) -> int | None:
    """First lens index from which the predicted grade stays the argmax."""
    stable = None
    for i in range(len(lens) - 1, -1, -1):
        probs = lens[i].get("probs") or []
        if probs and max(range(len(probs)), key=probs.__getitem__) == pred_index:
            stable = i
        else:
            break
    return stable


def _thumbnail(image: str | None, base_dir: Path | None) -> tuple[str | None, bool]:
    """Returns (data-uri, existed)."""
    if not image:
        return None, False
    path = Path(image)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        return None, False
    try:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((180, 180))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), True
    except Exception:  # noqa: BLE001 - a broken image must not break the report
        return None, True


def _read_cases(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    rows: dict[str, dict] = {}
    try:
        if p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[str(row.get("id"))] = row
        elif p.suffix == ".csv":
            import csv

            with p.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows[str(row.get("id"))] = row
    except (OSError, ValueError):
        return {}
    return rows


def _highlight(text: str, tokens: list[dict], vmax: float) -> str:
    """Escape text and colour substrings that match attributed tokens."""
    marks: list[tuple[int, int, float]] = []
    for tok in tokens:
        disp = (tok.get("display") or tok.get("token") or "").strip()
        if not disp or disp.startswith("#") or disp.startswith("<"):
            continue
        start = 0
        while True:
            i = text.find(disp, start)
            if i < 0:
                break
            if not any(a < i + len(disp) and i < b for a, b, _ in marks):
                marks.append((i, i + len(disp), tok.get("score") or 0.0))
            start = i + len(disp)
    marks.sort()
    out, pos = [], 0
    for a, b, s in marks:
        out.append(_e(text[pos:a]))
        out.append(f"<mark style='background:{_diverging(s, vmax)}' title='{s:+.3g}'>{_e(text[a:b])}</mark>")
        pos = b
    out.append(_e(text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- main


def build_visual_report(run_dir: Path, language: str | None = None, config: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    report = _load(run_dir / "report.json")
    if not isinstance(report, dict):
        from rba.errors import RBAError

        raise RBAError(f"{run_dir} 里没有可读的 report.json。", "先跑 rba analyze（或 rba demo），再对它的输出目录运行 rba report。")
    # analyze passes its live config (config.used.json is written after the report).
    config = config or _load(run_dir / "config.used.json", {}) or {}
    lang = language or config.get("language") or "zh"
    if lang not in _T:
        lang = "zh"
    html_text = render_visual(report, run_dir, config, lang)
    path = run_dir / "report.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def render_visual(report: dict, run_dir: Path, config: dict, lang: str = "zh") -> str:
    t = _T[lang]
    zh = lang == "zh"
    cache = run_dir / "cache"
    infer = _load(cache / "infer_bad.json", []) or []
    mech = {m["id"]: m for m in (_load(cache / "mech.json", []) or []) if isinstance(m, dict) and "id" in m}
    tok = {m["id"]: m for m in (_load(cache / "token_attr.json", []) or []) if isinstance(m, dict) and "id" in m}
    verify = _load(cache / "verify.json", {}) or {}
    grades = [str(g) for g in report.get("grades") or config.get("grades") or []]
    gidx = {g: i for i, g in enumerate(grades)}
    raw_rows = _read_cases(config.get("badcases"))
    base_dir = Path(config["badcases"]).parent if config.get("badcases") else None
    ov = report.get("overview") or {}
    groups = report.get("groups") or []

    sections: list[str] = []

    # ---------------------------------------------------------------- 1 overview
    conf = report.get("confusion") or {}
    cm = [[conf.get(g, {}).get(p) if conf else None for p in grades] for g in grades]
    errors = sorted(((v, g, p) for g in grades for p in grades if g != p for v in [conf.get(g, {}).get(p, 0) or 0] if v),
                    reverse=True)
    n_wrong = ov.get("n_bad_wrong", 0)
    if errors:
        v, g, p = errors[0]
        top_err = (f"最常见的错误是金标 {g} 被判成 {p}（{v} 条，占判错的 {v / max(1, n_wrong):.0%}）。" if zh else
                   f"Most common error: gold {g} predicted as {p} ({v} cases, {v / max(1, n_wrong):.0%} of errors).")
    else:
        top_err = "坏例里没有判错的样本。" if zh else "No wrong predictions among badcases."
    over_pred = {}
    under = 0
    for v, g, p in errors:
        try:
            if float(p) > float(g):
                over_pred[p] = over_pred.get(p, 0) + v
            else:
                under += v
        except ValueError:
            pass
    n_over = sum(over_pred.values())
    if zh:
        dirn = f"其中 {n_over} 条判高了、{under} 条判低了。" if errors else ""
    else:
        dirn = f"{n_over} over-graded, {under} under-graded." if errors else ""
    mass_low = [g["key"] for g in groups if (g.get("grade_mass") is not None and g["grade_mass"] < 0.5)]
    mass_note = ""
    if mass_low:
        mass_note = (f"注意：{', '.join(mass_low)} 组下一个 token 的概率大多不在等级词上，先核对模板/解码协议再看机制。" if zh
                     else f"Note: groups {', '.join(mass_low)} put little mass on grade tokens; check the template first.")
    top_conc = " ".join(x for x in [top_err, dirn, mass_note] if x)
    kpis = [
        (t["n_bad"], ov.get("n_bad")), (t["n_wrong"], n_wrong), (t["acc"], _pct(ov.get("bad_accuracy"))),
        (t["mae"], f"{ov['bad_mae']:.2f}" if ov.get("bad_mae") is not None else "—"), (t["n_good"], ov.get("n_good")),
        (t["invalid"], ov.get("n_invalid_bad", 0)),
    ]
    banner = f"<div class='banner'>{_e(report['banner'])}</div>" if report.get("banner") else ""
    sections.append(
        f"<section id='s1'><h2>{t['s1']}</h2>{banner}<div class='kpis'>"
        + "".join(f"<div class='kpi'><b>{_e(v)}</b><span>{_e(k)}</span></div>" for k, v in kpis)
        + "</div><div class='row'><div>"
        + svg_heatmap(grades, grades, cm, diverging=False, row_title=t["gold"], col_title=t["pred"], cell=44, fmt=lambda v: f"{int(v)}")
        + f"</div><div class='grow'>{_how(t, t['conf_how'])}{_concl(t, top_conc)}"
        + _group_table(groups, t)
        + "</div></div></section>"
    )

    # ---------------------------------------------------------------- 2 cards
    cards = []
    group_options = sorted({f"{r.get('label')}->{r.get('pred')}" for r in infer if str(r.get("label")) != str(r.get("pred"))})
    for row in infer:
        cid = str(row.get("id"))
        gold, pred = str(row.get("label")), str(row.get("pred"))
        wrong = gold != pred
        gkey = f"{gold}->{pred}" if wrong else "correct"
        probs = row.get("probs") or {}
        p_pred = probs.get(pred)
        raw = raw_rows.get(cid, {})
        thumb, existed = _thumbnail(raw.get("image") or raw.get("image_path"), base_dir)
        if thumb:
            img_html = f"<img class='thumb' alt='{_e(cid)}' src='{thumb}'/>"
        elif row.get("has_image") or existed:
            img_html = f"<div class='thumb none'>{t['img_missing']}</div>"
        else:
            img_html = f"<div class='thumb none'>{t['noimg']}</div>"
        ta = tok.get(cid)
        top_tokens = (ta or {}).get("top_text") or []
        tvmax = max((abs(x.get("score") or 0) for x in top_tokens), default=0) or 1.0
        q_html = _highlight(str(row.get("query") or ""), top_tokens, tvmax)
        d_html = _highlight(str(row.get("doc") or ""), top_tokens, tvmax)
        prob_bars = "".join(
            f"<div class='gp'><span class='gl'>{_e(g)}{' ★' if g == gold else ''}</span>"
            + _bar_html(probs.get(g), "#dc2626" if g == pred and wrong else ("#16a34a" if g == gold else "#94a3b8"),
                        f"{(probs.get(g) or 0):.3f}")
            + "</div>" for g in grades)
        m = mech.get(cid)
        if m and m.get("lens"):
            lens = m["lens"]
            xl = [str(x.get("layer")) for x in lens]
            series = [(f"{t['grade']} {g}", [(x.get("probs") or [0] * len(grades))[i] for x in lens], PALETTE[i % len(PALETTE)])
                      for i, g in enumerate(grades)]
            fl = _flip_layer(lens, gidx.get(pred, -1))
            lens_svg = svg_lines(xl, series, mark=fl, mark_label=f"{t['flip']} {xl[fl]}" if fl is not None else "")
            legend = "".join(f"<span class='lg'><i style='background:{PALETTE[i % len(PALETTE)]}'></i>{_e(g)}</span>" for i, g in enumerate(grades))
            if fl is not None:
                lens_c = (f"预测等级 {pred} 从 {xl[fl]} 开始一直领先" + ("，在最后一层才翻转——错误主要在末端读出。" if fl == len(lens) - 1 else "。")) if zh else \
                    f"Prediction {pred} leads from {xl[fl]} onward."
            else:
                lens_c = ""
            lens_html = f"<div class='mini'><b>Logit lens</b> {legend}{lens_svg}<p class='hint'>{_e(t['lens_how'])} {_e(lens_c)}</p></div>"
        else:
            fl = None
            lens_html = f"<div class='mini muted'>{_e(t['no_mech'])}</div>"
        if ta:
            ts, isc = ta.get("text_abs_share") or 0.0, ta.get("image_abs_share") or 0.0
            ti = (f"<div class='ti'><div style='width:{ts * 100:.1f}%' class='tt'>{t['text']} {ts:.0%}</div>"
                  f"<div style='width:{isc * 100:.1f}%' class='ii'>{t['image']} {isc:.0%}</div></div>")
            chips = "".join(
                f"<span class='chip' style='background:{_diverging(x.get('score') or 0, tvmax)}' title='{_e(x.get('context'))}'>"
                f"{_e(x.get('display') or x.get('token'))} <small>{(x.get('score') or 0):+.2g}</small></span>" for x in top_tokens)
            tok_html = (f"<div class='mini'><b>Token</b> <span class='hint'>{_e(t['tok_how'])}</span><div>{chips}</div>"
                        f"<b>{t['text']} vs {t['image']}</b> <span class='hint'>{_e(t['ti_how'])}</span>{ti}</div>")
        else:
            ts = None
            tok_html = f"<div class='mini muted'>{_e(t['no_tok'])}</div>"
        search = _e(f"{cid} {row.get('query', '')} {row.get('doc', '')}".lower())
        cards.append(
            f"<div class='card' data-id='{_e(cid)}' data-group='{_e(gkey)}' data-wrong='{int(wrong)}' "
            f"data-ppred='{(p_pred or 0):.6f}' data-flip='{fl if fl is not None else 999}' data-err='{_err(gold, pred)}' data-search='{search}' id='{_case_anchor(cid)}'>"
            f"<div class='ch'><b>{_e(cid)}</b> <span class='tag {'bad' if wrong else 'ok'}'>{t['gold']} {_e(gold)} → {t['pred']} {_e(pred)}</span></div>"
            f"<div class='row'>{img_html}<div class='grow'><p><span class='k'>{t['query']}</span> {q_html}</p>"
            f"<p><span class='k'>{t['doc']}</span> {d_html}</p></div><div class='probs'>{prob_bars}</div></div>"
            f"<div class='row'>{lens_html}{tok_html}</div></div>"
        )
    controls = (
        f"<div class='controls'>{t['filter']} <select id='fgroup'><option value=''>{t['all']}</option>"
        + "".join(f"<option>{_e(g)}</option>" for g in group_options)
        + "<option value='correct'>correct</option></select> "
        f"{t['sort']} <select id='fsort'><option value='id'>id</option><option value='ppred'>{t['pred']} prob ↓</option>"
        f"<option value='err'>|{t['gold']}−{t['pred']}| ↓</option><option value='flip'>{t['flip']} ↑</option></select> "
        f"<input id='fsearch' placeholder='{_e(t['search'])}'/> <span id='fcount'></span></div>"
    )
    sections.append(f"<section id='s2'><h2>{t['s2']}</h2>{_how(t, t['card_how'])}{controls}<div id='cards'>{''.join(cards)}</div></section>")

    # ---------------------------------------------------------------- 3 components
    mlist = list(mech.values())
    comp_html = ""
    if mlist:
        n_layers = max(len(m.get("head_attr") or []) for m in mlist)
        n_heads = max((len(h) for m in mlist for h in (m.get("head_attr") or []) if h), default=0)
        heat = [[None] * n_heads for _ in range(n_layers)]
        for li in range(n_layers):
            for hi in range(n_heads):
                vals = [m["head_attr"][li][hi] for m in mlist
                        if li < len(m.get("head_attr") or []) and m["head_attr"][li] and hi < len(m["head_attr"][li])]
                heat[li][hi] = sum(vals) / len(vals) if vals else None
        mlp = []
        for li in range(n_layers):
            vals = [m["mlp_attr"][li] for m in mlist if li < len(m.get("mlp_attr") or []) and m["mlp_attr"][li] is not None]
            mlp.append((f"L{li} MLP", sum(vals) / len(vals) if vals else 0.0))
        comps = [(f"L{li}.h{hi}", v) for li in range(n_layers) for hi in range(n_heads) if (v := heat[li][hi]) is not None]
        comps += mlp
        ranked = sorted(comps, key=lambda kv: kv[1], reverse=True)[:10]
        best = ranked[0] if ranked else None
        conc = ""
        if best:
            total = sum(v for _, v in comps if v > 0) or 1.0
            conc = (f"在 {len(mlist)} 条有逐层明细的坏例上，{best[0]} 的平均推错贡献最大（{best[1]:+.3g}，占全部正向贡献的 {best[1] / total:.0%}）。"
                    "这只是相关性，是否真的是原因看第 5 节的干预结果。" if zh else
                    f"Across {len(mlist)} detailed badcases, {best[0]} contributes most to the error ({best[1]:+.3g}).")
        comp_html = (
            f"<div class='row'><div><h3>{t['layer']} × {t['head']} DLA</h3>"
            + svg_heatmap([f"L{i}" for i in range(n_layers)], [f"h{j}" for j in range(n_heads)], heat, diverging=True,
                          row_title=t["layer"], col_title=t["head"], cell=36 if n_heads <= 16 else 14)
            + f"{_how(t, t['heat_how'])}</div><div class='grow'><h3>MLP DLA</h3>"
            + svg_hbars(mlp, signed=True, width=460) + _how(t, t["mlp_how"])
            + f"</div></div><h3>Top-{len(ranked)}</h3>" + svg_hbars(ranked, signed=True, width=560)
            + _how(t, t["topk_how"]) + _concl(t, conc)
        )
    else:
        comp_html = "<p class='muted'>—</p>"
    sections.append(f"<section id='s3'><h2>{t['s3']}</h2>{comp_html}</section>")

    # ---------------------------------------------------------------- 4 delta W
    dw = report.get("delta_w") or {}
    top = dw.get("top") or []
    dw_items = [(_short(x.get("name", "")), float(x.get("rel") or 0.0)) for x in top]
    per_layer: dict[str, float] = {}
    for x in top:
        mm = re.search(r"layers\.(\d+)\.", x.get("name", ""))
        key = f"L{mm.group(1)}" if mm else _short(x.get("name", "")).split(".")[0]
        per_layer[key] = per_layer.get(key, 0.0) + float(x.get("delta_fro") or 0.0)
    svd_items = []
    for x in top:
        for d in ((x.get("svd") or {}).get("directions") or []):
            if d.get("singular_value") is not None:
                svd_items.append((f"{_short(x.get('name', ''))} σ{d.get('which', 0) + 1}", float(d["singular_value"])))
    if any(v > 0 for _, v in dw_items):
        dw_items = [kv for kv in dw_items if kv[1] > 0]
    svd_items = [kv for kv in svd_items if kv[1] > 1e-6] or svd_items
    lmg = (dw.get("lm_head_grades") or {}).get("rows") or {}
    lm_items = [(f"{t['grade']} {g}", float((lmg.get(g) or {}).get("delta_norm") or 0.0)) for g in grades if g in lmg]
    lm_cos = "".join(f"<tr><td>{_e(g)}</td><td>{(lmg[g].get('cosine_to_base') or 0):.3f}</td><td>{(lmg[g].get('delta_norm') or 0):.3g}</td></tr>"
                     for g in grades if g in lmg)
    changed = [x for x in top if (x.get("rel") or 0) > 0]
    if changed:
        biggest = changed[0]
        changed_grade = [g for g in grades if g in lmg and (lmg[g].get("delta_norm") or 0) > 0]
        dw_conc = (f"共 {dw.get('n_parameters_changed', '?')} 个参数被改动（{_pct(dw.get('changed_fraction'))}）。改动最大的是 {_short(biggest['name'])}"
                   f"（相对变化 {biggest.get('rel', 0):.3g}）。" + (f"lm_head 里等级 {', '.join(changed_grade)} 的读出行被改动。" if changed_grade else "lm_head 等级行没有变化。")
                   if zh else f"{dw.get('n_parameters_changed', '?')} parameters changed; largest: {_short(biggest['name'])}.")
    else:
        dw_conc = "微调前后权重没有可见差异（或没有提供 base 模型）。" if zh else "No visible weight change."
    sections.append(
        f"<section id='s4'><h2>{t['s4']}</h2><div class='row'><div><h3>ΔW / W ({t['tensor']})</h3>"
        + svg_hbars(dw_items, signed=False, width=520) + _how(t, t["dw_how"])
        + f"<h3>‖ΔW‖ per {t['layer']}</h3>" + svg_hbars(sorted(per_layer.items()), signed=False, width=520)
        + f"</div><div class='grow'><h3>SVD</h3>" + svg_hbars(svd_items[:12], signed=False, width=480) + _how(t, t["svd_how"])
        + f"<h3>lm_head {t['grade']}</h3>" + svg_hbars(lm_items, signed=False, width=420)
        + f"<table><tr><th>{t['grade']}</th><th>{t['cos']}</th><th>‖Δ‖</th></tr>{lm_cos}</table>" + _how(t, t["lm_how"])
        + "</div></div>" + _concl(t, dw_conc) + "</section>"
    )

    # ---------------------------------------------------------------- 5 patches
    sections.append(f"<section id='s5'><h2>{t['s5']}</h2>{_patch_section(run_dir, groups, verify, t, zh)}</section>")

    # ---------------------------------------------------------------- 6 lead + hypotheses
    hyp = []
    for g in groups:
        items = "".join(f"<li><span class='tag'>{_e(h.get('status'))}</span> {_e(h.get('title'))}<div class='muted small'>{_e(h.get('evidence'))}</div></li>"
                        for h in (g.get("hypotheses") or [])[:4])
        hyp.append(f"<h3>{_e(g.get('key'))} · {_e(g.get('count'))}</h3><ol>{items}</ol>")
    sections.append(f"<section id='s6'><h2>{t['s6']}</h2><h3>{t['lead_hdr']}</h3><p>{_e(report.get('lead'))}</p>{''.join(hyp)}"
                    f"<p class='muted'>{_e(t['detail'])}</p></section>")

    # ---------------------------------------------------------------- 7 method
    sections.append(_method_section(report, config, groups, mech, tok, verify, t, zh))

    nav = "".join(f"<a href='#s{i}'>{_e(t['s' + str(i)])}</a>" for i in range(1, 8))
    return (
        f"<!DOCTYPE html><html lang='{'zh' if zh else 'en'}'><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>{_e(t['title'])}</title><style>{_CSS}</style></head><body>"
        f"<nav>{nav}</nav><main><h1>{_e(t['title'])}</h1><p class='muted'>{_e(report.get('output_dir'))}</p>"
        + "".join(sections) + f"</main><script>{_JS}</script></body></html>"
    )


def _err(gold: str, pred: str) -> str:
    try:
        return f"{abs(float(gold) - float(pred)):g}"
    except ValueError:
        return "0"


def _short(name: str) -> str:
    return name.replace("model.language_model.", "").replace(".weight", "")


def _how(t: dict, text: str) -> str:
    return f"<p class='how'><b>{_e(t['how'])}：</b>{_e(text)}</p>"


def _concl(t: dict, text: str) -> str:
    return f"<p class='concl'><b>{_e(t['concl'])}：</b>{_e(text)}</p>" if text else ""


def _group_table(groups: list[dict], t: dict) -> str:
    if not groups:
        return ""
    rows = "".join(
        f"<tr><td>{_e(g.get('key'))}</td><td>{_e(g.get('count'))}</td><td>{_e(g.get('flip_layer'))}</td>"
        f"<td>{_bar_html((g.get('token_attr') or {}).get('text_abs_share'), '#0d9488')}</td></tr>" for g in groups)
    return (f"<table><tr><th>{t['group']}</th><th>n</th><th>{t['flip']} ({t['layer']})</th><th>{t['text']} share</th></tr>{rows}</table>")


def _same_patch(a: dict, b: dict) -> bool:
    keys = ("method", "layer", "heads", "scale", "grade", "neurons")
    return all(a.get(k) == b.get(k) for k in keys if k in a or k in b)


def _patch_section(run_dir: Path, groups: list[dict], verify: dict, t: dict, zh: bool) -> str:
    evals = []
    for p in sorted(run_dir.glob("eval*.json")) + sorted((run_dir / "evals").glob("*.json")) + sorted((run_dir / "eval").glob("eval*.json")):
        e = _load(p)
        if isinstance(e, dict) and "patch" in e:
            evals.append(e)
    rows, scored = [], []
    for g in groups:
        for h in g.get("hypotheses") or []:
            if not h.get("patch_file"):
                continue
            patch = _load(run_dir / h["patch_file"], {}) or {}
            ver = None
            for v in verify.get(g.get("key"), []) or []:
                if v.get("control_type") == "target" and isinstance(v.get("patch"), dict) and _same_patch(v["patch"], patch):
                    ver = v
                    break
            per = (ver or {}).get("per_case") or []
            flip = (sum(1 for c in per if (c.get("after_diff") or 0) < 0) / len(per)) if per else None
            ev = next((e for e in evals if _same_patch(e.get("patch") or {}, patch)), None)
            fix = ((ev or {}).get("bad") or {}).get("fix_rate")
            reg = ((ev or {}).get("good") or {}).get("regression_rate")
            desc = f"{patch.get('method')} L{patch.get('layer')} {patch.get('heads') or ''} ×{patch.get('scale')}"
            scored.append((h.get("status") in ("消融有效", "已验证"), fix if fix is not None else -1, flip or 0, g.get("key"), desc, h, reg))
            rows.append(
                f"<tr><td>{_e(g.get('key'))}</td><td><code>{_e(desc)}</code><div class='muted small'>{_e(h.get('patch_file'))}</div></td>"
                f"<td><span class='tag'>{_e(h.get('status'))}</span></td>"
                f"<td>{_bar_html(flip, '#7c3aed', (_pct(flip) + f' ({len(per)})') if flip is not None else t['not_eval'])}</td>"
                f"<td>{_bar_html(fix, '#16a34a', '' if fix is not None else t['not_eval'])}</td>"
                f"<td>{_bar_html(reg, '#dc2626', '' if reg is not None else t['not_eval'])}</td></tr>")
    eval_html = ""
    for e in evals:
        b, gd = e.get("bad") or {}, e.get("good") or {}
        eval_html += (f"<div class='mini'><b>rba eval</b> <code>{_e(json.dumps({k: e['patch'].get(k) for k in ('method', 'layer', 'heads', 'scale')}, ensure_ascii=False))}</code>"
                      f"<div class='gp'><span class='gl'>{t['fix']}</span>{_bar_html(b.get('fix_rate'), '#16a34a')}</div>"
                      f"<div class='gp'><span class='gl'>{t['reg']}</span>{_bar_html(gd.get('regression_rate'), '#dc2626', '' if gd.get('regression_rate') is not None else '—')}</div></div>")
    if not rows:
        return _how(t, t["patch_how"]) + eval_html + "<p class='muted'>" + ("没有生成补丁候选。" if zh else "No patch candidates.") + "</p>"
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
    best = scored[0]
    if best[0] and best[1] > 0:
        rec = (f"优先验收 {best[3]} 组的 {best[4]}：干预复核有效，修复率 {_pct(best[1])}、回退率 {_pct(best[6])}。" if zh else f"Prefer {best[4]} ({best[3]}).")
    elif best[0]:
        rec = (f"{best[3]} 组的 {best[4]} 在抽样上干预有效，下一步用 rba eval 在验收集和对照集上跑修复率和回退率。" if zh
               else f"{best[4]} works on the sample; run rba eval next.")
    else:
        rec = ("目前没有补丁通过干预复核（状态都不是「消融有效」），不建议直接上线；补丁只作为探索方向，先扩大样本或用 rba finetune 做针对性训练。" if zh
               else "No patch passed intervention checks; treat them as exploratory.")
    cmd = best[5].get("command") or ""
    return (_how(t, t["patch_how"])
            + f"<table class='ptable'><tr><th>{t['group']}</th><th>{t['patch']}</th><th>{t['status']}</th><th>{t['flip_rate']}</th><th>{t['fix']}</th><th>{t['reg']}</th></tr>{''.join(rows)}</table>"
            + eval_html + f"<p class='concl'><b>{t['recommend']}：</b>{_e(rec)}</p>" + (f"<pre>{_e(cmd)}</pre>" if cmd else ""))


def write_eval_report(result: dict, out_dir: Path, language: str = "zh") -> Path:
    """Small standalone visual page for `rba eval`."""
    t = _T.get(language, _T["zh"])
    b, g = result.get("bad") or {}, result.get("good") or {}

    def block(name, d):
        before, after = d.get("before") or {}, d.get("after") or {}
        return (f"<div class='mini'><h3>{_e(name)} (n={_e(d.get('n'))})</h3>"
                f"<div class='gp'><span class='gl'>{t['fix']}</span>{_bar_html(d.get('fix_rate'), '#16a34a')}</div>"
                f"<div class='gp'><span class='gl'>{t['reg']}</span>{_bar_html(d.get('regression_rate'), '#dc2626', '' if d.get('regression_rate') is not None else '—')}</div>"
                f"<div class='gp'><span class='gl'>acc before</span>{_bar_html(before.get('accuracy'), '#94a3b8')}</div>"
                f"<div class='gp'><span class='gl'>acc after</span>{_bar_html(after.get('accuracy'), '#0d9488')}</div></div>")
    page = (f"<!DOCTYPE html><html lang='{language}'><head><meta charset='utf-8'/><title>rba eval</title><style>{_CSS}</style></head><body><main>"
            f"<h1>rba eval</h1><pre>{_e(json.dumps(result.get('patch'), ensure_ascii=False, indent=1))}</pre>"
            + _how(t, t["patch_how"]) + f"<div class='row'>{block('bad', b)}{block('good', g)}</div></main></body></html>")
    path = Path(out_dir) / "eval_report.html"
    path.write_text(page, encoding="utf-8")
    return path


_CSS = """
table.method td{vertical-align:top;font-size:13px}svg.flow{max-width:900px;display:block;margin:4px 0 10px}
body{font-family:"Noto Sans SC","PingFang SC","Source Han Sans SC",sans-serif;margin:0;background:#f6f4ef;color:#1c1917}
nav{position:sticky;top:0;background:#1c1917;padding:8px 16px;z-index:5}nav a{color:#fafaf9;margin-right:16px;text-decoration:none;font-size:14px}
main{max-width:1180px;margin:0 auto;padding:20px 20px 80px;line-height:1.5}
h2{border-bottom:2px solid #d6d3d1;padding-bottom:4px;margin-top:36px}section{scroll-margin-top:40px}
.row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}.grow{flex:1;min-width:280px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}.kpi{background:#fff;border-radius:10px;padding:10px 16px;min-width:110px;box-shadow:0 1px 2px #0001}
.kpi b{display:block;font-size:24px}.kpi span{font-size:13px;color:#57534e}
.how{background:#eef2ff;border-left:4px solid #6366f1;padding:6px 10px;font-size:13px;margin:6px 0}
.concl{background:#fef9c3;border-left:4px solid #ca8a04;padding:6px 10px;margin:8px 0}
.banner{background:#fef3c7;border:1px solid #f59e0b;padding:8px 12px;border-radius:8px}
.card{background:#fff;border-radius:12px;padding:12px 14px;margin:12px 0;box-shadow:0 1px 3px #0002}
.ch{margin-bottom:6px}.k{display:inline-block;font-size:12px;background:#e7e5e4;border-radius:4px;padding:0 6px;margin-right:4px}
.thumb{width:120px;height:120px;object-fit:cover;border-radius:8px;background:#e7e5e4}.thumb.none{display:flex;align-items:center;justify-content:center;font-size:12px;color:#78716c}
.probs{width:260px}.gp{display:flex;align-items:center;gap:6px;margin:2px 0}.gl{width:70px;font-size:13px;text-align:right}
.pbar{position:relative;flex:1;height:16px;background:#f1f5f9;border-radius:4px;overflow:hidden;min-width:120px}.pbar div{height:100%}
.pbar span{position:absolute;left:6px;top:0;font-size:11px;line-height:16px}
.mini{flex:1;min-width:320px;background:#fafaf9;border-radius:8px;padding:8px}
.hint{font-size:12px;color:#57534e}.muted{color:#78716c}.small{font-size:12px}
.chip{display:inline-block;padding:1px 6px;margin:2px;border-radius:4px;font-size:13px}
mark{border-radius:3px;padding:0 1px}
.ti{display:flex;height:18px;border-radius:4px;overflow:hidden;font-size:11px;color:#fff;margin-top:4px;background:#e7e5e4}
.ti .tt{background:#0d9488;white-space:nowrap;padding-left:4px}.ti .ii{background:#d97706;white-space:nowrap}
.tag{font-size:12px;padding:1px 8px;border-radius:999px;background:#e7e5e4;white-space:nowrap}.tag.ok{background:#dcfce7}.tag.bad{background:#fee2e2}
.controls{background:#fff;padding:8px;border-radius:8px;position:sticky;top:34px;z-index:4}
.controls input{width:220px}
table{border-collapse:collapse;margin:8px 0;font-size:14px}th,td{border:1px solid #d6d3d1;padding:4px 8px;text-align:left;vertical-align:middle}th{background:#e7e5e4}
.ptable td{min-width:90px}
pre{background:#1c1917;color:#fafaf9;padding:8px;border-radius:8px;overflow-x:auto;white-space:pre-wrap}
svg.chart{background:#fff;border-radius:8px;max-width:100%;height:auto}svg .ax{font-size:11px;fill:#44403c}svg .cv{font-size:10px;fill:#1c1917}svg .cv.inv{fill:#fff}
.lg{font-size:12px;margin-right:8px}.lg i{display:inline-block;width:10px;height:10px;margin-right:3px;border-radius:2px}
"""

_JS = """
(function(){
var g=document.getElementById('fgroup'),s=document.getElementById('fsort'),q=document.getElementById('fsearch'),
box=document.getElementById('cards'),n=document.getElementById('fcount');if(!box)return;
function apply(){var cards=Array.prototype.slice.call(box.children),gv=g.value,qv=q.value.toLowerCase(),k=s.value,shown=0;
cards.sort(function(a,b){if(k==='id')return a.dataset.id<b.dataset.id?-1:1;if(k==='ppred')return b.dataset.ppred-a.dataset.ppred;
if(k==='err')return b.dataset.err-a.dataset.err;return a.dataset.flip-b.dataset.flip;});
cards.forEach(function(c){var ok=(!gv||c.dataset.group===gv)&&(!qv||c.dataset.search.indexOf(qv)>=0);c.style.display=ok?'':'none';if(ok)shown++;box.appendChild(c);});
n.textContent=shown+' / '+cards.length;}
g.onchange=s.onchange=apply;q.oninput=apply;apply();})();
"""


# --------------------------------------------------------------------------- method section

def _case_anchor(cid) -> str:
    # Hex of the UTF-8 id: stable, HTML-safe and collision-free (different ids never share an anchor).
    return "case-" + str(cid).encode("utf-8").hex()


def _svg_flow(steps: list[tuple[str, str]], color: str, width: int = 860) -> str:
    """Horizontal box-and-arrow diagram; plain inline SVG, no external refs."""
    n = max(1, len(steps))
    gap = 18
    w = (width - gap * (n - 1)) / n
    parts = [f"<svg class='flow' viewBox='0 0 {width} 74' width='100%' role='img' aria-label='{_e(" → ".join(t for t, _ in steps))}'>"]
    for i, (title, sub) in enumerate(steps):
        x = i * (w + gap)
        parts.append(f"<rect x='{x:.1f}' y='4' width='{w:.1f}' height='64' rx='8' fill='#fff' stroke='{color}' stroke-width='1.5'/>"
                     f"<text x='{x + w / 2:.1f}' y='30' text-anchor='middle' font-size='13' font-weight='600'>{_e(title)}</text>"
                     f"<text x='{x + w / 2:.1f}' y='51' text-anchor='middle' font-size='11' fill='#4b5563'>{_e(sub)}</text>")
        if i < n - 1:
            ax = x + w
            parts.append(f"<path d='M{ax + 2:.1f} 36 L{ax + gap - 3:.1f} 36' stroke='{color}' stroke-width='1.6'/>"
                         f"<path d='M{ax + gap - 8:.1f} 31 L{ax + gap - 2:.1f} 36 L{ax + gap - 8:.1f} 41 Z' fill='{color}'/>")
    parts.append("</svg>")
    return "".join(parts)


_METHOD_FLOWS = {
    "zh": [
        ("归因：ΔW 先算，lens+DLA 同一次捕获前向，token 归因另做梯度前向", "#7c3aed", [
            ("分组", "金标→预测"), ("ΔW / SVD", "σ(v·x)(u·d')"), ("logit lens", "norm→W_U[等级]"),
            ("DLA", "写入 c · d'"), ("token 归因", "层0输入 ∇×h")]),
        ("排查：一次只动一个组件，看分差是否下降", "#0891b2", [
            ("候选", "DLA 前 k 个"), ("干预", "head×0 / MLP=0"), ("对照", "随机同类·identity"),
            ("度量", "Δ(pred−gold)"), ("状态", "n≥4·均值<−1e-4·≥60%")]),
        ("编辑：补丁 → 复评 → 二选一：导出（+等价性）或 微调", "#16a34a", [
            ("补丁", "临时前向/权重行干预"), ("rba eval", "修复率/回退率"), ("rba export", "烘焙进权重"),
            ("等价性", "钩子 vs 导出"), ("或 rba finetune", "CE+hinge+KL")]),
    ],
    "en": [
        ("Attribution: ΔW first, lens+DLA in one capture, token attr in gradient passes", "#7c3aed", [
            ("Group", "gold→pred"), ("ΔW / SVD", "σ(v·x)(u·d')"), ("Logit lens", "norm→W_U[grades]"),
            ("DLA", "write c · d'"), ("Token attr", "∇×h at layer 0")]),
        ("Diagnosis: intervene on one component at a time", "#0891b2", [
            ("Candidates", "top-k DLA"), ("Intervene", "head×0 / MLP=0"), ("Controls", "random peer·identity"),
            ("Measure", "Δ(pred−gold)"), ("Status", "n≥4·mean<−1e-4·≥60%")]),
        ("Editing: patch → eval → export (+equivalence) or finetune", "#16a34a", [
            ("Patch", "inference hook"), ("rba eval", "fix/regression"), ("rba export", "bake weights"),
            ("Equivalence", "hook vs export"), ("or rba finetune", "CE+hinge+KL")]),
    ],
}


def _method_section(report: dict, config: dict, groups: list[dict], mech: dict, tok: dict, verify: dict, t: dict, zh: bool) -> str:
    lang = "zh" if zh else "en"
    flows = "".join(f"<h3>{_e(title)}</h3>{_svg_flow(steps, color)}" for title, color, steps in _METHOD_FLOWS[lang])
    first_mech = next(iter(mech), None)
    first_tok = next(iter(tok), None)
    tok_methods = sorted({str(r.get("method")) for r in (tok or {}).values() if isinstance(r, dict) and r.get("method")})
    method = "/".join(tok_methods) or str(config.get("token_attr_method") or "gradxinput")
    steps = config.get("ig_steps")
    steps_txt = str(steps) if steps is not None else "?"

    def _rel(g):
        ok = g.get("attribution_reliable")
        err = g.get("reconstruction_error")
        err_txt = f"{err:.2g}" if isinstance(err, (int, float)) else "—"
        flag = ("可靠" if ok else "不可靠") if zh else ("reliable" if ok else "unreliable")
        if ok is None:
            flag = "—"
        return f"{g.get('key')}: {flag}（{err_txt}）" if zh else f"{g.get('key')}: {flag} ({err_txt})"
    rel = [_rel(g) for g in groups if "attribution_reliable" in g or "reconstruction_error" in g]
    gaps = [f"{g.get('key')}: {g['decomposition_gap']:.3g}" for g in groups if isinstance(g.get("decomposition_gap"), (int, float))]
    checks = [r for rows in (verify or {}).values() if isinstance(rows, list) for r in rows if isinstance(r, dict)]
    by_type: dict[str, int] = {}
    n_skipped = sum(1 for r in checks if r.get("skipped"))
    for r in checks:
        if r.get("skipped"):
            continue
        by_type[str(r.get("control_type", "target"))] = by_type.get(str(r.get("control_type", "target")), 0) + 1
    statuses: dict[str, int] = {}
    for g in groups:
        for h in g.get("hypotheses") or []:
            statuses[str(h.get("status"))] = statuses.get(str(h.get("status")), 0) + 1
    dw = report.get("delta_w") or {}
    delta = bool(dw.get("top") or dw.get("n_tensors") or (dw.get("lm_head_grades") or {}).get("rows"))

    def link(cid, label):
        return f"<a href='#{_case_anchor(cid)}'>{_e(label)}</a>" if cid is not None else ""

    if zh:
        ig_desc = ("积分梯度：基线 0 → 真实输入，α=s/steps（s=1..steps，共 {} 步）取梯度平均，再乘输入。".format(steps_txt)
                   if method == "integrated_gradients" else "grad×input（默认，等于只取 α=1 的一步）。")
        rows = [
            ("logit lens", "每层输出 h_ℓ（决策位置）→ 模型自己的 final_norm → lm_head 的等级行 → softmax；最后一层必须与真实输出一致。",
             f"{len(mech)} 条有逐层明细", "#s2", link(first_mech, f"例：{first_mech}")),
            ("DLA 组件归因", "d = W_U[pred] − W_U[gold]；每个头（o_proj/out_proj 输入按头切）和每个 MLP 的写入 c 贡献 c·d。"
             "norm-aware 版冻结最终 RMS 分母、乘 norm 增益。正 = 推向错误预测。",
             "读出重建（误差≤τ·(1+|分差|) 为可靠）：" + ("；".join(rel) if rel else "—")
             + ("；pre-norm 结构差距（>0.15 仅告警）" + "，".join(gaps) if gaps else ""), "#s3", ""),
            ("token 归因", f"分数 = logit[pred] − logit[gold]，对层 0 输入（视觉塔已写入图像 token）求梯度；{ig_desc}"
             "image = input_ids == image_token_id，其余非 pad 为文本。", f"方法 {method}，{len(tok)} 条", "#s2", link(first_tok, f"例：{first_tok}")),
            ("干预复核", "对 DLA 前 k 个组件逐个干预（头 scale=0 / MLP 清零，默认只在决策位置），另跑随机同类组件（头取同层其它头，MLP 取其它层 MLP）与 identity(scale=1，仅头) 对照；"
             "度量 Δ = (pred−gold)_后 − (pred−gold)_前，Δ<0 为改善。",
             f"复核记录 {len(checks)} 条：执行 " + ("，".join(f"{k} {v}" for k, v in sorted(by_type.items())) or "0")
             + f"，跳过 {n_skipped}；状态 "
             + ("，".join(f"{k} {v}" for k, v in statuses.items()) or "—"), "#s5", ""),
            ("ΔW / SVD", "ΔW = W_ft − W_base；rel = ‖ΔW‖/‖W_base‖；对 top 的二维权重做 SVD；写残差流的再用 u_k·d' 投影到错误方向（vocab 型看 u_k[pred]−u_k[gold]）；lm_head 等级行看余弦（需基座词表对齐）。几何证据不是因果。",
             "有 ΔW 数据" if delta else "无可用 ΔW 数据（未提供基座、配置关闭或无可比较项）", "#s4", ""),
        ]
        head = ("信号", "代码里怎么算", "本次运行", "看哪里")
        note = "完整推导、mermaid 流程图与决策树见仓库 docs/方法图解.md。"
    else:
        ig_desc = f"integrated gradients, {steps} steps from a zero baseline.".replace(str(steps), steps_txt) if method == "integrated_gradients" else "grad×input (default, one step at α=1)."
        rows = [
            ("logit lens", "Each layer output at the decision position → final_norm → grade rows of lm_head → softmax.", f"{len(mech)} cases", "#s2", link(first_mech, str(first_mech))),
            ("DLA", "d = W_U[pred] − W_U[gold]; each head/MLP write c contributes c·d (norm-aware variant freezes the final RMS).",
             "readout reconstruction: " + ("; ".join(rel) if rel else "—"), "#s3", ""),
            ("Token attribution", f"score = logit[pred] − logit[gold], gradient at the layer-0 input; {ig_desc}", f"{method}, {len(tok)} cases", "#s2", link(first_tok, str(first_tok))),
            ("Verification", "Ablate one top-k component at a time (decision position) with random and identity controls; Δ<0 means the error margin shrank.",
             f"{len(checks)} records ({n_skipped} skipped)", "#s5", ""),
            ("ΔW / SVD", "ΔW = W_ft − W_base, relative norm, SVD of residual writers projected onto W_U. Geometry, not causation.", "yes" if delta else "no usable ΔW data", "#s4", ""),
        ]
        head = ("Signal", "How the code computes it", "This run", "Where")
        note = "Full derivation and diagrams: docs/方法图解.md in the repository."
    body = "".join(
        f"<tr><td><b>{_e(a)}</b></td><td>{_e(b)}</td><td>{_e(c)}</td><td><a href='{d}'>{_e(t.get(d.lstrip('#'), d))}</a><br/>{e}</td></tr>" for a, b, c, d, e in rows)
    table = f"<table class='method'><thead><tr>{''.join(f'<th>{_e(h)}</th>' for h in head)}</tr></thead><tbody>{body}</tbody></table>"
    return f"<section id='s7'><h2>{_e(t['s7'])}</h2>{flows}{table}<p class='muted'>{_e(note)}</p></section>"
