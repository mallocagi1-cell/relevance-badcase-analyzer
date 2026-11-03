"""One Markdown report and one self-contained HTML report with the same content.

The Markdown is the single source: the HTML is rendered from it by a small
converter that knows exactly the constructs used here (headings, lists,
tables, fenced code, blockquotes, images, bold and inline code). User text and
LLM output are escaped before any formatting is applied.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

STATUS_LABELS: dict[str, str] = {
    "消融有效": "在该组抽样上（至少 4 条），决策位置的目标干预让 logit(原预测)−logit(金标) 下降，且多数样本同向，identity 对照接近 0。只说明干预有效，不是根因证明；选择样本与复核样本是否分开会写在证据里。",
    "已验证": "旧版本的叫法，等同于「消融有效」。",
    "仅相关": "归因或 ΔW 指向它，但还没做干预，或者只做了诊断性干预（例如 lm_head 行混合）。",
    "被复核否定": "干预反而让误差变大。留在列表里，不拿去当根因。",
    "消融无效果": "干预几乎不改变错误 logit。",
    "复核不稳定": "干预效果在样本之间方向不一致。",
    "证据不足": "复核样本少于 4 条、identity 对照非零，或最终读出分解重建失败。这类假设不给验收命令。",
    "先处理这个": "等级词上几乎没有概率质量，或全词表 argmax 不是等级词。先核对模板和解码协议，再看中间层。",
    "未执行": "这项复核没有运行（例如 lm_head 与 embedding 共享权重时的行编辑）。缺失不等于通过。",
}
_STATUS_CLASS = {
    "消融有效": "ok",
    "已验证": "ok",
    "先处理这个": "note",
    "被复核否定": "bad",
    "证据不足": "bad",
    "消融无效果": "mid",
    "复核不稳定": "mid",
    "仅相关": "mid",
    "未执行": "mid",
}


def write_reports(report: dict, folder: Path, plot_names: list[str]) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report, plot_names)
    md_path = folder / "report.md"
    html_path = folder / "report.html"
    md_path.write_text(markdown, encoding="utf-8")
    images = {}
    for name in plot_names:
        file = folder / "plots" / name
        if file.exists():
            images[name] = base64.b64encode(file.read_bytes()).decode("ascii")
    html_path.write_text(render_html(report, images, markdown=markdown), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, html_path


# --------------------------------------------------------------------------- markdown


def render_markdown(report: dict, plot_names: list[str]) -> str:
    lines: list[str] = ["# 相关性坏例分析报告", "", f"输出目录：`{report['output_dir']}`", ""]
    if report.get("banner"):
        lines += [f"> {report['banner']}", ""]
    lines += ["## 先看这里", "", report["lead"], ""]
    for group in report.get("groups") or []:
        lines.append(f"### {group['key']} · {group['count']} 条")
        lines.append("")
        if not group.get("hypotheses"):
            lines.append("这一组没有逐层证据，只在混淆表里计数。")
            lines.append("")
            continue
        shown = group["hypotheses"][:4]
        for item in shown:
            strength = item.get("evidence_strength") or item.get("status")
            lines.append(f"{item['rank']}. **[{item['status']}]** {item['title']}（证据强度：{strength}）")
            lines.append("")
            lines.append(item.get("evidence") or "")
            lines.append("")
            if item.get("weight_fraction") is not None:
                lines.append(f"若写成权重，大约动全部参数的 {item['weight_fraction']:.4%}。")
                lines.append("")
            if item.get("tied_note"):
                lines.append(f"注意：{item['tied_note']}")
                lines.append("")
            if item.get("command"):
                if item.get("exploratory"):
                    lines.append(f"探索性验收（{item.get('exploratory_note', '')}）：")
                else:
                    lines.append("下一步：")
                lines += ["", "```bash", item["command"], "```", ""]
            if item.get("evidence_ids"):
                lines.append("证据编号：" + "、".join(f"`{eid}`" for eid in item["evidence_ids"]))
                lines.append("")
            lines.append(f"会推翻它的观察：{item.get('would_refute', '')}")
            lines.append("")
        hidden = len(group["hypotheses"]) - len(shown)
        if hidden > 0:
            lines.append(f"其余 {hidden} 条在 report.json 里，没有单独的验收命令。")
            lines.append("")
        if group.get("clusters"):
            lines.append("簇（只按概率和长度分开，没有命名）：" + json.dumps(group["clusters"], ensure_ascii=False))
            lines.append("")

    lines += ["## 这次跑了什么", ""]
    overview = report["overview"]
    lines.append(f"- 坏例 {overview['n_bad']} 条，预测对 {overview['n_bad_correct']} 条，错 {overview['n_bad_wrong']} 条。")
    if overview.get("bad_accuracy") is not None:
        lines.append(
            f"- 坏例文件上的准确率 {overview['bad_accuracy']:.1%}，MAE {_fmt(overview.get('bad_mae'))}，"
            f"全词表 top 是等级词的比例 {_pct(overview.get('bad_legal_rate'))}。文件里本来就是错例时，准确率低是预期的。"
        )
    if overview.get("good_accuracy") is not None:
        lines.append(
            f"- 对照集 {overview['n_good']} 条，准确率 {overview['good_accuracy']:.1%}，MAE {_fmt(overview.get('good_mae'))}，"
            f"全词表 top 是等级词的比例 {_pct(overview.get('good_legal_rate'))}。"
        )
    if overview.get("n_invalid_bad") or overview.get("n_invalid_good"):
        lines.append(f"- 非法输出（全词表 argmax 不是等级词）：坏例 {overview.get('n_invalid_bad', 0)} 条，对照 {overview.get('n_invalid_good', 0)} 条，见下面「非法输出」。")
    if overview.get("n_excluded"):
        lines.append(f"- 剔除 {overview['n_excluded']} 条样本（图片读不到），见「剔除的样本」。")
    lines.append(
        f"- 协议：等级 token {report.get('grade_mode', 'single_token')}，解码 {report.get('decode_protocol', 'constrained')}，"
        f"输入 {report.get('input_mode', 'template')}。"
        + (" lm_head 与输入 embedding 共享权重：任何 lm_head 行编辑同时改输入表示。" if report.get("tied_lm_head") else "")
    )
    split = report.get("split") or {}
    if split:
        lines.append(f"- 验收集：{split.get('note', '')}" + (f" 留出 {split['n_holdout']}/{split['n_total']} 条。" if split.get("n_holdout") else ""))
    lines.append(f"- 设备 {report['device']}，dtype {report['dtype']}，类名 {report['architecture'].get('class_name')}。")
    lines.append(f"- 等级 token：`{json.dumps(report['grade_ids'], ensure_ascii=False)}`。")
    if report.get("warnings"):
        lines += ["", "注意：", ""]
        for item in report["warnings"]:
            lines.append(f"- {item}")
    lines.append("")

    invalid = report.get("invalid_rows") or []
    if invalid:
        lines += ["## 非法输出", "", "这些坏例的全词表 argmax 不是等级词。它们没有进入按等级索引的归因；先核对模板和解码协议。", ""]
        lines.append("| 样本 | 金标 | 全词表 top token | 等级内 argmax | 等级词概率质量 |")
        lines.append("| --- | --- | --- | --- | ---: |")
        for row in invalid:
            lines.append(f"| {row['id']} | {row['label']} | {row.get('full_top_token')} | {row.get('constrained_pred')} | {_fmt(row.get('grade_mass'))} |")
        lines.append("")
    excluded = report.get("excluded") or []
    if excluded:
        lines += ["## 剔除的样本", ""]
        for row in excluded:
            lines.append(f"- {row['id']}：{row['reason']}")
        lines.append("")

    lines += ["## 图", ""]
    for name in plot_names:
        lines.append(f"![{name}](plots/{name})")
        lines.append("")

    lines += ["## 证据", ""]
    for group in report.get("groups") or []:
        if not group.get("components") and not group.get("verifications"):
            continue
        lines.append(f"### {group['key']}")
        lines.append("")
        reliable = group.get("attribution_reliable")
        if reliable is not None:
            lines.append(
                f"读出分解{'可靠' if reliable else '不可靠'}（重建误差 {_fmt(group.get('reconstruction_error'))}）。"
                "归因是最终 RMSNorm 之后、按真实 norm 增益折算的读出贡献，正值支持原预测，负值支持金标。"
            )
            lines.append("")
        if group.get("components"):
            lines.append("| 组件 | 归因均值 | 标准差 | 绝对占比 |")
            lines.append("| --- | ---: | ---: | ---: |")
            for row in group["components"][:12]:
                lines.append(f"| {row['key']} | {row['mean_attr']:.4f} | {row['std_attr']:.4f} | {row['share']:.1%} |")
            lines.append("")
        token = group.get("token_attr") or {}
        if token:
            text = token.get("text_abs_share")
            image = token.get("image_abs_share")
            lines.append(
                f"Token 归因（{token.get('method')}，{token.get('n')} 条）：文本绝对占比 {_pct(text)}，图像绝对占比 {_pct(image)}，"
                f"文本带符号贡献 {_fmt(token.get('text_signed'))}，图像带符号贡献 {_fmt(token.get('image_signed'))}。"
                + (f" 完整性误差 {_fmt(token.get('completeness_gap'))}。" if token.get("completeness_gap") is not None else "")
                + (" 可靠性标记：不可靠。" if token.get("reliable") is False else "")
            )
            lines.append("")
        if group.get("verifications"):
            lines.append("干预复核。target 是被检验的组件，random 是同类随机对照，identity 是什么都不改的对照（应接近 0）。「范围」是干预真正作用的位置。")
            lines.append("")
            lines.append("| 复核 | 类型 | 范围 | 独立样本 | 错误 logit 变化 | 变小的条数 |")
            lines.append("| --- | --- | --- | --- | ---: | ---: |")
            for row in group["verifications"]:
                if row.get("skipped"):
                    lines.append(f"| {row['key']} | {row.get('control_type', '')} | {row.get('scope', '')} | — | 未执行：{row['skipped']} | — |")
                    continue
                lines.append(
                    f"| {row['key']} | {row.get('control_type', 'target')} | {row.get('scope', '')} | {'是' if row.get('selection_disjoint') else '否'} | "
                    f"{_fmt(row.get('mean_delta'))} | {row.get('n_improved')}/{row.get('n')} |"
                )
            lines.append("")
            lines.append("逐例的改前/改后差值在 report.json 的 `verifications[*].per_case`。")
            lines.append("")

    delta = report.get("delta_w") or {}
    if delta and (delta.get("top") or delta.get("n_tensors")):
        lines += ["### ΔW", ""]
        lines.append(
            f"比较了 {delta.get('n_tensors')} 个张量。非零 ΔW 元素占比 {(delta.get('changed_fraction') or 0):.4%}（共享的物理参数只计一次）。"
            + (f" {delta['tied_edit_note']}。" if delta.get("tied_edit_note") else "")
            + " 范数大不等于根因；奇异方向的符号是任意的。"
        )
        lines.append("")
        lines.append("| 参数 | 相对范数 | 绝对范数 | 奇异方向空间 | 解释能量 |")
        lines.append("| --- | ---: | ---: | --- | ---: |")
        for row in (delta.get("top") or [])[:12]:
            svd = row.get("svd") or {}
            direction = (svd.get("directions") or [{}])[0] if isinstance(svd, dict) else {}
            lines.append(
                f"| `{row['name']}` | {row['rel']:.4g} | {row['delta_fro']:.4g} | {direction.get('space') or svd.get('skipped') or '—'} | {_fmt(direction.get('explained_energy'))} |"
            )
        lines.append("")
        grades = (delta.get("lm_head_grades") or {}).get("rows") or {}
        if grades:
            lines.append("lm_head 等级行相对基座：")
            lines.append("")
            for grade, row in grades.items():
                lines.append(f"- {grade}: 余弦 {_fmt(row.get('cosine_to_base'))}，差范数 {_fmt(row.get('delta_norm'))}")
            lines.append("")
        if delta.get("n_missing"):
            lines.append(f"{delta['n_missing']} 个参数没能和基座对齐（名字或形状不同）。")
            lines.append("")

    probes = report.get("probes") or {}
    lines += ["### 探针", ""]
    if probes.get("skipped"):
        lines.append(probes["skipped"])
    else:
        lines.append(probes.get("note") or "")
        if probes.get("n") is not None:
            lines.append(f"样本 {probes.get('n')} 条（坏例 {probes.get('n_bad', '—')}，对照 {probes.get('n_good', '—')}，交错抽取）。")
        for layer in (probes.get("layers") or [])[:6]:
            bits = []
            for attr in layer["attributes"]:
                if attr["kind"] == "numeric":
                    bits.append(f"{attr['attribute']} 留出 R² {_fmt(attr.get('r2_heldout'))}")
                else:
                    baseline = attr.get("majority_accuracy_heldout", attr.get("majority_baseline"))
                    extra = f"（多数类基线 {_fmt(baseline)}，训练/留出 {attr.get('n_train', '—')}/{attr.get('n_test', '—')}）" if baseline is not None else ""
                    bits.append(f"{attr['attribute']} 留出准确率 {_fmt(attr.get('accuracy_heldout'))}{extra}")
            if bits:
                lines.append(f"- 层 {layer['layer']}: " + "；".join(bits))
    lines.append("")

    lines += ["## 给前沿模型的提示词", ""]
    mode = report.get("llm_mode") or ("dry_run" if report.get("dry_run") else "remote")
    if mode == "off":
        lines.append("配置关闭了大模型阶段，没有生成提示词。")
    elif mode == "dry_run":
        lines.append("这次没有调用接口。提示词（含证据包）在 `prompts/`：")
        for name in report.get("prompt_files") or []:
            lines.append(f"- `{name}`")
    else:
        if report.get("llm_results"):
            lines.append("模型回复在 `prompts/`，解析出的 JSON 在下面。这些是建议，不是已验证的根因；每条解释应引用证据编号。")
            lines.append("")
            for row in report["llm_results"]:
                lines += [f"### {row['group']}", "", "```json", json.dumps(row.get("parsed") or {"raw": row.get("raw", "")[:2000]}, ensure_ascii=False, indent=2), "```", ""]
        if report.get("llm_failures"):
            lines.append("调用失败的组（提示词仍在 `prompts/`，修好接口后重跑同一条命令即可）：")
            lines.append("")
            lines.append("| 组 | 错误 | 提示词 |")
            lines.append("| --- | --- | --- |")
            for row in report["llm_failures"]:
                lines.append(f"| {row['group']} | {row['error']} | `{row.get('prompt_file', '')}` |")
            lines.append("")
        if not report.get("llm_results") and not report.get("llm_failures"):
            lines.append("这次没有生成提示词（没有可分析的混淆组）。")
    lines.append("")

    lines += ["## 方法说明", "", report["methodology"], ""]
    lines += ["### 状态含义", "", "| 状态 | 含义 |", "| --- | --- |"]
    for status, meaning in STATUS_LABELS.items():
        if status == "已验证":
            continue
        lines.append(f"| {status} | {meaning} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- html


def render_html(report: dict, images: dict[str, str], markdown: str | None = None) -> str:
    if markdown is None:
        markdown = render_markdown(report, list(images))
    body = _markdown_to_html(markdown, images)
    style = """
    body { font-family: "Noto Sans SC", "PingFang SC", "Source Han Sans SC", sans-serif; margin: 0; background: #f6f4ef; color: #1c1917; }
    main { max-width: 960px; margin: 0 auto; padding: 32px 20px 80px; line-height: 1.55; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    h2 { margin-top: 36px; border-bottom: 1px solid #d6d3d1; padding-bottom: 4px; }
    h3 { margin-top: 28px; }
    blockquote { background: #fef3c7; border: 1px solid #f59e0b; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
    .tag { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #e7e5e4; margin-right: 4px; white-space: nowrap; }
    .tag.ok { background: #dcfce7; color: #14532d; }
    .tag.mid { background: #ffedd5; color: #7c2d12; }
    .tag.bad { background: #fee2e2; color: #7f1d1d; }
    .tag.note { background: #dbeafe; color: #1e3a8a; }
    pre { background: #1c1917; color: #fafaf9; padding: 10px 12px; border-radius: 8px; overflow-x: auto; }
    code { background: #e7e5e4; padding: 1px 4px; border-radius: 4px; font-size: 90%; }
    pre code { background: none; padding: 0; }
    table { border-collapse: collapse; margin: 10px 0 16px; font-size: 14px; max-width: 100%; display: block; overflow-x: auto; }
    th, td { border: 1px solid #d6d3d1; padding: 5px 8px; text-align: left; vertical-align: top; }
    th { background: #e7e5e4; }
    img { max-width: 100%; background: white; border-radius: 8px; display: block; margin: 8px 0; }
    ol > li { margin: 10px 0; }
    """
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>坏例分析报告</title><style>{style}</style></head><body><main>"
        + body
        + "</main></body></html>"
    )


_TAG_RE = re.compile(r"\*\*\[([^\]]+)\]\*\*")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")
_IMG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
_OL_RE = re.compile(r"^(\d+)\. (.*)$")


def _inline(text: str) -> str:
    escaped = html.escape(text, quote=False)

    def tag(match):
        status = html.unescape(match.group(1))
        klass = _STATUS_CLASS.get(status, "mid")
        return f"<span class='tag {klass}'>{match.group(1)}</span>"

    escaped = _TAG_RE.sub(tag, escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
    return escaped


def _markdown_to_html(markdown: str, images: dict[str, str]) -> str:
    out: list[str] = []
    lines = markdown.splitlines()
    i = 0
    list_kind: str | None = None

    def close_list():
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            close_list()
            i += 1
            continue
        if stripped.startswith("```"):
            close_list()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(code), quote=False) + "</code></pre>")
            continue
        heading = re.match(r"^(#{1,4}) (.*)$", stripped)
        if heading:
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if stripped.startswith("> "):
            close_list()
            out.append(f"<blockquote>{_inline(stripped[2:])}</blockquote>")
            i += 1
            continue
        if stripped.startswith("|"):
            close_list()
            rows: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            out.append(_table(rows))
            continue
        image = _IMG_RE.match(stripped)
        if image:
            close_list()
            name = Path(image.group(2)).name
            blob = images.get(name)
            src = f"data:image/png;base64,{blob}" if blob else html.escape(image.group(2))
            out.append(f"<figure><img alt='{html.escape(image.group(1))}' src='{src}'/><figcaption>{html.escape(image.group(1))}</figcaption></figure>")
            i += 1
            continue
        if stripped.startswith("- "):
            if list_kind != "ul":
                close_list()
                out.append("<ul>")
                list_kind = "ul"
            out.append(f"<li>{_inline(stripped[2:])}</li>")
            i += 1
            continue
        ordered = _OL_RE.match(stripped)
        if ordered:
            if list_kind != "ol":
                close_list()
                out.append("<ol>")
                list_kind = "ol"
            out.append(f"<li value='{ordered.group(1)}'>{_inline(ordered.group(2))}</li>")
            i += 1
            continue
        close_list()
        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines[i].strip()):
            paragraph.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(paragraph))}</p>")
    close_list()
    return "\n".join(out)


def _is_block_start(stripped: str) -> bool:
    return (
        stripped.startswith(("#", "> ", "- ", "|", "```", "!["))
        or bool(_OL_RE.match(stripped))
    )


def _table(rows: list[str]) -> str:
    def cells(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    if len(rows) < 2:
        return f"<p>{_inline(rows[0])}</p>"
    header = cells(rows[0])
    body_rows = rows[2:] if re.match(r"^\|?\s*:?-{3,}", rows[1]) else rows[1:]
    out = ["<table><thead><tr>" + "".join(f"<th>{_inline(cell)}</th>" for cell in header) + "</tr></thead><tbody>"]
    for row in body_rows:
        out.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in cells(row)) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


METHODOLOGY = """\
决策位置是最后一个非填充 token，等级是它的下一个 token。解码协议有两种：constrained 只在等级 token 里取 argmax；free 用全词表 argmax，落在等级外的样本记为非法输出，单独统计，不进入按等级索引的归因。两种协议都会报告全词表 top 是等级词的比例和等级词概率质量。若质量很低，先检查模板和等级 token 协议是否和训练一字不差，而不是先改中间层。

真实模型的输入全部走它自己的处理器：图片按原分辨率交给处理器，处理器决定图像 token 的数量；有图和无图样本用同一套模板或同一套对话模板。读不到的图片不会悄悄退化成纯文本，而是停下或剔除并写进报告。

Logit lens 把每一层的残差送进模型自己的最终 RMSNorm，再读等级行，同时保留等级 logit 和条件概率。最后一层的 lens 应当和模型真实的等级分布一致，这是接线检查。

读出归因的方向是 d = W_U[原预测] − W_U[金标]。报告里的组件归因是在最终 RMSNorm 之后、按真实 norm 增益和最终残差尺度折算的贡献：嵌入入口（视觉特征写入之后）加上每层注意力写入和 MLP 写入，应当重建真实的 logit 差；重建误差和「可靠」标记一起写出。旧的 norm 前点积只作为诊断保留。头的贡献来自注意力输出投影的分片：全注意力是 o_proj，线性注意力（GatedDeltaNet）是 out_proj。正值支持原预测，负值支持金标。这是归因，不是总因果效应。

干预复核在决策位置做：把某个头的输出切片乘系数、把某层 MLP 在该位置清零、或在该位置加 steering 向量。每个目标都带同类随机对照和 identity 对照（什么都不改，应接近 0）。「消融有效」的意思很窄：在该组抽样上，目标干预让 logit(原预测)−logit(金标) 下降且多数样本同向。若选择样本和复核样本没有分开、identity 非零或随机对照同样有效，状态是「证据不足」。lm_head 行混合是诊断性操作：它代数上必然缩小两行差距，不构成独立证据；lm_head 与输入 embedding 共享权重时，行编辑同时改输入表示，不会被推荐。

ΔW 只说明微调动过哪里。奇异向量成对保留（左、右、奇异值、解释能量），符号任意；只有输出维就是残差维的矩阵才投到等级词上，而且没有样本激活系数时，不能说某个方向「促进」了某个等级。范数大不是根因。

Token 归因是语言层入口表征上的 grad×input 或积分梯度，目标同样是错误 logit，按图像 token id 和文本切开；积分梯度写出零基线、端点分数和完整性误差，不收敛时标为不可靠。这不是像素梯度，也不是图像 token 的数量占比。

探针是用户属性上的岭回归（分类带截距），交错抽取坏例和对照，报告留出一半的结果和多数类基线。它不进入根因排序。

rba eval 报告两个条件率：坏例修复率和对照回退率，各带 95% Wilson 区间，再加净正确数变化。两个条件率来自不同人群，不能直接相减当作线上净收益；净收益取决于线上错误占比。只有设了 holdout_fraction 时，坏例侧才是没有参与选补丁的验收集；否则写成「发现集复评」，不代表泛化。

前沿模型拿到的是带证据编号的结构化证据包：观察、解释和待验证预测分开写，对照与反证优先保留。它的回复是建议，不是已验证的根因。
"""
