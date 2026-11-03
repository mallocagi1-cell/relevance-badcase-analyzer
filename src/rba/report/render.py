"""One markdown report and one self-contained HTML report.

Both open with the ranked hypotheses and the exact command to run next.
Evidence and the method notes come after that.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path


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
    html_path.write_text(render_html(report, images), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, html_path


def render_markdown(report: dict, plot_names: list[str]) -> str:
    lines = [f"# 相关性坏例分析报告", "", f"输出目录：`{report['output_dir']}`", ""]
    if report.get("banner"):
        lines += [f"> {report['banner']}", ""]
    lines += ["## 先看这里", ""]
    lines.append(report["lead"])
    lines.append("")
    for group in report.get("groups") or []:
        lines.append(f"### {group['key']} · {group['count']} 条")
        lines.append("")
        if not group.get("hypotheses"):
            lines.append("这一组没有逐层证据，只在混淆表里计数。")
            lines.append("")
            continue
        shown = group["hypotheses"][:4]
        for item in shown:
            lines.append(f"{item['rank']}. **[{item['status']}]** {item['title']}（置信度 {item['confidence']}）")
            lines.append("")
            lines.append(item["evidence"])
            lines.append("")
            if item.get("weight_fraction") is not None:
                lines.append(f"若写成权重，大约动全部参数的 {item['weight_fraction']:.4%}。")
                lines.append("")
            if item.get("command"):
                lines.append("下一步：")
                lines.append("")
                lines.append("```bash")
                lines.append(item["command"])
                lines.append("```")
                lines.append("")
            lines.append(f"会推翻它的观察：{item['would_refute']}")
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
        lines.append(f"- 坏例文件上的准确率 {overview['bad_accuracy']:.1%}，MAE { _fmt(overview.get('bad_mae')) }。文件里本来就是错例时，准确率低是预期的。")
    if overview.get("good_accuracy") is not None:
        lines.append(
            f"- 对照集 {overview['n_good']} 条，准确率 {overview['good_accuracy']:.1%}，MAE {_fmt(overview.get('good_mae'))}。"
        )
    lines.append(f"- 设备 {report['device']}，dtype {report['dtype']}，类名 {report['architecture'].get('class_name')}。")
    lines.append(f"- 等级 token：`{json.dumps(report['grade_ids'], ensure_ascii=False)}`。")
    if report.get("warnings"):
        lines.append("")
        lines.append("注意：")
        for item in report["warnings"]:
            lines.append(f"- {item}")
    lines += ["", "## 图", ""]
    for name in plot_names:
        lines.append(f"![{name}](plots/{name})")
        lines.append("")
    lines += ["## 证据", ""]
    for group in report.get("groups") or []:
        if not group.get("components"):
            continue
        lines.append(f"### {group['key']}")
        lines.append("")
        lines.append("| 组件 | 归因均值 | 标准差 | 占比 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in group["components"][:12]:
            lines.append(f"| {row['key']} | {row['mean_attr']:.4f} | {row['std_attr']:.4f} | {row['share']:.1%} |")
        lines.append("")
        token = group.get("token_attr") or {}
        if token:
            lines.append(
                f"Token 归因（{token.get('method')}）：文本绝对占比 {token.get('text_abs_share'):.1%}，"
                f"图像绝对占比 {token.get('image_abs_share'):.1%}。"
            )
            lines.append("")
        if group.get("verifications"):
            lines.append("| 复核 | 错误 logit 变化 | 变小的条数 |")
            lines.append("| --- | ---: | ---: |")
            for row in group["verifications"]:
                lines.append(f"| {row['key']} | {_fmt(row.get('mean_delta'))} | {row.get('n_improved')}/{row.get('n')} |")
            lines.append("")
    delta = report.get("delta_w") or {}
    if delta:
        lines.append("### ΔW")
        lines.append("")
        lines.append(
            f"比较了 {delta.get('n_tensors')} 个张量，相对变化超过 1e-8 的参数占比 {delta.get('changed_fraction') or 0:.4%}。"
        )
        lines.append("")
        lines.append("| 参数 | 相对范数 | 绝对范数 |")
        lines.append("| --- | ---: | ---: |")
        for row in (delta.get("top") or [])[:12]:
            lines.append(f"| `{row['name']}` | {row['rel']:.4g} | {row['delta_fro']:.4g} |")
        lines.append("")
        grades = delta.get("lm_head_grades", {}).get("rows", {})
        if grades:
            lines.append("lm_head 等级行相对基座：")
            lines.append("")
            for grade, row in grades.items():
                lines.append(
                    f"- {grade}: 余弦 {_fmt(row.get('cosine_to_base'))}，差范数 {_fmt(row.get('delta_norm'))}"
                )
            lines.append("")
    probes = report.get("probes") or {}
    lines.append("### 探针")
    lines.append("")
    if probes.get("skipped"):
        lines.append(probes["skipped"])
    else:
        lines.append(probes.get("note") or "")
        for layer in (probes.get("layers") or [])[:6]:
            bits = []
            for attr in layer["attributes"]:
                if attr["kind"] == "numeric":
                    bits.append(f"{attr['attribute']} 留出 R² {_fmt(attr.get('r2_heldout'))}")
                else:
                    bits.append(f"{attr['attribute']} 留出准确率 {_fmt(attr.get('accuracy_heldout'))}")
            if bits:
                lines.append(f"- 层 {layer['layer']}: " + "；".join(bits))
    lines.append("")
    lines += ["## 给前沿模型的提示词", ""]
    if report.get("dry_run"):
        lines.append("这次是干跑，没有调用接口。提示词在 `prompts/`。")
    elif report.get("llm_results"):
        lines.append("模型回复在 `prompts/`，解析出的 JSON 在下面。这些是建议，不是已验证的根因。")
        for row in report["llm_results"]:
            lines.append(f"### {row['group']}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(row.get("parsed") or {"raw": row.get("raw", "")[:2000]}, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    else:
        lines.append("这次没有生成提示词。")
    lines += ["", "## 方法说明", "", report["methodology"], ""]
    return "\n".join(lines)


def render_html(report: dict, images: dict[str, str]) -> str:
    body = []
    if report.get("banner"):
        body.append(f"<div class='banner'>{html.escape(report['banner'])}</div>")
    body.append("<h2>先看这里</h2>")
    body.append(f"<p class='lead'>{html.escape(report['lead'])}</p>")
    for group in report.get("groups") or []:
        body.append(f"<h3>{html.escape(group['key'])} · {group['count']} 条</h3>")
        if not group.get("hypotheses"):
            body.append("<p>这一组没有逐层证据。</p>")
            continue
        shown = group["hypotheses"][:4]
        for item in shown:
            klass = {"已验证": "ok", "先处理这个": "note", "被复核否定": "bad", "证据不足": "bad"}.get(item["status"], "mid")
            command = ""
            if item.get("command"):
                command = f"<pre>{html.escape(item['command'])}</pre>"
            fraction = ""
            if item.get("weight_fraction") is not None:
                fraction = f"<p>若写成权重，大约动全部参数的 {item['weight_fraction']:.4%}。</p>"
            body.append(
                f"<article class='card {klass}'>"
                f"<h4><span class='tag {klass}'>{html.escape(item['status'])}</span> "
                f"{html.escape(item['title'])} <small>置信度 {item['confidence']}</small></h4>"
                f"<p>{html.escape(item['evidence'])}</p>"
                f"{fraction}{command}"
                f"<p class='refute'>会推翻它的观察：{html.escape(item['would_refute'])}</p>"
                f"</article>"
            )
        hidden = len(group["hypotheses"]) - len(shown)
        if hidden:
            body.append(f"<p>其余 {hidden} 条在 report.json 里。</p>")
    overview = report["overview"]
    body.append("<h2>概览</h2><ul>")
    body.append(f"<li>坏例 {overview['n_bad']}，错 {overview['n_bad_wrong']}，对 {overview['n_bad_correct']}。</li>")
    if overview.get("good_accuracy") is not None:
        body.append(f"<li>对照集准确率 {overview['good_accuracy']:.1%}。</li>")
    body.append(f"<li>类名 {html.escape(str(report['architecture'].get('class_name')))}，设备 {html.escape(report['device'])}。</li>")
    body.append("</ul>")
    if images:
        body.append("<h2>图</h2>")
        for name, blob in images.items():
            body.append(f"<figure><img alt='{html.escape(name)}' src='data:image/png;base64,{blob}'/><figcaption>{html.escape(name)}</figcaption></figure>")
    body.append("<h2>方法说明</h2>")
    for paragraph in report["methodology"].split("\n\n"):
        body.append(f"<p>{html.escape(paragraph)}</p>")
    body.append("<details><summary>完整 JSON 摘要在 report.json</summary><p>同目录的 report.json 含混淆表、归因和补丁字段。</p></details>")
    style = """
    body { font-family: "Noto Sans SC", "PingFang SC", "Source Han Sans SC", sans-serif; margin: 0; background: #f6f4ef; color: #1c1917; }
    main { max-width: 920px; margin: 0 auto; padding: 32px 20px 80px; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    h2 { margin-top: 36px; }
    .banner { background: #fef3c7; border: 1px solid #f59e0b; padding: 12px 14px; border-radius: 10px; }
    .lead { font-size: 16px; line-height: 1.6; }
    .card { background: white; border-radius: 12px; padding: 14px 16px; margin: 12px 0; border-left: 6px solid #a8a29e; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .card.ok { border-left-color: #15803d; }
    .card.mid { border-left-color: #b45309; }
    .card.bad { border-left-color: #b91c1c; }
    .card.note { border-left-color: #1d4ed8; }
    .tag { font-size: 12px; padding: 2px 6px; border-radius: 999px; background: #e7e5e4; }
    .tag.ok { background: #dcfce7; }
    .tag.mid { background: #ffedd5; }
    .tag.bad { background: #fee2e2; }
    .tag.note { background: #dbeafe; }
    pre { background: #1c1917; color: #fafaf9; padding: 10px 12px; border-radius: 8px; overflow-x: auto; }
    .refute { color: #44403c; font-size: 14px; }
    img { max-width: 100%; background: white; border-radius: 8px; }
    figcaption { color: #78716c; font-size: 12px; }
    """
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>坏例分析报告</title><style>{style}</style></head><body><main>"
        "<h1>相关性坏例分析报告</h1>"
        + "\n".join(body)
        + "</main></body></html>"
    )


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


METHODOLOGY = """\
决策位置是最后一个非填充 token。等级是它的下一个 token。预测取五个等级 logit 的 argmax，同时单独报告全词表 argmax 是否落在等级词上。若等级词质量很低，先检查模板是不是和训练时一字不差，而不是先改中间层。

Logit lens 把每一层的残差送进模型自己的最终 RMSNorm，再读五个等级行。最后一层的 lens 应当和模型真实的等级分布一致，这是接线检查。

直接 logit 归因的方向是 d = W_U[预测] - W_U[金标]。某个残差写入 c 的贡献是 c·d，正号表示把决策推向错误等级。嵌入（语言层入口，视觉特征已经写入之后）加上每层注意力写入和 MLP 写入，应当等于进入最终 RMSNorm 的向量，点积也应当相加。这个和不等于 norm 之后的 logit 差，因为 RMSNorm 不是线性的。两边的数都会写在结果里。头的贡献来自注意力输出投影的分片：全注意力是 o_proj，线性注意力（GatedDeltaNet）是 out_proj。没有 bias 时，头的和等于该模块的写入。

ΔW 只说明微调动过哪里。奇异向量只有在输出维就是残差维（o_proj、out_proj、down_proj）时才投影到等级词。范数大不等于是根因。final norm 这类全局旋钮会标出来，不拿来当机制。

已验证的意思很窄：在该组抽样上，指定干预让 logit(预测)-logit(金标) 下降，并且多数样本同向。这不是对照集，也不是因果的充分条件。保留补丁之前要跑 rba eval，看坏例修复率是否高于对照回退率。消融让误差变大的组件会标成被复核否定，不会从列表里拿掉。

Token 归因是语言层入口表征上的 grad×input 或积分梯度，目标同样是错误 logit。图像和文本按 token id 切开。这不是像素梯度。注意力份额只在模块真的返回了权重时计算；线性注意力层没有注意力矩阵。

探针是用户属性上的岭回归。留出一半仅在样本够的时候做。它不进入根因排序。

前沿模型拿到的是上面这些压缩数字，不是权重本身。它的 JSON 建议和测量得到的排序分开写。提示词的结构跟权重空间编程那组实验一致：先说清任务和验收，再给证据，要求最小修改，并写明什么观察会推翻它。
"""
