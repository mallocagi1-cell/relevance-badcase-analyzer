"""Prompts in the style of the weight-space programming logs.

The blog's prompts are blunt and operational: name the architecture, say how
the edit will be scored, forbid training and global rescaling, give compact
numbers, and demand a minimal edit plus the observation that would refute it.
This module does the same job for one confusion group, in Chinese, inside a
token budget.
"""

from __future__ import annotations

import json


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / max(chars_per_token, 0.5)))


def fit_budget(parts: list[dict], budget_tokens: int, chars_per_token: float) -> tuple[str, list[str]]:
    """Keep non-droppable parts, then droppable parts in order, then truncate."""
    kept = []
    dropped = []
    used = 0
    for part in parts:
        cost = estimate_tokens(part["text"], chars_per_token)
        if not part.get("droppable", False):
            kept.append(part)
            used += cost
            continue
        if used + cost <= budget_tokens:
            kept.append(part)
            used += cost
        else:
            room = budget_tokens - used
            if room > 80:
                chars = int(room * chars_per_token)
                kept.append({**part, "text": part["text"][:chars] + "\n…（这一段因预算被截断）"})
                used = budget_tokens
            dropped.append(part["name"])
    # If the required header alone is over budget, hard-truncate the whole text.
    text = "\n\n".join(part["text"].strip() for part in kept if part["text"].strip())
    if estimate_tokens(text, chars_per_token) > budget_tokens:
        chars = int(budget_tokens * chars_per_token)
        text = text[:chars] + "\n…（整份提示词超过预算，已从末尾截断）"
        dropped.append("tail")
    return text, dropped


def build_group_prompt(
    *,
    architecture: dict,
    grades: list[str],
    grade_ids: dict[str, int],
    group: dict,
    mech: dict,
    hypotheses: list[dict],
    delta_top: list[dict],
    template: str,
    budget_tokens: int,
    chars_per_token: float,
) -> dict:
    layer_lines = []
    for layer in architecture.get("layers", []):
        layer_lines.append(
            f"  层 {layer['index']}: {layer['kind']} / {layer.get('attn_name')}，"
            f"头 {layer.get('heads')}，head_dim {layer.get('head_dim')}"
        )
    header = f"""任务。下面不是原始权重，而是从一个微调后的相关性模型上压出来的证据。给定查询和文档（可以有图），模型只解码一个 token，这个 token 必须是等级之一：{', '.join(grades)}。你要解释混淆组 {group['key']} 为什么会错，指出负责的组件，并给出一个最小、可复核的修改。

怎么算做成。修改只会用四种动作来验收：把某个注意力头的输出乘一个系数（head_scale）、把某一层 MLP 的输出清零（mlp_zero，只用于定位）、在残差上加一个 steering 向量、或把 lm_head 里错误等级的那一行向金标等级的行混合。验收看两件事：坏例上原来判错、改完判对的比例（修复率），以及对照集上原来判对、改完判错的比例（回退率）。修复率应当高于回退率。你看不到验收集的再训练，也不许建议再训练。

什么算数。要具体到层、头、MLP 或 lm_head 的某一行，并说明证据。只改错误原因，不要动整个张量的尺度，不要动 final RMSNorm 来把分布调尖。直接 logit 归因是残差写到 (W_U[预测]-W_U[金标]) 上的点积，发生在最终 RMSNorm 之前，所以它不等于 norm 之后的 logit 差。只有消融让错误 logit 下降，才写成已验证；否则就说证据不够。抽样复核不是对照集。

架构。类名 {architecture.get('class_name')}，{architecture.get('n_layers')} 层，hidden {architecture.get('hidden_size')}，词表 {architecture.get('vocab_size')}，参数量 {architecture.get('n_parameters')}。
等级 token id：{json.dumps(grade_ids, ensure_ascii=False)}。
层：
{chr(10).join(layer_lines)}
路径：{json.dumps(architecture.get('paths'), ensure_ascii=False)}
训练模板（下一个 token 才是等级，模板里不含答案）：
{template.strip()}
"""
    measured = _measured_block(group, mech, hypotheses)
    delta = _delta_block(delta_top, group)
    examples = _examples_block(group)
    lens = _lens_block(mech)
    ask = """请这样回答。先用四五句中文说明：这组错在哪、你相信哪个组件、什么观察会推翻它。然后给一个 JSON 对象，不要别的字段：
{
  "group": "金标->预测",
  "mechanism": "这个组件在决策位置做了什么",
  "components": [{"kind": "attn_head|mlp|lm_head|image", "layer": 0, "head": 0, "why": "..."}],
  "minimal_fix": {"method": "head_scale|mlp_zero|steering|lm_head_blend", "layer": 0, "heads": [0], "scale": 0.5, "grade": "0", "toward": "3", "alpha": 0.5, "from_grade": "0", "to_grade": "3", "layers": [0]},
  "weight_note": "大约动多少参数，推理期干预就写 0",
  "verification": "rba eval 应看到修复率高于回退率；以及会推翻你的那条观察",
  "confidence": 0.0,
  "insufficient_evidence": false
}
method 用不到的字段省略。证据不够就把 insufficient_evidence 设为 true，minimal_fix 设为 null。不要打印大段词表，不要建议梯度训练。
"""
    parts = [
        {"name": "header", "text": header, "droppable": False},
        {"name": "measured", "text": measured, "droppable": False},
        {"name": "delta", "text": delta, "droppable": True},
        {"name": "lens", "text": lens, "droppable": True},
        {"name": "examples", "text": examples, "droppable": True},
        {"name": "ask", "text": ask, "droppable": False},
    ]
    text, dropped = fit_budget(parts, budget_tokens, chars_per_token)
    return {"group": group["key"], "text": text, "dropped": dropped, "estimated_tokens": estimate_tokens(text, chars_per_token)}


def _measured_block(group: dict, mech: dict, hypotheses: list[dict]) -> str:
    lines = [
        f"这组证据。{group['key']} 共 {group['count']} 条，其中 {group['n_detail']} 条做了逐层分析。",
        f"整组条件概率均值：{json.dumps(group.get('mean_probs'), ensure_ascii=False)}。",
        f"带图比例 {group.get('has_image_fraction')}。等级词概率质量 {mech.get('grade_mass')}。全词表 argmax 落在等级词上的比例 {mech.get('full_top_grade_rate')}。",
        f"错误 logit（预测减金标）均值 {mech.get('logit_diff_mean')}。残差分解相对差距 {mech.get('decomposition_gap')}。",
        f"logit lens 中预测概率第一次超过金标的层：{mech.get('flip_layer')}。",
        f"图像 token 占比 {mech.get('image_token_fraction')}，注意力打到图像上的比例 {mech.get('attn_image_share')}（用到的注意力层数 {mech.get('attn_layers_used')}）。",
        "程序自己排的假设（已验证表示抽样消融让错误 logit 下降；不是对照集结果）：",
    ]
    if not hypotheses:
        lines.append("  （没有排出假设）")
    for item in hypotheses[:5]:
        lines.append(
            f"  {item['rank']}. [{item['status']}] {item['title']} 置信度 {item['confidence']} 分数 {item['score']:.2f}"
        )
        lines.append(f"     {item['evidence']}")
    lines.append("归因最高的组件：")
    for component in (mech.get("components") or [])[:8]:
        lines.append(
            f"  {component['key']} 均值 {component['mean_attr']:.4f} 标准差 {component['std_attr']:.4f} 占比 {component['share']:.2f}"
        )
    return "\n".join(lines)


def _delta_block(delta_top: list[dict], group: dict) -> str:
    if not delta_top:
        return "权重差。这次没有基座对照，或 ΔW 被关掉了。"
    lines = ["权重差。下面是相对范数最大的矩阵，不是因果排序。"]
    for row in delta_top[:8]:
        svd = row.get("svd") or {}
        directions = (svd.get("directions") or [])[:1]
        extra = ""
        if directions:
            extra = f" 最大奇异值 {directions[0].get('singular_value')} 空间 {directions[0].get('space')} 等级投影 {directions[0].get('grades')}"
        lines.append(
            f"  {row['name']} 相对范数 {row['rel']:.4g} 绝对范数 {row['delta_fro']:.4g} 形状 {row['shape']}{extra}"
        )
    return "\n".join(lines) + f"\n本组金标 {group['label']}，预测 {group['pred']}。"


def _lens_block(mech: dict) -> str:
    rows = mech.get("lens") or []
    if not rows:
        return ""
    # Keep endpoints and the flip, drop the middle if the model is deep.
    if len(rows) > 12:
        flip = mech.get("flip_layer")
        keep = []
        for row in rows:
            if row["layer"] < 2 or row["layer"] >= rows[-1]["layer"] - 1 or row["layer"] == flip:
                keep.append(row)
        rows = keep
    lines = ["logit lens（最终 RMSNorm 之后，五个等级上的条件概率）："]
    for row in rows:
        lines.append(f"  层 {row['layer']}: P(预测)={row['p_pred']:.3f} P(金标)={row['p_gold']:.3f}")
    return "\n".join(lines)


def _examples_block(group: dict) -> str:
    lines = ["样例（截断）："]
    for example in group.get("examples") or []:
        lines.append(
            f"  {example['id']} 图={example['has_image']} 查询={example['query']} 文档={example['doc']}"
        )
    return "\n".join(lines)
