"""生成 docs/images/method_*.png（方法图解用的静态示意图）。

纯示意：不加载模型，数字都是为了说明形状/方向而编的。运行：
    .venv/bin/python scripts/make_method_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
CJK = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"]
available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in CJK if f in available] or ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, GREY, PURPLE, GREEN, ORANGE = "#dc2626", "#2563eb", "#6b7280", "#7c3aed", "#16a34a", "#ea580c"


def box(ax, x, y, w, h, text, color="#eef2ff", edge=PURPLE, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, wrap=True)


def arrow(ax, a, b, color=GREY, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, linestyle=ls))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ 1 overview
def overview():
    fig, ax = canvas(13, 4.6, "三段式方法：归因 → 排查 → 编辑")
    cols = [
        ("① 归因（rba analyze）", "#eef2ff", PURPLE,
         ["按 金标→预测 分组", "logit lens：每层读等级", "DLA：头/MLP 写入·d", "token 归因：文本 vs 图像", "ΔW / SVD：微调改了哪"]),
        ("② 排查（干预复核）", "#ecfeff", "#0891b2",
         ["前 k 个组件逐个干预", "随机同类组件 + identity 对照", "决策位置 scope", "n≥4，均值 Δ<−1e-4\n且 ≥60% 同向", "→ 状态：消融有效/无效果/…"]),
        ("③ 编辑（eval / export / finetune）", "#f0fdf4", GREEN,
         ["补丁 = 推理时钩子", "rba eval：修复率 / 回退率", "rba export：烘焙进权重", "等价性检查", "rba finetune：LoRA 定向训练"]),
    ]
    for i, (title, fc, ec, items) in enumerate(cols):
        x = 0.3 + i * 4.3
        box(ax, x, 3.3, 3.8, 0.7, title, fc, ec, 12, "bold")
        for j, it in enumerate(items):
            box(ax, x + 0.15, 2.65 - j * 0.55, 3.5, 0.45, it, "white", ec, 9.5)
        if i < 2:
            arrow(ax, (x + 3.85, 3.65), (x + 4.25, 3.65), ec, lw=2)
    ax.text(6.5, 0.0, "归因只给相关性；排查用干预检验；编辑必须在独立验收集 + 对照集上量修复率与回退率。",
            ha="center", fontsize=10, color=GREY)
    save(fig, "method_overview.png")


# ------------------------------------------------------------------ 2 residual stream / lens / DLA
def residual():
    fig, ax = canvas(13, 6, "残差流、logit lens 与直接 logit 归因（DLA）")
    y = 3.0
    ax.add_patch(Rectangle((0.4, y - 0.08), 10.2, 0.16, color="#c7d2fe"))
    ax.text(0.4, y - 0.45, "残差流 h（决策位置 = 最后一个非 pad token）", fontsize=10, color=PURPLE)
    box(ax, 0.2, y - 1.4, 1.3, 0.6, "embed\n(含图像 token)", "#fef3c7", ORANGE, 9)
    arrow(ax, (0.85, y - 0.8), (0.85, y - 0.1), ORANGE)
    xs = [2.2, 4.6, 7.0]
    for k, x in enumerate(xs):
        name = ["层 0", "层 ℓ", "层 L-1"][k]
        box(ax, x, y + 0.9, 1.0, 0.7, f"{name}\nattn（各头和）", "#fee2e2", RED, 9)
        box(ax, x + 1.15, y + 0.9, 0.9, 0.7, "MLP\ndown_proj", "#dcfce7", GREEN, 9)
        arrow(ax, (x + 0.5, y + 0.9), (x + 0.5, y + 0.1), RED)
        arrow(ax, (x + 1.6, y + 0.9), (x + 1.6, y + 0.1), GREEN)
        ax.text(x + 0.5, y + 0.45, "+c", color=RED, fontsize=9, ha="left")
        # logit lens tap
        arrow(ax, (x + 2.1, y - 0.1), (x + 2.1, y - 1.0), PURPLE, ls="--")
        box(ax, x + 1.5, y - 1.7, 1.2, 0.65, "final_norm\n→ W_U[等级行]", "#f5f3ff", PURPLE, 8)
    box(ax, 10.7, y - 0.35, 2.1, 0.7, "final_norm → lm_head\n→ 等级 logit", "#f5f3ff", PURPLE, 9, "bold")
    arrow(ax, (10.6, y), (10.7, y), PURPLE)
    ax.text(0.3, 0.75, "logit lens：对每层输出 h_ℓ 做 softmax(W_U[grades] · RMSNorm(h_ℓ))，看“如果此刻就输出”选哪个等级；最后一层必须与真实等级 logit 一致（图中只画 3 层，每层同理）。",
            fontsize=9.5)
    ax.text(0.3, 0.35, "DLA：d = W_U[pred] − W_U[gold]（预测减金标）。组件写入 c 的贡献 = c·d（pre-norm 代理）；"
            "norm-aware 版冻结最终 RMS 分母并乘 norm 增益 g：c·(d⊙g/rms)。正值 = 推向错误预测。", fontsize=9.5)
    ax.text(0.3, -0.05, "每个头的写入 = o_proj.weight[:, h·hd:(h+1)·hd] @ x[h·hd:(h+1)·hd]（在 o_proj / out_proj 输入上按头切）。",
            fontsize=9.5, color=GREY)
    save(fig, "method_residual_dla.png")


# ------------------------------------------------------------------ 3 IG
def ig():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("token 归因：层 0 输入上的 grad×input（默认）/ 积分梯度（可选）", fontsize=14, weight="bold")
    ax = axes[0]
    a = np.linspace(0, 1, 101)
    f = 1 / (1 + np.exp(-8 * (a - 0.45))) * 2.3 - 0.2
    ax.plot(a, f, color=PURPLE, lw=2)
    steps = 8
    for s in range(1, steps + 1):
        al = s / steps
        ax.plot([al, al], [-.4, 1 / (1 + np.exp(-8 * (al - 0.45))) * 2.3 - 0.2], color=GREY, lw=.8, ls=":")
        ax.plot(al, 1 / (1 + np.exp(-8 * (al - 0.45))) * 2.3 - 0.2, "o", color=ORANGE, ms=5)
    ax.set_xlabel("α（基线 0 → 真实输入 h0；α = s/steps，s=1..steps，右端点，梯度取平均）")
    ax.set_ylabel("分数 = logit[pred] − logit[gold]")
    ax.set_title("沿路径 α·h0 取梯度，再求平均", fontsize=11)
    ax.text(0.02, 1.85, "IG_t = Σ_dim h0[t] ⊙ mean_α ∇_{α·h0} score\n（gradxinput 时只取 α=1）", fontsize=10)
    ax = axes[1]
    toks = ["查询", "词", "<img>", "<img>", "<img>", "文档", "标题", "<img>"]
    vals = np.array([.3, -.15, .55, .4, .2, -.35, .1, .25])
    colors = [ORANGE if t == "<img>" else (RED if v > 0 else BLUE) for t, v in zip(toks, vals)]
    ax.bar(range(len(toks)), vals, color=colors)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(len(toks)), toks)
    ax.set_title("按 token 汇总：image = input_ids == image_token_id", fontsize=11)
    ax.text(-.4, -.55, "text_signed / image_signed = 分组求和；abs_share = |IG| 占比\n正 = 推向错误预测（红），负 = 拉回金标（蓝）；橙 = 图像 token",
            fontsize=9)
    ax.set_ylim(-.7, .7)
    fig.tight_layout()
    save(fig, "method_ig.png")


# ------------------------------------------------------------------ 4 SVD
def svd():
    fig, ax = canvas(13, 4.2, "ΔW = W_ft − W_base 的低秩分解与读出投影")
    box(ax, 0.3, 1.5, 1.6, 1.6, "ΔW\n[out, in]", "#fef3c7", ORANGE, 12, "bold")
    ax.text(2.1, 2.25, "≈ Σ_k σ_k ·", fontsize=13)
    box(ax, 3.6, 1.3, 0.45, 2.0, "u_k", "#fee2e2", RED, 10)
    box(ax, 4.25, 2.05, 2.0, 0.45, "v_k^T", "#dbeafe", BLUE, 10)
    ax.text(6.5, 2.9, "u_k 在输出空间：若该矩阵写残差流（o_proj/out_proj/down_proj，out=hidden），\n"
            "  就用 W_U · u_k 看它最推高/压低哪些 token、哪几个等级 → space=residual", fontsize=9.5)
    ax.text(6.5, 1.95, "lm_head 本身：u_k 直接是词表向量，读等级 token 的分量 → space=vocab", fontsize=9.5)
    ax.text(6.5, 1.25, "逐样本：系数 = v_k·x（x 为实际捕获的模块输入），\n  residual 型：readout_logit_diff = σ_k·(v_k·x)·(u_k·d')，d' 含 norm 增益与 RMS\n  vocab 型：readout_logit_diff = σ_k·(v_k·x)·(u_k[pred]−u_k[gold])；W_U·u_k 仅几何投影", fontsize=9.5)
    ax.text(0.3, 0.45, "explained_energy = σ_k² / ‖ΔW‖_F²；另报 rel = ‖ΔW‖/‖W_base‖、lm_head 等级行余弦。几何投影不是因果效应。", fontsize=9.5, color=GREY)
    save(fig, "method_svd.png")


# ------------------------------------------------------------------ 5 patches in forward
def patches():
    fig, ax = canvas(13.2, 6.4, "补丁在前向里改了哪里（全部是 context manager，退出即恢复）")
    chain = [("input_ids", 0.2), ("embed\n(视觉塔写入\n图像 token)", 2.05), ("层 0 输入", 3.9), ("层 ℓ：attn\n→ o_proj", 5.75),
             ("层 ℓ：MLP", 7.6), ("层输出 h", 9.45), ("final_norm\n→ lm_head", 11.3)]
    for name, x in chain:
        box(ax, x, 3.3, 1.5, 0.9, name, "#f8fafc", GREY, 8)
    for i in range(len(chain) - 1):
        arrow(ax, (chain[i][1] + 1.5, 3.75), (chain[i + 1][1], 3.75))
    notes = [
        (3.9, "ablate_tokens/image\n层 0 pre-hook：\ninput_ids∈token_ids\n处 hidden 置 0", ORANGE, 1.6),
        (5.75, "head_scale\no_proj 输入中头 h\n的切片 *= scale", RED, 4.75),
        (7.6, "mlp_zero\nMLP forward 返回全 0", GREEN, 1.6),
        (9.45, "steering\n层 forward hook：\nh += α·v，默认 v =\n(W_U[to]−W_U[from])/‖·‖", PURPLE, 4.75),
        (11.3, "lm_head_blend\nW_U[src] ←\n(1−α)W_U[src]+αW_U[toward]", BLUE, 1.6),
    ]
    for x, text, color, yy in notes:
        box(ax, x - 0.2, yy, 1.9, 1.3, text, "white", color, 7.5)
        arrow(ax, (x + 0.75, yy + (1.3 if yy < 3 else 0)), (x + 0.75, 3.3 if yy < 3 else 4.2), color, ls="--")
    ax.text(0.3, 0.6, "scope：复核默认 decision（只改决策位置，head_scale/mlp_zero/steering 用 _decision_position_patch）；"
            "其余为 all_positions；lm_head_blend 记为 readout，ablate_* 记为 token_positions。", fontsize=9.5)
    ax.text(0.3, 0.2, "复核时 ablate_image 被改写成 ablate_tokens(token_ids=[image_token_id])。", fontsize=9.5, color=GREY)
    save(fig, "method_patches.png")


# ------------------------------------------------------------------ 6 bake
def bake():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.suptitle("rba export 烘焙：每种补丁改的是哪块权重", fontsize=14, weight="bold")
    rng = np.random.default_rng(0)

    def mat(ax, m, title, xl, yl):
        ax.imshow(m, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    m = rng.normal(0, .6, (12, 16))
    m[:, 4:8] *= 0
    mat(axes[0], m, "head_scale：o_proj/out_proj\n头 h 的输入列 ×scale（图示 scale=0）", "输入列（n_heads×head_dim）", "hidden（输出行）")
    axes[0].add_patch(Rectangle((3.5, -.5), 4, 12, fill=False, ec=RED, lw=2))
    m = np.zeros((12, 16))
    mat(axes[1], m, "mlp_zero：down_proj\nweight（及 bias）全清零", "intermediate", "hidden")
    m = rng.normal(0, .6, (10, 12))
    m[3] = .5 * m[3] + .5 * m[7]
    mat(axes[2], m, "lm_head_blend：lm_head 一行\nW[src]←(1−α)W[src]+αW[toward]", "hidden", "词表行")
    axes[2].add_patch(Rectangle((-.5, 2.5), 12, 1, fill=False, ec=BLUE, lw=2))
    axes[2].add_patch(Rectangle((-.5, 6.5), 12, 1, fill=False, ec=GREY, lw=1, ls="--"))
    axes[3].axis("off")
    axes[3].set_title("steering：down_proj.bias += α·v", fontsize=10.5)
    axes[3].text(0, .85, "仅当 down_proj 已有 bias 且\n层后没有 post-MLP norm 才可烘焙。\nQwen3.5 线性层无 bias → 拒绝。", fontsize=9.5)
    axes[3].text(0, .45, "ablate_tokens/image：依赖输入内容，\n没有等价的权重修改 → 拒绝。", fontsize=9.5)
    axes[3].text(0, .08, "tied embedding 时 lm_head_blend 需\n--untie-embeddings；head_scale/mlp_zero/steering 若\nscope≠all_positions 需 --allow-all-positions。", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "method_bake.png")


# ------------------------------------------------------------------ 7 finetune loss
def loss():
    fig, ax = canvas(13, 4.6, "rba finetune 的损失（只在决策位置的等级 logit z ∈ R^G 上）")
    box(ax, 0.0, 2.65, 4.1, 1.4, "坏例（按混淆组均衡抽样，w=1）\nw·CE(z, gold)\n+ λ_hn·w·relu(m − (z_gold − z_pred_before))\n（仅当改前预测合法且≠金标）", "#fee2e2", RED, 8.5)
    box(ax, 0.0, 1.05, 4.1, 1.4, "对照回放（w=1；难对照 w=2）\nw·CE(z, label) + λ_kl·KL(p_ref ‖ p_θ)\n难对照另加 λ_hn·w·relu(m − (z_label − z_neg))\nneg = 最高的非金标等级", "#dbeafe", BLUE, 8.5)
    box(ax, 4.6, 2.0, 2.4, 1.1, "loss / len(chosen)\n→ backward\n→ clip_grad_norm", "#f5f3ff", PURPLE, 9.5)
    arrow(ax, (4.1, 3.3), (4.6, 2.7), RED)
    arrow(ax, (4.1, 1.7), (4.6, 2.4), BLUE)
    box(ax, 7.6, 2.0, 2.3, 1.1, "LoRA（默认 r=8, α=16）\n只挂在归因定位到的\n层/模块（localized）", "#f0fdf4", GREEN, 9.5)
    arrow(ax, (7.0, 2.55), (7.6, 2.55), GREEN)
    box(ax, 10.4, 2.0, 2.3, 1.1, "merge_lora → 保存\n→ 验收集/对照集\n修复率·回退率", "#fef3c7", ORANGE, 9.5)
    arrow(ax, (9.9, 2.55), (10.4, 2.55), ORANGE)
    ax.text(0.3, 0.05, "z = 最后位置 logits 取等级 token 列；p_ref = 训练前的当前微调模型在该对照样本上的等级分布快照（不是基座）。"
            "\n难对照 = 对照样本的 (label, 最高非金标等级) 落在训练坏例的混淆方向里。默认 margin m=1.0，λ_hn=1.0，λ_kl=1.0，replay_ratio=0.5。", fontsize=9.5)
    save(fig, "method_finetune_loss.png")


# ------------------------------------------------------------------ 8 eval 2x2
def eval2x2():
    fig, ax = canvas(12, 4.8, "rba eval：配对比较，修复率与回退率用不同分母")
    cells = [(0, 1, "修复 f\n(错 → 对)", "#dcfce7", GREEN), (0, 0, "仍错", "#f3f4f6", GREY),
             (1, 1, "保持", "#f3f4f6", GREY), (1, 0, "回退 r\n(对 → 错)", "#fee2e2", RED)]
    for row, col, text, fc, ec in cells:
        box(ax, 2.2 + col * 2.2, 2.6 - row * 1.4, 2.0, 1.2, text, fc, ec, 11, "bold" if ec != GREY else "normal")
    ax.text(3.2, 4.0, "改后错", ha="center", fontsize=10)
    ax.text(5.4, 4.0, "改后对", ha="center", fontsize=10)
    ax.text(2.0, 3.2, "改前错\n(集合 W)", ha="right", va="center", fontsize=10)
    ax.text(2.0, 1.8, "改前对\n(集合 R)", ha="right", va="center", fontsize=10)
    ax.text(7.0, 3.3, "修复率 = f / |W|", fontsize=12, color=GREEN, weight="bold")
    ax.text(7.0, 2.7, "回退率 = r / |R|", fontsize=12, color=RED, weight="bold")
    ax.text(7.0, 2.0, "各带 95% Wilson 区间；空分母记为“无定义”", fontsize=9.5)
    ax.text(7.0, 1.55, "净正确数 = f − r；另报新增非法输出", fontsize=9.5)
    ax.text(7.0, 1.1, "W/R 按改前预测划分，不按文件名（坏例/对照）划分", fontsize=9.5, color=GREY)
    ax.text(0.3, 0.3, "验收集（有 holdout 时）报修复率，对照集报回退率；两个率不能直接相减当净收益。", fontsize=9.5)
    save(fig, "method_eval.png")


if __name__ == "__main__":
    for fn in (overview, residual, ig, svd, patches, bake, loss, eval2x2):
        fn()
