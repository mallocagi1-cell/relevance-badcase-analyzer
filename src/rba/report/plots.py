"""Matplotlib figures. Missing a plot is a warning, not a failed run."""

from __future__ import annotations

from pathlib import Path

from rba.console import warn

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_plots(report: dict, folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    jobs = [
        ("confusion.png", _confusion),
        ("logit_lens.png", _lens),
        ("dla.png", _dla),
        ("delta_w.png", _delta),
        ("token_share.png", _token_share),
    ]
    for name, drawer in jobs:
        path = folder / name
        try:
            if drawer(report, path):
                written.append(name)
        except Exception as exc:
            warn(f"图 {name} 没有画出来：{exc}")
    return written


def _confusion(report, path) -> bool:
    table = report.get("confusion") or {}
    grades = report.get("grades") or []
    if not grades or not table:
        return False
    import numpy as np

    grid = np.asarray([[table.get(label, {}).get(pred, 0) for pred in grades] for label in grades], dtype=float)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(grades)), grades)
    ax.set_yticks(range(len(grades)), grades)
    ax.set_xlabel("预测")
    ax.set_ylabel("金标")
    ax.set_title("混淆矩阵")
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            ax.text(col, row, str(int(grid[row, col])), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _lens(report, path) -> bool:
    groups = [group for group in report.get("groups") or [] if group.get("lens")]
    if not groups:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for group in groups[:4]:
        xs = [row["layer"] for row in group["lens"]]
        ax.plot(xs, [row["p_pred"] for row in group["lens"]], marker="o", label=f"{group['key']} 预测")
        ax.plot(xs, [row["p_gold"] for row in group["lens"]], marker="x", linestyle="--", label=f"{group['key']} 金标")
    ax.set_xlabel("层")
    ax.set_ylabel("条件概率")
    ax.set_ylim(0, 1)
    ax.set_title("Logit lens")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _dla(report, path) -> bool:
    group = next((item for item in report.get("groups") or [] if item.get("components")), None)
    if group is None:
        return False
    rows = group["components"][:10]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = [row["key"] for row in rows][::-1]
    values = [row["mean_attr"] for row in rows][::-1]
    colors = ["#b45309" if value > 0 else "#1d4ed8" for value in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#444", linewidth=0.6)
    ax.set_title(f"直接 logit 归因 · {group['key']}")
    ax.set_xlabel("推向错误等级（正）或金标（负）")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _delta(report, path) -> bool:
    rows = (report.get("delta_w") or {}).get("top") or []
    rows = [row for row in rows if row.get("rel", 0) > 0][:10]
    if not rows:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = [_short(row["name"]) for row in rows][::-1]
    ax.barh(labels, [row["rel"] for row in rows][::-1], color="#0f766e")
    ax.set_xlabel("||ΔW|| / ||W_base||")
    ax.set_title("相对变化最大的权重")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _token_share(report, path) -> bool:
    groups = [group for group in report.get("groups") or [] if group.get("token_attr")]
    if not groups:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = [group["key"] for group in groups]
    text = [group["token_attr"].get("text_abs_share") or 0 for group in groups]
    image = [group["token_attr"].get("image_abs_share") or 0 for group in groups]
    import numpy as np

    xs = np.arange(len(labels))
    ax.bar(xs, text, label="文本", color="#334155")
    ax.bar(xs, image, bottom=text, label="图像", color="#d97706")
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("绝对归因占比")
    ax.set_title("Token 归因：文本 vs 图像")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _short(name: str) -> str:
    if len(name) <= 42:
        return name
    return "…" + name[-40:]
