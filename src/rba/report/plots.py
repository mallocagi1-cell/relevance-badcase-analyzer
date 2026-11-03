"""Matplotlib figures. Missing a plot is a warning, not a failed run.

Fonts: the first CJK font found is appended to the sans-serif fallback list, so
ASCII glyphs still come from DejaVu Sans and Chinese from the CJK font. Only the
known "glyph missing" warning is filtered, and only when a CJK font was found;
other matplotlib warnings stay visible. If no CJK font exists, one hint is
printed and the Chinese labels will render as boxes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from rba.console import warn

_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
)
_state: dict = {"configured": False, "cjk": None}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    if not _state["configured"]:
        _state["configured"] = True
        cjk_name = None
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    font_manager.fontManager.addfont(path)
                    cjk_name = font_manager.FontProperties(fname=path).get_name()
                    break
                except Exception:
                    continue
        if cjk_name is None:
            installed = {font.name for font in font_manager.fontManager.ttflist}
            for name in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "PingFang SC", "Microsoft YaHei", "SimHei", "Source Han Sans SC"):
                if name in installed:
                    cjk_name = name
                    break
        _state["cjk"] = cjk_name
        plt.rcParams["font.family"] = "sans-serif"
        base = ["DejaVu Sans"]
        plt.rcParams["font.sans-serif"] = ([cjk_name] if cjk_name else []) + base + list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["axes.unicode_minus"] = False
        if cjk_name is None:
            warn("没有找到中文字体，图里的中文会显示成方块。安装 fonts-noto-cjk 或 wqy-microhei 后重跑即可。")
    return plt


def _quiet_glyph_warnings():
    """Filter only the known glyph warning, and only when a CJK font is in use."""
    manager = warnings.catch_warnings()
    manager.__enter__()
    if _state.get("cjk"):
        warnings.filterwarnings("ignore", message=r"Glyph \d+ .*missing from font", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"Tight layout not applied", category=UserWarning)
    return manager


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
    _plt()
    manager = _quiet_glyph_warnings()
    try:
        for name, drawer in jobs:
            path = folder / name
            try:
                if drawer(report, path):
                    written.append(name)
            except Exception as exc:
                warn(f"图 {name} 没有画出来：{exc}")
    finally:
        manager.__exit__(None, None, None)
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
    ax.set_title("混淆矩阵（只含等级内预测）")
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
    ax.set_ylabel("等级内条件概率")
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
    ax.set_title(f"最终读出归因 · {group['key']}")
    ax.set_xlabel("推向原预测（正）或金标（负）")
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
    ax.set_xlabel("||ΔW|| / ||W_base||（范数大不等于根因）")
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
