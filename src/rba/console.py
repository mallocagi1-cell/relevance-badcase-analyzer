"""Chinese console output. Lines are kept so the run log can be written at the end."""

from __future__ import annotations

import sys

_lines: list[str] = []


def reset() -> None:
    _lines.clear()


def lines() -> list[str]:
    return list(_lines)


def info(message: str) -> None:
    _lines.append(message)
    print(message, flush=True)


def warn(message: str) -> None:
    text = f"注意：{message}"
    _lines.append(text)
    print(text, file=sys.stderr, flush=True)


def error(message: str, hint: str = "") -> None:
    print(f"错误：{message}", file=sys.stderr, flush=True)
    if hint:
        print(f"建议：{hint}", file=sys.stderr, flush=True)
