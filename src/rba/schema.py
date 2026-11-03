"""Badcase / goodcase readers. JSONL or CSV, validated before any model load."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from rba.errors import RBAError

STANDARD_COLUMNS = {"id", "query", "doc", "document", "image", "image_path", "label", "prediction", "pred", "attributes", "note"}


@dataclass
class Case:
    id: str
    query: str
    doc: str
    image: str | None
    label: str
    prediction: str | None = None
    attributes: dict = field(default_factory=dict)
    note: str = ""
    source: str = ""


def _as_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_grade(value, grades: list[str], *, what: str, where: str) -> str:
    if value is None or _as_text(value) == "":
        raise RBAError(
            f"{where} 的 {what} 是空的。",
            f"填上等级。当前允许的等级：{', '.join(grades)}。",
        )
    text = _as_text(value)
    if text in grades:
        return text
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None and number == int(number):
        as_int = str(int(number))
        if as_int in grades:
            return as_int
    raise RBAError(
        f"{where} 的 {what}={value!r} 不在等级表里。",
        f"把 {what} 改成其中之一：{', '.join(grades)}。若微调用了别的记号，改配置里的 grades。",
    )


def _optional_grade(value, grades: list[str], where: str) -> str | None:
    if value is None or _as_text(value) == "":
        return None
    try:
        return normalize_grade(value, grades, what="prediction", where=where)
    except RBAError as exc:
        raise RBAError(
            exc.message,
            "prediction 可以留空，程序会重新推断。若填写，必须是等级字符串。",
        ) from exc


def _attributes_from_row(row: dict) -> dict:
    attrs = {}
    raw = row.get("attributes")
    if isinstance(raw, dict):
        attrs.update(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RBAError(
                "attributes 不是合法 JSON。",
                'JSONL 里写成 "attributes": {"domain": "food"}；CSV 里用 attr_ 前缀的列，例如 attr_domain。',
            ) from exc
        if not isinstance(parsed, dict):
            raise RBAError("attributes 必须是对象。", '例如 {"domain": "food", "has_image": 1}。')
        attrs.update(parsed)
    for key, value in row.items():
        if key.startswith("attr_") and _as_text(value) != "":
            attrs[key[len("attr_") :]] = _coerce_attr(value)
    return attrs


def _coerce_attr(value):
    text = _as_text(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if number == int(number) and "." not in text and "e" not in text.lower():
        return int(number)
    return number


def _case_from_mapping(row: dict, grades: list[str], where: str, index: int) -> Case:
    if not isinstance(row, dict):
        raise RBAError(
            f"{where} 第 {index} 条不是对象。",
            "JSONL 每行一个 JSON 对象。CSV 需要表头。",
        )
    query = _as_text(row.get("query"))
    doc = _as_text(row.get("doc", row.get("document")))
    image = _as_text(row.get("image", row.get("image_path"))) or None
    if not query:
        raise RBAError(
            f"{where} 第 {index} 条缺少 query。",
            "每条至少要有 query，以及 doc 或 image 之一。",
        )
    if not doc and not image:
        raise RBAError(
            f"{where} 第 {index} 条既没有 doc 也没有 image。",
            "文本文档放 doc，图片放 image（本地路径）。两者可以同时有。",
        )
    label = normalize_grade(row.get("label"), grades, what="label", where=f"{where} 第 {index} 条")
    prediction = _optional_grade(row.get("prediction", row.get("pred")), grades, f"{where} 第 {index} 条")
    case_id = _as_text(row.get("id")) or f"c{index:04d}"
    return Case(
        id=case_id,
        query=query,
        doc=doc,
        image=image,
        label=label,
        prediction=prediction,
        attributes=_attributes_from_row(row),
        note=_as_text(row.get("note")),
        source=where,
    )


def load_cases(path: str | Path, grades: list[str], *, what: str = "坏例文件") -> list[Case]:
    path = Path(path)
    if not path.exists():
        raise RBAError(
            f"找不到{what} {path}。",
            "确认路径。相对路径是相对于配置文件所在目录；命令行传入的路径相对于当前目录。示例：examples/badcases.jsonl。",
        )
    if path.suffix.lower() == ".csv":
        cases = _load_csv(path, grades, what)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        cases = _load_jsonl(path, grades, what)
    else:
        raise RBAError(
            f"{what} {path.name} 的扩展名不是 jsonl 或 csv。",
            "用 .jsonl（每行一个对象）或带表头的 .csv。",
        )
    if not cases:
        raise RBAError(f"{what}是空的。", "至少放一条带 query 和 label 的样本。")
    seen: dict[str, int] = {}
    base = path.parent
    for case in cases:
        seen[case.id] = seen.get(case.id, 0) + 1
        if seen[case.id] > 1:
            case.id = f"{case.id}#{seen[case.id]}"
        if case.image and not Path(case.image).is_absolute():
            beside = base / case.image
            if beside.exists():
                case.image = str(beside.resolve())
            elif Path(case.image).exists():
                case.image = str(Path(case.image).resolve())
    return cases


def _load_jsonl(path: Path, grades: list[str], what: str) -> list[Case]:
    cases = []
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" and text.lstrip().startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RBAError(f"{what} 不是合法 JSON：{exc}", "JSON 数组里每个元素是一条样本，或改用 JSONL。") from exc
        if not isinstance(rows, list):
            raise RBAError(f"{what} 的顶层不是数组。", "用 JSON 数组或 JSONL。")
        for index, row in enumerate(rows, start=1):
            cases.append(_case_from_mapping(row, grades, what, index))
        return cases
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RBAError(
                f"{what} 第 {line_no} 行不是合法 JSON：{exc}",
                "每行一个完整对象，行尾不要多逗号。对照 examples/badcases.jsonl。",
            ) from exc
        cases.append(_case_from_mapping(row, grades, f"{what} 第 {line_no} 行", line_no))
    return cases


def _load_csv(path: Path, grades: list[str], what: str) -> list[Case]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RBAError(f"{what} 没有表头。", "第一行写成 query,doc,image,label,prediction。")
        missing = {"query", "label"} - set(reader.fieldnames)
        has_doc = "doc" in reader.fieldnames or "document" in reader.fieldnames or "image" in reader.fieldnames or "image_path" in reader.fieldnames
        if missing or not has_doc:
            raise RBAError(
                f"{what} 缺少列：{', '.join(sorted(missing or {'doc 或 image'}))}。",
                "表头至少包含 query,label，以及 doc 或 image。可选 prediction、id、attr_ 开头的属性列。",
            )
        cases = []
        for index, row in enumerate(reader, start=2):
            cases.append(_case_from_mapping(row, grades, f"{what} 第 {index} 行", index))
        return cases
