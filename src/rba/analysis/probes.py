"""Optional linear probes from the decision-position residual to user attributes.

These are descriptive. A high train R² on a handful of badcases does not mean
the layer causes the attribute. When there are enough rows we also score a
held-out half and say so.
"""

from __future__ import annotations

import numpy as np


def _numeric(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def probe_attributes(records: list[dict], hidden: np.ndarray, attributes: list[str] | None) -> dict:
    """hidden: [n, layers, hidden]. records aligned with axis 0."""
    if hidden.size == 0 or len(records) == 0:
        return {"skipped": "没有可用于探针的隐状态。", "layers": []}
    keys = attributes
    if keys is None:
        counts: dict[str, int] = {}
        for record in records:
            for key in (record.get("attributes") or {}):
                counts[key] = counts.get(key, 0) + 1
        keys = [key for key, count in counts.items() if count >= max(4, len(records) // 5)]
    if not keys:
        return {"skipped": "样本没有可用的 attributes，已跳过探针。要启用就在 JSONL 里加 attributes 对象，或在 CSV 里加 attr_ 列。", "layers": []}
    report = {"skipped": None, "layers": [], "n": int(hidden.shape[0]), "note": "探针只说明残差和属性是否线性相关，不是因果。"}
    n_layers = hidden.shape[1]
    for layer in range(n_layers):
        layer_rows = []
        features = hidden[:, layer, :]
        for key in keys:
            result = _probe_one(features, records, key)
            if result is not None:
                layer_rows.append(result)
        if layer_rows:
            report["layers"].append({"layer": layer, "attributes": layer_rows})
    if not report["layers"]:
        report["skipped"] = "属性列没有变化，或样本太少（至少 4 条同时带这个属性），探针没有拟合。"
    return report


def _probe_one(features: np.ndarray, records: list[dict], key: str) -> dict | None:
    paired = []
    for row, record in enumerate(records):
        if key not in (record.get("attributes") or {}):
            continue
        paired.append((row, record["attributes"][key]))
    if len(paired) < 4:
        return None
    numeric = [_numeric(value) for _, value in paired]
    if all(value is not None for value in numeric):
        target = np.asarray(numeric, dtype=np.float64)
        if float(target.std()) < 1e-8:
            return None
        rows = np.stack([features[row] for row, _ in paired], axis=0)
        return {"attribute": key, "kind": "numeric", **_ridge_split(rows, target)}
    labels = [str(value) for _, value in paired]
    unique = sorted(set(labels))
    if len(unique) < 2 or len(unique) > 8:
        return None
    mapping = {label: index for index, label in enumerate(unique)}
    target = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    rows = np.stack([features[row] for row, _ in paired], axis=0)
    scored = _classify_split(rows, target)
    scored.update({"attribute": key, "kind": "categorical", "classes": unique})
    return scored


def _ridge_fit(features: np.ndarray, target: np.ndarray, lam: float):
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    gram = centered.T @ centered
    gram = gram + lam * np.eye(gram.shape[0])
    weight = np.linalg.solve(gram, centered.T @ (target - target.mean()))
    return weight, mean, float(target.mean())


def _predict(features, weight, mean, bias):
    return (features - mean) @ weight + bias


def _r2(pred, target) -> float:
    ss_tot = float(((target - target.mean()) ** 2).sum())
    ss_res = float(((target - pred) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _ridge_split(features: np.ndarray, target: np.ndarray) -> dict:
    n_rows, dim = features.shape
    lam = 10.0 if n_rows < dim else 1.0
    if n_rows < 8:
        weight, mean, bias = _ridge_fit(features, target, lam)
        pred = _predict(features, weight, mean, bias)
        return {"r2_train": _r2(pred, target), "r2_heldout": None, "n": int(n_rows), "underdetermined": n_rows < dim}
    perm = np.random.default_rng(0).permutation(n_rows)
    cut = n_rows // 2
    train, test = perm[:cut], perm[cut:]
    weight, mean, bias = _ridge_fit(features[train], target[train], lam)
    return {
        "r2_train": _r2(_predict(features[train], weight, mean, bias), target[train]),
        "r2_heldout": _r2(_predict(features[test], weight, mean, bias), target[test]),
        "n": int(n_rows),
        "underdetermined": n_rows < dim,
    }


def _classify_split(features: np.ndarray, target: np.ndarray) -> dict:
    classes = int(target.max()) + 1
    eye = np.eye(classes)[target]
    n_rows, dim = features.shape
    lam = 10.0 if n_rows < dim else 1.0

    def fit_predict(train_x, train_y, eval_x):
        weight_parts = []
        mean = train_x.mean(axis=0, keepdims=True)
        centered = train_x - mean
        gram = centered.T @ centered + lam * np.eye(centered.shape[1])
        for column in range(train_y.shape[1]):
            weight_parts.append(np.linalg.solve(gram, centered.T @ (train_y[:, column] - train_y[:, column].mean())))
        weight = np.stack(weight_parts, axis=1)
        scores = (eval_x - mean) @ weight
        return scores.argmax(axis=1)

    train_acc = float((fit_predict(features, eye, features) == target).mean())
    heldout = None
    if n_rows >= 8:
        perm = np.random.default_rng(0).permutation(n_rows)
        cut = n_rows // 2
        pred = fit_predict(features[perm[:cut]], eye[perm[:cut]], features[perm[cut:]])
        heldout = float((pred == target[perm[cut:]]).mean())
    return {"accuracy_train": train_acc, "accuracy_heldout": heldout, "n": int(n_rows), "underdetermined": n_rows < dim}
