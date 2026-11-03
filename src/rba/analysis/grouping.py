"""Confusion groups and a small k-means over grade probabilities."""

from __future__ import annotations

import numpy as np


def confusion(records: list[dict], grades: list[str]) -> dict[str, dict[str, int]]:
    table = {label: {pred: 0 for pred in grades} for label in grades}
    for record in records:
        label = record["label"]
        pred = record["pred"]
        if label not in table:
            table[label] = {grade: 0 for grade in grades}
        table[label][pred] = table[label].get(pred, 0) + 1
    return table


def group_errors(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record["label"] == record["pred"]:
            continue
        key = f"{record['label']}->{record['pred']}"
        groups.setdefault(key, []).append(record)
    return dict(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])))


def pick_detail(records: list[dict], limit: int, seed: int) -> list[dict]:
    if len(records) <= limit:
        return list(records)
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(records), size=limit, replace=False)
    return [records[int(index)] for index in sorted(indexes)]


def kmeans(features: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows = features.shape[0]
    k = max(1, min(k, n_rows))
    if k == 1:
        return np.zeros(n_rows, dtype=np.int64)
    centers = features[rng.choice(n_rows, size=k, replace=False)]
    labels = np.zeros(n_rows, dtype=np.int64)
    for _ in range(iters):
        distance = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = distance.argmin(axis=1)
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members) == 0:
                centers[cluster] = features[int(rng.integers(0, n_rows))]
            else:
                centers[cluster] = members.mean(axis=0)
    return labels


def cluster_group(records: list[dict], grades: list[str], k: int, seed: int, min_size: int = 6) -> list[dict] | None:
    if len(records) < min_size:
        return None
    rows = []
    for record in records:
        probs = [float(record["probs"].get(grade, 0.0)) for grade in grades]
        query_len = np.log1p(len(record.get("query") or ""))
        doc_len = np.log1p(len(record.get("doc") or ""))
        rows.append(probs + [query_len, doc_len, 1.0 if record.get("has_image") else 0.0])
    features = np.asarray(rows, dtype=np.float64)
    spread = features.std(axis=0).sum()
    if spread < 1e-6:
        return [{"cluster": 0, "size": len(records), "image_fraction": float(np.mean([r.get("has_image") for r in records]))}]
    labels = kmeans(features, min(k, len(records) // 2), seed)
    summaries = []
    for cluster in sorted(set(labels.tolist())):
        members = [record for record, label in zip(records, labels) if label == cluster]
        for record, label in zip(records, labels):
            if label == cluster:
                record["cluster"] = int(cluster)
        summaries.append(
            {
                "cluster": int(cluster),
                "size": len(members),
                "image_fraction": float(np.mean([1.0 if item.get("has_image") else 0.0 for item in members])),
            }
        )
    return summaries
