"""Stage cache so a rerun of the same data and models skips finished work."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rba.util import to_builtin


class RunCache:
    def __init__(self, root: Path, fingerprint: str, resume: bool):
        self.root = root
        self.cache_dir = root / "cache"
        self.fingerprint = fingerprint
        self.resume = resume
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def has(self, stage: str) -> bool:
        if not self.resume:
            return False
        meta = self.manifest.get(stage)
        if not meta or meta.get("fingerprint") != self.fingerprint:
            return False
        return Path(meta["path"]).exists()

    def load_json(self, stage: str):
        path = Path(self.manifest[stage]["path"])
        return json.loads(path.read_text(encoding="utf-8"))

    def load_torch(self, stage: str):
        path = Path(self.manifest[stage]["path"])
        return torch.load(path, map_location="cpu", weights_only=False)

    def save_json(self, stage: str, payload) -> None:
        path = self.cache_dir / f"{stage}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(to_builtin(payload), ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
        self.manifest[stage] = {"fingerprint": self.fingerprint, "path": str(path), "format": "json"}
        self._save_manifest()

    def save_torch(self, stage: str, payload) -> None:
        path = self.cache_dir / f"{stage}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        self.manifest[stage] = {"fingerprint": self.fingerprint, "path": str(path), "format": "pt"}
        self._save_manifest()

    def clear(self) -> None:
        if self.cache_dir.exists():
            for child in self.cache_dir.iterdir():
                if child.is_file():
                    child.unlink()
        self.manifest = {}
        self._save_manifest()
