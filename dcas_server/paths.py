from __future__ import annotations

from pathlib import Path


class Storage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_rel(self, rel: str) -> Path:
        rel = rel.replace("\\", "/").lstrip("/")
        candidate = (self.root / rel).resolve()
        root = self.root.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError("invalid path")
        return candidate

    def relpath(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    def ensure_dir(self, rel: str) -> Path:
        p = self.resolve_rel(rel)
        p.mkdir(parents=True, exist_ok=True)
        return p

