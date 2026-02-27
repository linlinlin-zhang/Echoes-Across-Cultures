from __future__ import annotations

import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dcas.serialization_json import read_json, write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_tokens(text: str) -> set[str]:
    toks = re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
    return {t for t in toks if t}


def _token_overlap_score(a: str, b: str) -> float:
    ta = _normalize_tokens(a)
    tb = _normalize_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter / max(1, union))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        "concepts": [],
        "relations": [],
        "annotations": [],
    }


class OntologyStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            write_json(self.path, _empty_state())

    def _load(self) -> dict[str, Any]:
        return read_json(self.path)

    def _save(self, state: dict[str, Any]) -> None:
        write_json(self.path, state)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._load()

    def add_concept(
        self,
        name: str,
        description: str | None = None,
        parent_id: str | None = None,
        aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        name = str(name).strip()
        if not name:
            raise ValueError("name is required")
        aliases = [str(a).strip() for a in (aliases or []) if str(a).strip()]
        with self._lock:
            state = self._load()
            if any(c["name"].lower() == name.lower() for c in state["concepts"]):
                raise ValueError(f"concept already exists: {name}")
            if parent_id is not None and not any(c["id"] == parent_id for c in state["concepts"]):
                raise ValueError("parent concept not found")
            obj = {
                "id": _new_id("c"),
                "name": name,
                "description": str(description).strip() if description is not None else "",
                "parent_id": parent_id,
                "aliases": aliases,
                "created_at": _now_iso(),
            }
            state["concepts"].append(obj)
            self._save(state)
            return obj

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        source_id = str(source_id).strip()
        target_id = str(target_id).strip()
        relation_type = str(relation_type).strip()
        if not source_id or not target_id or not relation_type:
            raise ValueError("source_id/target_id/relation_type are required")
        with self._lock:
            state = self._load()
            concept_ids = {c["id"] for c in state["concepts"]}
            if source_id not in concept_ids or target_id not in concept_ids:
                raise ValueError("source or target concept not found")
            obj = {
                "id": _new_id("r"),
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "weight": float(weight),
                "created_at": _now_iso(),
            }
            state["relations"].append(obj)
            self._save(state)
            return obj

    def add_annotation(
        self,
        track_id: str,
        concept_id: str,
        confidence: float = 1.0,
        source: str = "expert",
        rationale: str | None = None,
    ) -> dict[str, Any]:
        track_id = str(track_id).strip()
        concept_id = str(concept_id).strip()
        if not track_id or not concept_id:
            raise ValueError("track_id and concept_id are required")
        with self._lock:
            state = self._load()
            concept_ids = {c["id"] for c in state["concepts"]}
            if concept_id not in concept_ids:
                raise ValueError("concept not found")
            obj = {
                "id": _new_id("a"),
                "track_id": track_id,
                "concept_id": concept_id,
                "confidence": float(confidence),
                "source": str(source).strip() or "expert",
                "rationale": str(rationale).strip() if rationale is not None else "",
                "created_at": _now_iso(),
            }
            state["annotations"].append(obj)
            self._save(state)
            return obj

    def suggest_concepts(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q = str(query).strip()
        if not q:
            return []
        with self._lock:
            state = self._load()
        scored: list[tuple[float, dict[str, Any]]] = []
        for c in state["concepts"]:
            hay = " ".join([c.get("name", ""), c.get("description", ""), " ".join(c.get("aliases", []))])
            score = _token_overlap_score(q, hay)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda t: (-t[0], t[1]["name"]))
        out: list[dict[str, Any]] = []
        for score, c in scored[: int(top_k)]:
            item = dict(c)
            item["score"] = float(score)
            out.append(item)
        return out

