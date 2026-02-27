from __future__ import annotations

import json
import itertools
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

    def export_pairwise_constraints(
        self,
        min_confidence: float = 0.5,
        max_pairs_per_concept: int = 200,
    ) -> list[dict[str, Any]]:
        with self._lock:
            state = self._load()
        concept_by_id = {c["id"]: c for c in state["concepts"]}
        by_concept: dict[str, dict[str, float]] = {}
        for a in state["annotations"]:
            conf = float(a.get("confidence", 0.0))
            if conf < float(min_confidence):
                continue
            cid = str(a.get("concept_id", "")).strip()
            tid = str(a.get("track_id", "")).strip()
            if not cid or not tid:
                continue
            by_concept.setdefault(cid, {})
            prev = by_concept[cid].get(tid, 0.0)
            by_concept[cid][tid] = max(prev, conf)

        out: list[dict[str, Any]] = []
        for cid, tracks_conf in by_concept.items():
            track_ids = sorted(tracks_conf.keys())
            pairs = itertools.combinations(track_ids, 2)
            count = 0
            for a, b in pairs:
                if count >= int(max_pairs_per_concept):
                    break
                c = concept_by_id.get(cid, {})
                rationale = f"shared ontology concept: {c.get('name', cid)}"
                out.append(
                    {
                        "track_id_a": a,
                        "track_id_b": b,
                        "similar": True,
                        "rationale": rationale,
                    }
                )
                count += 1
        return out

    def save_pairwise_constraints(
        self,
        path: str | Path,
        min_confidence: float = 0.5,
        max_pairs_per_concept: int = 200,
    ) -> dict[str, Any]:
        pairs = self.export_pairwise_constraints(
            min_confidence=min_confidence,
            max_pairs_per_concept=max_pairs_per_concept,
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for obj in pairs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return {
            "path": str(p),
            "count": int(len(pairs)),
        }
