from __future__ import annotations

import argparse
import json

from dcas.ontology import OntologyStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="ontology state json path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_state = sub.add_parser("state")

    ap_concept = sub.add_parser("add-concept")
    ap_concept.add_argument("--name", required=True)
    ap_concept.add_argument("--description", default="")
    ap_concept.add_argument("--parent_id", default=None)
    ap_concept.add_argument("--aliases", default="", help="comma-separated aliases")

    ap_rel = sub.add_parser("add-relation")
    ap_rel.add_argument("--source_id", required=True)
    ap_rel.add_argument("--target_id", required=True)
    ap_rel.add_argument("--relation_type", required=True)
    ap_rel.add_argument("--weight", type=float, default=1.0)

    ap_ann = sub.add_parser("add-annotation")
    ap_ann.add_argument("--track_id", required=True)
    ap_ann.add_argument("--concept_id", required=True)
    ap_ann.add_argument("--confidence", type=float, default=1.0)
    ap_ann.add_argument("--source", default="expert")
    ap_ann.add_argument("--rationale", default="")

    ap_sug = sub.add_parser("suggest")
    ap_sug.add_argument("--query", required=True)
    ap_sug.add_argument("--top_k", type=int, default=5)

    args = ap.parse_args()
    store = OntologyStore(args.state)

    if args.cmd == "state":
        print(json.dumps(store.state(), ensure_ascii=False))
        return
    if args.cmd == "add-concept":
        aliases = [x.strip() for x in args.aliases.split(",") if x.strip()]
        out = store.add_concept(
            name=args.name,
            description=args.description,
            parent_id=args.parent_id,
            aliases=aliases,
        )
        print(json.dumps(out, ensure_ascii=False))
        return
    if args.cmd == "add-relation":
        out = store.add_relation(
            source_id=args.source_id,
            target_id=args.target_id,
            relation_type=args.relation_type,
            weight=args.weight,
        )
        print(json.dumps(out, ensure_ascii=False))
        return
    if args.cmd == "add-annotation":
        out = store.add_annotation(
            track_id=args.track_id,
            concept_id=args.concept_id,
            confidence=args.confidence,
            source=args.source,
            rationale=args.rationale,
        )
        print(json.dumps(out, ensure_ascii=False))
        return
    if args.cmd == "suggest":
        out = {"items": store.suggest_concepts(query=args.query, top_k=args.top_k)}
        print(json.dumps(out, ensure_ascii=False))
        return

    raise RuntimeError(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

