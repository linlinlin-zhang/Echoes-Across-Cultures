from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dcas.ontology import OntologyStore
from dcas.pipelines import build_tracks_with_culturemert, generate_toy, pal_tasks, recommend, style_transfer, train_model

from .paths import Storage
from .schemas import (
    DatasetBuildRequest,
    OntologyAnnotationCreateRequest,
    OntologyConceptCreateRequest,
    OntologyExportConstraintsRequest,
    OntologyRelationCreateRequest,
    OntologySuggestRequest,
    PalRequest,
    RecommendRequest,
    StyleTransferRequest,
    ToyGenerateRequest,
    TrainRequest,
)


def create_app() -> FastAPI:
    app = FastAPI(title="DCAS API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    storage = Storage(root=Path("storage"))
    storage.ensure_dir("datasets")
    storage.ensure_dir("models")
    storage.ensure_dir("uploads")
    storage.ensure_dir("pal")
    storage.ensure_dir("style")
    storage.ensure_dir("ontology")
    ontology = OntologyStore(storage.resolve_rel("ontology/state.json"))

    @app.get("/api/health")
    def health():
        return {"ok": True, "time": time.time()}

    @app.get("/api/files")
    def list_files():
        root = storage.root.resolve()
        files: list[str] = []
        for p in root.rglob("*"):
            if p.is_file():
                files.append(storage.relpath(p))
        files.sort()
        return {"files": files}

    @app.get("/api/files/download")
    def download(path: str):
        try:
            p = storage.resolve_rel(path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(p))

    @app.post("/api/files/upload")
    async def upload(file: UploadFile = File(...), dir: str = "uploads"):
        try:
            target_dir = storage.ensure_dir(dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid dir")
        name = Path(file.filename or "file.bin").name
        dest = (target_dir / name).resolve()
        if storage.root.resolve() not in dest.parents:
            raise HTTPException(status_code=400, detail="invalid filename")
        content = await file.read()
        dest.write_bytes(content)
        return {"path": storage.relpath(dest), "size": int(len(content))}

    @app.post("/api/toy/generate")
    def api_generate_toy(req: ToyGenerateRequest):
        dataset_dir = storage.ensure_dir(f"datasets/{req.name}")
        out = generate_toy(out_dir=dataset_dir, n_tracks=req.n_tracks, dim=req.dim, seed=req.seed)
        return {
            "dir": storage.relpath(Path(out["dir"])),
            "tracks": storage.relpath(Path(out["tracks"])),
            "interactions": storage.relpath(Path(out["interactions"])),
            "meta": storage.relpath(Path(out["meta"])),
        }

    @app.post("/api/dataset/build_from_audio")
    def api_build_dataset(req: DatasetBuildRequest):
        try:
            metadata_path = storage.resolve_rel(req.metadata_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="metadata not found")
        out_path = storage.resolve_rel(f"datasets/{Path(req.out_name).name}")
        result = build_tracks_with_culturemert(
            metadata_csv=str(metadata_path),
            out_tracks_path=str(out_path),
            model_id=req.model_id,
            device=req.device,
            pooling=req.pooling,
            max_seconds=req.max_seconds,
            limit=req.limit,
            skip_errors=req.skip_errors,
        )
        result["out"] = storage.relpath(Path(str(result["out"])))
        if "manifest" in result:
            result["manifest"] = storage.relpath(Path(str(result["manifest"])))
        return result

    @app.post("/api/train")
    def api_train(req: TrainRequest):
        try:
            tracks_path = storage.resolve_rel(req.tracks_path)
            constraints_path = storage.resolve_rel(req.constraints_path) if req.constraints_path else None
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")
        if constraints_path is not None and not constraints_path.exists():
            raise HTTPException(status_code=404, detail="constraints not found")

        out_path = storage.resolve_rel(f"models/{Path(req.out_name).name}")
        result = train_model(
            tracks_path=str(tracks_path),
            out_path=str(out_path),
            constraints_path=str(constraints_path) if constraints_path else None,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            seed=req.seed,
            prefer_cuda=req.prefer_cuda,
            lambda_constraints=req.lambda_constraints,
            constraint_margin=req.constraint_margin,
        )
        result["checkpoint"] = storage.relpath(Path(result["checkpoint"]))
        return result

    @app.post("/api/recommend")
    def api_recommend(req: RecommendRequest):
        try:
            model_path = storage.resolve_rel(req.model_path)
            tracks_path = storage.resolve_rel(req.tracks_path)
            interactions_path = storage.resolve_rel(req.interactions_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")
        if not interactions_path.exists():
            raise HTTPException(status_code=404, detail="interactions not found")

        return recommend(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            interactions_path=str(interactions_path),
            user_id=req.user_id,
            target_culture=req.target_culture,
            k=req.k,
            prefer_cuda=req.prefer_cuda,
            epsilon=req.epsilon,
            iters=req.iters,
        )

    @app.post("/api/style/transfer")
    def api_style_transfer(req: StyleTransferRequest):
        try:
            model_path = storage.resolve_rel(req.model_path)
            tracks_path = storage.resolve_rel(req.tracks_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")

        out_path = storage.resolve_rel(f"style/{Path(req.out_name).name}")
        out = style_transfer(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            source_track_id=req.source_track_id,
            style_track_id=req.style_track_id,
            out_path=str(out_path),
            target_culture=req.target_culture,
            alpha=req.alpha,
            k=req.k,
            prefer_cuda=req.prefer_cuda,
        )
        out["artifact"] = storage.relpath(Path(str(out["artifact"])))
        return out

    @app.post("/api/pal")
    def api_pal(req: PalRequest):
        try:
            model_path = storage.resolve_rel(req.model_path)
            tracks_path = storage.resolve_rel(req.tracks_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")
        out_path = storage.resolve_rel(f"pal/{Path(req.out_name).name}")
        result = pal_tasks(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            out_path=str(out_path),
            n=req.n,
            prefer_cuda=req.prefer_cuda,
        )
        result["tasks"] = storage.relpath(Path(result["tasks"]))
        return result

    @app.get("/api/ontology/state")
    def api_ontology_state():
        return ontology.state()

    @app.post("/api/ontology/concepts")
    def api_ontology_add_concept(req: OntologyConceptCreateRequest):
        try:
            obj = ontology.add_concept(
                name=req.name,
                description=req.description,
                parent_id=req.parent_id,
                aliases=req.aliases,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obj

    @app.post("/api/ontology/relations")
    def api_ontology_add_relation(req: OntologyRelationCreateRequest):
        try:
            obj = ontology.add_relation(
                source_id=req.source_id,
                target_id=req.target_id,
                relation_type=req.relation_type,
                weight=req.weight,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obj

    @app.post("/api/ontology/annotations")
    def api_ontology_add_annotation(req: OntologyAnnotationCreateRequest):
        try:
            obj = ontology.add_annotation(
                track_id=req.track_id,
                concept_id=req.concept_id,
                confidence=req.confidence,
                source=req.source,
                rationale=req.rationale,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obj

    @app.post("/api/ontology/suggest")
    def api_ontology_suggest(req: OntologySuggestRequest):
        return {"items": ontology.suggest_concepts(query=req.query, top_k=req.top_k)}

    @app.post("/api/ontology/export_constraints")
    def api_ontology_export_constraints(req: OntologyExportConstraintsRequest):
        out_path = storage.resolve_rel(f"ontology/{Path(req.out_name).name}")
        result = ontology.save_pairwise_constraints(
            path=out_path,
            min_confidence=req.min_confidence,
            max_pairs_per_concept=req.max_pairs_per_concept,
        )
        result["path"] = storage.relpath(Path(str(result["path"])))
        return result

    dist = Path("web/dist")
    dist.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


app = create_app()
