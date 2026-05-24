from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dcas.ontology import OntologyStore
from dcas.pipelines import (
    build_tracks_with_culturemert,
    generate_toy,
    pal_tasks,
    recommend,
    style_transfer,
    style_transfer_waveform,
    train_model,
)

from .mainline_platform import MainlineWeights, get_mainline_platform
from .paths import Storage
from .prototype_api import create_prototype_router
from .schemas import (
    DatasetBuildRequest,
    KimiChatRequest,
    MainlineRecommendRequest,
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
    WaveStyleTransferRequest,
)


def create_app() -> FastAPI:
    app = FastAPI(title="DCAS API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
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
    storage.ensure_dir("prototype")
    ontology = OntologyStore(storage.resolve_rel("ontology/state.json"))
    app.include_router(create_prototype_router(storage))

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
            lambda_domain=req.lambda_domain,
            lambda_contrast=req.lambda_contrast,
            lambda_cov=req.lambda_cov,
            lambda_tc=req.lambda_tc,
            lambda_hsic=req.lambda_hsic,
            beta_kl=req.beta_kl,
            shared_encoder=req.shared_encoder,
            regularizer_warmup_epochs=req.regularizer_warmup_epochs,
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

    @app.get("/api/mainline/status")
    def api_mainline_status(prefer_cuda: bool = False):
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            return platform.status()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/cultures")
    def api_mainline_cultures(prefer_cuda: bool = False):
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            return platform.cultures()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/catalog")
    def api_mainline_catalog(
        culture: str | None = None,
        source_dataset: str | None = None,
        q: str | None = None,
        limit: int = 24,
        random_seed: int | None = 42,
        exclude_low_signal: bool = True,
        prefer_cuda: bool = False,
    ):
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            return platform.catalog(
                culture=culture,
                source_dataset=source_dataset,
                q=q,
                limit=limit,
                random_seed=random_seed,
                exclude_low_signal=exclude_low_signal,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/random")
    def api_mainline_random(
        culture: str | None = None,
        source_dataset: str | None = None,
        random_seed: int | None = 42,
        exclude_low_signal: bool = True,
        prefer_cuda: bool = False,
    ):
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            return platform.random_track(
                culture=culture,
                source_dataset=source_dataset,
                random_seed=random_seed,
                exclude_low_signal=exclude_low_signal,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/tracks/{track_id}")
    def api_mainline_track(track_id: str, prefer_cuda: bool = False):
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            return platform.track(track_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/audio/{track_id}")
    def api_mainline_audio(track_id: str, prefer_cuda: bool = False):
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            path, media_type = platform.audio_file(track_id)
            return FileResponse(str(path), media_type=media_type, headers={"Accept-Ranges": "bytes"})
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/mainline/recommend")
    def api_mainline_recommend(req: MainlineRecommendRequest):
        seed_track_ids = list(req.seed_track_ids)
        if req.seed_track_id:
            seed_track_ids.insert(0, req.seed_track_id)
        weights = MainlineWeights(
            relevance=req.relevance_weight,
            novelty=req.novelty_weight,
            target_affinity=req.target_affinity_weight,
            minority=req.minority_weight,
            source=req.source_weight,
            diversity_lambda=req.diversity_lambda,
        )
        try:
            platform = get_mainline_platform(storage, prefer_cuda=req.prefer_cuda)
            return platform.recommend(
                seed_track_ids=seed_track_ids,
                seed_culture=req.seed_culture,
                target_culture=req.target_culture,
                mode=req.mode,
                k=req.k,
                recall_k=req.recall_k,
                random_seed=req.random_seed,
                exclude_same_artist=req.exclude_same_artist,
                exclude_low_signal=req.exclude_low_signal,
                weights=weights,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/mainline/upload_recommend")
    async def api_mainline_upload_recommend(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        artist: str | None = Form(default=None),
        seed_culture: str | None = Form(default=None),
        target_culture: str | None = Form(default=None),
        mode: str = Form(default="open"),
        k: int = Form(default=10),
        recall_k: int = Form(default=900),
        random_seed: int | None = Form(default=42),
        prefer_cuda: bool = Form(default=False),
        exclude_same_artist: bool = Form(default=False),
        exclude_low_signal: bool = Form(default=True),
        max_seconds: float = Form(default=30.0),
        window_count: int = Form(default=1),
        window_strategy: str = Form(default="single"),
        window_aggregate: str = Form(default="mean"),
    ):
        filename = Path(file.filename or "uploaded_audio").name
        suffix = Path(filename).suffix.lower()[:12]
        saved_name = f"{uuid4().hex}_{Path(filename).stem[:80] or 'audio'}{suffix}"
        upload_dir = storage.ensure_dir("uploads/mainline")
        dest = (upload_dir / saved_name).resolve()
        max_bytes = 200 * 1024 * 1024
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="uploaded audio is larger than 200MB")
        dest.write_bytes(content)

        weights = MainlineWeights()
        rel_upload_path = storage.relpath(dest)
        upload_info = {
            "track_id": f"upload_{dest.stem[:48]}",
            "filename": filename,
            "title": title or Path(filename).stem,
            "artist": artist or "Uploaded audio",
            "path": rel_upload_path,
            "size_bytes": int(len(content)),
            "content_type": file.content_type or "",
            "audio_api_url": f"/api/files/download?path={rel_upload_path}",
        }
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            result = platform.recommend_audio_file(
                audio_path=dest,
                upload_info=upload_info,
                seed_culture=seed_culture,
                target_culture=target_culture,
                mode=mode,
                k=k,
                recall_k=recall_k,
                random_seed=random_seed,
                exclude_same_artist=exclude_same_artist,
                exclude_low_signal=exclude_low_signal,
                weights=weights,
                max_seconds=max_seconds,
                window_count=window_count,
                window_strategy=window_strategy,
                window_aggregate=window_aggregate,
            )
            return result
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/ai/kimi/chat")
    def api_kimi_chat(req: KimiChatRequest):
        endpoint = req.endpoint.strip()
        if not endpoint.startswith("https://"):
            raise HTTPException(status_code=400, detail="Kimi endpoint must use https")
        payload = {
            "model": req.model.strip() or "kimi-k2.6",
            "messages": req.messages,
            "max_completion_tokens": int(req.max_completion_tokens),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            import urllib.error
            import urllib.request

            upstream = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {req.api_key.strip()}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(upstream, timeout=float(req.timeout_seconds)) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:4000]
            raise HTTPException(status_code=e.code, detail=detail)
        except urllib.error.URLError as e:
            raise HTTPException(status_code=502, detail=str(e.reason))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Kimi returned invalid JSON")
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise HTTPException(status_code=502, detail="Kimi returned no message content")
        return {"ok": True, "content": content, "raw": data}

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

    @app.post("/api/style/transfer_waveform")
    def api_style_transfer_waveform(req: WaveStyleTransferRequest):
        try:
            source_audio = storage.resolve_rel(req.source_audio_path)
            style_audio = storage.resolve_rel(req.style_audio_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not source_audio.exists():
            raise HTTPException(status_code=404, detail="source audio not found")
        if not style_audio.exists():
            raise HTTPException(status_code=404, detail="style audio not found")

        out_path = storage.resolve_rel(f"style/{Path(req.out_name).name}")
        out = style_transfer_waveform(
            source_audio_path=str(source_audio),
            style_audio_path=str(style_audio),
            out_wav_path=str(out_path),
            alpha=req.alpha,
            target_sr=req.target_sr,
            n_fft=req.n_fft,
            hop_length=req.hop_length,
            win_length=req.win_length,
            max_seconds=req.max_seconds,
            peak_norm=req.peak_norm,
        )
        out["artifact"] = storage.relpath(Path(str(out["artifact"])))
        out["source_audio_path"] = storage.relpath(source_audio)
        out["style_audio_path"] = storage.relpath(style_audio)
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

    # --- PAL Annotation Web UI endpoints ---

    @app.get("/api/pal/tasks")
    def get_pal_tasks():
        """Load enriched PAL tasks JSON."""
        tasks_path = Path("storage/pal/v4_main_annotation/pal_tasks.json")
        if not tasks_path.exists():
            raise HTTPException(status_code=404, detail="PAL tasks not found. Run pal_tasks first.")
        with open(tasks_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    @app.get("/api/pal/audio")
    def stream_audio(path: str):
        """Stream an audio file by absolute or relative path."""
        p = Path(path)
        if not p.is_absolute():
            p = Path("E:/Desktop/Echo") / p
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail=f"audio not found: {path}")
        ext = p.suffix.lower()
        media_type = "audio/mpeg" if ext == ".mp3" else "audio/wav" if ext == ".wav" else "audio/octet-stream"
        return FileResponse(str(p), media_type=media_type, headers={"Accept-Ranges": "bytes"})

    @app.post("/api/pal/annotate")
    def save_annotation(ann: dict):
        """Save a single annotation to JSONL file."""
        out_dir = Path("storage/pal/v4_main_annotation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "annotations.jsonl"
        record = {
            "task_id": ann.get("task_id", ""),
            "track_id_a": ann.get("track_id_a", ""),
            "track_id_b": ann.get("track_id_b", ""),
            "judgment": ann.get("judgment", ""),  # "a", "b", or "neither"
            "rationale": ann.get("rationale", ""),
            "annotator": ann.get("annotator", "web"),
            "timestamp": time.time(),
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "saved": int(out_file.stat().st_size > 0)}

    @app.get("/api/pal/progress")
    def get_pal_progress():
        """Return how many annotations have been saved."""
        out_file = Path("storage/pal/v4_main_annotation/annotations.jsonl")
        if not out_file.exists():
            return {"count": 0, "total": 60}
        with open(out_file, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        return {"count": count, "total": 60}

    @app.post("/api/pal/export")
    def export_annotations():
        """Export annotations as CSV for constraint building."""
        out_file = Path("storage/pal/v4_main_annotation/annotations.jsonl")
        csv_file = Path("storage/pal/v4_main_annotation/annotated.csv")
        if not out_file.exists():
            raise HTTPException(status_code=404, detail="no annotations yet")

        import csv as _csv
        # Load annotations
        anns = []
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    anns.append(json.loads(line))

        # Load tasks for enrichment
        tasks_map = {}
        tasks_json = Path("storage/pal/v4_main_annotation/pal_tasks.json")
        if tasks_json.exists():
            with open(tasks_json, "r", encoding="utf-8") as f:
                for t in json.load(f):
                    tasks_map[(t["track_id_a"], t["track_id_b"])] = t

        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["task_id", "track_id_a", "track_id_b", "judgment", "similar", "rationale", "annotator"])
            for a in anns:
                judgment = a.get("judgment", "")
                similar = "yes" if judgment in ("a", "b") else ("no" if judgment == "neither" else "")
                writer.writerow([
                    a["task_id"], a["track_id_a"], a["track_id_b"],
                    judgment, similar, a.get("rationale", ""), a.get("annotator", ""),
                ])
        return {"csv": str(csv_file), "count": len(anns)}

    web_dir = Path("web")
    dist = Path("web/dist")
    prototype_dir = Path("web_prototype")
    if prototype_dir.exists():
        app.mount("/prototype", StaticFiles(directory=str(prototype_dir), html=True), name="prototype")
    # PAL annotation UI
    pal_html = dist / "pal.html"
    if pal_html.exists():
        app.mount("/pal", StaticFiles(directory=str(pal_html.parent), html=False), name="pal")
    if (web_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    elif (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
    elif prototype_dir.exists():
        app.mount("/", StaticFiles(directory=str(prototype_dir), html=True), name="prototype-root")
    else:
        dist.mkdir(parents=True, exist_ok=True)
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web-empty")

    return app


app = create_app()
