"""FastAPI application.

Endpoints (all under /api):
    GET  /health           live Gemma 4 / Ollama status (surfaced, never hidden)
    GET  /stats            collection statistics
    POST /upload           upload one media file (multipart) -> detected kind
    POST /ingest           Server-Sent-Events stream of live ingestion progress
    POST /search           cross-modal search
    POST /chat             temporal RAG follow-up
    GET  /media            stream a media file (HTTP range -> video seeking)
    GET  /thumbnail        serve a keyframe thumbnail

The frontend (Next.js) calls these. Ingest/search/chat refuse to run on a dead
backend (HTTP 503) instead of silently degrading, so the UI can show a real error.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from deepsearch.api import schemas
from deepsearch.config import PROJECT_ROOT, get_settings
from deepsearch.ingestion import IngestionPipeline
from deepsearch.ingestion.media_probe import media_kind
from deepsearch.logging_utils import get_logger
from deepsearch.models import get_model_client
from deepsearch.query import get_engine
from deepsearch.rag import get_rag
from deepsearch.vectorstore.store import get_store

log = get_logger(__name__)

# Only one ingest runs at a time (GPU/CPU guard for concurrent long-video uploads).
_INGEST_SEM = threading.Semaphore(1)

MEDIA_DIR = (PROJECT_ROOT / "data" / "media").resolve()
CACHE_DIR = (PROJECT_ROOT / "data" / "cache").resolve()
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Deep Search in Media Library", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_live_backend() -> None:
    """Raise 503 if the real Gemma 4 backend is not usable (unless stub mode)."""
    cfg = get_settings()
    if cfg.runtime.use_stub:
        return
    health = get_model_client().health()
    if not health["ok"]:
        raise HTTPException(status_code=503, detail=health["detail"] or "Gemma 4 backend unavailable.")


def _safe_media_path(raw: str) -> Path:
    """Resolve a media/thumbnail path, refusing anything outside the allowed dirs."""
    p = Path(raw).resolve()
    if not (str(p).startswith(str(MEDIA_DIR)) or str(p).startswith(str(CACHE_DIR))):
        raise HTTPException(status_code=403, detail="Path outside media directory.")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return p


def _media_url(file_path: str) -> str:
    return f"/api/media?path={quote(file_path)}"


def _thumb_url(thumb: str | None) -> str | None:
    return f"/api/thumbnail?path={quote(thumb)}" if thumb else None


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    return schemas.HealthResponse(**get_model_client().health())


@app.get("/api/stats", response_model=schemas.StatsResponse)
def stats() -> schemas.StatsResponse:
    try:
        return schemas.StatsResponse(**get_store().stats())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Index unavailable: {exc}") from exc


@app.get("/api/assets", response_model=list[schemas.AssetOut])
def assets() -> list[schemas.AssetOut]:
    out = []
    for a in get_store().list_assets():
        out.append(
            schemas.AssetOut(
                asset_id=a["asset_id"], asset_name=a["asset_name"], file_path=a["file_path"],
                media_url=_media_url(a["file_path"]), modalities=a["modalities"],
                has_audio=a["has_audio"],
            )
        )
    return sorted(out, key=lambda x: x.asset_name.lower())


@app.delete("/api/asset/{asset_id}", response_model=schemas.DeleteResponse)
def delete_asset(asset_id: str) -> schemas.DeleteResponse:
    """Remove an asset's vectors, its media file, and its keyframe cache."""
    store = get_store()
    asset = next((a for a in store.list_assets() if a["asset_id"] == asset_id), None)
    store.delete_asset(asset_id)
    if asset:
        try:
            fp = Path(asset["file_path"]).resolve()
            if str(fp).startswith(str(MEDIA_DIR)) and fp.is_file():
                fp.unlink()
        except Exception as exc:  # pragma: no cover
            log.warning(f"Could not delete media file: {exc}")
    cache = (CACHE_DIR / asset_id)
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)
    return schemas.DeleteResponse(ok=True, asset_id=asset_id)


# ─────────────────────────────────────────────────────────────────────────────
# Upload + ingest
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/upload", response_model=schemas.UploadResponse)
async def upload(file: UploadFile = File(...)) -> schemas.UploadResponse:
    name = Path(file.filename or "upload").name
    dst = MEDIA_DIR / name
    data = await file.read()
    dst.write_bytes(data)
    kind = media_kind(dst)
    return schemas.UploadResponse(
        filename=name, kind=kind, size_bytes=len(data), supported=kind != "unknown"
    )


@app.post("/api/ingest")
def ingest(req: schemas.IngestRequest) -> StreamingResponse:
    _require_live_backend()
    path = (MEDIA_DIR / Path(req.filename).name).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Uploaded file not found: {req.filename}")

    events: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def worker() -> None:
        try:
            # Serialize ingests — only one heavy media job touches the GPU at a
            # time, so uploading several long videos queues instead of thrashing.
            if not _INGEST_SEM.acquire(blocking=False):
                events.put({"stage": "audio", "message": "Queued — another file is indexing…"})
                _INGEST_SEM.acquire()
            try:
                pipe = IngestionPipeline()
                result = pipe.ingest_file(path, req.category, on_progress=events.put)
            finally:
                _INGEST_SEM.release()
            events.put({
                "stage": "result",
                "error": result.error,
                "fragments": result.fragments,
                "by_modality": result.by_modality,
                "asset_id": result.asset_id,
                "asset_name": result.asset_name,
            })
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Ingest worker failed")
            events.put({"stage": "error", "message": str(exc)})
        finally:
            events.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def sse() -> Iterator[str]:
        while True:
            ev = events.get()
            if ev is _SENTINEL:
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# Search + chat
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/search", response_model=schemas.SearchResponse)
def search(req: schemas.SearchRequest) -> schemas.SearchResponse:
    _require_live_backend()
    resp = get_engine().search(
        req.query,
        categories=req.categories,
        modalities=req.modalities,
        weights_override=req.weights,
        asset_recall=req.asset_recall,
    )
    fusion = "weighted" if req.weights else get_settings().retrieval.fusion
    hits = [
        schemas.Hit(
            point_id=h.point_id, asset_id=h.asset_id, asset_name=h.asset_name,
            file_path=h.file_path, media_url=_media_url(h.file_path),
            thumbnail_url=_thumb_url(h.thumbnail_path), modality=h.modality,
            text=h.text, start_s=h.start_s, end_s=h.end_s,
            timestamp_label=h.timestamp_label, score=h.score, contributions=h.contributions,
        )
        for h in resp.hits
    ]
    return schemas.SearchResponse(
        query=req.query, intent=resp.intent.intent, weights=resp.intent.weights,
        fusion=fusion, blocked=resp.blocked, message=resp.message, hits=hits,
    )


@app.post("/api/ask/stream")
def ask_stream(req: schemas.AskRequest) -> StreamingResponse:
    """Stream the grounded answer token-by-token (SSE), then a final sources event."""
    _require_live_backend()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    def sse() -> Iterator[str]:
        try:
            for kind, data in get_rag().stream(req.question, history):
                if kind == "token":
                    yield f"data: {json.dumps({'type': 'token', 'data': data})}\n\n"
                else:  # sources
                    payload = [
                        {
                            "asset_id": s.asset_id, "asset_name": s.asset_name,
                            "media_url": _media_url(s.file_path), "modality": s.modality,
                            "start_s": s.start_s, "timestamp_label": s.timestamp_label,
                        }
                        for s in data
                    ]
                    yield f"data: {json.dumps({'type': 'sources', 'data': payload})}\n\n"
        except Exception as exc:  # pragma: no cover
            log.exception("ask_stream failed")
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/ask", response_model=schemas.AskResponse)
def ask(req: schemas.AskRequest) -> schemas.AskResponse:
    """Library-wide retrieval-grounded answer with supporting sources."""
    _require_live_backend()
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = get_rag().answer(req.question, history)
    sources = [
        schemas.AskSource(
            asset_id=s.asset_id, asset_name=s.asset_name, media_url=_media_url(s.file_path),
            modality=s.modality, start_s=s.start_s, timestamp_label=s.timestamp_label,
        )
        for s in result["sources"]
    ]
    return schemas.AskResponse(answer=result["answer"], sources=sources)


# ─────────────────────────────────────────────────────────────────────────────
# Media serving (range-enabled FileResponse -> video/audio seeking)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/media")
def media(path: str) -> FileResponse:
    return FileResponse(_safe_media_path(path))


@app.get("/api/thumbnail")
def thumbnail(path: str) -> FileResponse:
    return FileResponse(_safe_media_path(path), media_type="image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
def run() -> None:
    """Console entry point: `deepsearch-api`."""
    import uvicorn

    uvicorn.run("deepsearch.api.main:app", host="127.0.0.1", port=8000, reload=False)
