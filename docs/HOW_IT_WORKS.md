# How It Works — Operation, Usage & Troubleshooting

> Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (which covers *why* the system is built this
> way). This document is the practical walkthrough: how a file flows from disk to a searchable,
> chat-able, timestamp-accurate result, plus setup, usage, configuration and troubleshooting.

---

## 1. Setup

### Prerequisites
- macOS (Apple Silicon recommended) or Linux
- [`uv`](https://docs.astral.sh/uv/), [`ffmpeg`](https://ffmpeg.org/),
  [Ollama](https://ollama.com) ≥ 0.30

```bash
make setup     # uv venv (Python 3.12) + editable install (+ dev tools)
make model     # ollama pull gemma4:e4b && nomic-embed-text  (~10 GB for e4b)
make app       # Streamlit → http://localhost:8501
```

No GPU / models yet? Run the whole pipeline against the deterministic stub:

```bash
DEEPSEARCH_USE_STUB=true make app
```

---

## 2. The ingestion pipeline (offline)

For each file, `ingestion/pipeline.py:IngestionPipeline.ingest_file` emits **progress events**
(`{stage, message, current, total}`) consumed by the UI for a live status display.

1. **Probe & route** — `media_probe` inspects the extension and (via `ffprobe`) streams to
   classify the file as `video` / `audio` / `image` / `text`, and reads duration.
2. **Replace, not duplicate** — `store.delete_asset(asset_id)` removes any prior points for this
   file so a re-ingest replaces rather than doubles them.
3. **Video → scene keyframes** — `video.py` runs PySceneDetect's content detector to find true
   scene cuts (not fixed intervals), grabs each scene's middle frame with OpenCV, resizes, and
   writes a JPEG thumbnail tagged with its timestamp. Falls back to uniform sampling if needed.
4. **Audio → overlapping chunks** — `audio.py` extracts 16 kHz mono WAV with FFmpeg (optional
   `afftdn` noise reduction), then slices 30-second windows with a 5-second overlap so a phrase
   on a boundary survives.
5. **Captions / text** — `text.py` parses sidecar `.srt`/`.vtt` (timed) or plain `.txt`/`.json`
   (untimed) into segments.
6. **Describe-then-embed** — Gemma 4 describes each keyframe (vision) and transcribes each audio
   chunk (audio). Generated text is sanitized (`safety.sanitize_generated`) and embedded by
   `nomic-embed-text` into a 768-d L2-normalized vector. This is the linchpin that makes
   cross-modal text search work.
7. **Upsert** — `store.upsert_fragments` writes one Qdrant point per fragment (populating one
   named vector + payload with offsets). Then `store.upsert_asset_multivectors` writes one
   asset-level MaxSim point holding all frame/audio vectors. Embedding is batched
   (`ingestion.batch_size`) to bound memory.

## 3. The query pipeline (real-time)

`query/search.py:SearchEngine.search`:

1. **Safety** — `safety.check_query` rejects empty/oversized queries and prompt-injection
   patterns before any vector work.
2. **Intent → weights** — `intent.py` asks Gemma 4 to classify the query (visual / spoken /
   conceptual / mixed) and propose per-space weights, with a keyword heuristic fallback.
3. **Embed & prefetch** — the query is embedded into the unified space; the top `prefetch_limit`
   candidates are pulled from each weighted named space, optionally constrained by a Qdrant
   payload pre-filter (category / modality / recency). With `asset_recall=True`, MaxSim asset
   recall runs first and narrows the search to the winning assets.
4. **Fusion** — `fusion.py` merges per-space lists: **RRF** by default, or **weighted** when the
   UI provides manual weight overrides (so the sliders actually bite).
5. **Rerank** — `rerank.py` sends the top candidates' texts to Gemma 4 as a listwise prompt for a
   best-first ordering (falls back to fused order).
6. **Present** — each `SearchHit` carries `start_s`; the UI seeks the native player to that
   exact second.

## 4. Temporal RAG

`rag/chat.py` pulls all fragments for the selected asset, interleaves them into one
**time-ordered transcript** (VIDEO/AUDIO/TEXT tagged, timestamped), and feeds it into Gemma 4's
context window with a strict "answer only from this timeline" system prompt. That ordering lets
the model answer temporal follow-ups like *"what was said right after the blue car appeared?"*.

## 5. Using the app

The Streamlit UI (cream/amber theme) has three tabs:

- **Ingest** — one auto-detecting uploader (video/audio/image/text). After clicking **Ingest**,
  each file shows a live status card: progress bar + current step + streaming event log
  (`probe → keyframes → describing keyframe i/n → audio → transcribing i/n → embedding → MaxSim
  → done`) ending in per-modality counts. Nothing leaves your machine.
- **Search** — type a natural-language query. Leave *Auto weights* on for Gemma intent routing,
  or open *Manual weights & filters* to set vector weights (forces weighted fusion) and
  modality/category filters. Toggle *MaxSim recall* for two-stage retrieval. Pick a result to
  jump the player to the exact second; ask follow-ups in the grounded RAG chat.
- **Library** — collection stats (assets, fragments, modality breakdown).

### CLI

```bash
uv run deepsearch ingest ./data/media --category meetings   # index a file/folder
uv run deepsearch search "the chart with the budget spike"  # one-off search
uv run deepsearch stats                                      # collection stats
uv run deepsearch reset                                      # destroy & recreate the collection
uv run deepsearch serve                                      # launch the Streamlit app
```

## 6. Configuration

All settings live in [`config/default.yaml`](../config/default.yaml), overridable by environment
variables (see [`.env.example`](../.env.example)). Common ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEPSEARCH_GEMMA_MODEL` | `gemma4:e4b` | Multimodal model (`e2b`/`e4b`/`12b`/…) |
| `DEEPSEARCH_EMBED_MODEL` | `nomic-embed-text` | Unified-space embedder |
| `DEEPSEARCH_EMBED_STRATEGY` | `describe_then_embed` | or `native` (embed via Gemma 4) |
| `DEEPSEARCH_QDRANT_URL` | *(empty)* | Set to use a Qdrant server instead of local mode |
| `DEEPSEARCH_USE_STUB` | `false` | Run without weights (CI / offline) |
| `DEEPSEARCH_AUDIO_CHUNK_SECONDS` / `_OVERLAP_SECONDS` | `30` / `5` | Audio slicing |
| `DEEPSEARCH_TOP_K` / `DEEPSEARCH_RERANK` | `10` / `true` | Retrieval |

### Optional: Qdrant as a server (for concurrency)
```bash
docker compose up -d
export DEEPSEARCH_QDRANT_URL=http://localhost:6333
make app
```

## 7. Runtime backends

`models/gemma.py` selects a backend at startup:
- **Ollama** if reachable — real Gemma 4 + embeddings; the embedding dimension is probed once.
- **Stub** if `runtime.use_stub` is set or Ollama is down — deterministic hash embeddings and
  canned descriptions so ingestion, indexing, search and RAG all still run (used by the test
  suite and offline dev).

## 8. Testing & quality

```bash
make test     # pytest on the stub backend — no weights/GPU needed
make lint     # ruff + mypy
make fmt      # ruff format + autofix
```

The suite covers fusion correctness, intent routing, safety, an ingest→search end-to-end,
re-ingest deduplication, manual-weights→weighted-fusion, and image+MaxSim paths. CI runs it on a
`uv` venv with the stub backend.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Storage folder … already accessed by another instance` | Two app processes — embedded Qdrant is single-process | `pkill -f "streamlit run"; make app`, or use server mode |
| Sidebar shows `Index busy` / `—` | Same lock contention | Close other instances and click **↻ Refresh** |
| `pull model manifest: 412 … requires a newer version of Ollama` | Ollama daemon too old for `gemma4` | `brew upgrade ollama && brew services restart ollama` |
| Backend badge shows `stub (offline)` | Ollama unreachable or `DEEPSEARCH_USE_STUB=true` | Start Ollama / unset the flag |
| Ingest produces 0 audio chunks | The clip has no audio stream | Expected — not an error |
| Search returns weak/odd matches after changing `embed_strategy` | Dimension/space changed under an existing collection | `uv run deepsearch reset`, then re-ingest |
| FFmpeg/codec errors during ingest | Missing FFmpeg or unsupported codec | Install FFmpeg; the pipeline degrades gracefully and reports the file |

## 10. Privacy

Both the model runtime (Ollama) and the vector database (embedded Qdrant) run locally. Media
binaries are never stored in the index — only file paths and timestamp offsets. The query path
applies a prompt-injection pre-filter before any lookup, and generated descriptions are
sanitized before indexing.
