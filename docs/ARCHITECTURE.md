# Architecture & Design Rationale

> **Project:** Deep Search in Media Library Using Gemma 4 and Qdrant Multivectors
> **Author:** Sarvesh Talele
> This document covers *what the system is*, *why it is built this way*, the *advantages* of
> each decision, the *gap analysis* of the original proposal, and a *requirements coverage*
> matrix. For the step-by-step runtime walkthrough and usage, see
> [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md).

---

## 1. Objective

Ingest heterogeneous local media (video, audio, images, text/captions) and let a user find the
**exact moment** something happens using a natural-language query — entirely on local hardware,
with no cloud egress. The system unifies three modalities into one searchable space and serves
timestamp-accurate playback plus a grounded conversational (RAG) interface.

## 2. Technology stack (fixed by requirement)

| Component | Technology | Role |
|-----------|------------|------|
| Multimodal AI | **Gemma 4** (`gemma4:e4b`) via **Ollama** | Vision + audio description, query intent, reranking, RAG |
| Embeddings | **nomic-embed-text** via Ollama | Unified 768-d text-embedding space |
| Vector DB | **Qdrant** (embedded local mode) | Named vectors, MaxSim multivectors, payload filtering |
| Media extraction | **FFmpeg** + **OpenCV** + **PySceneDetect** | Keyframes, audio slicing, probing |
| Web UI | **Streamlit** | Ingest, search, RAG chat, live status |
| Packaging | **uv** + hatchling | Env + build |

## 3. High-level architecture

```
                         ┌──────────────────────── Gemma 4 (Ollama, local) ────────────────────────┐
                         │  vision · audio · text generation · intent · reranking · RAG chat        │
                         └───────────────▲──────────────────────────────────────────▲──────────────┘
   OFFLINE INGESTION                     │ describe / transcribe                     │ reason / rerank   REAL-TIME QUERY
 ┌───────────────────────────────────────┴───────┐               ┌───────────────────┴───────────────────────┐
 │ media file                                     │               │ user query (Streamlit)                     │
 │   ├─ video → scene keyframes (PySceneDetect)   │               │   ├─ safety pre-filter                     │
 │   ├─ video/audio → 30s/5s chunks (FFmpeg)      │               │   ├─ intent analysis → dynamic weights     │
 │   ├─ image → single described frame            │               │   ├─ embed query (unified space)           │
 │   └─ text → SRT/VTT/plain segments             │               │   ├─ per-named-vector prefetch (+ filter)  │
 │            │ describe-then-embed               │               │   ├─ fusion (RRF / weighted)               │
 │   nomic-embed-text → 768-d unified vectors     │               │   ├─ Gemma listwise rerank                 │
 │            │                                   │               │   └─ timestamped hits + RAG chat           │
 └────────────┼───────────────────────────────────┘               └───────────────────┼───────────────────────┘
              ▼                                                                         ▼
        ┌──────────────────────────── Qdrant (local, on-disk) ───────────────────────────┐
        │ collection: media_library                                                       │
        │  single-vector named spaces (fragment points): text_descriptions |              │
        │       video_frames | audio_chunks   (768-d, cosine)                             │
        │  MaxSim multivector spaces (asset points): frames_maxsim | audio_maxsim         │
        │  payload: asset_id, file_path, start_s, end_s, duration_s, category, ingested_at │
        │  payload indexes: asset_id/modality/category (keyword) · ingested_at (int)       │
        │  scalar quantization (INT8, always_ram)                                          │
        └─────────────────────────────────────────────────────────────────────────────────┘
```

The platform is split into two decoupled domains — the **offline ingestion pipeline** and the
**real-time query pipeline** — sharing one Qdrant collection and one Gemma 4 runtime.

## 4. Module map

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `deepsearch.config` | YAML + env layered settings (cached) |
| Logging | `deepsearch.logging_utils` | loguru + in-memory ring buffer for the UI |
| Models | `deepsearch.models.gemma` | Single Gemma/embedding client; Ollama or deterministic stub |
| Safety | `deepsearch.models.safety` | Query pre-filter + generated-metadata sanitizer |
| Ingestion | `deepsearch.ingestion.*` | probe · video · audio · text · pipeline (with progress events) |
| Vector store | `deepsearch.vectorstore.*` | schema, named vectors, MaxSim, upsert/search/stats/delete |
| Query | `deepsearch.query.*` | intent · fusion · rerank · engine |
| RAG | `deepsearch.rag.chat` | time-ordered timeline → grounded follow-up Q&A |
| UI / CLI | `deepsearch.streamlit_app`, `deepsearch.cli` | Streamlit app + Typer commands |

---

## 5. Key design decisions & advantages

### 5.1 Describe-then-embed (the core decision)

**Decision.** Gemma 4 *describes* every keyframe and audio chunk into text; that text — along
with captions and the user query — is embedded by `nomic-embed-text` into **one** 768-d space.

**Why.** Gemma 4 is a *generative* multimodal decoder, **not** a contrastive dual-encoder
(CLIP/ImageBind). Its internal image/audio activations are optimized for next-token prediction,
**not** for cosine proximity to text. Indexing raw frame activations and querying them with a
text vector would retrieve near-noise. Collapsing all modalities to text first puts the query
and every fragment in the same metric space.

**Advantage.** A single natural-language query genuinely retrieves across video, audio and text.
No separate Whisper/OCR model is needed — Gemma 4 transcribes audio and reads on-screen text as
part of one description call. The only dedicated extra model is a small text embedder (something
*must* define the metric space; a generative decoder should not).

> The alternative `native` strategy (embed via Gemma 4 itself) is available behind a config flag
> for same-modal experiments; the dimensionality is auto-detected at startup.

### 5.2 Fragment-level points (not asset multivectors as the primary unit)

**Decision.** The primary retrieval unit is a **fragment** — one Qdrant point per keyframe /
audio chunk / caption — each carrying `start_s`/`end_s`.

**Why.** Asset-level MaxSim returns a single score per asset and **loses which sub-vector
matched**, which is incompatible with "jump to the exact second". Fragment points make the
matching moment intrinsic to the hit.

**Advantage.** Timestamp-accurate playback and temporal RAG come for free; the named vector
spaces and payload pre-filtering still apply.

### 5.3 MaxSim multivectors as a secondary recall layer

**Decision.** The collection also declares `frames_maxsim` / `audio_maxsim` multivector spaces;
ingestion writes one asset-level point per file holding all its keyframe/audio vectors. An
optional two-stage mode (`search(asset_recall=True)`, UI toggle) runs MaxSim asset recall first,
then restricts fragment search to the winning assets.

**Why / advantage.** Satisfies the requirement of a single asset holding N keyframe vectors with
ColBERT/MaxSim semantics, while keeping temporal precision. MaxSim is computed **client-side**
(`store.search_maxsim`) because the embedded Qdrant client's native multivector query path is
unreliable — this makes it portable across embedded *and* server Qdrant.

### 5.4 Client-side score fusion

**Decision.** Per-named-space candidate lists are fused in the client: **RRF** (default,
rank-based, scale-free) or **weighted** (normalized-score weighted sum). Manual weight overrides
from the UI **force** weighted fusion so the sliders actually affect ranking.

**Why.** Qdrant has no single-call "weighted sum across named vectors" knob. The Query API fuses
candidate lists you define, not a per-named-vector scalar.

### 5.5 LLM-as-reranker (not a trained cross-encoder)

**Decision.** Reranking is a **listwise** Gemma 4 prompt returning a best-first ordering, with
graceful fallback to fused order.

**Why.** Gemma 4 is generative; calling it a "cross-encoder" is a misnomer. The listwise prompt
gives deep-reasoning reranking honestly.

### 5.6 Single runtime (Ollama) + deterministic stub

**Decision.** All model access goes through one client. If Ollama is unreachable or
`runtime.use_stub` is set, a deterministic hash-based stub powers the entire pipeline.

**Advantage.** Reproducible, offline-capable, CI-safe — the full ingest→search→RAG path runs
with **no weights**. (The original proposal mixed GGUF/llama.cpp *and* PyTorch+MPS, two
incompatible runtimes; this standardizes on one.)

### 5.7 Embedded Qdrant by default

**Decision.** On-disk local mode, no server/Docker. Set `DEEPSEARCH_QDRANT_URL` to switch to a
server with zero code changes.

**Trade-off.** Embedded mode locks the storage directory to **one process** — two app instances
cannot share it (use server mode for concurrency). Payload indexes are a no-op in embedded mode
(filtering still works, just unindexed).

### 5.8 Production guardrails

- **Scalar quantization** (INT8, `always_ram`) shrinks the footprint; product quantization is
  opt-in (needs rescoring).
- **Batched embedding** bounds unified-memory pressure during ingestion.
- **Safety**: a pre-retrieval input filter (length cap + prompt-injection patterns) and a
  generated-metadata sanitizer that redacts injected directives before indexing.
- **Asset-path independence**: raw media binaries are never stored in payloads — only file paths
  and offsets. Re-ingesting a file **replaces** its points (`store.delete_asset`) rather than
  duplicating them.

---

## 6. Gap analysis — original proposal vs. this implementation

The tech stack was fixed; only the **solution design** was corrected where the original would
not work reliably. ✅ = as specified · ➕ = delivered with a corrected mechanism.

| # | Original claim | Correction |
|---|----------------|------------|
| 1 | "Gemma 4 E2B/E4B/31B" exposes per-modality embedding endpoints | ➕ Gemma 4 ships on Ollama and is multimodal, but it is **generative**, not a contrastive embedder. Used for description/intent/rerank/RAG, not as the metric space. E2B/E4B are *effective*-parameter variants; default `gemma4:e4b`. |
| 2 | Compare a **text** query vector against raw `video_frames`/`audio_chunks` vectors | ➕ Misaligned spaces → noise. **Describe-then-embed** unifies all modalities in one text space (§5.1). |
| 3 | Gemma emits retrieval embeddings directly | ➕ Dedicated `nomic-embed-text` owns the space; vectors L2-normalized for cosine. |
| 4 | "Weighted combination of vector spaces" in one Qdrant call | ➕ No such knob; **client-side fusion** (RRF/weighted), manual weights force weighted (§5.4). |
| 5 | Asset multivector **and** second-accurate retrieval | ➕ Both built: fragment points for the exact second + MaxSim multivectors for asset recall (§5.2–5.3). |
| 6 | "Gemma cross-encoder reranking" | ➕ **LLM-as-reranker**, listwise (§5.5). |
| 7 | GGUF (llama.cpp) **and** PyTorch+MPS | ➕ One runtime: **Ollama** + deterministic stub (§5.6). |
| 8 | "Eliminate ASR" via direct audio→vector | ✅ No separate Whisper/OCR — Gemma 4 audio transcribes+characterizes in one call, then text is embedded. |
| 9 | Quantization named, recall impact ignored | ✅ Scalar default; PQ opt-in with rescoring caveat. |
| 10 | "Native safety prompt structures" (vague) | ✅ Concrete pre-retrieval filter + generated-metadata sanitizer. |
| 11 | No evaluation harness | ✅ `pytest` suite (fusion, intent, safety, ingest→search e2e, re-ingest dedup, MaxSim/image) runnable on the stub. |

---

## 7. Requirements coverage matrix

| Requirement | Status | Where |
|-------------|--------|-------|
| Ingest video / audio / image / text | ✅ | `ingestion/pipeline.py` |
| Unified cross-modal NL search | ✅ | `query/search.py` |
| Scene-boundary keyframe detection | ✅ | `ingestion/video.py` (PySceneDetect) |
| 30s audio chunks, 5s overlap, mono/16k, noise reduction | ✅ | `ingestion/audio.py` (`afftdn`) |
| Titles/tags/descriptions, SRT/VTT captions | ✅ | `ingestion/text.py` |
| Gemma 4 vision on keyframes | ➕ | `gemma.describe_image` (describe-then-embed) |
| Gemma 4 audio, no mandatory ASR | ✅ | `gemma.describe_audio` |
| Visual / audio / textual vectors | ✅ | three named spaces |
| Single collection, named vectors | ✅ | `store.ensure_collection` |
| Multivector arrays per asset, MaxSim | ✅ | `store.upsert_asset_multivectors`, `search_maxsim` |
| Payload: paths, timestamps, durations, categories | ✅ | `schema.Fragment.payload` |
| Payload indexes (keyword/int) for pre-filter | ✅ | `asset_id/modality/category`, `ingested_at` |
| Intent analysis (visual/spoken/conceptual) | ✅ | `query/intent.py` |
| Weighted combination of spaces | ➕ | `query/fusion.py` (weighted forced on override) |
| Qdrant fusion / external RRF | ✅ | `fusion.rrf_fuse` (default) |
| Gemma reranking of top-N | ➕ | `query/rerank.py` (LLM-as-reranker) |
| Timestamp-level retrieval (payload offsets) | ✅ | fragment `start_s`; `st.video(start_time=…)` |
| Temporal RAG follow-ups | ✅ | `rag/chat.py` (time-ordered timeline) |
| Scalar/Product quantization | ✅ | `store._quantization` |
| Batched ingestion | ✅ | `pipeline._embed_and_upsert` |
| Content safety on queries + generated metadata | ✅ | `safety.check_query`, `safety.sanitize_generated` |
| No binaries in payload | ✅ | paths + offsets only |

---

## 8. Known limitations

- **Describe-then-embed adds ingestion latency** — each frame/chunk is a Gemma generation;
  mitigated by scene-based keyframing and batched embedding. Acceptable for an offline pipeline.
- **Description quality bounds recall** — if Gemma omits a detail it is unsearchable
  (tunable via the prompts in `models/gemma.py`).
- **Embedded Qdrant is single-process** — one app instance at a time, or use server mode.
- **Switching `embed_strategy` after a collection exists** can change the dimension; recreate the
  collection (`deepsearch reset`) when changing the embedding model.
- **The stub backend** is for offline/CI only — its hash embeddings have no semantics.
