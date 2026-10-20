# Multimodal RAG implementation phases

Source: `MULTIMODAL_RAG_INTEGRATION_PLAN_V2.md`.

Status: all six phases completed. Phases 0–3 completed on 2026-10-19 and Phases 4–6 on 2026-10-20. All 114 automated tests pass locally and in the pinned Linux image, plus 12 browser checks. Real-service upload, shared indexing/retrieval, grounded answering, image serving, re-ingestion, and restart persistence passed. Phase 6 used exactly three DeepSeek requests/1,647 tokens; combined with Phase 3, recorded live validation totals five requests/2,538 tokens. Phases 4–5 used zero paid calls.

## Goal

Extend the application (now in the repository root) so users can upload a CV/PDF, ask text questions about its text and visuals, and receive grounded answers with filename/page citations and original images when relevant.

Use caption-mediated retrieval: extract images, generate factual descriptions with a genuine Vision LLM during ingestion, and embed both text chunks and image descriptions with the existing Ollama `nomic-embed-text` model into one Qdrant collection. Retain FastAPI, LlamaIndex, DeepSeek answer generation, and the existing HTML/JS UI.

Implementation paths below are now relative to the repository root. Historical references to `text-rag/` describe its former location. The unused gallery retrieval project is preserved in `archive/cross-modal-retrieval-flickr30k/`; its image-to-text endpoint retrieves existing gallery captions and cannot generate descriptions of unseen CV images. The superseded text-only roadmap is preserved as `archive/text-rag-original-PLAN.md`.

Repository cleanup on 2026-10-20 moved the active app, configuration, credentials, data, tests, scripts, and reports to this root without deleting files. Compose explicitly retains the `text-rag` project name to reuse its existing Qdrant volume. `README.md` documents the current layout and commands.

## Phase 0 — Capture the baseline and settle the shared contract

- [x] Read repository instructions and the current configuration, ingestion, retrieval, API, UI, dependencies, Docker configuration, README, and tests before editing application code.
- [x] Capture repository status and baseline test results; use an implementation branch where Git is available. Neither supplied folder contains Git metadata, so branch creation is not applicable to these copies.
- [x] Confirm current upload locations, sample loading, full-collection rebuild behavior, response schemas, and chat history handling.
- [x] Define stable document and record IDs, using filename plus content hash where appropriate.
- [x] Standardize metadata: `file_name`, `document_id`, `page`, `modality`, and an optional validated relative `image_path`. Use 1-based PDF pages; define an explicit absent-page convention for TXT/MD sources.
- [x] Define ingestion counts and warning reporting, plus backward-compatible source fields: `file`, `score`, optional `page`, `modality`, and `image_url`.
- [x] Plan upload migration to `data/documents/`, preserving or explicitly migrating `data/sample.md`; keep derived images under `data/images/`.

Exit condition: baseline behavior and failures are recorded, and extraction, ingestion, API, and UI have an agreed metadata contract.

### Phase 0 findings and baseline

Reviewed `app/config.py`, `app/ingest.py`, `app/query.py`, `app/main.py`, the complete HTML UI, all three test modules, requirements, Dockerfile, Compose, environment template, ignore files, README, and the original `text-rag/PLAN.md`. No `AGENTS.md` was found in the project trees or checked ancestor directories. The old plan's second-collection image stretch goal is superseded by the V2 integration plan's one-collection requirement. Keep the original plan unchanged.

Project 2's `server.py`, `cmrs/model.py`, `export_deploy.py`, and README confirm fixed-gallery dual-encoder retrieval. `/api/image-to-text` ranks stored captions; it does not generate captions. No `.pt` or `.pth` checkpoints were found in the supplied tree. Do not require its training stack, gallery, dataset, or checkpoints for this integration.

| Area | Observed baseline | Consequence for later phases |
|---|---|---|
| Input storage | `DATA_DIR = Path("data")`; uploads use `Path(filename).name` and overwrite a matching filename. Current input contains `sample.md` and `.gitkeep`. | Coordinate upload migration and discovery; document same-name replacement. |
| Document loading | `SimpleDirectoryReader` accepts TXT/MD/PDF; no custom image extraction or canonical page/modality contract exists. | Preserve TXT/MD behavior; introduce PDF-specific extraction and normalize metadata. |
| Rebuild | Existing collection is deleted before a new index is embedded and written; returned integer is the collection point count. | Keep the full-rebuild policy explicit; a failed rebuild can leave the index missing or incomplete. |
| Models and retrieval | Ollama `nomic-embed-text`, DeepSeek `deepseek-chat`, Qdrant `documents`, chunk size/overlap 512/50, top-k 3. | Keep these defaults and the existing wrappers. |
| Source response | `answer` plus `sources` containing `file` and rounded numeric `score`; no image route. | Add optional fields without removing or renaming existing fields. |
| Chat | Browser-local sessions; UI maps bot turns to `assistant`; server retains the latest 8 history messages and uses `condense_plus_context`. | Preserve client history shape, trimming, and follow-up question behavior. |
| Upload response | `{"file": "name.ext", "chunks_in_collection": N}`; UI reads the count directly. | Retain both keys; add detailed counts and warnings. |
| Errors and health | Unsupported type 400, invalid question/history 422, caught service connection errors 503; query `ValueError` is broadly mapped to 404. Health checks Qdrant only. | Do not interpret health as model readiness; later PDF handling must distinguish malformed input from connection failures. |
| Runtime | Python 3.12 Docker base, API + Qdrant Compose, host Ollama, bind-mounted `./data`. Requirements are unpinned; `.env` is absent. | Mocked tests need no keys. Live model access and reproducible build validation remain later gates. |

Baseline execution from `text-rag/` (PowerShell):

```powershell
& 'C:\Users\benma\anaconda3\envs\deep_learning\python.exe' -B -m pytest -q -p no:cacheprovider
```

Result: **11 passed in 6.30 seconds**, exit code 0. These cover eight API tests, two chunking tests, and one PDF-reader import test. They do not demonstrate live ingestion, retrieval quality, actual PDF extraction, or model connectivity. `-B` and the disabled pytest cache avoid writing bytecode/test caches into the supplied projects.

Environment discovery: the default `python` command is an unusable WindowsApps alias, `py -0p` reports no registered Python, and base Anaconda Python lacks pytest. The existing `deep_learning` environment has Python **3.12.9** and the dependencies needed for the passing baseline, so no installation was necessary.

Observed package versions for reproducing the baseline (not new dependency pins):

```text
pytest==9.1.1
fastapi==0.115.12
llama-index-core==0.14.25
llama-index-readers-file==0.7.0
llama-index-embeddings-ollama==0.10.0
llama-index-llms-deepseek==0.4.1
llama-index-vector-stores-qdrant==0.10.4
qdrant-client==1.19.1
pydantic==2.13.4
pydantic-settings==2.13.1
pypdf==6.7.4
httpx==0.28.1
```

Git status checks fail because the workspace and both project copies have no `.git` repository. No repository was initialized and no branch or commit was created. Docker CLI 27.3.1 is installed, but this session cannot read its user configuration and cannot reach the default daemon pipe; container execution was not available. Neither issue prevents completing this source-review and mocked-test phase.

### Shared contract selected for implementation

These are design decisions for later phases, not behavior already implemented.

**Identity and node boundaries**

- Compute `document_id` as the full lowercase SHA-256 hex digest of the sanitized basename's UTF-8 bytes, a zero-byte separator, and the source file bytes. Do not include its absolute directory, so moving uploads does not change identity. A rename or content change creates a new identity.
- Build each node ID as a deterministic UUIDv5 using `uuid.NAMESPACE_URL` and a name of the form `multimodal-rag:{document_id}:p{page_or_0}:{modality}:{ordinal}`. Ordinals are 1-based and deterministic within each page/modality; 0 is an internal identity marker for unpaginated TXT/MD only. UUID node IDs are suitable for Qdrant point IDs.
- Split each PDF page's text independently so a text node has one unambiguous source page. Split TXT/MD normally. Chunk ordinals may change when text or chunking settings change; identical input and extraction/chunking settings must reproduce IDs.
- Keep each successful image description as one node; never run captions through the text splitter. Use deterministic extraction order and image ordinals, preserving the corresponding saved-image pointer.

**Node metadata and image references**

| Field | Contract |
|---|---|
| `file_name` | Sanitized original basename including extension; map to API `file`. |
| `document_id` | Stable document hash defined above, on every node. |
| `page` | Positive 1-based integer for PDF nodes. TXT/MD use no page: omit the stored key and serialize API `page` as `null`. Never invent page 1 for unpaginated documents. |
| `modality` | Exactly `text` or `image`. Image means an image-description node, not an image embedding. |
| `image_path` | Present only for image nodes; app-root-relative POSIX path such as `data/images/<document_id>/page_2_img_0001.png`. Use a validated generated filename with the saved artifact's actual extension. No absolute paths or image blobs. |

Node text contains the actual chunk or generated caption. Metadata must survive splitting and Qdrant serialization. Keep image paths available for rendering but exclude them and opaque IDs from embedding text; use the same explicit embedding-metadata policy for both modalities.

Derive public image URLs such as `/images/<document_id>/page_2_img_0001.png` from validated metadata. Resolve files beneath the configured image root and check containment before serving. A URL is not a filesystem path supplied by a client. Never return absolute disk paths in the query response.

**API and ingestion compatibility**

- Keep `GET /health`, `POST /documents`, `POST /query`, and `GET /`, plus existing question/history validation and the eight-message history cap.
- Retain `{"answer": str, "sources": [...]}` and required source fields `file: str`, `score: float`. Add optional `page: int | null`, `modality: "text" | "image" | null`, and `image_url: str | null`, with defaults that accept legacy source dictionaries. Do not fabricate missing PDF pages in old indexed data; a rebuild populates the new metadata.
- Add collection-wide successful-record statistics: `text_chunks`, `image_descriptions`, and `total_vectors`, where `total_vectors = text_chunks + image_descriptions`. Retain upload `file` and `chunks_in_collection`, making the legacy count equal to `total_vectors` after rebuilding.
- Use `warnings` as a list of objects with `code`, `file`, optional `page` and `record_id`, and a concise `message`. Caption failures or empty captions skip that image's node, preserve eligible text, and produce warnings. Avoid exposing credentials, raw provider responses, or absolute paths.
- Keep `ingest_directory(...) -> int` as a compatibility wrapper; a small internal ingestion function can return detailed statistics for the upload route. No new data-model framework is needed.
- Do not report partial/failed embedding or database writes as a successful rebuild. Caption warnings are distinct from a fatal ingestion failure. Full-rebuild recovery remains limited in the local v1 demo and must be documented.

Illustrative future upload response (counts are examples, not measured results):

```json
{
  "file": "resume.pdf",
  "chunks_in_collection": 12,
  "text_chunks": 10,
  "image_descriptions": 2,
  "total_vectors": 12,
  "warnings": []
}
```

**Input migration and derived files**

1. In Phase 4, save new uploads in `data/documents/`; retain the existing same-basename replacement policy for this single-user demo and rebuild the entire input set after upload.
2. Preserve `data/sample.md` in place and include it explicitly alongside supported files directly inside `data/documents/`. Do not recursively scan `data/` or ingest generated images/metadata.
3. Before switching discovery, migrate any legacy top-level TXT/MD/PDF uploads except `sample.md` into `data/documents/`. If a target exists with identical bytes, index it once; if bytes differ, report the collision and require a rename rather than silently overwrite either file. The supplied folder currently has no legacy upload to migrate.
4. Keep generated visual evidence in `data/images/<document_id>/`. Rebuild/reconcile derived files against the current source set and ensure every indexed image pointer resolves; remove stale artifacts only within the generated-image root. The existing data bind mount covers both new directories.
5. Preserve the sample and placeholder ignore exceptions. Uploaded documents and derived images stay untracked; verify the existing ignore rules still cover the new directories when they are introduced.

Phase 0 outcome: source review, passing test baseline, compatibility constraints, identity/schema decisions, and migration plan are recorded. Only this planning document was changed. No credentials or additional uploads are needed for Phase 0. A genuine Vision LLM choice remains for Phase 1; live credentials and Docker access are later integration requirements.

## Phase 1 — Define the Vision LLM boundary

- [x] Isolate captioning behind a small provider-configurable interface, such as `describe_image(image_path: Path, context: str = "") -> str` in `app/image_processor.py`.
- [x] Identify a genuine vision-capable generative provider/model; record required configuration, image input support, runtime requirements, and expected cost or local hardware needs.
- [x] Define a prompt that captures visible labels, diagram components, relationships, and purpose while marking unreadable or uncertain details.
- [x] Establish mocked success/failure behavior so extraction and ingestion work can proceed without provider credentials.
- [x] Add only necessary settings in `app/config.py` and placeholders in `.env.example`; keep secrets out of source control.

Exit condition: the adapter contract, configuration needs, and mocked behavior are settled. Confirm the actual provider and request any required credentials at the live integration boundary in Phase 3.

### Phase 1 implementation and provider decision

The user changed the captioning choice to **DeepSeek's hosted vision API**, reusing their existing DeepSeek API key. The current official [vision guide](https://api-docs.deepseek.com/guides/vision/) supports image input with `deepseek-flash`. Use this documented model name for captions, independently of the existing answer-model setting. Images leave the local machine for caption generation and normal hosted API usage charges apply; see [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/). No local vision-model download or vision GPU is required. Actual account access, network connectivity, caption quality, and costs remain live integration checks.

`app/image_processor.py` exposes `describe_image(image_path: Path, context: str = "") -> str`. Small provider-specific transports are isolated in `_describe_with_deepseek` and `_describe_with_ollama`; both share sanitized transport errors and caption validation. Unsupported providers fail explicitly. The adapter is not invoked by uploads or queries yet.

The DeepSeek transport sends a text block plus an inline base64 image data URL in the user message to `/chat/completions`, with bearer authentication from `DEEPSEEK_API_KEY`. It disables streaming and thinking, caps the output at 512 tokens, and accepts only a completed text caption. Image headers determine the MIME type for PNG, JPEG, GIF, and WebP; unsupported headers and files exceeding the 32 MiB inline image limit fail before the request. It reuses the existing `httpx` dependency. Base64 is transient request data and is not stored in vector metadata. The [thinking-mode guide](https://api-docs.deepseek.com/guides/thinking_mode/) documents the disabled-thinking request option.

Ollama remains an optional local captioning backend: set `VISION_PROVIDER=ollama` and `VISION_MODEL=gemma3:4b` together, pull that model separately, and use a reachable Ollama server. Its [vision API](https://docs.ollama.com/capabilities/vision) receives the same image bytes through the provider's native `/api/chat` format. No automatic provider fallback occurs, so hosted/local behavior stays explicit.

The caption prompt requests concise factual descriptions, readable labels, components, directions, relationships, and uncertainty. Neighboring text is context only, and instructions inside the image/context are treated as document content. Temperature is set to zero to reduce sampling variation; this is not a claim of deterministic or correct live captions.

Configuration added to `app/config.py` and `.env.example`:

| Variable | Default | Meaning |
|---|---|---|
| `VISION_PROVIDER` | `deepseek` | Hosted captioning; `ollama` is also implemented. |
| `VISION_MODEL` | `deepseek-flash` | Generative image-description model, separate from the embedding and answer models. |
| `VISION_BASE_URL` | empty | Uses `https://api.deepseek.com` for DeepSeek, or `OLLAMA_BASE_URL` for Ollama. Explicit values override the provider default. |
| `VISION_TIMEOUT` | `120` | Positive HTTP timeout in seconds. |
| `DEEPSEEK_API_KEY` | empty | Existing secret setting, reused for hosted captions; missing/blank values fail before network access. |

Errors use `ImageDescriptionError` with a stable `code` and a safe message. Cases include unsupported provider, missing model/key setting, unreadable/empty image file, unsupported/oversized hosted image, network/timeout/HTTP failure, invalid/incomplete response, and empty caption. The ingestion layer will attach source information and convert these to visible warnings in later phases; this module does not silently return fake evidence or log raw response bodies/image bytes.

Validation from `text-rag/`:

```powershell
& 'C:\Users\benma\anaconda3\envs\deep_learning\python.exe' -B -m pytest -q -p no:cacheprovider
```

Result after the provider change: **48 passed in 5.17 seconds**, exit code 0 (the original Phase 1 Ollama result was 28 passed). `tests/test_image_processor.py` cases intercept HTTP using `httpx.MockTransport`; they check original image bytes, context/model settings, authentication, URL fallback/override, configured timeout, sanitized connection/timeout/401/429/other HTTP failures, invalid/empty/incomplete captions, and rejection before HTTP for missing credentials or invalid inputs. No live model calls or real credentials were used by tests.

Runtime probe: `D:\Ollama\ollama.exe list` returned `could not locate ollama app`; a direct request to `http://localhost:11434/api/tags` also failed with `ConnectError`. Installed models and live vision capability could not be verified. No models were downloaded and no services were started.

For the Phase 3 live captioning check, set these values in the ignored `text-rag/.env` file (the key below is a placeholder):

```dotenv
DEEPSEEK_API_KEY=<your-existing-key>
VISION_PROVIDER=deepseek
VISION_MODEL=deepseek-flash
VISION_BASE_URL=
```

The runtime needs network access to DeepSeek and account access/balance for the selected model. No real key was read, created, or requested in chat, and no live hosted requests were made. Ollama is still required for the existing `nomic-embed-text` embeddings during end-to-end ingestion/retrieval; replacing captioning does not replace embeddings. Phase 2 PDF extraction needs neither a running Ollama server nor a hosted key. Phase 3 will validate real unseen document images and caption quality/failures; its live acceptance gate remains pending.

## Phase 2 — Extract PDF text and useful visuals

- [x] Add `app/pdf_processor.py`, preferably using PyMuPDF, to extract page text and raster images with source metadata.
- [x] Save original extracted images beneath `data/images/<document_id>/` and preserve the correct filename and page for each result.
- [x] Filter tiny decorative images and repeated logos where feasible without removing useful evidence.
- [x] Provide a controlled page-region or rendered-page fallback for vector diagrams; define when it runs and avoid captioning every page indiscriminately.
- [x] Handle malformed PDFs clearly and document limited support for scanned/image-only PDFs; do not add an OCR pipeline.
- [x] Add fixture-based tests for two-page text extraction, image persistence, source paths, page numbering, and the selected vector-diagram fallback.

Exit condition: extractor tests pass and raster images plus the chosen vector-diagram case retain correct source references.

### Phase 2 implementation and validation

`app/pdf_processor.py` now exposes `extract_pdf_content(pdf_path: Path, images_dir: Path | None = None) -> PdfContent`. The small dataclass holds `text_documents` (LlamaIndex Documents, one per nonempty PDF page), `images` (uncaptained visual candidates), and `warnings` (source-aware dictionaries). It does not call a model, chunk text, connect to Qdrant, or alter the upload route. This separation keeps the later captioning and unified-indexing steps independently testable.

Identity follows the Phase 0 contract: SHA-256 of basename plus separator plus file bytes, and deterministic UUIDv5 record IDs. Source-page text documents use a `page_text` identity namespace so they cannot collide with the eventual `text` chunk IDs. Each image candidate contains `id`, `metadata`, and up to 2,000 characters of page text as `context`; it has no caption yet. Metadata includes filename, document ID, 1-based page, modality, app-relative POSIX image path, and `image_kind` (`raster` or `vector_crop`). Opaque document IDs are excluded from text Document embedding/LLM metadata.

Raster extraction retains supported embedded bytes where possible, restores transparency from soft masks, and normalizes other formats to PNG for the caption interface. Generated paths resolve beneath the configured image root and application directory. Deterministic filenames live in `data/images/<document_id>/`; repeated extraction rewrites the same files without creating extra copies. Rebuild cleanup of obsolete files remains Phase 4 work.

Filtering decisions: skip raster images with a source dimension below `PDF_MIN_IMAGE_PIXELS` (default 64), or a visible page dimension below 24 points. Suppress same-page duplicate image content. Across pages, suppress a repeated small raster only when it occupies at most 5% of the page, retaining the first occurrence; larger repeated diagrams retain each page's citation. These rules are heuristics and can suppress useful small evidence, so document the tradeoff and adjust thresholds against real CVs later.

The vector fallback groups nearby drawing paths and requires at least `PDF_VECTOR_MIN_PATHS` (default 3), a region at least 96 by 48 points, and a bounding area between 1% and 80% of the page. It pads accepted regions by 16 points, clips them to the page, and renders only those regions. Rotated pages use transformed crop coordinates. `PDF_RENDER_DPI` defaults to 144 (allowed 72–300), with longest-side resolution capped near 2,048 pixels. Text-only pages and simple borders/rules are not automatically rendered. `PDF_VECTOR_FALLBACK=false` disables the fallback entirely.

Coverage limits: connected-path clustering is not diagram understanding. Sparse, disconnected, single-path, or near-full-page diagrams may be missed; decorative clusters may be included. A page-sized frame can group otherwise useful drawings into an excluded region. These limits are explicit rather than expanding to automatic full-page captioning. Vector crops preserve the original PDF's visible evidence but are rendered artifacts, not embedded raster originals. See PyMuPDF's [image extraction and rendering guide](https://pymupdf.readthedocs.io/en/latest/recipes-images.html) for the underlying operations.

`PdfProcessingError` distinguishes unreadable, malformed, password-protected PDFs, and invalid output directories. Image decoding/saving and vector detection/rendering failures become warnings instead of discarding text already extracted. Image-only/visual pages without extractable text receive a `no_extractable_text` warning; no OCR engine or perfect scanned-PDF text support was added. API mapping of document errors and visible warning presentation remain later integration tasks.

Added extraction settings and examples in `app/config.py` / `.env.example`: `IMAGE_DIR`, `PDF_MIN_IMAGE_PIXELS`, `PDF_VECTOR_FALLBACK`, `PDF_VECTOR_MIN_PATHS`, and `PDF_RENDER_DPI`. Added `pymupdf>=1.26.6,<2` to requirements. The existing Python environment already had PyMuPDF 1.26.6, so no package installation was needed; exact environment pins and Docker build validation remain Phase 6. README documents current standalone status, options, and limitations.

Validation from `text-rag/`:

```powershell
& 'C:\Users\benma\anaconda3\envs\deep_learning\python.exe' -B -m pytest -q -p no:cacheprovider
```

Result: **66 passed in 5.50 seconds**, exit code 0. The 18 new `tests/test_pdf_processor.py` cases create real synthetic PDFs in temporary directories and verify two-page text/raster extraction, metadata and saved paths, stable IDs across input migration/re-extraction, changed identity after rename/content changes, small/decorative/duplicate filtering, restored alpha, vector crops and rotated coordinates, fallback disablement, near-full-page frame suppression, coexisting raster/vector evidence, image-only warnings, invalid/encrypted/missing input, output-root containment, and image-save failure with preserved text. No private documents, live services, API keys, or persistent fixture images were needed.

Phase 2 is complete. Upload ingestion still follows its original text-only route until Phase 4; this phase adds the extractor rather than silently changing existing uploads. Next: Phase 3 will connect extraction to caption preparation and warning handling, with a live DeepSeek check when credentials/account access are available.

## Phase 3 — Implement and validate image captioning

- [x] Implement the selected provider through the Phase 1 adapter; add a provider client dependency only when needed.
- [x] Generate concise searchable descriptions at ingestion time, optionally using neighboring page text as context.
- [x] Handle failures and empty descriptions explicitly: skip the affected image, preserve text ingestion, and report a useful warning.
- [x] Test requests, deterministic mock descriptions, and failure reporting without external model calls in unit tests.
- [x] Confirm provider access, credentials, and networking from the intended runtime; test caption generation on previously unseen, non-sensitive CV images. Verified from local Python; container access remains a Phase 6 gate.

Exit condition: mocked tests pass, live captions are demonstrated with the selected model, and failures do not prevent otherwise valid text ingestion. Record a live-service blocker explicitly if this gate cannot be completed.

### Phase 3 implementation and validation

The user configured the DeepSeek key in the ignored `text-rag/.env` and authorized minimal live API use, preferably two or three calls. The key was loaded through existing settings and never printed or copied into source, test fixtures, or reports. No credentials or additional inputs are currently needed for this phase.

Added `prepare_pdf_documents(pdf_path, images_dir=None)` to `app/ingest.py`. It composes the Phase 2 extractor and Phase 1 caption adapter, returning `PreparedPdf(text_documents, image_nodes, warnings)`. Text remains unsplit for Phase 4; each successful image becomes a complete LlamaIndex `TextNode` with its stable image record ID, filename, page, modality, and original saved-image path. The optional page context goes to captioning, separately from the caption stored as searchable node text.

Image nodes record `caption_provider` and `caption_model`. Opaque IDs, image paths/kinds, and provider/model provenance are excluded from embedding and LLM metadata. Failures, including empty provider captions, skip only the failed image and append a warning with code, filename, page, record ID, and safe message. Extractor warnings and all extracted text remain available even if every image caption fails. There is at most one provider call per image and no automatic retry. Upload responses and Qdrant indexing are deliberately unchanged until Phase 4; warning presentation there will consume this preparation result.

Automated validation from `text-rag/`:

```powershell
& 'C:\Users\benma\anaconda3\envs\deep_learning\python.exe' -B -m pytest -q -p no:cacheprovider
```

Result: **71 passed in 5.73 seconds**, exit code 0. This includes all prior 66 tests and five new preparation cases in `tests/test_multimodal.py`: complete unsplit long captions, source/provenance preservation and metadata exclusions, single caption failure/empty-caption warnings with continued processing, all-captions-fail text preservation, and zero caption calls for text-only PDFs. Model calls are mocked; Qdrant/embedding calls are explicitly forbidden in preparation tests.

Added the opt-in `scripts/validate_vision.py --live` smoke check. It creates a fresh synthetic CV/PDF containing a raster pipeline on page 1 and a vector workflow on page 2. Text outside the diagrams contains no diagram labels, so the live caption checks cannot pass by merely repeating page context. A separate offline preflight verified exactly two image candidates and both page texts before enabling network access. Temporary artifacts are removed after the check, and the script writes a safe report under `reports/`.

Live results from local Python with `deepseek-flash`:

| Request | Evidence | HTTP | Input tokens | Output tokens | Total tokens | Result |
|---|---|---|---|---|---|---|
| 1 | Embedded raster: PDF → Text + Captions → Qdrant | 200 | 347 | 99 | 446 | Correct labels and left-to-right relationships. |
| 2 | Vector crop: Question → Retrieval → Answer | 200 | 340 | 105 | 445 | Correct labels and left-to-right relationships. |
| Total | Two synthetic images; no retries | — | 687 | 204 | 891 | Two text pages preserved, two captions produced, no warnings. |

**DeepSeek API usage for Phase 3: exactly 2 requests**, both successful. No extra model-list, answer-generation, diagnostic, or retry requests were made. Outputs stayed below the existing 512-token cap. Token totals are server-reported usage; no currency charge is inferred. All other validation used local PDFs and mocks. The smoke-check script caps each execution at two requests and stops after its first HTTP/network error; it never runs under ordinary pytest.

Evidence: [safe live validation report](text-rag/reports/phase3_vision_validation.json), containing request count, status codes, usage totals, synthetic captions, and label checks. Manual review of the captions confirmed the synthetic diagrams' components and directions. This demonstrates live caption generation for these two cases, not general CV accuracy or retrieval quality.

Phase 3 is complete for the local Python runtime. Docker/container connectivity remains unverified because the daemon was unavailable at baseline; retain that check in Phase 6. Next, Phase 4 will split text and index text plus the prepared caption nodes through the same embedding model/collection, and expose statistics/warnings. No additional live caption requests were made after the successful two-request run.

## Phase 4 — Index text and captions in one collection

- [x] Refactor `app/ingest.py` while retaining `get_embed_model()` and `get_qdrant_client()`.
- [x] Discover only supported `.txt`, `.md`, and `.pdf` input documents; exclude generated images and metadata from discovery.
- [x] Preserve the TXT/MD reader and split text using the existing `SentenceSplitter` defaults of 512 tokens and 50 overlap.
- [x] Preserve metadata through text chunking and keep each image caption as one unsplit node.
- [x] Embed both node types through the same embedding model and persist them into `settings.qdrant_collection` with matching vector dimensions.
- [x] Keep the single-user full-rebuild policy explicit: recreate the collection and regenerate or reconcile image artifacts consistently, without duplicate vectors or broken references.
- [x] Return text chunk, image description, total vector, and warning counts while preserving existing callers where practical.
- [x] Update upload paths and sample handling consistently across ingestion, API, and tests.
- [x] Test shared embedding/collection usage, metadata preservation, caption node boundaries, document discovery, caption failures, and repeated ingestion.

Exit condition: a mixed PDF produces text and image-caption nodes in one collection; re-ingestion preserves consistent counts and valid image pointers.

### Phase 4 implementation and validation

Completed on 2026-10-20. `ingest_directory_with_stats(...)` now orchestrates discovery, PDF extraction/caption preparation, TXT/MD reading, text-only splitting, one shared embedding batch, and one Qdrant collection rebuild. The existing embedding/client wrappers remain unchanged, and `ingest_directory(...) -> int` remains a compatibility wrapper returning the total point count. CLI ingestion prints text/image/vector counts and source-aware warnings.

Discovery accepts supported files directly inside `data/documents/` plus the retained root `data/sample.md`. It excludes nested folders, generated images, metadata, unsupported extensions, and symlinks. Migration preflights all legacy filename conflicts before moving files: identical target content is retained once, while different bytes require a rename. The supplied workspace's data still contains only its original sample/placeholder; actual user documents were not moved during tests.

TXT/MD still use `SimpleDirectoryReader` and `SentenceSplitter` with the unchanged 512/50 defaults. Metadata is normalized to the shared contract, with no invented page for unpaginated inputs. PDF pages are split independently; deterministic chunk UUIDs use document ID, page, modality, and chunk ordinal. Caption nodes bypass the splitter entirely, including long mocked captions. All text and captions pass through one `get_embed_model()` instance and embedding batch. The app checks vector count, consistent positive dimensions, and finite numeric values before touching the existing index.

The shared `QdrantVectorStore` writes the already embedded nodes directly, avoiding another embedding pass. The configured collection is rebuilt and its exact point count must match prepared node count before success is reported. Existing `VectorStoreIndex.from_vector_store(...)` retrieval remains compatible; the local integration test reconstructs that index and retrieves both modalities together. This uses the standard [LlamaIndex Qdrant adapter](https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/qdrant/); tests use [Qdrant's in-memory mode](https://github.com/qdrant/qdrant-client) without a server.

Preparation, empty-input rejection, and embedding failures happen before collection deletion. Collection replacement itself remains nontransactional; a failure during database writes can leave a missing/partial index and requires a rerun. Generated-image reconciliation runs only after a successful counted rebuild. It retains active image pointers and removes only recognized generated filenames in hash-named document directories within the validated image root; unrelated files and symlinks are preserved. Failed caption images have no indexed pointers and may be removed during reconciliation after an otherwise successful text rebuild.

`POST /documents` now writes to `data/documents/` and retains `file` and `chunks_in_collection`. It adds `text_chunks`, `image_descriptions`, `total_vectors`, `warnings`, and `warning_count`. Source-aware caption failures can yield a successful text-only response with warnings. Handled malformed/encrypted PDF/ingestion failures return a clear error and remove a failed new upload or restore prior same-name bytes. Caught embedding timeouts are handled as service-unavailable responses; migration conflicts return 409 without overwriting inputs. Rich query source fields, image serving, and UI warning/image rendering remain Phase 5 work.

Automated validation from `text-rag/`:

```powershell
& 'C:\Users\benma\anaconda3\envs\deep_learning\python.exe' -B -m pytest -q -p no:cacheprovider
```

Result: **91 passed, 1 warning in 13.69 seconds**, exit code 0. The warning is expected: in-memory Qdrant does not enforce payload indexes. Added 20 cases across `tests/test_ingestion.py` and `tests/test_api.py`, exercising actual LlamaIndex/PDF processing/Qdrant serialization and local retrieval with mocked provider/embedding calls. Checks include mixed modalities, one embedding batch/collection, matching dimensions, metadata persistence, long captions remaining unsplit, stable IDs/rebuild counts, excluded generated/nested inputs, caption failure warnings, source removal/stale cleanup, malformed/empty/embedding failure preservation, invalid vector output, migration conflicts/deduplication, API compatibility, and failed-upload rollback. A database-write failure test confirms cleanup does not run prematurely.

**DeepSeek API usage for Phase 4: 0 requests.** Paid model and network calls were blocked in the ingestion tests. No real embedding service, external Qdrant server, live answer generation, or prior live-validation script was invoked. The previous Phase 3 report remains unchanged at two successful requests. Full rebuilds will re-caption images during real ingestion; no caption cache or incremental upserts were introduced, and the cost is explicit in README.

Phase 4 is complete against the local integration harness. No new key, upload, or service access is needed now. Phase 5 can proceed independently; the real Ollama/Qdrant/container workflow and end-to-end demo remain Phase 6 acceptance gates.

## Phase 5 — Return grounded answers and display original evidence

- [x] Extend `app/query.py` source serialization with page, modality, and a safe derived image URL, retaining `file` and `score`.
- [x] Preserve `VectorStoreIndex.from_vector_store(...)`, DeepSeek, and `condense_plus_context` chat history behavior.
- [x] Instruct answer generation to use retrieved evidence and distinguish caption observations from verified details. Raw image pixels need not be sent to the answer model in v1.
- [x] Extend the `Source` schema in `app/main.py` while preserving `/health`, `/documents`, `/query`, and `/`.
- [x] Add read-only image serving restricted to trusted generated files under `data/images/`; validate resolved paths and reject traversal and missing files without exposing unrelated files.
- [x] Validate upload extensions and malformed PDFs, and consider a reasonable upload-size limit.
- [x] Update `app/static/index.html` to show filename, page, modality, and original images beside citations, with graceful missing-image behavior.
- [x] Preserve upload interaction, text answers, source scores, and chat history in the existing single-page UI.
- [x] Test mixed source responses, legacy text-only behavior, image serving, path traversal rejection, and missing-image handling.

Exit condition: retrieval can return both modalities, the API remains compatible with text-only clients, and the UI displays the original visual evidence with useful citations.

### Phase 5 implementation and validation

Completed on 2026-10-20. `app/query.py` now serializes source `file`, `score`, `page`, `modality`, and a safe derived `image_url`. Invalid/missing pages remain null; legacy metadata does not acquire invented page numbers. Existing `VectorStoreIndex.from_vector_store(...)`, DeepSeek, top-k configuration, history conversion, and `condense_plus_context` behavior remain in place. A system instruction restricts answers to retrieved evidence, asks for missing-evidence acknowledgment, and distinguishes model-generated caption observations from verified extracted-text details. It does not claim the answer model sees or verifies pixels.

The FastAPI `Source` schema accepts optional page/modality/image URL fields. `response_model_exclude_unset=True` preserves the exact filename/score-only shape when a legacy source dictionary omits the new fields, while newly serialized results include them. The existing health/upload/query/root endpoints remain; uploads also enforce configurable `MAX_UPLOAD_BYTES` (default 20 MiB) before saving or any model call. Earlier extension/PDF validation and rollback behavior remain covered by tests.

Added `app/image_paths.py` to share validation between URL derivation and `GET /images/{document_id}/{filename}`. Only hash document IDs and recognized generated raster filenames are accepted. Paths must resolve within a dedicated image root under the application directory; root/folder/file symlink guards and containment checks reject escapes. The route is read-only, returns 404 for unrecognized or missing images, and sets no-store/nosniff headers. URLs contain only generated identifiers, not local disk paths. The shared helper honors a configured image root without exposing arbitrary files or mounting the whole data directory.

The existing HTML/JS UI now shows filename/page/modality citations, responsive saved-image previews and links, unavailable-image fallbacks, upload text/image counts, and source-aware warnings. Rendered vector crops display the saved PDF evidence rather than synthesized replacement images. Client rendering accepts only same-origin generated image URLs and renders filenames/warnings as text. Old filename/score-only chats still render; notes/warnings persist in local storage without becoming model chat history.

Automated validation from `text-rag/`:

```powershell
& 'C:\Users\benma\anaconda3\envs\deep_learning\python.exe' -B -m pytest -q -p no:cacheprovider
```

Result: **114 passed, 1 warning in 16.01 seconds**, exit code 0. The unchanged warning concerns payload indexes in in-memory Qdrant. The 23 added cases cover grounded chat configuration and history semantics, mixed/legacy source serialization, rejection of untrusted paths, original image bytes, read-only serving, encoded traversal/private-file rejection, missing images, configured/external image roots, symlink guards, mixed API schemas, and upload-limit rejection before persistence/ingestion. The symlink-guard unit case stubs filesystem link detection, so it does not claim an operating-system symlink attack test. No live model calls were made.

Added an optional local smoke-check script, `scripts/validate_ui.py`, using a temporary Uvicorn server, synthetic PDF visuals, mocked API services, and a fresh headless Chromium profile. Its development prerequisites are local Windows Chrome/Edge and `websocket-client`; no new production dependency was added. Sandbox browser-control connections were reset, so the browser smoke check ran outside the sandbox. No approval rejection occurred. Windows emitted an intermittent connection-reset message during browser connection shutdown; the final check exited successfully.

The final headless Chrome run passed all **12 browser checks**: original image load, filename/page/modality citations, mobile sidebar collapse, no horizontal mobile overflow, missing-image fallback, legacy message rendering, upload counts/warnings, external URL rejection, filename text rendering, actual history sent to the API, chat/warning persistence after reload, and exclusion of notes from model history. Desktop and 390-pixel mobile screenshots were visually inspected; the screenshot capture waits for the mobile sidebar transition to finish.

Evidence: [browser report](text-rag/reports/phase5_ui_validation.json), [desktop screenshot](text-rag/reports/phase5_ui.png), and [mobile screenshot](text-rag/reports/phase5_ui_mobile.png). All evidence uses synthetic data; temporary uploads, derived files, and browser profiles were cleaned up. The original workspace sample and credential file were not changed.

**DeepSeek API usage for Phase 5: 0 requests.** All answer generation and ingestion services were mocked for browser checks, and provider HTTP POST calls were explicitly blocked. The Phase 3 report still records exactly two successful requests and 891 tokens. Phase 5 demonstrates local API/UI behavior, not live answer quality; the real Qdrant/Ollama/DeepSeek/Docker workflow remains Phase 6 work. No user input is needed now.

## Phase 6 — Validate the environment and complete the demo

- [x] Run the existing pytest suite plus the new PDF, captioning, ingestion, metadata, and API tests with external clients mocked where appropriate.
- [x] Update requirements and verify a successful environment build before pinning dependency versions for reproducibility.
- [x] Check Docker permissions, host Ollama access, Vision LLM access, and persistence through the existing `./data:/app/data` volume.
- [x] Ignore uploaded CVs and derived images as needed; document required configuration without committing private files or credentials.
- [x] Update README installation, configuration, rebuild policy, demo instructions, and known limitations.
- [x] Run the end-to-end acceptance sequence below and record actual outcomes, warnings, and unresolved limitations.

Exit condition: automated checks pass, the container workflow is verified, and the documented demo succeeds using real model integrations.

### Phase 6 implementation and validation

Completed on 2026-10-20. Docker access worked outside the sandbox; Docker Desktop 4.36.0 / Engine 27.3.1 provided a Linux/amd64 runtime. Host Ollama was initially unavailable. The official Ollama image exposed a roughly 3.8 GB download, which was stopped before completion; the user chose to start host Ollama instead. Host and container readiness checks then confirmed `nomic-embed-text:latest` availability and container access through `host.docker.internal`. No optional Ollama Compose service was retained, preserving the original host-embedding architecture.

Environment checks/builds:

- Existing local environment: **114 passed, 1 warning in 14.52 seconds** using Python 3.12.9. The warning is the known in-memory Qdrant payload-index notice.
- Clean Docker build from then-unpinned requirements: **114 passed, 2 warnings in 4.67 seconds**, with networking disabled, plus successful `pip check`.
- Only after that success, captured 109 package versions into `text-rag/requirements.lock`, pinned direct dependencies in `requirements.txt`, and added the lock as a pip constraint. Pinned the tested Python base digest in Dockerfile.
- Rebuilt pinned image: **114 passed, 2 warnings in 4.24 seconds**, again with networking disabled, and `pip check` reported no broken requirements. Every one of the 109 installed lock versions was then checked for equality. Runtime Python is 3.12.15, FastAPI 0.150.0, PyMuPDF 1.28.2, Qdrant client 1.19.1, and LlamaIndex core 0.14.25. The second warning is a Starlette TestClient HTTPX-backend deprecation; it did not prevent API or live checks.
- Final validated image ID: `sha256:1ac956f16537de991ed6089ce18474db9a288a054e114eb50c65f2a503830b3a`. Docker inspection confirmed no `/app/.env` in the image. Reports/scripts were excluded from build context alongside the existing secret/data/test exclusions.

Added `scripts/validate_e2e.py`: an explicit `--live` command capped at three paid requests, plus a free `--write-fixture` mode. The live script creates two text pages and one raster diagram, exercises real FastAPI handlers through TestClient inside the Linux image, and uses real host Ollama, server Qdrant, and DeepSeek. It sets its own temporary data/image directories and unique collection, which are cleaned afterward. No caption or answer is stubbed. The answer adapter is only wrapped to cap output at 512 tokens and disable SDK retries during validation. No ordinary pytest run invokes this script.

To honor the user's low-usage preference, text-only, image-focused, and mixed questions were bundled as three numbered questions in **one live query**, rather than three independent answer requests. The synthetic PDF's labels are in the diagram; its parsed text separately lists FastAPI and Qdrant and identifies the document-question-answering project. The live answer correctly cited page 1 text versus page 2 image-description evidence, acknowledged that no arrowheads establish flow direction, and avoided inventing how FastAPI is used. Manual review checked it against the actual PDF and saved visual. This is limited synthetic-case validation rather than a broad visual-accuracy claim.

| Live request | Purpose | HTTP | Input tokens | Output tokens | Total |
|---|---|---|---|---|---|
| 1 | Fresh image caption during upload | 200 | 321 | 109 | 430 |
| 2 | Three numbered question types in one answer | 200 | 452 | 331 | 783 |
| 3 | Fresh caption during same-file re-ingestion | 200 | 321 | 113 | 434 |
| Total | No retries/diagnostic model calls | — | 1,094 | 553 | 1,647 |

**DeepSeek API usage for Phase 6: exactly 3 successful requests.** The earlier Phase 3 report remains two requests/891 tokens, so all recorded live validation across the work totals **5 requests/2,538 tokens**. No currency charge is inferred from token counts, and no further paid calls were made after this successful run.

Live acceptance results: upload returned two text chunks, one image description, three total vectors, and no warnings. Both modalities occupied the one isolated collection with matching **768-dimensional** embeddings. Retrieval returned page-1/page-2 text and an image source; the image endpoint returned original PNG evidence. Re-ingestion preserved record IDs and counts with no duplicates or broken image URLs. The prior separate Phase 2 vector-diagram extraction tests and Phase 3 live vector-caption evidence remain part of coverage; all vector extraction tests also passed inside the pinned image.

Real HTTP checks then restarted the API and Qdrant containers using only owned synthetic probes. Health recovered; original image bytes remained available through the public image route; the bind-mounted file stayed visible on the host; and a test Qdrant point survived the restart in its named volume. Those persistence probes were removed afterward. The API image/service had access to the key at runtime, but no key or private CV data was copied into source/reports. An actual Git ignore check in a temporary repository confirmed `.env`, uploaded documents, and derived images are ignored, while `data/sample.md` and `.gitkeep` remain permitted. The supplied projects themselves still have no Git history; none was initialized or committed.

Evidence under `text-rag/reports/`:

- [Live acceptance report](text-rag/reports/phase6_e2e_validation.json): counted requests, real token usage, vector checks, grounded answer, and citations.
- [Runtime persistence report](text-rag/reports/phase6_runtime_validation.json): API/Qdrant restarts and bind-mount checks, zero paid calls.
- [Environment report](text-rag/reports/phase6_environment_validation.json): installed pin equality and no baked-in environment file.
- [Ignore-rule report](text-rag/reports/phase6_ignore_validation.json): actual ignored/retained path checks.
- [Demo PDF](text-rag/reports/demo_cv.pdf) and [source visual](text-rag/reports/phase6_source.png): non-sensitive evidence for reproducing the demo.

README now documents the pinned environment, host/container configuration, demo upload/questions/re-ingestion, optional bounded live check, and remaining limitations. Temporary `rag-phase6` containers/network were stopped after checks; named volumes and the user's running host Ollama were retained. The validated image was tagged for normal `text-rag` Compose use. Original sample/credential files were unchanged. Start the normal demo with `docker compose up --build -d` from `text-rag/`.

All phases are complete. Remaining limitations are the documented v1 boundaries: caption/extraction quality, no OCR, full-rebuild caption costs, nontransactional collection replacement, local single-user scope, and synthetic rather than broad CV benchmarking. They are not unverified claims of production readiness.

## End-to-end acceptance sequence

1. Start Qdrant, Ollama with the embedding model, the API, DeepSeek access, and the selected Vision LLM.
2. Upload a non-sensitive PDF containing text and a raster diagram; use a separate vector-diagram PDF to assess the configured rendering fallback.
3. Inspect the shared `documents` collection and confirm both text and image-caption nodes with the expected metadata and vector dimensions.
4. Ask a text-focused question and verify its answer against the source page.
5. Ask a diagram-focused question and verify the caption-grounded answer and displayed original image.
6. Ask a mixed question and verify retrieval across text and visual descriptions with correct filename/page/modality citations.
7. Re-ingest and confirm consistent rebuilding, no duplicate vectors, and valid image URLs.
8. Verify that a captioning failure produces a visible warning while valid text remains ingestible; check missing-image behavior and path traversal rejection.
9. Record test results and demo evidence. Mark implementation complete only after these checks actually pass.

## Dependencies and decisions

- Phase 0 establishes the contract for all later work. Phase 2 and mocked captioning can proceed while the live vision provider remains undecided.
- Phase 4 depends on extraction and the captioner interface; its live validation requires Phase 3. Phase 5 consumes the resulting metadata and saved images. Phase 6 validates the complete workflow.
- Resolve provider/model selection, rendering fallback triggers, image filtering thresholds, upload/sample migration, and warning response format during their respective phases.
- Keep the implementation small and reviewable. Caption quality limits visual retrieval, and vector diagrams may require rendering; document coverage limits instead of claiming complete PDF understanding.

## V1 boundaries

Do not introduce CLIP retrieval, a second embedding space or vector store, score fusion, CMRS training/gallery dependencies, uploaded-image queries, OCR, table reconstruction, rerankers, agents, queues, microservices, cloud storage, multi-user authentication, incremental ingestion, or a frontend framework. Preserve the existing application architecture and defer those extensions.

## Definition of done

One PDF upload supports text and visual-description retrieval through one embedding model and one Qdrant collection, produces grounded answers with correct source metadata, and displays the original retrieved images. Existing functionality remains usable, tests pass, Docker/runtime integration is verified, and README instructions reproduce the demonstrated workflow.
