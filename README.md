# RAG Document Q&A

A simple retrieval-augmented generation (RAG) system that answers questions about your own documents (`.txt`, `.md`, `.pdf`), with sources, through a REST API and a small demo web page.

**Stack:** Python 3.12, LlamaIndex, Qdrant, FastAPI, Docker Compose. DeepSeek writes answers and image descriptions; a local Ollama model (`nomic-embed-text`) embeds both text chunks and descriptions.

**Validation status:** all six integration phases are complete. The pinned Linux/Python 3.12.15 image passed 114 tests and a real-service upload/query/re-ingestion check with host Ollama, server Qdrant, and DeepSeek. Desktop/mobile behavior was separately checked in a headless browser. See `plan.md` and the safe reports under `reports/`.

## Repository layout

The application now runs directly from this repository root. Run all commands below here; the former `text-rag/` folder has been flattened.

- `app/`: FastAPI backend, ingestion, retrieval, and chatbot UI.
- `data/`: existing documents, sample text, and extracted images; preserved locally and excluded from the Docker image.
- `tests/` and `scripts/`: automated tests and optional validation tools.
- `reports/`: validation evidence and the synthetic `demo_cv.pdf` used for demonstrations.
- `plan.md` and `MULTIMODAL_RAG_INTEGRATION_PLAN_V2.md`: integration progress and design.
- `archive/`: the unused Flickr30k project and superseded text RAG plan, retained for reference and excluded from Docker builds.

The existing `.env` was retained. Compose keeps the project name `text-rag`, so the existing `text-rag_qdrant_data` volume is reused when starting from this root. Uploaded documents now live under this root's `data/` folder.

## How it works
1. **Ingest:** files directly in `data/documents/`, plus `data/sample.md`, are processed. Text is split into chunks (512 tokens, 50 overlap); PDF visuals are extracted and captioned at ingestion time. Both use the same text embedding model and Qdrant collection.
2. **Query:** the question is embedded, the 3 most similar chunks are retrieved, and DeepSeek answers using only those chunks.
3. The answer comes back with source filenames, pages, modalities, and similarity scores. The UI displays saved original visuals beside image citations.

## Prerequisites
- Docker (with Compose)
- [Ollama](https://ollama.com) running on the host, with the embedding model pulled: `ollama pull nomic-embed-text`
- A [DeepSeek API key](https://platform.deepseek.com)

## Run it
```bash
git clone https://github.com/benmanafe/multimodal-rag.git
cd multimodal-rag
cp .env.example .env          # then put your DEEPSEEK_API_KEY in .env
docker compose up --build -d
docker compose exec api python -m app.ingest   # index the sample document (data/sample.md)
```
- Demo page: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

Stop with `docker compose down` (add `-v` to also delete the stored vectors).

## Choose the files used by a chat

Open **RAG documents** in the sidebar to see source filenames, indexed text/image counts, and files that still need indexing. Upload a file from this panel or with the paperclip; upload still rebuilds the full index and can incur caption API calls.

Keep **All indexed files** checked to search the entire collection, including future uploads. To search specific files, adjust the file checkboxes and click **Apply to chat**. The **Searching** button above the conversation shows the applied scope and opens the panel on mobile. Selections are saved per chat in this browser. Applying a different scope to an existing conversation starts a new chat so earlier answers cannot influence the new scope. Empty selections cannot be applied.

Selecting files only filters retrieval from the existing index; it does not re-embed documents or call DeepSeek. Index-only sources remain searchable even if their original file was removed from disk. Files marked **Not indexed** must be uploaded before selection. If a selected file was replaced or removed from the index, refresh and select again; requests fail rather than silently searching other files.

`GET /documents` returns filenames, source IDs, availability, index status, and text/image counts. `POST /query` accepts optional `document_ids`: omit it or send `null` for all indexed files, or provide a nonempty array of source IDs from the inventory to restrict retrieval. Filtering happens in Qdrant before ranking and applies to both text and image captions. Existing query clients remain compatible.

The document-selection update passed 120 automated tests, including retrieval against real in-memory Qdrant with deterministic embedding/answer models. Browser checks cover inventory, selected-file requests and citations, separate chat scopes, selection persistence, and mobile panel access. These checks use no paid model calls.

## API examples
```bash
# Health
curl http://localhost:8000/health

# Upload and index a document (rebuilds the full document collection)
curl -F "file=@notes.pdf" http://localhost:8000/documents

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Why does chunk size matter?"}'
# {"answer": "...", "sources": [{"file": "sample.md", "score": 0.67, "page": null, "modality": "text", "image_url": null}]}
```
Uploads retain `file` and `chunks_in_collection` and additionally return `text_chunks`, `image_descriptions`, `total_vectors`, `warnings`, and `warning_count`. The legacy count now includes both text and caption vectors. Caption failures can return a successful text-only rebuild with source-aware warnings; inspect these warnings rather than assuming every image was indexed.

Errors include `400` unsupported file type/malformed or encrypted PDF/no searchable content, `409` legacy filename collision, `413` upload too large, `422` empty question, `404` nothing ingested yet or missing image, and `503` caught Qdrant/Ollama connection or timeout failures. Uploads default to a 20 MiB file limit (`MAX_UPLOAD_BYTES`), checked before saving or caption requests. Failed new uploads are removed, or the previous same-name file restored, for handled ingestion/service errors.

## Run without Docker for the API
```bash
pip install -r requirements.txt
docker compose up -d qdrant     # Qdrant still runs in Docker
python -m app.ingest
uvicorn app.main:app --reload
```
Outside Docker, `.env` should keep `QDRANT_HOST=localhost` and `OLLAMA_BASE_URL=http://localhost:11434`.

Docker Desktop was validated with API-to-host Ollama access through `host.docker.internal`. If that address is unavailable on native Linux, add an API `extra_hosts` mapping of `host.docker.internal:host-gateway` and ensure Ollama listens on an interface the container can reach. Keep the Compose `QDRANT_HOST=qdrant` override inside containers.

## Reproduce the multimodal demo

1. Start host Ollama and ensure `nomic-embed-text` is installed (`ollama pull nomic-embed-text`). Keep your existing `.env`; for a fresh setup, create it from `.env.example` and set `DEEPSEEK_API_KEY`. The caption defaults are `VISION_PROVIDER=deepseek` and `VISION_MODEL=deepseek-flash`.
2. Run `docker compose up --build -d` from the repository root. Check `http://localhost:8000/health` for API/Qdrant readiness. Health does not check embedding/model access; confirm Ollama separately before uploading.
3. Open `http://localhost:8000/` and upload the supplied non-sensitive `reports/demo_cv.pdf`. It has library/project text on page 1 and a raster architecture diagram on page 2. The retained `data/sample.md` is also included in ordinary ingestion, so normal collection counts can differ from the isolated validation report.
4. Ask which libraries are listed, what the diagram contains, and how the text and diagram relate. To minimize answer calls, combine those as three numbered questions in one message. Check the answer against the visible file/page/modality citations and source image. Ask separate focused questions when evaluating retrieval quality across a larger input set.
5. Upload the same PDF again and confirm consistent counts and usable image references. This rebuild makes another caption request; each answer also incurs API usage. Files and vectors persist across API/Qdrant container restarts.
6. Stop services with `docker compose down` when finished; this retains the named Qdrant volume. The temporary `rag-phase6` validation containers were stopped after testing, and host Ollama was left running.

The diagram deliberately uses connecting lines without arrowheads. The validated answer identified that uncertainty and did not claim visual verification. A caption/answer can still be wrong: inspect the displayed source and avoid inferring details absent from the PDF.

To regenerate the free synthetic PDF with a configured local Python environment:

```bash
python scripts/validate_e2e.py --write-fixture reports/demo_cv.pdf
```

For an optional real-service acceptance check inside the running Compose environment:

```bash
docker compose run --rm --no-deps --entrypoint python --volume "$PWD/scripts:/validation:ro" --volume "$PWD/reports:/app/reports" api -B /validation/validate_e2e.py --live
```

This opt-in script caps paid requests at three: initial caption, one combined numbered answer, and re-ingestion caption. It checks local services before spending tokens, limits answer output to 512 tokens, disables answer SDK retries for the check, and records HTTP statuses/usage. It exercises the real FastAPI handlers through TestClient inside Linux, real Ollama embeddings, server Qdrant, and DeepSeek; no caption or answer is mocked. It uses temporary inputs and an isolated collection, leaving your normal `documents` index intact. Each rerun incurs fresh charges and replaces `reports/phase6_e2e_validation.json`; ordinary pytest never runs it.

The recorded Phase 6 run passed with **3 requests and 1,647 tokens**, indexing two text chunks and one caption with matching 768-dimensional vectors. All calls returned HTTP 200. The previous Phase 3 check used 2 requests/891 tokens; total recorded live validation usage is **5 requests/2,538 tokens**. Restart/persistence and ignore-rule checks used no paid requests. Reports include `phase6_e2e_validation.json`, `phase6_runtime_validation.json`, `phase6_environment_validation.json`, and `phase6_ignore_validation.json`.

## Dependency reproducibility

`requirements.txt` pins direct dependencies and constrains transitive versions through the 109-entry `requirements.lock`. Both were captured only after a clean image build passed tests and `pip check`; the pinned image was then rebuilt and retested. The Dockerfile pins the tested Python base digest. The validated platform is Linux/amd64 on Docker Desktop with Python 3.12.15; the prior local Python 3.12.9 environment also passed all 114 tests but was not modified to match the new lock.

To run the suite inside the pinned image without model/network access:

```bash
docker compose build api
docker compose run --rm --no-deps --volume "$PWD/tests:/app/tests:ro" api python -B -m pytest -q -p no:cacheprovider
```

Validation additionally ran the suite with Docker networking disabled: 114 passed in 4.24 seconds. Two non-failing warnings were observed: Starlette's TestClient deprecates its current HTTPX backend, and local in-memory Qdrant ignores payload indexes. Production API operations and the real-service check succeeded. Update dependency pins only after building and testing the replacement environment; regenerating a lock from an unrelated workstation environment is not a substitute for that check.

## Tests
```bash
pytest
```
The tests mock model/network calls and use mocked or in-memory Qdrant, so they need no services or API keys. The in-memory adapter can emit an expected warning that payload indexes have no effect locally; server behavior is a later runtime check.

## Configuration
All settings live in `.env` (see `.env.example`): model names, Qdrant host and collection, chunk size and overlap, top-k.

## Multimodal development: PDF extraction

The standalone `app/pdf_processor.py` extractor returns unsplit page text, saved image candidates, and warnings. Extraction needs no model server or API key. `app.ingest.prepare_pdf_documents(...)` additionally creates one unsplit text node per successful image caption and preserves page text and source-aware warnings when a caption fails. Uploads and `python -m app.ingest` use this preparation workflow and index both modalities in the same collection.

### Answers, citations, and images

`POST /query` preserves `answer`, `sources`, `file`, and `score`, and adds optional `page`, `modality`, and `image_url` source fields. TXT/MD sources have no page; legacy source dictionaries remain accepted. The existing `condense_plus_context` history behavior is preserved, with an explicit grounding instruction to use retrieved evidence, acknowledge missing information, and distinguish generated image descriptions from verified extracted text. The answer model receives text evidence, not raw image pixels.

The single-page UI shows filename/page/modality citations, responsive saved-image previews, a missing-image fallback, upload counts, and source-aware ingestion warnings. These remain available in browser-local chat history. Source images open through `GET /images/<document_id>/<generated_filename>`; URL derivation and serving share checks for generated names, image-root containment, and symlinks. No arbitrary disk paths, environment files, or external image URLs are served by this route. Responses disable caching and content-type sniffing.

The preview shows the saved embedded raster or rendered vector crop, rather than an AI-generated replacement. Existing filename/score-only messages still render. Image files may become unavailable after a later rebuild removes their source, in which case older chats show the fallback.

PDF pages retain their original filename, stable document ID, and 1-based page number. Useful raster images are saved beneath `IMAGE_DIR` (default `data/images`) with relative paths. Small images and duplicate logos are filtered; larger repeated visuals retain their page citations. Supported raster bytes are retained where possible, transparency is restored from PDF masks, and other image formats are normalized to PNG.

`PDF_VECTOR_FALLBACK=true` enables a heuristic cropped-render fallback for connected drawing clusters. It requires at least `PDF_VECTOR_MIN_PATHS` paths (default 3), a substantial region, and excludes near-full-page frames. `PDF_RENDER_DPI` defaults to 144, with rendered crops capped near 2048 pixels on their longest side. Set the fallback to `false` to extract rasters and text only. `PDF_MIN_IMAGE_PIXELS` defaults to 64 on each source-image dimension. See `.env.example` for all options.

The vector heuristic can miss sparse, disconnected, or page-sized diagrams and can mistake decorative clusters for useful visuals. Crops are identified as `vector_crop`, distinguishing them from extracted raster evidence. There is no OCR pipeline: scanned PDFs may yield image evidence but no searchable page text; the extractor reports a warning. Malformed and password-protected PDFs produce explicit document errors. Image-level decode/save/render failures produce warnings while preserving extracted text. Generated files must stay inside a dedicated image folder in the application directory.

### Input migration and rebuild behavior

New uploads go to `data/documents/`. On ingestion, legacy top-level TXT/MD/PDF files move there, while the original `data/sample.md` remains in place and included. Identical legacy/destination files are retained once; differing same-name files produce a collision error before migration moves anything. Discovery is nonrecursive and excludes derived-image folders, metadata, unsupported extensions, and symlinks. Uploading an existing basename replaces that document in this single-user demo.

Text and captions are fully prepared and embedded before the old Qdrant collection is removed. Every node uses the same embedding batch/model, and vector counts/dimensions are checked. Text chunks retain their source page; captions stay single nodes. Empty input or preparation/embedding failure preserves the current index. Rebuilds reproduce IDs and avoid duplicate vectors for unchanged inputs/settings. Collection replacement is not transactional: a database write failure after deletion can leave the index incomplete, requiring another ingestion run.

After a successful counted rebuild, the app removes only recognized stale generated images beneath hash-named document folders in `IMAGE_DIR`. Referenced images, unrecognized files, and symlinks are retained. The existing `data/*` ignore rule covers uploads and generated images, while preserving the original sample/placeholder exceptions.

**Cost:** each full rebuild captions the current PDF images again, once each, with no automatic retries or caption cache. Real uploads or CLI ingestion can therefore incur DeepSeek charges even for unchanged PDFs. Development tests use mocks; Phase 4 validation made zero live DeepSeek calls.

### Captioning validation

Configure `DEEPSEEK_API_KEY` in the ignored `.env`, with `VISION_PROVIDER=deepseek` and `VISION_MODEL=deepseek-flash` (see `.env.example`). Caption preparation sends each extracted image to DeepSeek once, with no automatic retries, and returns image-level warnings for failures. Existing Ollama embeddings remain separate; preparation alone does not use them or Qdrant. Caption nodes retain their original image pointer and model/provider provenance; opaque IDs, paths, and provenance are excluded from searchable caption text.

All ordinary `pytest` tests use mocks for model calls. An optional live check generates a non-sensitive two-page CV with a raster pipeline and a vector workflow, makes **at most two paid caption requests**, and writes a safe report to `reports/phase3_vision_validation.json`:

```bash
python scripts/validate_vision.py --live
```

Run this command only when another live check is needed: each execution can incur fresh API charges and replaces the prior report. It stops after the first HTTP/network failure, has no retries, prints no key, and removes its generated documents/images afterward. The script checks expected labels and preserves captions for manual review; those checks are a smoke test, not a general visual-accuracy benchmark. The Phase 3 recorded run used **2 requests and 891 tokens**; both synthetic captions passed. Container-to-provider access and end-to-end ingestion/retrieval remain later checks.

For an optional zero-model-cost browser check, `python scripts/validate_ui.py` starts a temporary local API with mocked services and a fresh headless Chrome/Edge profile. This developer script requires a local Windows Chrome/Edge installation and `websocket-client`; it is separate from API runtime dependencies and pytest. It checks desktop/mobile rendering, image loading/fallbacks, upload warnings, legacy messages, and history persistence. It writes safe synthetic evidence to `reports/phase5_ui_validation.json`, `reports/phase5_ui.png`, and `reports/phase5_ui_mobile.png`. Local browser-control connections may require permission outside a restrictive sandbox.

## Project layout
- `app/main.py`: FastAPI routes and the demo page
- `app/ingest.py`: load, chunk, embed, store
- `app/query.py`: retrieve and generate the answer
- `app/config.py`: settings from the environment
- `app/static/index.html`: demo web page
- `app/pdf_processor.py`: PDF page text, raster extraction, and vector crops
- `app/image_processor.py`: DeepSeek/Ollama captioning boundary
- `app/image_paths.py`: shared image URL/path validation
- `data/documents/`: uploaded documents (gitignored)
- `data/images/`: generated visual evidence (gitignored)
- `data/sample.md`: retained sample input
- `tests/`: tests

## Limitations
- Each upload rebuilds the collection from all inputs and regenerates captions (slow and potentially costly for large folders).
- Caption quality limits visual retrieval. The answer model cannot visually verify image pixels; inspect the displayed source evidence.
- Live validation used one synthetic CV, with earlier separate raster/vector caption checks; this is a working demo, not a general CV-accuracy benchmark.
- No authentication; intended for local use.
