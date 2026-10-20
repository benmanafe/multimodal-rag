# Multimodal CV RAG — Integration Plan for Codex

**Status:** Planning / implementation handoff  
**Sources reviewed:** Project 1 `multimodal-rag-main.zip`; Project 2 `cross-modal-retrieval-flickr30k-main.zip` (October 2026 snapshots)  
**Scope:** Extend Project 1 to process image-containing CV/PDF documents. Project 2 has been inspected: it contains a trained CLIP-style dual-encoder retrieval system, **not a generative Vision LLM**. A new Vision LLM adapter is required.

## 1. Goal and guiding constraints

Build a clear, demo-friendly **multimodal document RAG system** that answers questions from textual and visual contents of CVs/resumes and other PDFs.

- Use **one Qdrant collection** and **one text embedding model** for all searchable representations.
- For images, generate **text descriptions with a Vision LLM**, embed those descriptions with the existing text embedding model, and keep a pointer to the **original image**.
- Query that same vector collection for text chunks and image descriptions; answer with retrieved evidence, source file/page, and display original image(s) when applicable.
- Prefer minimal, readable changes to existing Python modules over an architecture rewrite.
- Avoid CLIP-based parallel retrieval, cross-modal vector alignment, microservices, queues, complex orchestration, OCR, and cloud storage in **v1**.
- This is *caption-mediated multimodal RAG*, not true image-query/image-embedding similarity search. Caption quality limits visual retrieval quality.

## 2. Verified current repository

```text
app/
  __init__.py
  config.py
  ingest.py
  main.py
  query.py
  static/index.html
data/sample.md
requirements.txt
Dockerfile
docker-compose.yml
.env.example
README.md
PLAN.md
tests/test_api.py
tests/test_chunking.py
tests/test_pdf_reader.py
```

| Capability | Existing implementation | Decision |
|---|---|---|
| Python | Python 3.12 Docker base | Keep |
| Web API | FastAPI in `app/main.py` | Keep and extend |
| Demo UI | HTML/JS in `app/static/index.html` | Keep and enhance |
| Text loading | LlamaIndex `SimpleDirectoryReader` in `app/ingest.py` | Keep for TXT/MD; use PDF-specific extraction |
| Chunking | `SentenceSplitter`, size 512, overlap 50 | Keep for text |
| Embeddings | Ollama `nomic-embed-text` via `get_embed_model()` | Reuse for text and captions |
| Vectors | Qdrant collection `documents` via `get_qdrant_client()` | Keep |
| Answer LLM | DeepSeek `deepseek-chat` | Keep for text-grounded answers |
| Retrieval/chat | LlamaIndex `condense_plus_context` in `app/query.py` | Keep, add source metadata |
| Containerization | API + Qdrant Docker Compose; Ollama on host | Keep |
| Tests | API, chunking, PDF reader import | Extend |

**Current paths:** `app/ingest.py` loads `.txt`, `.md`, `.pdf`, splits documents, **deletes the existing Qdrant collection on every ingestion**, then stores a fresh index. `app/query.py` returns `answer` and sources containing `file` and `score` only. `app/main.py` exposes `GET /health`, `POST /documents`, `POST /query`, and `GET /`. Uploads are saved in `data/` and trigger re-indexing. The UI is served by the FastAPI backend.

**Review scope:** This is a source review, not a verified runtime integration test. No Project 2 code has been assessed.

## 3. Target architecture

```text
CV PDF / TXT / MD
       |
       +--> PDF parser --> page text ----------> SentenceSplitter --> text records --+
       |                                                                           |
       +--> PDF parser --> image crops/files --> Vision LLM --> image descriptions -+
       |                                                                           |
       +--> TXT/MD reader ----------------------> SentenceSplitter --> text records --+
                                                                                   |
                           shared Ollama nomic-embed-text embedding model <--------+
                                                  |
                                           Qdrant `documents`
                                                  |
User query --> embed query --> similarity search (text + image-description nodes)
                                                  |
                 DeepSeek text answer grounded in retrieved descriptions/chunks
                                                  |
                Answer + source filenames/page numbers + optional image URLs
                                                  |
                                      Existing FastAPI HTML/JS UI
```

**Important distinction:** The answer LLM need not receive raw image pixels in v1. Captions are text context; the app displays saved source images. Later we may optionally send retrieved images to a vision-capable answer model for visual verification.

## 4. Required shared data contract

Use LlamaIndex `Document` or nodes with a consistent metadata schema; avoid adding a separate data-model framework unless useful.

Minimum conceptual representation:

```python
{
    "id": "<stable_document_id>_p2_image_01",  # ID of record/node
    "text": "Architecture diagram showing ...",  # chunk or vision-generated caption
    "metadata": {
        "file_name": "resume.pdf",
        "document_id": "<stable_document_id>",
        "page": 2,                  # 1-based PDF page
        "modality": "image",        # "text" or "image"
        "image_path": "data/images/<document_id>/page_2_img_01.png",
    },
}
```

- A **text** record uses `modality="text"`, `image_path` omitted or `None`, and an actual text chunk.
- An **image** record uses `modality="image"`, `text` set to a concise, factual vision-generated description, and a relative, validated image path.
- Keep raw image files on disk; **do not** store image blobs/base64 in Qdrant metadata.
- Maintain page number and original file name on every record; avoid losing metadata when chunking.
- Stable document IDs can be derived from a filename + content hash; exact scheme can be chosen during implementation.
- Store searchable caption text and original image path separately. Include enough context in captions to make them searchable (e.g. diagram labels, relationships, project name); avoid hallucinating unreadable content.
- **Image captions should remain single nodes** rather than being split by the text chunker. Split text only.
- Use the same embedding model and vector dimension for both. When changing models, rebuild the collection.

## 5. Proposed minimal file layout

```text
app/
  config.py                # MODIFY: vision model, image path, PDF extraction options
  ingest.py                # MODIFY: orchestration, shared Qdrant writes
  query.py                 # MODIFY: return modality/page/image pointer
  main.py                  # MODIFY: safe image serving and expanded API response
  pdf_processor.py         # ADD: PDF page text + image extraction
  image_processor.py       # ADD: vision captioning interface
  static/index.html        # MODIFY: render matched images with cited sources

data/
  documents/               # ADD: uploaded files, except optional sample
  images/                  # ADD: persistent derived images
  sample.md                # EXISTING

tests/
  test_api.py              # MODIFY
  test_chunking.py         # EXISTING
  test_pdf_reader.py       # EXISTING
  test_pdf_processor.py    # ADD
  test_image_processor.py  # ADD
  test_multimodal.py       # ADD

requirements.txt           # MODIFY: PyMuPDF + whichever vision client is selected
.env.example               # MODIFY: new env-var names, no credentials
.gitignore                # MODIFY if needed to ignore new data paths
Dockerfile                 # CHECK packages, permissions
 docker-compose.yml        # CHECK mounted folders and host access
README.md                  # MODIFY: install, run, demo and limitations
```

> Keep the current `app/` architecture; splitting into many more packages is unnecessary for this demo. If a Project 2 module already handles PDF extraction/captioning cleanly, reuse it instead of adding duplicate implementations.

## 6. Specific code changes, by file

### `app/config.py` — MODIFY

Add minimally necessary env-backed settings, e.g. `vision_model`, `vision_base_url` or provider credentials (depending on Project 2), and `image_dir`. Keep existing `embed_model`, `ollama_base_url`, Qdrant, DeepSeek, chunking, and retrieval settings. **Do not select a new vision provider until inspecting Project 2.**

### `app/pdf_processor.py` — ADD

Implement a readable PDF parser, likely with **PyMuPDF (`pymupdf`)**, which returns page text and useful image paths with page numbers.

Suggested functions:

```python
extract_pdf_content(pdf_path: Path, images_dir: Path) -> list[Document]
# or two clearly named functions:
extract_page_text(pdf_path: Path) -> list[Document]
extract_page_images(pdf_path: Path, images_dir: Path) -> list[dict]
```

Requirements:
- Page numbers must be 1-based.
- Avoid decorative tiny images / duplicate logos when feasible.
- Some PDF diagrams are **vector graphics, not embedded raster images**. Embedded-image extraction alone can miss them; consider a page-region or full-page rendered-image fallback, with clear control over when it runs. Avoid indexing every full page blindly because it duplicates text and adds captioning cost.
- No OCR in the first version; scanned PDF handling is explicitly deferred or documented as limited.

### `app/image_processor.py` — ADD (or reuse from Project 2)

Suggested interface:

```python
def describe_image(image_path: Path, context: str = "") -> str:
    """Return a factual, searchable description of an image."""
```

Requirements:
- Reuse Project 2's Vision LLM when possible.
- Prompt for visible text/labels, diagram components, directions/relationships, and purpose; state uncertainty when unreadable.
- Captioning occurs at **ingest time**, not on every query.
- Handle model failures explicitly: log and skip an image without breaking all text ingestion, or report a clear ingest warning.
- Add mocks for tests; avoid hitting paid/model APIs in unit tests.

### `app/ingest.py` — MODIFY

Keep `get_embed_model()` and `get_qdrant_client()` as working wrappers. Refactor `ingest_directory()` to:

1. Discover `.txt`, `.md`, `.pdf` from the **document input directory only**. Do not accidentally ingest generated image folders or metadata.
2. For TXT/MD, preserve current reader and splitter behavior.
3. For PDFs, extract text and images with document/page metadata.
4. Split **text** into chunks using `SentenceSplitter` (512/50 by default).
5. Caption images, and make each caption its own LlamaIndex document/node.
6. Embed both types of text using **`get_embed_model()`**.
7. Persist both types to **the same Qdrant collection** (`settings.qdrant_collection`).
8. Return useful ingestion statistics: text chunks, image descriptions, total vectors, warnings (keep backward compatibility if practical).

**Index refresh policy for v1:** The existing full-rebuild behavior is acceptable for the demo. Make it explicit and consistent: remove/recreate the Qdrant collection, regenerate or reconcile image files, and avoid duplicate vectors. Don't accidentally erase other users' indexed documents; this is a *single-user local demo*. More sophisticated per-document upserts can be a later milestone.

**API compatibility consideration:** Existing `POST /documents` saves files at the root of `data/`. If changing uploads to `data/documents/`, update all code paths and tests together. Preserve existing `data/sample.md` behavior or clearly migrate it. Avoid recursive ingestion of `data/images/`.

### `app/query.py` — MODIFY

Keep `VectorStoreIndex.from_vector_store(...)`, `DeepSeek`, and `condense_plus_context` history semantics. Do not add query routers, separate indices, or CLIP retrieval in v1.

Expand returned source metadata to include `file`, `page`, `modality`, `score`, and a safe, **derived** image URL when appropriate. Show that results can mix modalities. Retain existing answer-generation prompting style, with an explicit instruction to answer only from retrieved evidence and distinguish image-description observations from verified details.

### `app/main.py` — MODIFY

- Preserve `/health`, `/documents`, `/query`, and `/`.
- Extend `Source` Pydantic schema with optional `page`, `modality`, `image_url`; keep `file` and `score` to avoid breaking clients.
- Provide a read-only image serving route (or verified static mount), mapping **trusted generated relative paths** to files under `data/images`. Validate against path traversal; never serve arbitrary user-specified filesystem paths.
- Consider a file-size limit for uploads; validate file extension and handle malformed PDFs with a clear error. Existing filename sanitization (`Path(...).name`) is helpful but not sufficient for production safety.
- Do not expose `.env` or private files over the image route.

Example response shape (illustrative only):

```json
{
  "answer": "The candidate built a PDF-to-vector RAG pipeline...",
  "sources": [
    {"file": "resume.pdf", "page": 1, "modality": "text", "score": 0.77, "image_url": null},
    {"file": "resume.pdf", "page": 2, "modality": "image", "score": 0.73, "image_url": "/images/<safe-image-id>"}
  ]
}
```

Scores are **examples only**, not measured retrieval results.

### `app/static/index.html` — MODIFY

- Continue supporting upload, text answer, sources, and chat history.
- For each source, show filename, page, and `text`/`image` modality.
- For image sources, render the referenced original image below/near the citation.
- Provide graceful fallback if image file is missing.
- Keep the existing single-page HTML/JS approach; **do not introduce Streamlit or a frontend framework** for v1.

### `requirements.txt`, `.env.example`, Docker, README — MODIFY AS NEEDED

- Add PDF imaging support (e.g. `pymupdf`); add Vision LLM package only when provider is known.
- Keep Ollama embedding dependency, DeepSeek dependency, Qdrant, FastAPI, LlamaIndex.
- Verify host Ollama accessibility from the Docker API service (existing Compose uses `host.docker.internal`; portability may require platform-specific setup).
- Ensure `data/images` persists via the existing `./data:/app/data` volume.
- Document that ingestion requires Vision LLM access and that input CVs and derived images should not be committed.
- After a successful environment build, pin versions for reproducibility.

## 7. Tests and acceptance criteria

**Unit tests (mock external clients):**

- PDF extraction returns the correct text and 1-based page numbers for a two-page fixture.
- PDF image extraction saves an image and records a valid source path.
- Captioning receives an image and returns deterministic mock text; failures produce a visible warning.
- Text chunks and image captions carry `file_name`, `document_id`, `page`, `modality`, and image path when relevant.
- Ingestion sends both types through the **same embedding model** and **same collection**.
- Ingestion never picks up generated images as source documents.
- Retrieval serializes text and image source metadata correctly, and old text-only responses remain valid.
- API image path rejects traversal / missing images.
- Existing `pytest` suite remains passing.

**Manual end-to-end demo** (one non-sensitive test CV/PDF with text and a diagram):

1. Start Qdrant, Ollama embedding model, API, DeepSeek configuration, and Vision LLM.
2. Upload the PDF; verify both text nodes and image-caption nodes appear in `documents` collection.
3. Ask a **text-only** question (e.g., “Which Python libraries are mentioned?”).
4. Ask an **image-focused** question (e.g., “What is shown in the architecture diagram?”).
5. Ask a **mixed** question (e.g., “Explain the project using its text and diagram.”).
6. Confirm citations identify file/page/modality, and visual matches display the **original image**.
7. Re-ingest and confirm the collection is rebuilt consistently without duplicate vectors or broken image paths.

**Done means:** one PDF upload, one shared Qdrant collection, correct text/image description retrieval, grounded answer, original-image display, passing tests, and a clear README demo sequence. Do **not** claim this until actually tested.

## 8. Suggested implementation sequence

| Phase | Work | Exit condition |
|---|---|---|
| 0 | Create branch, run baseline tests, document current behavior | Baseline captured |
| 1 | Select a new Vision LLM caption provider/model and implement a tiny adapter | Provider confirmed; captions created for unseen CV images |
| 2 | Add PDF text/image extractor and source metadata | Extractor tests pass |
| 3 | Add image captioning adapter and mock tests | Image becomes a searchable caption |
| 4 | Refactor ingestion to unified text + caption records in Qdrant | Both modality types indexed |
| 5 | Extend retrieved sources, safe image-serving API, and UI | Image sources displayed |
| 6 | Test, update README, validate Docker and one real PDF | Demonstrable end-to-end workflow |

**Project 2 has now been inspected; it does not supply a generative Vision LLM. Do not try to integrate its caption-retrieval endpoint as a caption generator.** Independent work on PDF extraction, metadata contracts, and tests can begin now. Keep each phase reviewable and backward-compatible as much as possible.

## 9. Explicit non-goals for the first version

- CLIP joint-embedding search or advanced text-image alignment.
- Running dual vector stores with weighted score fusion.
- Agents, orchestration graphs, rerankers, asynchronous worker queues.
- OCR, handwriting extraction, table structure reconstruction, or perfect coverage of diagrams inside PDFs.
- Multi-user auth, distributed deployment, complex storage, incremental ingestion, or expensive monitoring.
- Replacing FastAPI, the HTML demo, Qdrant, Ollama embeddings, or DeepSeek without a demonstrated need.

## 10. CMRS audit: what Project 2 actually contains

Repo: `cross-modal-retrieval-flickr30k-main` (inspected ZIP). It is a Flickr30k **contrastive dual-encoder retrieval** experiment:

- `cmrs/model.py`: custom `ImageEncoder` (ConvNeXt-Tiny or EfficientNet-B0), `TextEncoder` (MiniLM transformer), `CrossModalModel`, learned projection into a normalized shared embedding space (best experiment: 1024 dimensions).
- `cmrs/data.py`: JPEG loading, PIL conversion/transforms, Flickr30k Karpathy splits, image cache, dataset/data loaders. Extract conceptual image validation/preprocessing patterns if helpful; **do not copy training-specific datasets into the CV app**.
- `cmrs/losses.py`, `cmrs/train.py`, `cmrs/evaluate.py`, `configs/`, `runs/`: contrastive training and retrieval evaluation; retain in original repository, **do not import into v1 RAG**.
- `server.py`: FastAPI demo with `/api/text-to-image`, `/api/image-to-text`, fixed precomputed gallery embedding tensors and image static file serving. Its image-to-text route retrieves **nearest existing captions**, it does **not** generate new captions for arbitrary images.
- `export_deploy.py`: packages trained model, precomputed captions/gallery embeddings, and sample gallery images. Not applicable to newly uploaded CVs.
- `web/index.html`: optional reference for displaying retrieved images/similarity scores; continue using Project 1's existing UI instead.
- No PDF parsing and **no generative Vision LLM client or prompt** was found.
- README reports best experiment `exp3_convnext` text→image R@1=47.3 and image→text R@1=58.0 on Flickr30k test. These are dataset-specific reported results, not evidence of CV diagram accuracy.
- Model checkpoints and Flickr30k dataset are **not present in the ZIP**; CMRS deployment is not directly runnable with the supplied archive alone.

**Integration decision:** Keep CMRS as a separate learning project and optional comparative extension. V1 should use Project 1's text embeddings for both document chunks and newly generated vision captions, with **no dual-encoder model, second embedding space, gallery, or score fusion**. CMRS contributes conceptual image handling/UI patterns rather than production code that must be imported.

## 11. Remaining decisions and technical caveats

1. **Choose a genuine Vision LLM:** e.g., an API or local vision-capable model. Do not equate CMRS image→caption retrieval with caption generation. Make a narrow `describe_image(image_bytes, context=None) -> str` adapter so choice is isolated.
2. **PDF visual coverage:** extract raster images with PyMuPDF; also consider rendering page/region crops for vector-drawn diagrams, but avoid indiscriminately describing every page (duplicated text, cost).
3. **Image relevance:** skip tiny logos/decorative elements; link caption records to the original image file, PDF name, and page number. Save a small optional neighboring-text context.
4. **Metadata:** preserve `modality`, `source_file`, `page_number`, `image_path`/safe ID, and stable record ID; filenames alone may collide. Preserve image pointers in LlamaIndex/Qdrant nodes.
5. **Runtime:** test image client credentials/network from the API container, while avoiding checked-in CVs/API keys. Original-image read route must prevent `../` traversal.
6. **Demo boundaries:** initial interface is **text question → search in text and image descriptions**; arbitrary uploaded-image query → semantic answer is explicitly out of scope.
7. **Quality assurance:** test PDF with true raster image and separate PDF with vector diagram; evaluate answers against known page references, and show original evidence rather than assuming the caption is always correct.

## 12. Instructions to Codex

> You are extending the existing Python repository, **not creating a new project from scratch**. Read `app/config.py`, `app/ingest.py`, `app/query.py`, `app/main.py`, the HTML UI, requirements, Docker files, and tests first. Follow this plan incrementally. Reuse Qdrant, Ollama `nomic-embed-text`, DeepSeek, LlamaIndex, FastAPI, and the existing UI. Implement one collection containing chunk text and Vision LLM image captions, with metadata that links to saved original images. Keep code small and readable. Preserve or carefully migrate existing API behavior. Write mocked tests and verify each change. **Project 2 is a Flickr30k-trained dual-encoder retrieval system, NOT a Vision LLM. Do not copy its training code into v1 and do not treat `/api/image-to-text` as caption generation. Implement a separate provider-configurable Vision LLM adapter to produce captions for previously unseen CV images.** Begin with extraction/metadata/mocked captioner and ask for provider credentials only at the integration boundary. Show a change summary, test results, and any unresolved assumptions at the end.

---

*This plan is derived from both supplied ZIPs. The exact Vision LLM provider remains a separate decision; the CMRS repository does not include one.*
