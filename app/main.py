"""FastAPI app: routes only. The real work lives in ingest.py and query.py.

Run from the project root:  uvicorn app.main:app --reload
Interactive docs:           http://localhost:8000/docs
"""
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import settings
from app.ingest import (DATA_DIR, SUPPORTED_EXTS, IngestionError, get_qdrant_client,
                        ingest_directory_with_stats, migrate_legacy_documents)
from app.pdf_processor import PdfProcessingError
from app.query import DocumentSelectionError, answer_question
from app.image_paths import generated_image_path
from app.documents import document_inventory

app = FastAPI(title="RAG Document Q&A")

# Qdrant or Ollama being down shows up as one of these connection errors.
UNREACHABLE = (ResponseHandlingException, httpx.ConnectError, httpx.TimeoutException, ConnectionError)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


MAX_HISTORY = 8  # keep only the latest turns to limit tokens/cost


class QueryRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []  # earlier turns of this chat session; empty = new chat
    document_ids: list[Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]] | None = Field(default=None, max_length=100)

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class Source(BaseModel):
    file: str
    score: float
    page: int | None = Field(default=None, ge=1)
    modality: Literal["text", "image"] | None = None
    image_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    try:
        get_qdrant_client().get_collections()
    except UNREACHABLE:
        raise HTTPException(503, "Qdrant is unreachable")
    return {"status": "ok", "qdrant": "up"}


@app.get("/documents")
def list_documents():
    try:
        return {"documents": document_inventory(DATA_DIR, get_qdrant_client())}
    except UNREACHABLE:
        raise HTTPException(503, "Qdrant is unreachable; document index status is unavailable") from None


@app.post("/documents")
def upload_document(file: UploadFile):
    name = Path((file.filename or "").replace("\\", "/")).name
    if Path(name).suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(
            400, f"Unsupported file type. Allowed: {', '.join(SUPPORTED_EXTS)}"
        )
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "Document exceeds the configured upload size limit")
    try:
        documents_dir = migrate_legacy_documents(DATA_DIR)
    except IngestionError as exc:
        raise HTTPException(409, str(exc)) from None
    target = documents_dir / name
    if target.is_symlink():
        raise HTTPException(400, "Upload destination must not be a symlink")
    previous = target.read_bytes() if target.exists() else None
    target.write_bytes(content)
    try:
        stats = ingest_directory_with_stats(DATA_DIR)
    except UNREACHABLE:
        if previous is None:
            target.unlink()
        else:
            target.write_bytes(previous)
        raise HTTPException(503, "Qdrant or Ollama is unreachable")
    except (PdfProcessingError, IngestionError) as exc:
        if previous is None:
            target.unlink()
        else:
            target.write_bytes(previous)
        raise HTTPException(400, str(exc)) from None
    return {"file": name, "chunks_in_collection": stats.total_vectors,
            **asdict(stats), "warning_count": len(stats.warnings)}


@app.post("/query", response_model=QueryResponse, response_model_exclude_unset=True)
def query(req: QueryRequest):
    try:
        history = [t.model_dump() for t in req.history[-MAX_HISTORY:]]
        if req.document_ids is not None:
            if not req.document_ids:
                raise HTTPException(400, "Select at least one indexed document.")
            indexed = {item["id"] for item in document_inventory(DATA_DIR, get_qdrant_client()) if item["indexed"]}
            if not set(req.document_ids).issubset(indexed):
                raise HTTPException(409, "Selected documents are no longer indexed. Refresh the document list and select again.")
            return answer_question(req.question, history, document_ids=req.document_ids)
        return answer_question(req.question, history)
    except UNREACHABLE:
        raise HTTPException(503, "Qdrant or Ollama is unreachable")
    except DocumentSelectionError as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError:
        # Raised by Qdrant client when the collection doesn't exist yet.
        raise HTTPException(404, "No documents ingested yet. POST a file to /documents.")


@app.get("/images/{document_id}/{filename}", response_class=FileResponse)
def source_image(document_id: str, filename: str):
    path = generated_image_path(document_id, filename)
    if path is None or not path.is_file():
        raise HTTPException(404, "Source image not found")
    return FileResponse(path, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})


@app.get("/", include_in_schema=False)
def demo_page():
    """Single-page demo UI (upload + ask), calling the endpoints above."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")
