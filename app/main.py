"""FastAPI app: routes only. The real work lives in ingest.py and query.py.

Run from the project root:  uvicorn app.main:app --reload
Interactive docs:           http://localhost:8000/docs
"""
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import settings
from app.ingest import DATA_DIR, SUPPORTED_EXTS, get_qdrant_client, ingest_directory
from app.query import answer_question

app = FastAPI(title="RAG Document Q&A")

# Qdrant or Ollama being down shows up as one of these connection errors.
UNREACHABLE = (ResponseHandlingException, httpx.ConnectError, ConnectionError)


class QueryRequest(BaseModel):
    question: str

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


@app.post("/documents")
def upload_document(file: UploadFile):
    # Path(...).name strips any directory parts so a filename can't escape data/.
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(
            400, f"Unsupported file type. Allowed: {', '.join(SUPPORTED_EXTS)}"
        )
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / name).write_bytes(file.file.read())
    try:
        # Re-indexes everything in data/ (see the note in ingest.py).
        chunks = ingest_directory()
    except UNREACHABLE:
        raise HTTPException(503, "Qdrant or Ollama is unreachable")
    return {"file": name, "chunks_in_collection": chunks}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return answer_question(req.question)
    except UNREACHABLE:
        raise HTTPException(503, "Qdrant or Ollama is unreachable")
    except ValueError:
        # Raised by Qdrant client when the collection doesn't exist yet.
        raise HTTPException(404, "No documents ingested yet. POST a file to /documents.")


@app.get("/", include_in_schema=False)
def demo_page():
    """Single-page demo UI (upload + ask), calling the endpoints above."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")
