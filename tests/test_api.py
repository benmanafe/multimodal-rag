"""API tests. Qdrant and the LLM are mocked, so no services or API keys are needed."""
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.main import app
from app.ingest import IngestionStats
from app.pdf_processor import PdfProcessingError

client = TestClient(app)


def test_health_ok():
    with patch("app.main.get_qdrant_client", return_value=MagicMock()):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_when_qdrant_down():
    with patch("app.main.get_qdrant_client", side_effect=ResponseHandlingException("down")):
        assert client.get("/health").status_code == 503


def test_query_returns_answer_and_sources():
    fake = {"answer": "Canberra.", "sources": [{"file": "geo.txt", "score": 0.72}]}
    with patch("app.main.answer_question", return_value=fake) as mock:
        r = client.post("/query", json={"question": "Capital of Australia?"})
    assert r.status_code == 200
    assert r.json() == fake
    mock.assert_called_once_with("Capital of Australia?", [])


def test_query_rejects_empty_question():
    assert client.post("/query", json={"question": "   "}).status_code == 422


def test_upload_rejects_unsupported_type():
    r = client.post("/documents", files={"file": ("bad.exe", b"x")})
    assert r.status_code == 400


def test_demo_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "RAG Document Q&amp;A" in r.text


def test_query_passes_history_and_trims_it():
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(12)
    ]
    fake = {"answer": "ok", "sources": []}
    with patch("app.main.answer_question", return_value=fake) as mock:
        r = client.post("/query", json={"question": "and then?", "history": history})
    assert r.status_code == 200
    sent = mock.call_args.args[1]
    assert len(sent) == 8 and sent[-1]["content"] == "m11"


def test_query_rejects_bad_history_role():
    body = {"question": "hi", "history": [{"role": "system", "content": "x"}]}
    assert client.post("/query", json=body).status_code == 422


def test_upload_uses_documents_folder_and_returns_compatible_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path / "data")
    warning = {"code": "empty_caption", "file": "notes.txt", "page": 1, "message": "Empty caption."}
    stats = IngestionStats(text_chunks=3, image_descriptions=1, total_vectors=4, warnings=[warning])
    with patch("app.main.ingest_directory_with_stats", return_value=stats) as ingest:
        response = client.post("/documents", files={"file": ("../notes.txt", b"hello")})
    assert response.status_code == 200
    assert response.json() == {"file": "notes.txt", "chunks_in_collection": 4,
                               "text_chunks": 3, "image_descriptions": 1,
                               "total_vectors": 4, "warning_count": 1, "warnings": [warning]}
    assert (tmp_path / "data/documents/notes.txt").read_bytes() == b"hello"
    ingest.assert_called_once_with(tmp_path / "data")


def test_bad_pdf_upload_is_removed_and_returns_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path / "data")
    with patch("app.main.ingest_directory_with_stats", side_effect=PdfProcessingError("invalid_pdf", "Could not parse PDF.")):
        response = client.post("/documents", files={"file": ("broken.pdf", b"bad")})
    assert response.status_code == 400 and response.json()["detail"] == "Could not parse PDF."
    assert not (tmp_path / "data/documents/broken.pdf").exists()


def test_failed_replacement_restores_previous_upload(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path / "data")
    folder = tmp_path / "data/documents"
    folder.mkdir(parents=True)
    (folder / "resume.pdf").write_bytes(b"old content")
    with patch("app.main.ingest_directory_with_stats", side_effect=PdfProcessingError("invalid_pdf", "Invalid PDF.")):
        response = client.post("/documents", files={"file": ("resume.pdf", b"bad replacement")})
    assert response.status_code == 400
    assert (folder / "resume.pdf").read_bytes() == b"old content"


def test_embedding_timeout_returns_503_and_removes_new_upload(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path / "data")
    with patch("app.main.ingest_directory_with_stats", side_effect=httpx.ReadTimeout("down")):
        response = client.post("/documents", files={"file": ("notes.txt", b"new content")})
    assert response.status_code == 503
    assert not (tmp_path / "data/documents/notes.txt").exists()


def test_migration_collision_returns_409_without_overwriting_input(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr("app.main.DATA_DIR", root)
    (root / "documents").mkdir(parents=True)
    (root / "notes.txt").write_bytes(b"old")
    (root / "documents/notes.txt").write_bytes(b"different")
    with patch("app.main.ingest_directory_with_stats") as ingest:
        response = client.post("/documents", files={"file": ("notes.txt", b"new")})
    assert response.status_code == 409
    ingest.assert_not_called()
    assert (root / "notes.txt").read_bytes() == b"old"
    assert (root / "documents/notes.txt").read_bytes() == b"different"


def test_query_preserves_new_mixed_source_fields():
    fake = {"answer": "Based on the retrieved description...", "sources": [
        {"file": "resume.pdf", "score": 0.8, "page": 1, "modality": "text", "image_url": None},
        {"file": "resume.pdf", "score": 0.7, "page": 2, "modality": "image",
         "image_url": "/images/" + "a" * 64 + "/page_2_img_0001.png"},
    ]}
    with patch("app.main.answer_question", return_value=fake):
        response = client.post("/query", json={"question": "Explain the diagram"})
    assert response.status_code == 200 and response.json() == fake


def test_upload_size_limit_rejects_before_saving_or_ingestion(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("app.main.settings.max_upload_bytes", 4)
    with patch("app.main.ingest_directory_with_stats") as ingest:
        response = client.post("/documents", files={"file": ("large.pdf", b"12345")})
    assert response.status_code == 413
    ingest.assert_not_called()
    assert not (tmp_path / "data").exists()
