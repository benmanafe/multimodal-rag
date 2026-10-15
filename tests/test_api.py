"""API tests. Qdrant and the LLM are mocked, so no services or API keys are needed."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.main import app

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
