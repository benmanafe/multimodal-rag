"""Source selection tested with real local Qdrant and free deterministic models."""
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import httpx
import pytest

from app import documents, main, query
from app.config import Settings
from app.pdf_processor import get_document_id


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Settings(_env_file=None, qdrant_collection="selection", deepseek_api_key="mock")
    for module in (documents, query):
        monkeypatch.setattr(module, "settings", config)
    client = QdrantClient(":memory:")
    monkeypatch.setattr(main, "DATA_DIR", Path("data"))
    monkeypatch.setattr(main, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(query, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(query, "get_embed_model", lambda: MockEmbedding(embed_dim=8))
    monkeypatch.setattr(query, "DeepSeek", lambda **kwargs: MockLLM(max_tokens=32))
    root = Path("data/documents")
    root.mkdir(parents=True)
    identities = []
    for name, text in [("first.txt", "First document evidence."), ("second.txt", "Second document evidence.")]:
        path = root / name
        path.write_text(text)
        identities.append(get_document_id(path, path.read_bytes()))
    nodes = [TextNode(text="Evidence", embedding=[0.5] * 8,
                      metadata={"document_id": identity, "file_name": name, "modality": modality})
             for identity, name, modality in [(identities[0], "first.txt", "text"),
                                               (identities[0], "first.txt", "image"),
                                               (identities[1], "second.txt", "text")]]
    QdrantVectorStore(client=client, collection_name=config.qdrant_collection).add(nodes)
    yield client, identities
    client.close()


def test_inventory_reports_indexed_unindexed_and_index_only_sources_without_moving_files(library):
    client, identities = library
    Path("data/legacy.md").write_text("Not indexed yet")
    Path("data/documents/second.txt").unlink()
    Path("data/images").mkdir()
    Path("data/images/hidden.txt").write_text("Not a source")
    rows = documents.document_inventory(Path("data"), client)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {"first.txt", "second.txt", "legacy.md"}
    assert by_name["first.txt"]["id"] == identities[0]
    assert by_name["first.txt"]["text_chunks"] == 1
    assert by_name["first.txt"]["image_descriptions"] == 1
    assert by_name["second.txt"]["indexed"] and not by_name["second.txt"]["available"]
    assert not by_name["legacy.md"]["indexed"]
    assert Path("data/legacy.md").exists()  # GET must not migrate files.


def test_inventory_without_a_collection_and_changed_content(library):
    client, identities = library
    Path("data/documents/first.txt").write_text("Changed content")
    rows = documents.document_inventory(Path("data"), client)
    versions = [row for row in rows if row["name"] == "first.txt"]
    assert len(versions) == 2
    assert any(row["id"] == identities[0] and row["indexed"] and not row["available"] for row in versions)
    client.delete_collection("selection")
    assert all(not row["indexed"] for row in documents.document_inventory(Path("data"), client))


def test_selected_documents_filter_real_qdrant_before_answering(library):
    _, identities = library
    for selected, expected in [([identities[0]], {"first.txt"}),
                               ([identities[1]], {"second.txt"}),
                               (identities, {"first.txt", "second.txt"})]:
        result = query.answer_question("What is the evidence?", document_ids=selected)
        assert {source["file"] for source in result["sources"]} == expected
        if selected == [identities[0]]:
            assert {source["modality"] for source in result["sources"]} == {"text", "image"}


def test_inventory_paginates_all_index_records(library):
    client, identities = library
    nodes = [TextNode(text="Evidence", embedding=[0.5] * 8,
                      metadata={"document_id": identities[0], "file_name": "first.txt", "modality": "text"})
             for _ in range(260)]
    QdrantVectorStore(client=client, collection_name="selection").add(nodes)
    rows = documents.document_inventory(Path("data"), client)
    first = next(row for row in rows if row["id"] == identities[0])
    assert first["text_chunks"] == 261 and first["image_descriptions"] == 1
    assert "node_ids" not in first


def test_document_routes_and_filter_validation(library, monkeypatch):
    _, identities = library
    client = TestClient(main.app)
    response = client.get("/documents")
    assert response.status_code == 200
    assert {row["id"] for row in response.json()["documents"]} == set(identities)
    answer = Mock(return_value={"answer": "Selected evidence", "sources": []})
    monkeypatch.setattr(main, "answer_question", answer)
    response = client.post("/query", json={"question": "Evidence?", "document_ids": [identities[1]]})
    assert response.status_code == 200
    answer.assert_called_once_with("Evidence?", [], document_ids=[identities[1]])
    answer.reset_mock()
    for selection, status in [([], 400), (["f" * 64], 409), (["../../.env"], 422), (["a" * 64] * 101, 422)]:
        assert client.post("/query", json={"question": "Evidence?", "document_ids": selection}).status_code == status
    answer.assert_not_called()


def test_inventory_unavailable_does_not_fall_back_to_all_documents(library, monkeypatch):
    _, identities = library
    monkeypatch.setattr(main, "get_qdrant_client", Mock(side_effect=httpx.ConnectError("offline")))
    answer = Mock()
    monkeypatch.setattr(main, "answer_question", answer)
    client = TestClient(main.app)
    assert client.get("/documents").status_code == 503
    assert client.post("/query", json={"question": "Evidence?", "document_ids": identities}).status_code == 503
    answer.assert_not_called()
