"""Retrieval response/prompt checks with no embedding, vector server, or LLM calls."""
from pathlib import Path
from unittest.mock import MagicMock, Mock

from llama_index.core.schema import NodeWithScore, TextNode
import pytest

from app import image_paths, query
from app.config import Settings


@pytest.fixture
def query_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Settings(_env_file=None, deepseek_api_key="mock-test-key", image_dir=Path("data/images"))
    monkeypatch.setattr(query, "settings", config)
    monkeypatch.setattr(image_paths, "settings", config)
    for name in ["get_embed_model", "get_qdrant_client", "QdrantVectorStore", "DeepSeek"]:
        monkeypatch.setattr(query, name, Mock())
    index = Mock()
    monkeypatch.setattr(query.VectorStoreIndex, "from_vector_store", Mock(return_value=index))
    response = MagicMock()
    response.__str__.return_value = "The diagram is described as a PDF-to-Qdrant pipeline."
    index.as_chat_engine.return_value.chat.return_value = response
    return index, response


def test_mixed_sources_include_safe_urls_pages_and_modalities_and_preserve_chat(query_runtime):
    index, response = query_runtime
    document_id = "a" * 64
    response.source_nodes = [
        NodeWithScore(node=TextNode(text="Python", metadata={
            "file_name": "resume.pdf", "page": 1, "modality": "text",
        }), score=0.77777),
        NodeWithScore(node=TextNode(text="PDF to Qdrant", metadata={
            "file_name": "resume.pdf", "page": 2, "modality": "image",
            "image_path": f"data/images/{document_id}/page_2_img_0001.png",
        }), score=0.73),
    ]
    result = query.answer_question("What does it show?", [
        {"role": "user", "content": "Explain the project."},
        {"role": "assistant", "content": "It uses retrieval."},
    ])
    assert result["sources"] == [
        {"file": "resume.pdf", "page": 1, "modality": "text", "score": 0.7778, "image_url": None},
        {"file": "resume.pdf", "page": 2, "modality": "image", "score": 0.73,
         "image_url": f"/images/{document_id}/page_2_img_0001.png"},
    ]
    options = index.as_chat_engine.call_args.kwargs
    assert options["chat_mode"] == "condense_plus_context"
    assert options["similarity_top_k"] == query.settings.top_k
    assert "Answer only from the retrieved document evidence" in options["system_prompt"]
    assert "model-generated image description" in options["system_prompt"]
    history = index.as_chat_engine.return_value.chat.call_args.kwargs["chat_history"]
    assert [turn.role.value for turn in history] == ["user", "assistant"]
    assert history[0].content == "Explain the project."
    query.get_embed_model.assert_called_once()
    query.VectorStoreIndex.from_vector_store.assert_called_once()


def test_legacy_source_has_no_invented_page_or_modality(query_runtime):
    _, response = query_runtime
    response.source_nodes = [NodeWithScore(node=TextNode(text="Legacy text", metadata={"file_name": "sample.md"}))]
    result = query.answer_question("What is RAG?")
    assert result["sources"] == [{"file": "sample.md", "score": 0.0, "page": None,
                                  "modality": None, "image_url": None}]


@pytest.mark.parametrize("path", [".env", "../../.env", "/etc/passwd", "C:\\private\\image.png",
                                  "data/images/../../.env", "https://example.com/picture.png"])
def test_untrusted_image_metadata_never_becomes_a_url(query_runtime, path):
    node = NodeWithScore(node=TextNode(text="Image caption", metadata={
        "file_name": "resume.pdf", "page": -1, "modality": "image", "image_path": path,
    }))
    result = query.serialize_source(node)
    assert result["image_url"] is None and result["page"] is None
