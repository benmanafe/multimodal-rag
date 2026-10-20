"""PDF-to-caption preparation tests never make live provider requests."""
from pathlib import Path
from unittest.mock import Mock

from llama_index.core import Document
from llama_index.core.schema import MetadataMode
import pytest

from app import ingest
from app.config import Settings
from app.image_processor import ImageDescriptionError
from app.pdf_processor import PdfContent, make_record_id


@pytest.fixture
def extracted(monkeypatch):
    document_id = "a" * 64
    text = Document(text="Python and FastAPI.", metadata={
        "file_name": "resume.pdf", "document_id": document_id, "page": 1, "modality": "text",
    })
    images = [{
        "id": make_record_id(document_id, page, "image", 1),
        "metadata": {
            "file_name": "resume.pdf", "document_id": document_id, "page": page,
            "modality": "image", "image_path": f"data/images/{document_id}/page_{page}_img_0001.png",
            "image_kind": "raster" if page == 1 else "vector_crop",
        },
        "context": f"Project on page {page}",
    } for page in [1, 2]]
    result = PdfContent([text], images, [])
    monkeypatch.setattr(ingest, "extract_pdf_content", Mock(return_value=result))
    monkeypatch.setattr(ingest, "settings", Settings(
        _env_file=None, vision_provider="deepseek", vision_model="deepseek-flash",
    ))
    # Any accidental Qdrant/embedding calls fail the test before contacting a service.
    monkeypatch.setattr(ingest, "get_qdrant_client", Mock(side_effect=AssertionError("No DB in preparation")))
    monkeypatch.setattr(ingest, "get_embed_model", Mock(side_effect=AssertionError("No embedding in preparation")))
    return result


def test_preparation_preserves_sources_and_keeps_long_captions_unsplit(extracted, monkeypatch):
    caption = "PDF feeds Qdrant. " * 600
    describe = Mock(side_effect=[caption, "A vector diagram shows retrieval."])
    monkeypatch.setattr(ingest, "describe_image", describe)
    result = ingest.prepare_pdf_documents(Path("resume.pdf"))
    assert result.text_documents == extracted.text_documents
    assert len(result.image_nodes) == 2
    assert result.image_nodes[0].text == caption
    assert result.image_nodes[0].id_ == extracted.images[0]["id"]
    assert result.image_nodes[1].metadata["page"] == 2
    assert result.image_nodes[1].metadata["image_kind"] == "vector_crop"
    assert result.image_nodes[0].metadata["caption_model"] == "deepseek-flash"
    assert result.image_nodes[0].metadata["caption_provider"] == "deepseek"
    describe.assert_any_call(Path(extracted.images[0]["metadata"]["image_path"]), "Project on page 1")
    assert describe.call_count == 2
    assert result.warnings == []
    embedding_text = result.image_nodes[0].get_content(metadata_mode=MetadataMode.EMBED)
    assert "data/images" not in embedding_text and "deepseek-flash" not in embedding_text
    assert extracted.images[0]["metadata"]["document_id"] not in embedding_text
    assert "caption_model" not in extracted.images[0]["metadata"]


@pytest.mark.parametrize("code,message", [
    ("caption_request_failed", "Vision request timed out."),
    ("empty_caption", "Vision server returned an empty description."),
])
def test_failed_caption_skips_only_one_image_and_reports_source_warning(extracted, monkeypatch, code, message):
    describe = Mock(side_effect=[ImageDescriptionError(code, message), "Second image description."])
    monkeypatch.setattr(ingest, "describe_image", describe)
    result = ingest.prepare_pdf_documents(Path("resume.pdf"))
    assert result.text_documents == extracted.text_documents
    assert len(result.image_nodes) == 1 and result.image_nodes[0].metadata["page"] == 2
    assert result.warnings == [{
        "code": code, "file": "resume.pdf", "page": 1,
        "record_id": extracted.images[0]["id"], "message": message,
    }]
    assert describe.call_count == 2  # No retry of the failed image.


def test_all_captions_can_fail_without_discarding_text_or_extractor_warnings(extracted, monkeypatch):
    original_warning = {"code": "image_save_failed", "file": "resume.pdf", "page": 3, "message": "Save failed."}
    extracted.warnings.append(original_warning)
    monkeypatch.setattr(ingest, "describe_image", Mock(side_effect=ImageDescriptionError(
        "vision_not_configured", "Set DEEPSEEK_API_KEY before using DeepSeek captions.",
    )))
    result = ingest.prepare_pdf_documents(Path("resume.pdf"))
    assert result.text_documents == extracted.text_documents
    assert result.image_nodes == []
    assert len(result.warnings) == 3 and result.warnings[0] == original_warning
    assert extracted.warnings == [original_warning]


def test_text_only_pdf_requires_no_caption_call(extracted, monkeypatch):
    extracted.images.clear()
    describe = Mock(side_effect=AssertionError("No images to caption"))
    monkeypatch.setattr(ingest, "describe_image", describe)
    result = ingest.prepare_pdf_documents(Path("resume.pdf"))
    assert result.text_documents == extracted.text_documents
    assert result.image_nodes == [] and result.warnings == []
    describe.assert_not_called()
