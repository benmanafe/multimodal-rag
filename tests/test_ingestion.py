"""Exercise the real LlamaIndex/Qdrant adapter with local vectors and synthetic PDFs."""
from pathlib import Path
from unittest.mock import Mock

import httpx
import pymupdf
import pytest
from qdrant_client import QdrantClient
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app import ingest, pdf_processor
from app.config import Settings
from app.image_processor import ImageDescriptionError
from app.pdf_processor import PdfProcessingError


@pytest.fixture
def local_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Settings(_env_file=None, qdrant_collection="documents", image_dir=Path("data/images"))
    monkeypatch.setattr(ingest, "settings", config)
    monkeypatch.setattr(pdf_processor, "settings", config)
    client = QdrantClient(":memory:")
    embedding = Mock()
    embedding.get_text_embedding_batch.side_effect = lambda texts: [[1.0, float(len(text)), 0.5, 0.25] for text in texts]
    monkeypatch.setattr(ingest, "get_embed_model", Mock(return_value=embedding))
    monkeypatch.setattr(ingest, "get_qdrant_client", Mock(return_value=client))
    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("No paid/network requests in ingestion tests")))
    monkeypatch.setattr(ingest, "describe_image", Mock(return_value="Architecture diagram sends PDF captions to Qdrant. " * 600))
    yield client, embedding
    client.close()


def make_pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((72, 72), "Python and FastAPI on the first page.")
        second = pdf.new_page()
        second.insert_text((72, 72), "Project architecture on the second page.")
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 160, 120), False)
        image.clear_with(120)
        second.insert_image(pymupdf.Rect(80, 180, 320, 360), stream=image.tobytes("png"))
        pdf.save(path)


def points(client):
    return client.scroll("documents", limit=1000, with_vectors=True)[0]


def nodes(client):
    return [metadata_dict_to_node(point.payload) for point in points(client)]


def test_shared_embedding_collection_and_metadata_survive_real_qdrant_roundtrip(local_index):
    client, embedding = local_index
    make_pdf(Path("data/documents/resume.pdf"))
    Path("data/sample.md").write_text("Sample document about retrieval.")
    Path("data/notes.txt").write_text("Python retrieval sentence. " * 1200)
    # Derived files, unsupported files, and nested inputs must never become documents.
    Path("data/images").mkdir()
    Path("data/images/not_a_source.md").write_text("DO NOT INDEX GENERATED METADATA")
    Path("data/documents/nested").mkdir()
    Path("data/documents/nested/hidden.md").write_text("DO NOT INDEX NESTED FILE")
    Path("data/documents/ignored.json").write_text("{}")
    stats = ingest.ingest_directory_with_stats()
    stored = nodes(client)
    assert stats.image_descriptions == 1 and stats.text_chunks > 3
    assert stats.total_vectors == stats.text_chunks + 1 == len(stored)
    assert stats.warnings == []
    assert [entry.name for entry in client.get_collections().collections] == ["documents"]
    ingest.get_embed_model.assert_called_once()
    embedding.get_text_embedding_batch.assert_called_once()
    batch = embedding.get_text_embedding_batch.call_args.args[0]
    assert len(batch) == stats.total_vectors
    assert not any("DO NOT INDEX" in text for text in batch)
    caption = next(node for node in stored if node.metadata["modality"] == "image")
    assert caption.text == ingest.describe_image.return_value  # Long caption was never split.
    assert caption.metadata["page"] == 2 and Path(caption.metadata["image_path"]).is_file()
    assert caption.metadata["file_name"] == "resume.pdf"
    assert len(caption.metadata["document_id"]) == 64
    pdf_pages = {node.metadata["page"] for node in stored if node.metadata["file_name"] == "resume.pdf"
                 and node.metadata["modality"] == "text"}
    assert pdf_pages == {1, 2}
    assert all("page" not in node.metadata for node in stored if node.metadata["file_name"].endswith((".txt", ".md")))
    assert Path("data/documents/notes.txt").exists() and not Path("data/notes.txt").exists()
    assert Path("data/sample.md").exists()
    assert {len(vector) for point in points(client) for vector in
            (point.vector.values() if isinstance(point.vector, dict) else [point.vector])} == {4}
    index = VectorStoreIndex.from_vector_store(
        QdrantVectorStore(client=client, collection_name="documents"),
        embed_model=MockEmbedding(embed_dim=4),
    )
    retrieved = index.as_retriever(similarity_top_k=stats.total_vectors).retrieve("diagram")
    assert {hit.metadata["modality"] for hit in retrieved} == {"text", "image"}


def test_rebuild_has_stable_ids_no_duplicate_vectors_and_reconciles_only_generated_images(local_index):
    client, _ = local_index
    make_pdf(Path("data/documents/resume.pdf"))
    first = ingest.ingest_directory_with_stats()
    ids = {point.id for point in points(client)}
    stale = Path("data/images") / ("f" * 64)
    stale.mkdir()
    (stale / "page_1_img_0001.png").write_bytes(b"stale generated image")
    (stale / "keep.txt").write_text("Not a generated image; preserve this file.")
    second = ingest.ingest_directory_with_stats()
    assert first == second
    assert {point.id for point in points(client)} == ids
    assert not (stale / "page_1_img_0001.png").exists()
    assert (stale / "keep.txt").exists()
    assert all(Path(node.metadata["image_path"]).exists() for node in nodes(client) if node.metadata["modality"] == "image")
    assert ingest.describe_image.call_count == 2  # Mocked: one caption per rebuild, no retry.


def test_caption_failure_indexes_text_and_returns_warning(local_index, monkeypatch):
    client, _ = local_index
    make_pdf(Path("data/documents/resume.pdf"))
    monkeypatch.setattr(ingest, "describe_image", Mock(side_effect=ImageDescriptionError("empty_caption", "Empty caption.")))
    stats = ingest.ingest_directory_with_stats()
    assert stats.text_chunks == stats.total_vectors == 2
    assert stats.image_descriptions == 0
    assert stats.warnings[0]["code"] == "empty_caption" and stats.warnings[0]["page"] == 2
    assert all(node.metadata["modality"] == "text" for node in nodes(client))


def test_malformed_pdf_preserves_existing_collection(local_index):
    client, _ = local_index
    Path("data").mkdir()
    Path("data/sample.md").write_text("Original indexed content.")
    ingest.ingest_directory_with_stats()
    old_ids = {point.id for point in points(client)}
    Path("data/documents/broken.pdf").write_bytes(b"not a PDF")
    with pytest.raises(PdfProcessingError):
        ingest.ingest_directory_with_stats()
    assert {point.id for point in points(client)} == old_ids


def test_embedding_failure_preserves_existing_vectors_and_image_files(local_index):
    client, embedding = local_index
    make_pdf(Path("data/documents/resume.pdf"))
    ingest.ingest_directory_with_stats()
    old_ids = {point.id for point in points(client)}
    image_path = Path(next(node.metadata["image_path"] for node in nodes(client) if node.metadata["modality"] == "image"))
    Path("data/documents/resume.pdf").unlink()
    Path("data/documents/notes.txt").write_text("Changed document set.")
    embedding.get_text_embedding_batch.side_effect = httpx.ConnectError("embedding server down")
    with pytest.raises(httpx.ConnectError):
        ingest.ingest_directory_with_stats()
    assert {point.id for point in points(client)} == old_ids
    assert image_path.exists()  # Cleanup waits for a successful write.


def test_empty_rebuild_does_not_delete_existing_collection(local_index):
    client, _ = local_index
    Path("data").mkdir()
    sample = Path("data/sample.md")
    sample.write_text("Original indexed content.")
    ingest.ingest_directory_with_stats()
    old_ids = {point.id for point in points(client)}
    sample.unlink()
    with pytest.raises(ingest.IngestionError, match="No searchable"):
        ingest.ingest_directory_with_stats()
    assert {point.id for point in points(client)} == old_ids


def test_migration_checks_all_collisions_before_moving_any_file(local_index):
    Path("data/documents").mkdir(parents=True)
    Path("data/first.txt").write_text("Unmoved input.")
    Path("data/clash.txt").write_text("Old version.")
    Path("data/documents/clash.txt").write_text("Different version.")
    with pytest.raises(ingest.IngestionError) as caught:
        ingest.migrate_legacy_documents(Path("data"))
    assert caught.value.code == "document_collision"
    assert Path("data/first.txt").exists() and Path("data/clash.txt").read_text() == "Old version."
    assert Path("data/documents/clash.txt").read_text() == "Different version."


def test_identical_legacy_files_and_sample_are_indexed_once(local_index):
    Path("data/documents").mkdir(parents=True)
    for path in ["data/a.txt", "data/documents/a.txt"]:
        Path(path).write_text("Same document.")
    for path in ["data/sample.md", "data/documents/sample.md"]:
        Path(path).write_text("Same sample.")
    paths = ingest.discover_documents(Path("data"))
    assert len(paths) == 2
    assert not Path("data/a.txt").exists() and Path("data/sample.md").exists()


@pytest.mark.parametrize("vectors", [[], [[1.0], [1.0]], [[float("nan")]], [[]]])
def test_invalid_embedding_output_is_rejected_before_collection_mutation(local_index, vectors):
    client, embedding = local_index
    Path("data").mkdir()
    Path("data/sample.md").write_text("One short chunk.")
    embedding.get_text_embedding_batch.side_effect = None
    embedding.get_text_embedding_batch.return_value = vectors
    with pytest.raises(ingest.IngestionError) as caught:
        ingest.ingest_directory_with_stats()
    assert caught.value.code == "invalid_embeddings"
    assert not client.collection_exists("documents")


def test_integer_compatibility_wrapper(local_index):
    Path("data").mkdir()
    Path("data/sample.md").write_text("One short chunk.")
    assert ingest.ingest_directory() == 1


def test_successful_source_removal_cleans_stale_visuals(local_index):
    client, _ = local_index
    source = Path("data/documents/resume.pdf")
    make_pdf(source)
    ingest.ingest_directory_with_stats()
    old_image = Path(next(node.metadata["image_path"] for node in nodes(client) if node.metadata["modality"] == "image"))
    source.unlink()
    Path("data/sample.md").write_text("Only this text remains.")
    stats = ingest.ingest_directory_with_stats()
    assert stats.image_descriptions == 0 and stats.total_vectors == 1
    assert not old_image.exists()


def test_database_write_failure_does_not_run_image_cleanup(local_index, monkeypatch):
    make_pdf(Path("data/documents/resume.pdf"))
    stale = Path("data/images") / ("f" * 64)
    stale.mkdir(parents=True)
    old_image = stale / "page_1_img_0001.png"
    old_image.write_bytes(b"Previous evidence")
    monkeypatch.setattr(ingest.QdrantVectorStore, "add", Mock(side_effect=RuntimeError("DB write failed")))
    with pytest.raises(RuntimeError, match="DB write failed"):
        ingest.ingest_directory_with_stats()
    assert old_image.exists()
