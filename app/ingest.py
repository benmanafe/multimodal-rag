"""Ingestion: load files -> split into chunks -> embed -> store in Qdrant.

Run from the project root:  python -m app.ingest
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings
from app.image_processor import ImageDescriptionError, describe_image
from app.pdf_processor import extract_pdf_content, get_document_id, make_record_id

DATA_DIR = Path("data")
SUPPORTED_EXTS = [".txt", ".md", ".pdf"]


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class IngestionStats:
    text_chunks: int
    image_descriptions: int
    total_vectors: int
    warnings: list[dict] = field(default_factory=list)


@dataclass
class PreparedPdf:
    """Text awaits splitting; image captions are already complete, unsplit nodes."""

    text_documents: list[Document]
    image_nodes: list[TextNode]
    warnings: list[dict]


def prepare_pdf_documents(pdf_path: Path, images_dir: Path | None = None) -> PreparedPdf:
    """Prepare PDF evidence at ingest time without embedding or rebuilding Qdrant.

    Caption failures skip only the affected visual. The caller receives source-aware
    warnings alongside preserved text and successful image nodes. Each image makes
    at most one provider request; there are no automatic retries or query-time calls.
    """
    extracted = extract_pdf_content(pdf_path, images_dir)
    warnings = list(extracted.warnings)
    image_nodes = []
    excluded_keys = ["document_id", "image_path", "image_kind", "caption_provider", "caption_model"]
    for image in extracted.images:
        metadata = dict(image["metadata"])
        try:
            caption = describe_image(Path(metadata["image_path"]), image.get("context", ""))
        except ImageDescriptionError as exc:
            warnings.append({
                "code": exc.code, "file": metadata["file_name"], "page": metadata["page"],
                "record_id": image["id"], "message": str(exc),
            })
            continue
        metadata.update(caption_provider=settings.vision_provider, caption_model=settings.vision_model)
        image_nodes.append(TextNode(
            id_=image["id"], text=caption, metadata=metadata,
            excluded_embed_metadata_keys=excluded_keys,
            excluded_llm_metadata_keys=excluded_keys,
        ))
    return PreparedPdf(extracted.text_documents, image_nodes, warnings)


def get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=settings.embed_model, base_url=settings.ollama_base_url
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def migrate_legacy_documents(data_dir: Path) -> Path:
    """Move legacy inputs after checking every collision; preserve root sample.md."""
    data_dir = Path(data_dir).resolve()
    documents_dir = data_dir / "documents"
    if documents_dir.is_symlink():
        raise IngestionError("invalid_document_directory", "Document input directory must not be a symlink.")
    legacy = sorted(path for path in data_dir.glob("*") if path.is_file()
                    and not path.is_symlink() and path.suffix.lower() in SUPPORTED_EXTS
                    and path.name != "sample.md")
    for source in legacy:
        target = documents_dir / source.name
        if target.exists() and (target.is_symlink() or not target.is_file()
                                or target.read_bytes() != source.read_bytes()):
            raise IngestionError("document_collision", "Legacy document name collision; rename the conflicting file before ingestion.")
    documents_dir.mkdir(parents=True, exist_ok=True)
    for source in legacy:
        target = documents_dir / source.name
        if target.exists():
            source.unlink()  # Identical bytes were verified above; retain one copy.
        else:
            source.rename(target)
    return documents_dir


def discover_documents(data_dir: Path) -> list[Path]:
    documents_dir = migrate_legacy_documents(data_dir)
    paths = [path for path in documents_dir.iterdir() if path.is_file()
             and not path.is_symlink() and path.suffix.lower() in SUPPORTED_EXTS]
    sample = Path(data_dir) / "sample.md"
    if sample.is_file() and not sample.is_symlink():
        paths.append(sample)
    unique = {}
    for path in sorted(paths, key=lambda item: item.as_posix()):
        unique.setdefault(get_document_id(path, path.read_bytes()), path)
    return list(unique.values())


def _image_root() -> Path:
    root = settings.image_dir.resolve()
    app_root = Path.cwd().resolve()
    if root == app_root or not root.is_relative_to(app_root):
        raise IngestionError("invalid_image_directory", "Image directory must be a dedicated folder within the application directory.")
    return root


def _reconcile_images(image_nodes: list[TextNode], warnings: list[dict]) -> None:
    """Remove only recognized generated files, and only after a successful rebuild."""
    root = _image_root()
    keep = {Path(node.metadata["image_path"]).resolve() for node in image_nodes}
    if not root.exists():
        return
    for folder in root.iterdir():
        if folder.is_symlink() or not folder.is_dir() or not re.fullmatch("[a-f0-9]{64}", folder.name):
            continue
        for image in folder.iterdir():
            if image.is_symlink() or not image.is_file() or not re.fullmatch(
                r"page_[1-9][0-9]*_img_[0-9]{4,}\.(png|jpeg|jpg|gif|webp)", image.name,
            ):
                continue
            resolved = image.resolve()
            if not resolved.is_relative_to(root) or resolved in keep:
                continue
            try:
                image.unlink()
            except OSError:
                warnings.append({"code": "image_cleanup_failed", "file": image.name,
                                 "message": "Could not remove a stale generated image."})
        try:
            folder.rmdir()  # Only removes an empty generated directory.
        except OSError:
            pass


def ingest_directory_with_stats(data_dir: Path = DATA_DIR) -> IngestionStats:
    """Rebuild the local demo's one collection from text chunks and caption nodes.

    Preparation/embedding complete before deleting the old collection. Database
    replacement itself is not transactional; a write failure still requires a rerun.
    """
    _image_root()
    documents, image_nodes, warnings = [], [], []
    for source in discover_documents(Path(data_dir)):
        if source.suffix.lower() == ".pdf":
            prepared = prepare_pdf_documents(source)
            documents.extend(prepared.text_documents)
            image_nodes.extend(prepared.image_nodes)
            warnings.extend(prepared.warnings)
        else:
            document_id = get_document_id(source, source.read_bytes())
            loaded = SimpleDirectoryReader(input_files=[str(source)]).load_data()
            for ordinal, document in enumerate(loaded, start=1):
                document.id_ = make_record_id(document_id, 0, "document_text", ordinal)
                document.metadata = {"file_name": source.name, "document_id": document_id, "modality": "text"}
                document.excluded_embed_metadata_keys = ["document_id"]
                document.excluded_llm_metadata_keys = ["document_id"]
                documents.append(document)

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    text_nodes = []
    ordinals = {}
    for document in documents:
        key = (document.metadata["document_id"], document.metadata.get("page", 0))
        for node in splitter.get_nodes_from_documents([document]):
            if not node.text.strip():
                continue
            ordinals[key] = ordinals.get(key, 0) + 1
            node.id_ = make_record_id(key[0], key[1], "text", ordinals[key])
            text_nodes.append(node)
    nodes = text_nodes + image_nodes
    if not nodes:
        raise IngestionError("no_searchable_content", "No searchable text or image descriptions were produced; existing index was preserved.")

    embed_model = get_embed_model()
    embeddings = embed_model.get_text_embedding_batch([
        node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
    ])
    dimensions = {len(vector) for vector in embeddings}
    if len(embeddings) != len(nodes) or len(dimensions) != 1 or not next(iter(dimensions), 0):
        raise IngestionError("invalid_embeddings", "Embedding model returned missing or inconsistent vector dimensions.")
    if not all(math.isfinite(value) for vector in embeddings for value in vector):
        raise IngestionError("invalid_embeddings", "Embedding model returned nonfinite vector values.")
    for node, vector in zip(nodes, embeddings):
        node.embedding = vector

    client = get_qdrant_client()
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    vector_store = QdrantVectorStore(
        client=client, collection_name=settings.qdrant_collection
    )
    vector_store.add(nodes)  # All nodes already carry embeddings from the same model.
    total = client.count(settings.qdrant_collection, exact=True).count
    if total != len(nodes):
        raise IngestionError("incomplete_rebuild", "Collection count does not match prepared nodes; rerun ingestion.")
    _reconcile_images(image_nodes, warnings)
    return IngestionStats(len(text_nodes), len(image_nodes), total, warnings)


def ingest_directory(data_dir: Path = DATA_DIR) -> int:
    """Compatibility wrapper returning the shared collection's total vector count."""
    return ingest_directory_with_stats(data_dir).total_vectors


if __name__ == "__main__":
    stats = ingest_directory_with_stats()
    print(f"Stored {stats.total_vectors} vectors: {stats.text_chunks} text chunks, {stats.image_descriptions} image descriptions")
    for warning in stats.warnings:
        print("Warning: " + json.dumps(warning))
