"""Ingestion: load files -> split into chunks -> embed -> store in Qdrant.

Run from the project root:  python -m app.ingest
"""
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings

DATA_DIR = Path("data")
SUPPORTED_EXTS = [".txt", ".md", ".pdf"]


def get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=settings.embed_model, base_url=settings.ollama_base_url
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ingest_directory(data_dir: Path = DATA_DIR) -> int:
    """Index every supported file in data_dir. Returns the number of chunks stored."""
    documents = SimpleDirectoryReader(
        input_dir=str(data_dir), required_exts=SUPPORTED_EXTS
    ).load_data()

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    client = get_qdrant_client()
    # Simple approach: rebuild the collection each run so re-ingesting the same
    # files doesn't create duplicate chunks. (Smarter upserts can come later.)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)

    vector_store = QdrantVectorStore(
        client=client, collection_name=settings.qdrant_collection
    )
    VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        transformations=[splitter],
        embed_model=get_embed_model(),
    )
    return client.count(settings.qdrant_collection).count


if __name__ == "__main__":
    n = ingest_directory()
    print(f"Stored {n} chunks in collection '{settings.qdrant_collection}'")
