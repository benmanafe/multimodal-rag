"""Query: embed the question -> retrieve top-k chunks from Qdrant -> LLM answers.

Try it from the project root:  python -m app.query "What is RAG?"
"""
import sys

from llama_index.core import VectorStoreIndex
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.config import settings
from app.ingest import get_embed_model, get_qdrant_client


def answer_question(question: str) -> dict:
    """Return {"answer": str, "sources": [{"file": str, "score": float}]}."""
    # Connect to the collection that ingest.py already filled (nothing is re-embedded).
    vector_store = QdrantVectorStore(
        client=get_qdrant_client(), collection_name=settings.qdrant_collection
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=get_embed_model()
    )
    llm = DeepSeek(model=settings.llm_model, api_key=settings.deepseek_api_key)
    engine = index.as_query_engine(llm=llm, similarity_top_k=settings.top_k)

    response = engine.query(question)
    return {
        "answer": str(response),
        "sources": [
            {
                "file": node.metadata.get("file_name", "unknown"),
                "score": round(node.score or 0.0, 4),
            }
            for node in response.source_nodes
        ],
    }


if __name__ == "__main__":
    result = answer_question(" ".join(sys.argv[1:]) or "What is RAG?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  {s['file']}  (score {s['score']})")
