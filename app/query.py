"""Query: embed the question -> retrieve top-k chunks from Qdrant -> LLM answers.

Supports chat: pass earlier turns as `history` and follow-up questions like
"tell me more about that" are rewritten into standalone questions before retrieval.

Try it from the project root:  python -m app.query "What is RAG?"
"""
import sys

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.config import settings
from app.ingest import get_embed_model, get_qdrant_client


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    """Return {"answer": str, "sources": [{"file": str, "score": float}]}.

    history: earlier turns as [{"role": "user" | "assistant", "content": str}, ...].
    """
    # Connect to the collection that ingest.py already filled (nothing is re-embedded).
    vector_store = QdrantVectorStore(
        client=get_qdrant_client(), collection_name=settings.qdrant_collection
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=get_embed_model()
    )
    llm = DeepSeek(model=settings.llm_model, api_key=settings.deepseek_api_key)

    # condense_plus_context: with history it first rewrites the question into a
    # standalone one (1 extra LLM call), then retrieves and answers. With no history
    # it skips the rewrite, so a first question costs a single LLM call.
    engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        similarity_top_k=settings.top_k,
    )
    chat_history = [
        ChatMessage(
            role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
            content=m["content"],
        )
        for m in (history or [])
    ]
    response = engine.chat(question, chat_history=chat_history)
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
