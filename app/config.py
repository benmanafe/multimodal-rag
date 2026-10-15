"""Settings loaded from environment variables / the .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (used from M4)
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # Embeddings (local Ollama)
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Chunking / retrieval
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3


settings = Settings()
