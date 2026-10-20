"""Settings loaded from environment variables / the .env file."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (used from M4)
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # Embeddings (local Ollama)
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # Image captions; blank URL uses the selected provider's default server.
    vision_provider: str = "deepseek"
    vision_model: str = "deepseek-flash"
    vision_base_url: str = ""
    vision_timeout: float = Field(default=120.0, gt=0)

    # PDF extraction; image paths must remain inside the application directory.
    image_dir: Path = Path("data/images")
    pdf_min_image_pixels: int = Field(default=64, ge=1)
    pdf_vector_fallback: bool = True
    pdf_vector_min_paths: int = Field(default=3, ge=2)
    pdf_render_dpi: int = Field(default=144, ge=72, le=300)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Chunking / retrieval
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3


settings = Settings()
