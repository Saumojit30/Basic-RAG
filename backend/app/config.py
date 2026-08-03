"""App configuration loaded from environment variables.

Every value can be overridden with an environment variable, so the same
code runs locally, on Render, or anywhere else without changes.
"""

import os
from pathlib import Path


class Settings:
    # --- LLM (any OpenAI-compatible API: Groq, OpenAI, Together, Ollama...) ---
    api_key: str = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1_5")
    chat_model: str = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")

    # --- Retrieval ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    # --- Storage ---
    db_path: str = os.getenv("DB_PATH", str(Path(__file__).parent.parent / "data" / "vectors.db"))

    # --- API ---
    cors_origins: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    @property
    def mock_mode(self) -> bool:
        """No API key -> run the pipeline with fake embeddings and echo answers."""
        return not self.api_key.strip()


settings = Settings()
