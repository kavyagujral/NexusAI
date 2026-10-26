import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Nexus Spatiotemporal RAG Engine - Global Application Settings
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core Application
    APP_NAME: str = "Nexus Spatiotemporal RAG Engine"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # Relational Database (Application Metadata: videos, jobs, status)
    POSTGRES_USER: str = "nexus_admin"
    POSTGRES_PASSWORD: str = "nexus_secure_password"
    POSTGRES_DB: str = "nexus_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql+asyncpg://nexus_admin:nexus_secure_password@localhost:5432/nexus_db"

    # ChromaDB Vector Database (Multimodal Embeddings & Spatiotemporal Indexing)
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    CHROMA_COLLECTION_NAME: str = "nexus_spatiotemporal_chunks"

    # Storage Paths (Base path relative to project root)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: str = "./data"
    VIDEOS_DIR: str = "./data/videos"
    FRAMES_DIR: str = "./data/frames"
    AUDIO_DIR: str = "./data/audio"

    # Video Processing Parameters
    FRAME_SAMPLE_INTERVAL_SECONDS: float = 2.0
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_VIDEO_EXTENSIONS: str = ".mp4,.mkv,.mov,.avi,.webm"

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.ALLOWED_VIDEO_EXTENSIONS.split(",")]

    # Machine Learning Models & Device Configuration
    DEVICE: str = "auto"  # 'auto', 'cuda', 'mps', or 'cpu'
    WHISPER_MODEL: str = "base"  # 'tiny', 'base', 'small', 'medium', etc.
    VISION_MODEL_BACKBONE: str = "resnet50"
    TEXT_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GRU_HIDDEN_DIM: int = 256
    GRU_BIDIRECTIONAL: bool = True
    EMBEDDING_DIM: int = 512
    ATTENTION_HEADS: int = 1  # Primary single-head formulation from report
    MIN_RETRIEVAL_SIMILARITY: float = 0.12  # Strict relevance gate separating grounded moments from unlinked queries




    def resolve_path(self, relative_path: str) -> Path:
        """Helper to resolve storage directories relative to BASE_DIR"""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.BASE_DIR / relative_path

    def get_torch_device(self) -> str:
        """Detect best available PyTorch execution device."""
        if self.DEVICE != "auto":
            return self.DEVICE
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


# Global cached settings instance
settings = Settings()
