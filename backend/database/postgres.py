"""
Nexus Spatiotemporal RAG Engine - PostgreSQL + pgvector Database Layer
======================================================================
Provides asynchronous database sessions, tables for video metadata, processing jobs,
and spatiotemporal multimodal chunks indexed with pgvector vector extension.
"""

from datetime import datetime
from typing import AsyncGenerator, Optional, List
import json
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Text,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    text
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from pgvector.sqlalchemy import Vector
from backend.config import settings
from backend.utils.logger import logger

Base = declarative_base()


class VideoModel(Base):
    """Stores uploaded video metadata and high-level file properties."""
    __tablename__ = "videos"

    id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chunks = relationship("SpatiotemporalChunkModel", back_populates="video", cascade="all, delete-orphan")
    jobs = relationship("ProcessingJobModel", back_populates="video", cascade="all, delete-orphan")


class ProcessingJobModel(Base):
    """Tracks asynchronous processing status and pipeline execution stages."""
    __tablename__ = "processing_jobs"

    id = Column(String(64), primary_key=True, index=True)
    video_id = Column(String(64), ForeignKey("videos.id"), nullable=False, index=True)
    status = Column(String(32), default="QUEUED", index=True)  # QUEUED, EXTRACTING, ANALYZING, ALIGNING, COMPLETED, FAILED
    progress_percent = Column(Float, default=0.0)
    current_stage = Column(String(128), default="Video Uploaded")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    video = relationship("VideoModel", back_populates="jobs")


class SpatiotemporalChunkModel(Base):
    """
    Multimodal Spatiotemporal Chunk indexed with pgvector for high-performance similarity search.
    """
    __tablename__ = "spatiotemporal_chunks"

    id = Column(String(64), primary_key=True, index=True)
    video_id = Column(String(64), ForeignKey("videos.id"), nullable=False, index=True)
    timestamp_start = Column(Float, nullable=False, index=True)
    timestamp_end = Column(Float, nullable=False, index=True)
    transcript = Column(Text, nullable=False)
    visual_labels = Column(JSON, default=list)  # Stored as JSON array: ["Architecture Diagram", ...]
    frame_reference = Column(String(512), nullable=False)
    
    # Vector Embeddings using pgvector Vector type (dim=512)
    audio_embedding = Column(Vector(settings.EMBEDDING_DIM), nullable=True)
    visual_embedding = Column(Vector(settings.EMBEDDING_DIM), nullable=True)
    multimodal_embedding = Column(Vector(settings.EMBEDDING_DIM), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("VideoModel", back_populates="chunks")


# Async Engine & Sessionmaker
try:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
        connect_args={"timeout": 1.5}
    )
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
except Exception as e:
    logger.warning(f"Could not create database engine with url {settings.DATABASE_URL}: {e}")
    engine = None
    AsyncSessionLocal = None


async def init_db() -> bool:
    """
    Initializes PostgreSQL database, ensures vector extension is enabled,
    and creates table schemas.
    """
    if engine is None:
        logger.warning("Database engine is not configured.")
        return False

    try:
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema and pgvector extension initialized successfully.")
        return True
    except Exception as e:
        logger.warning(f"Database connection / initialization failed (Postgres may not be running yet): {e}")
        return False


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injector yielding database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
