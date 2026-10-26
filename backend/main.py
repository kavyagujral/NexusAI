"""
Nexus Spatiotemporal RAG Engine - FastAPI Main Application
==========================================================
Unified API service providing video ingestion, spatiotemporal processing,
NumPy cross-modal attention alignment, and pgvector semantic retrieval.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.utils.logger import logger
from backend.utils.file_utils import ensure_directories
from backend.database.postgres import init_db
from backend.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager: sets up storage paths and initializes database connections.
    """
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode")
    ensure_directories()
    
    # Initialize database schemas and pgvector extension
    db_ok = await init_db()
    if not db_ok:
        logger.warning("Running with in-memory fallback store until PostgreSQL service is started.")
    
    yield
    
    logger.info(f"Shutting down {settings.APP_NAME}")


# FastAPI Application Instance
app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal Spatiotemporal RAG Engine for Educational Videos and Zoom Recordings",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local data storage as static directory for direct frame & video preview in frontend
storage_path = settings.resolve_path(settings.STORAGE_DIR)
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/static/data", StaticFiles(directory=str(storage_path)), name="static_data")

# Mount API Routers under /api
app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["Health & Diagnostics"])
async def health_check():
    """Service health and diagnostics endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "vision_backbone": settings.VISION_MODEL_BACKBONE,
        "gru_hidden_dim": settings.GRU_HIDDEN_DIM,
        "embedding_dim": settings.EMBEDDING_DIM,
        "custom_attention": "NumPy (CLO01/CLO02)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
