"""API Routes Package for Nexus"""
from fastapi import APIRouter
from backend.api.upload import router as upload_router
from backend.api.processing import router as processing_router
from backend.api.query import router as query_router

api_router = APIRouter()
api_router.include_router(upload_router, prefix="/videos", tags=["Video Ingestion"])
api_router.include_router(processing_router, prefix="/processing", tags=["Processing & Status"])
api_router.include_router(query_router, prefix="/query", tags=["RAG Search & Query"])

__all__ = ["api_router"]
