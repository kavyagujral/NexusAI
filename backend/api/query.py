"""
Nexus Spatiotemporal RAG Engine - Search & Query API
====================================================
Accepts natural language user questions and returns spatiotemporally aligned
answers, timestamps, transcripts, and corresponding video frames.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from backend.rag.retriever import SpatiotemporalRetriever, RetrievalResult
from backend.utils.logger import logger

router = APIRouter()
retriever = SpatiotemporalRetriever()


class QueryRequest(BaseModel):
    """User question payload."""
    query: str = Field(..., json_schema_extra={"example": "When did the professor explain the microservice architecture?"})
    video_id: Optional[str] = Field(None, description="Optional filter for a specific video ID")
    top_k: int = Field(default=3, ge=1, le=10)
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold to consider relevant")


class QueryResponse(BaseModel):
    """RAG query answer containing multimodal evidence and synthesized explanation."""
    query: str
    total_results: int
    results: List[RetrievalResult]
    synthesized_explanation: Optional[str] = None
    topic_start_time: Optional[str] = None
    topic_start_seconds: Optional[float] = None
    message: Optional[str] = None


@router.post("/", response_model=QueryResponse)
async def query_video_knowledge(payload: QueryRequest):
    """
    Search indexed spatiotemporal video representations for timestamped answers.
    Only returns answers for questions linked to the video content above the similarity threshold.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    logger.info(f"Received spatiotemporal query: '{payload.query}' (video_id={payload.video_id}, min_sim={payload.min_similarity})")
    
    results, explanation, topic_start_time, topic_start_seconds = await retriever.query(
        query_text=payload.query,
        video_id=payload.video_id,
        top_k=payload.top_k,
        min_similarity=payload.min_similarity
    )

    message = None
    if not results:
        message = (
            f"No relevant moments found for '{payload.query}'. "
            "This question does not appear to be linked to the content of this video."
        )

    return QueryResponse(
        query=payload.query,
        total_results=len(results),
        results=results,
        synthesized_explanation=explanation,
        topic_start_time=topic_start_time,
        topic_start_seconds=topic_start_seconds,
        message=message
    )


