"""
Nexus Spatiotemporal RAG Engine - Spatiotemporal Retriever
==========================================================
Retrieves timestamped multimodal moments matching user queries from ChromaDB and constructs
structured, explainable RAG responses with temporal markers, visual context, and transcripts.
"""

from typing import List, Dict, Any, Optional
import asyncio
from pydantic import BaseModel, Field
from backend.config import settings
from backend.rag.embeddings import QueryEmbedder
from backend.rag.vector_store import vector_store, ChromaVectorStore
from backend.rag.synthesizer import (
    format_timestamp,
    group_windows_into_topics,
    synthesize_comprehensive_explanation,
    TopicSegment
)
from backend.utils.logger import logger


class RetrievalResult(BaseModel):
    """Structured topic-level response returned by the Spatiotemporal RAG engine."""
    video_id: str
    answer: str
    timestamp_start: float
    timestamp_end: float
    topic_start_time: str = Field(default="00:00", description="e.g. '00:00'")
    formatted_timestamp: str = Field(..., description="e.g. 'Starts at 00:00'")
    transcript: str
    visual_context: List[str]
    relevant_frame: str
    similarity_score: float


class SpatiotemporalRetriever:
    """
    Coordinates semantic vector retrieval in ChromaDB, merges contiguous 2-second windows
    into cohesive topic segments with clear start times, and synthesizes structured explanations.
    """

    def __init__(self, embedder: Optional[QueryEmbedder] = None, store: Optional[ChromaVectorStore] = None):
        self.embedder = embedder or QueryEmbedder()
        self.vector_store = store or vector_store
        logger.info("Initialized SpatiotemporalRetriever with ChromaDB backend & Topic Synthesizer")

    async def query(
        self,
        query_text: str,
        video_id: Optional[str] = None,
        top_k: int = 3,
        min_similarity: Optional[float] = None
    ):
        """
        Executes spatiotemporal search for a given natural language query.
        Filters unlinked questions, merges 2.0s discretization windows into cohesive topics,
        and synthesizes a comprehensive explanation with topic starting time.
        
        Returns:
            Tuple: (results, synthesized_explanation, topic_start_time, topic_start_seconds)
        """
        logger.info(f"Executing query: '{query_text}' on video_id={video_id} (min_similarity={min_similarity})")
        query_vector = self.embedder.embed_query(query_text)
        
        threshold = min_similarity if min_similarity is not None else settings.MIN_RETRIEVAL_SIMILARITY
        
        # Query ChromaDB for candidate windows (expanding k to capture contiguous topic context)
        candidate_k = max(top_k * 4, 12)
        raw_matches = await asyncio.to_thread(
            self.vector_store.search_similar,
            query_embedding=query_vector,
            video_id=video_id,
            top_k=candidate_k,
            min_similarity=threshold
        )

        if not raw_matches:
            logger.info(f"No video segments met the relevance threshold ({threshold:.2f}) for query: '{query_text}'")
            return [], None, None, None

        # Group granular 2-second windows into cohesive topic segments
        topic_segments = group_windows_into_topics(raw_matches, proximity_seconds=8.0)
        
        if not topic_segments:
            return [], None, None, None

        # Synthesize direct, comprehensive explanation for the question
        explanation = synthesize_comprehensive_explanation(query_text, topic_segments)
        
        # Primary topic start marker
        first_topic = topic_segments[0]
        topic_start_time = first_topic.topic_start_time
        topic_start_seconds = first_topic.topic_start_seconds

        # Map merged topic segments to RetrievalResult objects (capped at top_k)
        results: List[RetrievalResult] = []
        for seg in topic_segments[:top_k]:
            results.append(
                RetrievalResult(
                    video_id=seg.video_id,
                    answer=seg.explanation,
                    timestamp_start=seg.topic_start_seconds,
                    timestamp_end=seg.topic_end_seconds,
                    topic_start_time=seg.topic_start_time,
                    formatted_timestamp=seg.display_timestamp,
                    transcript=seg.transcript,
                    visual_context=seg.visual_context,
                    relevant_frame=seg.relevant_frame,
                    similarity_score=seg.similarity_score
                )
            )

        return results, explanation, topic_start_time, topic_start_seconds

