"""
Nexus Spatiotemporal RAG Engine - Processing Status & Management API
===================================================================
Tracks asynchronous video processing pipelines and provides real-time stage updates.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from backend.utils.logger import logger

router = APIRouter()

# In-memory pipeline job registry
job_registry: Dict[str, Dict[str, Any]] = {}
video_registry: Dict[str, Dict[str, Any]] = {}


class JobStatusResponse(BaseModel):
    """Schema for video processing pipeline status."""
    job_id: str
    video_id: str
    status: str = Field(..., description="QUEUED, EXTRACTING, ANALYZING, ALIGNING, COMPLETED, FAILED")
    progress_percent: float
    current_stage: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    frames_extracted: int = 0
    audio_segments_count: int = 0
    chunks_indexed: int = 0


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_processing_status(job_id: str):
    """
    Get real-time status and stage progress for a specific processing job.
    """
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job with ID '{job_id}' not found."
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        video_id=job["video_id"],
        status=job["status"],
        progress_percent=job["progress_percent"],
        current_stage=job["current_stage"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        error_message=job.get("error_message"),
        frames_extracted=job.get("frames_extracted", 0),
        audio_segments_count=job.get("audio_segments_count", 0),
        chunks_indexed=job.get("chunks_indexed", 0)
    )


@router.get("/video/{video_id}", response_model=JobStatusResponse)
async def get_status_by_video_id(video_id: str):
    """
    Lookup latest processing status associated with a given video ID.
    """
    # Find matching job
    for job in reversed(list(job_registry.values())):
        if job["video_id"] == video_id:
            return JobStatusResponse(
                job_id=job["job_id"],
                video_id=job["video_id"],
                status=job["status"],
                progress_percent=job["progress_percent"],
                current_stage=job["current_stage"],
                created_at=job["created_at"],
                updated_at=job["updated_at"],
                error_message=job.get("error_message"),
                frames_extracted=job.get("frames_extracted", 0),
                audio_segments_count=job.get("audio_segments_count", 0),
                chunks_indexed=job.get("chunks_indexed", 0)
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No processing job found for video ID '{video_id}'."
    )
