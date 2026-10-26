"""
Nexus Spatiotemporal RAG Engine - Video Ingestion API & ML Pipeline Orchestrator
================================================================================
Handles video uploads, disk streaming, and orchestrates the end-to-end real ML pipeline:
1. Audio Extraction (FFmpeg)
2. Speech-to-Text Transcription (Whisper)
3. 2-Second Frame Sampling (OpenCV)
4. Visual Feature Extraction & Classification (ResNet-50 CNN)
5. Temporal Audio Context Recurrence (PyTorch GRU)
6. Spatiotemporal Stream Alignment (2.0s Windows)
7. Cross-Modal Attention Fusion (Pure NumPy Single-Head Attention)
8. Persistent Vector Indexing (ChromaDB)
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from backend.config import settings
from backend.utils.logger import logger
from backend.utils.file_utils import save_uploaded_file
from backend.api.processing import job_registry, video_registry
from backend.vision.frame_processor import FrameProcessor
from backend.vision.visual_encoder import VisualEncoder
from backend.audio.audio_processor import AudioProcessor
from backend.models.gru_encoder import TemporalGRUEncoder
from backend.models.temporal_alignment import SpatiotemporalAligner2s
from backend.models.multimodal_attention import SpatiotemporalAligner, SpatiotemporalChunk
from backend.rag.vector_store import vector_store

router = APIRouter()


class VideoUploadResponse(BaseModel):
    """Response returned upon successful video upload and job dispatch."""
    video_id: str
    job_id: str
    filename: str
    file_size_bytes: int
    duration_seconds: float = 0.0
    status: str
    message: str


class VideoListItem(BaseModel):
    """Metadata item for listing uploaded videos."""
    video_id: str
    filename: str
    file_size_bytes: int
    duration_seconds: float
    created_at: str
    status: str


async def run_pipeline_task(video_id: str, job_id: str, video_path: Path):
    """
    Asynchronous pipeline orchestrator executing the real multimodal machine learning pipeline.
    """
    job = job_registry[job_id]
    try:
        # --- Stage 1: Audio Extraction (FFmpeg) ---
        job["status"] = "EXTRACTING_AUDIO"
        job["current_stage"] = "Extracting 16kHz mono audio stream via FFmpeg"
        job["progress_percent"] = 10.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 1: FFmpeg Audio Extraction")
        
        audio_processor = AudioProcessor()
        audio_path = await asyncio.to_thread(audio_processor.extract_audio_from_video, video_path, video_id)

        # --- Stage 2: Whisper Speech-to-Text Transcription ---
        job["status"] = "WHISPER_TRANSCRIPTION"
        job["current_stage"] = f"Running Whisper ({settings.WHISPER_MODEL}) speech-to-text transcription"
        job["progress_percent"] = 25.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 2: Whisper Transcription")

        audio_segments = await asyncio.to_thread(audio_processor.process_and_segment, audio_path, video_id)
        job["audio_segments_count"] = len(audio_segments)

        # --- Stage 3: Frame Sampling Every 2 Seconds (OpenCV) ---
        job["status"] = "FRAME_SAMPLING"
        job["current_stage"] = "Sampling video frames at exact 2.0-second intervals"
        job["progress_percent"] = 40.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 3: Frame Sampling (2s intervals)")

        frame_processor = FrameProcessor(sample_interval_seconds=settings.FRAME_SAMPLE_INTERVAL_SECONDS)
        extracted_frames = await asyncio.to_thread(frame_processor.extract_frames, video_path, video_id)
        job["frames_extracted"] = len(extracted_frames)

        # Inspect total video duration from OpenCV metadata
        total_duration = 0.0
        if extracted_frames:
            total_duration = max(f.timestamp for f in extracted_frames) + settings.FRAME_SAMPLE_INTERVAL_SECONDS
        if total_duration <= 0.0:
            total_duration = 10.0

        # --- Stage 4: CNN Visual Analysis (ResNet-50) ---
        job["status"] = "CNN_ANALYSIS"
        job["current_stage"] = "Running ResNet-50 CNN visual feature extractor & educational classifier"
        job["progress_percent"] = 55.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 4: ResNet-50 Visual Classification")

        visual_encoder = VisualEncoder()
        frame_paths = [f.frame_path for f in extracted_frames]
        timestamps = [f.timestamp for f in extracted_frames]
        visual_features = await asyncio.to_thread(visual_encoder.batch_extract, frame_paths, timestamps)

        # --- Stage 5: Recurrent GRU Temporal Audio Modeling (PyTorch) ---
        job["status"] = "GRU_ENCODING"
        job["current_stage"] = "Computing sequential temporal hidden states with PyTorch GRU"
        job["progress_percent"] = 70.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 5: PyTorch GRU Recurrent Sequence Processing")

        gru_encoder = TemporalGRUEncoder()
        if audio_segments:
            raw_segment_embeds = [s.embedding for s in audio_segments]
            gru_context_embeds = await asyncio.to_thread(gru_encoder.encode_sequence, raw_segment_embeds)
        else:
            gru_context_embeds = np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)

        # --- Stage 6: Spatiotemporal Stream Alignment (2s Windows) ---
        job["status"] = "TEMPORAL_ALIGNMENT"
        job["current_stage"] = "Aligning multimodal streams into discrete 2.0-second temporal windows"
        job["progress_percent"] = 80.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 6: 2-Second Temporal Stream Alignment")

        stream_aligner = SpatiotemporalAligner2s(window_duration=settings.FRAME_SAMPLE_INTERVAL_SECONDS)
        aligned_windows = await asyncio.to_thread(
            stream_aligner.align_streams,
            video_id=video_id,
            total_duration_seconds=total_duration,
            visual_features=visual_features,
            audio_segments=audio_segments,
            gru_contextual_embeddings=gru_context_embeds
        )

        # --- Stage 7: Pure NumPy Cross-Modal Attention Fusion ---
        job["status"] = "NUMPY_ATTENTION_FUSION"
        job["current_stage"] = "Executing pure-NumPy single-head cross-modal attention fusion (CLO01/CLO02)"
        job["progress_percent"] = 90.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 7: NumPy Cross-Modal Attention Fusion")

        multimodal_aligner = SpatiotemporalAligner()
        chunks: List[SpatiotemporalChunk] = []

        for win in aligned_windows:
            fused_vec, _ = multimodal_aligner.fuse_features(
                audio_embed=np.array(win.audio_feature, dtype=np.float32),
                visual_embed=np.array(win.visual_feature, dtype=np.float32),
                timestamp_seconds=win.timestamp_start
            )

            chunks.append(
                SpatiotemporalChunk(
                    video_id=video_id,
                    timestamp_start=win.timestamp_start,
                    timestamp_end=win.timestamp_end,
                    transcript=win.transcript,
                    visual_label=win.visual_label,
                    visual_labels=win.visual_labels,
                    frame_reference=win.frame_path,
                    audio_embedding=win.audio_feature,
                    visual_embedding=win.visual_feature,
                    multimodal_embedding=fused_vec.tolist()
                )
            )

        # --- Stage 8: ChromaDB Indexing ---
        job["status"] = "CHROMADB_INDEXING"
        job["current_stage"] = "Persisting spatiotemporal embeddings into ChromaDB vector database"
        job["progress_percent"] = 95.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Step 8: ChromaDB Vector Indexing")

        indexed_count = await asyncio.to_thread(vector_store.insert_chunks, chunks)
        job["chunks_indexed"] = indexed_count

        # --- Completed ---
        job["status"] = "COMPLETED"
        job["current_stage"] = "Pipeline complete. Video spatiotemporally indexed in ChromaDB."
        job["progress_percent"] = 100.0
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Pipeline completed successfully. Indexed {indexed_count} chunks in ChromaDB.")

    except Exception as e:
        logger.exception(f"[{job_id}] Video pipeline execution failed: {str(e)}")
        job["status"] = "FAILED"
        job["current_stage"] = "Processing pipeline encountered an error"
        job["error_message"] = str(e)
        job["updated_at"] = datetime.utcnow().isoformat()


@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts video file upload, streams it to local storage, and kicks off real ML background processing.
    """
    video_id = f"vid_{uuid.uuid4().hex[:12]}"
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now_str = datetime.utcnow().isoformat()

    logger.info(f"Receiving video upload: {file.filename} (assigning video_id={video_id}, job_id={job_id})")
    
    # Save file to disk
    dest_path, file_size = await save_uploaded_file(file, video_id)

    # Inspect video metadata with OpenCV
    duration_sec = 0.0
    try:
        cap = cv2.VideoCapture(str(dest_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration_sec = round(total_frames / fps, 2)
            cap.release()
    except Exception as e:
        logger.warning(f"Could not read duration for {dest_path}: {e}")

    # Register Video Record
    video_registry[video_id] = {
        "video_id": video_id,
        "filename": file.filename,
        "file_path": str(dest_path),
        "file_size_bytes": file_size,
        "duration_seconds": duration_sec,
        "created_at": now_str,
        "status": "QUEUED"
    }

    # Register Processing Job Record
    job_registry[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "status": "QUEUED",
        "progress_percent": 0.0,
        "current_stage": "Video file saved to storage. Queued for pipeline.",
        "created_at": now_str,
        "updated_at": now_str,
        "frames_extracted": 0,
        "audio_segments_count": 0,
        "chunks_indexed": 0,
        "error_message": None
    }

    # Dispatch background pipeline
    background_tasks.add_task(run_pipeline_task, video_id, job_id, dest_path)

    return VideoUploadResponse(
        video_id=video_id,
        job_id=job_id,
        filename=file.filename or "video.mp4",
        file_size_bytes=file_size,
        duration_seconds=duration_sec,
        status="QUEUED",
        message="Video uploaded successfully. Spatiotemporal ML pipeline running in background."
    )


@router.get("/", response_model=List[VideoListItem])
async def list_videos():
    """
    List all uploaded videos and their high-level statuses.
    """
    results = []
    for vid_id, data in video_registry.items():
        v_status = "QUEUED"
        for job in job_registry.values():
            if job["video_id"] == vid_id:
                v_status = job["status"]

        results.append(
            VideoListItem(
                video_id=vid_id,
                filename=data["filename"],
                file_size_bytes=data["file_size_bytes"],
                duration_seconds=data["duration_seconds"],
                created_at=data["created_at"],
                status=v_status
            )
        )
    return results
