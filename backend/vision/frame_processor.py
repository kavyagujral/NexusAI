"""
Nexus Spatiotemporal RAG Engine - Frame Extraction Processor
============================================================
CLO04: Computer Vision Pipeline (Video ingestion, temporal sampling, and frame persistence).
Samples frames every N seconds (default 2.0s) and retains accurate temporal mapping.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import cv2
from pydantic import BaseModel, Field
from backend.config import settings
from backend.utils.logger import logger


class ExtractedFrame(BaseModel):
    """Metadata container for a sampled video frame."""
    frame_index: int
    timestamp: float = Field(..., description="Timestamp in seconds")
    frame_path: str = Field(..., description="Local filesystem path to saved frame image")
    width: int
    height: int


class FrameProcessor:
    """
    Extracts frames periodically from video files using OpenCV / FFmpeg.
    """

    def __init__(self, sample_interval_seconds: float = settings.FRAME_SAMPLE_INTERVAL_SECONDS):
        self.sample_interval = sample_interval_seconds
        logger.info(f"Initialized FrameProcessor (sample_interval={self.sample_interval}s)")

    def extract_frames(self, video_path: Path, video_id: str) -> List[ExtractedFrame]:
        """
        Samples frames at regular time intervals from the provided video file.

        Args:
            video_path: Path to the target video file.
            video_id: Unique identifier for the video session.

        Returns:
            List of ExtractedFrame objects containing relative paths and timestamps.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # Ensure destination frame directory exists
        video_frames_dir = settings.resolve_path(settings.FRAMES_DIR) / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file via OpenCV: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        logger.info(f"Video [{video_id}] FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

        extracted: List[ExtractedFrame] = []
        frame_interval = int(round(fps * self.sample_interval)) if fps > 0 else 60
        frame_interval = max(1, frame_interval)

        current_frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % frame_interval == 0:
                timestamp = current_frame_idx / fps if fps > 0 else float(current_frame_idx)
                frame_filename = f"frame_{saved_count:05d}_{timestamp:.2f}s.jpg"
                frame_file_path = video_frames_dir / frame_filename

                # Save compressed JPEG image
                cv2.imwrite(str(frame_file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                h, w, _ = frame.shape

                extracted.append(
                    ExtractedFrame(
                        frame_index=saved_count,
                        timestamp=round(timestamp, 2),
                        frame_path=str(frame_file_path),
                        width=w,
                        height=h
                    )
                )
                saved_count += 1

            current_frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(extracted)} frames for video [{video_id}] to {video_frames_dir}")
        return extracted
