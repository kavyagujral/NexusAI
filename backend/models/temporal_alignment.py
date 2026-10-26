"""
Nexus Spatiotemporal RAG Engine - Spatiotemporal Temporal Alignment Module
==========================================================================
Aligns asynchronous visual frame streams and spoken audio transcript streams
into discrete, uniform 2.0-second temporal windows:
    Window_k = [t_start, t_end) = [2k, 2k + 2]

Handles temporal boundary overlaps for speech segments that span multiple windows
by computing exact intersection durations and weighted feature aggregations.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field

from backend.vision.visual_encoder import FrameFeature
from backend.audio.audio_processor import AudioTranscriptSegment
from backend.config import settings
from backend.utils.logger import logger


class AlignedTemporalWindow(BaseModel):
    """
    Unified multimodal record for a single 2.0-second time window.
    """
    window_index: int
    video_id: str
    timestamp_start: float = Field(..., description="Start of window in seconds (e.g. 0.0, 2.0, 4.0)")
    timestamp_end: float = Field(..., description="End of window in seconds (e.g. 2.0, 4.0, 6.0)")
    
    # Visual modality
    visual_label: str
    visual_labels: List[str]
    frame_path: str
    visual_feature: List[float]
    visual_confidence: float
    
    # Audio modality
    transcript: str
    audio_feature: List[float]
    overlapping_segments_count: int


def calculate_temporal_overlap(
    win_start: float,
    win_end: float,
    seg_start: float,
    seg_end: float
) -> float:
    """
    Calculates the duration of overlap (in seconds) between two time intervals.
    Overlap = max(0.0, min(win_end, seg_end) - max(win_start, seg_start))
    """
    intersection_start = max(win_start, seg_start)
    intersection_end = min(win_end, seg_end)
    return max(0.0, intersection_end - intersection_start)


class SpatiotemporalAligner2s:
    """
    Performs temporal discretization and stream alignment into 2-second windows.
    """

    def __init__(
        self,
        window_duration: float = settings.FRAME_SAMPLE_INTERVAL_SECONDS,
        embedding_dim: int = settings.EMBEDDING_DIM
    ):
        self.window_duration = window_duration
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized SpatiotemporalAligner2s (window_duration={window_duration}s, dim={embedding_dim})")

    def align_streams(
        self,
        video_id: str,
        total_duration_seconds: float,
        visual_features: List[FrameFeature],
        audio_segments: List[AudioTranscriptSegment],
        gru_contextual_embeddings: Optional[np.ndarray] = None
    ) -> List[AlignedTemporalWindow]:
        """
        Aligns sampled visual frames and contextualized audio segments into 2-second windows.

        Args:
            video_id: Unique identifier of video.
            total_duration_seconds: Total duration of the video file.
            visual_features: List of FrameFeature objects extracted every 2 seconds.
            audio_segments: List of AudioTranscriptSegment objects from Whisper.
            gru_contextual_embeddings: (N_seg, embedding_dim) contextual states from GRU.

        Returns:
            List of AlignedTemporalWindow instances spanning the entire video.
        """
        # Determine total number of 2-second windows
        effective_duration = max(total_duration_seconds, 2.0)
        num_windows = int(np.ceil(effective_duration / self.window_duration))
        
        # Ensure at least as many windows as extracted visual frames
        if visual_features:
            max_frame_ts = max(vf.timestamp for vf in visual_features)
            num_windows = max(num_windows, int(np.ceil((max_frame_ts + self.window_duration) / self.window_duration)))

        logger.info(f"Aligning [{video_id}] ({effective_duration:.2f}s) into {num_windows} discrete 2-second windows")

        aligned_windows: List[AlignedTemporalWindow] = []

        for win_idx in range(num_windows):
            win_start = round(win_idx * self.window_duration, 2)
            win_end = round(win_start + self.window_duration, 2)

            # -------------------------------------------------------------
            # 1. Align Visual Modality
            # Find the visual frame closest to win_start (or within [win_start, win_end])
            # -------------------------------------------------------------
            best_visual = None
            min_diff = float("inf")
            for vf in visual_features:
                diff = abs(vf.timestamp - win_start)
                if diff < min_diff:
                    min_diff = diff
                    best_visual = vf

            if best_visual:
                vis_label = best_visual.visual_label
                vis_labels = best_visual.visual_labels
                vis_frame_path = best_visual.frame_path
                vis_embed = best_visual.embedding
                vis_conf = best_visual.confidence
            else:
                vis_label = "slide"
                vis_labels = ["slide"]
                vis_frame_path = ""
                vis_embed = [0.0] * self.embedding_dim
                vis_conf = 0.5

            # -------------------------------------------------------------
            # 2. Align Audio Modality & Temporal Overlap Weighting
            # -------------------------------------------------------------
            overlapping_texts: List[str] = []
            weighted_audio_embed = np.zeros(self.embedding_dim, dtype=np.float32)
            total_overlap_sec = 0.0
            overlap_count = 0

            for seg_idx, segment in enumerate(audio_segments):
                overlap = calculate_temporal_overlap(win_start, win_end, segment.start, segment.end)
                if overlap > 0.0:
                    overlapping_texts.append(segment.text)
                    total_overlap_sec += overlap
                    overlap_count += 1

                    # Get representation (either from GRU output or segment embedding)
                    if gru_contextual_embeddings is not None and seg_idx < len(gru_contextual_embeddings):
                        seg_vec = gru_contextual_embeddings[seg_idx]
                    else:
                        seg_vec = np.array(segment.embedding, dtype=np.float32)

                    # Weight embedding by overlap fraction
                    weighted_audio_embed += overlap * seg_vec

            # Normalize audio representation if overlap occurred
            if total_overlap_sec > 1e-6:
                audio_vec = weighted_audio_embed / total_overlap_sec
                norm = np.linalg.norm(audio_vec)
                if norm > 1e-9:
                    audio_vec = audio_vec / norm
                audio_embed_list = audio_vec.tolist()
                combined_transcript = " ".join(overlapping_texts).strip()
            else:
                # No spoken audio in this window
                audio_embed_list = [0.0] * self.embedding_dim
                combined_transcript = f"[{vis_label} shown on screen]"

            aligned_windows.append(
                AlignedTemporalWindow(
                    window_index=win_idx,
                    video_id=video_id,
                    timestamp_start=win_start,
                    timestamp_end=win_end,
                    visual_label=vis_label,
                    visual_labels=vis_labels,
                    frame_path=vis_frame_path,
                    visual_feature=vis_embed,
                    visual_confidence=vis_conf,
                    transcript=combined_transcript,
                    audio_feature=audio_embed_list,
                    overlapping_segments_count=overlap_count
                )
            )

        logger.info(f"Successfully constructed {len(aligned_windows)} aligned temporal windows for [{video_id}]")
        return aligned_windows
