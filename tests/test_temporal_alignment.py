"""
Unit Tests for Spatiotemporal 2-Second Temporal Alignment
=========================================================
Validates uniform 2.0-second discretization, interval overlap weighting,
and correct visual/audio assignment.
"""

import numpy as np
import pytest
from backend.models.temporal_alignment import (
    calculate_temporal_overlap,
    SpatiotemporalAligner2s,
    AlignedTemporalWindow
)
from backend.vision.visual_encoder import FrameFeature
from backend.audio.audio_processor import AudioTranscriptSegment


class TestTemporalAlignment:
    """Test suite for stream discretization and overlap math."""

    def test_calculate_temporal_overlap(self):
        """Verify mathematical boundary overlap between intervals."""
        # Case 1: Identical intervals
        assert calculate_temporal_overlap(0.0, 2.0, 0.0, 2.0) == 2.0

        # Case 2: Partial overlap
        assert calculate_temporal_overlap(0.0, 2.0, 1.0, 3.0) == 1.0

        # Case 3: Contained segment
        assert calculate_temporal_overlap(0.0, 4.0, 1.0, 2.5) == 1.5

        # Case 4: Disjoint intervals
        assert calculate_temporal_overlap(0.0, 2.0, 3.0, 5.0) == 0.0

        # Case 5: Adjacent touching boundary
        assert calculate_temporal_overlap(0.0, 2.0, 2.0, 4.0) == 0.0

    def test_2_second_window_alignment(self):
        """Verify that a 6.0s video produces exactly 3 discrete windows: [0,2), [2,4), [4,6)."""
        aligner = SpatiotemporalAligner2s(window_duration=2.0, embedding_dim=512)

        # Mock visual frames sampled every 2 seconds
        visual_features = [
            FrameFeature(
                timestamp=0.0,
                frame_path="/data/frames/v1/frame_00000_0.00s.jpg",
                visual_label="slide",
                visual_labels=["slide"],
                confidence=0.95,
                embedding=[0.1] * 512
            ),
            FrameFeature(
                timestamp=2.0,
                frame_path="/data/frames/v1/frame_00001_2.00s.jpg",
                visual_label="architecture diagram",
                visual_labels=["architecture diagram"],
                confidence=0.92,
                embedding=[0.2] * 512
            ),
            FrameFeature(
                timestamp=4.0,
                frame_path="/data/frames/v1/frame_00002_4.00s.jpg",
                visual_label="code snippet",
                visual_labels=["code snippet"],
                confidence=0.88,
                embedding=[0.3] * 512
            ),
        ]

        # Mock audio segments spanning across boundaries
        audio_segments = [
            AudioTranscriptSegment(
                segment_index=0,
                start=0.5,
                end=3.5,
                text="This is the architecture diagram explanation.",
                embedding=[0.5] * 512
            )
        ]

        windows = aligner.align_streams(
            video_id="test_video_1",
            total_duration_seconds=6.0,
            visual_features=visual_features,
            audio_segments=audio_segments
        )

        assert len(windows) == 3
        # Window 0: 0.0s - 2.0s
        assert windows[0].timestamp_start == 0.0
        assert windows[0].timestamp_end == 2.0
        assert windows[0].visual_label == "slide"
        assert windows[0].overlapping_segments_count == 1

        # Window 1: 2.0s - 4.0s
        assert windows[1].timestamp_start == 2.0
        assert windows[1].timestamp_end == 4.0
        assert windows[1].visual_label == "architecture diagram"
        assert windows[1].overlapping_segments_count == 1

        # Window 2: 4.0s - 6.0s
        assert windows[2].timestamp_start == 4.0
        assert windows[2].timestamp_end == 6.0
        assert windows[2].visual_label == "code snippet"
        assert windows[2].overlapping_segments_count == 0
