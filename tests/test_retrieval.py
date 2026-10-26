"""
Unit Tests for ChromaDB Vector Storage and Grounding Retrieval
==============================================================
Validates ChromaDB vector insertion, top-k semantic query search,
Recall@k metrics computation, and graceful empty evaluations.
"""

import pytest
import numpy as np
from backend.models.multimodal_attention import SpatiotemporalChunk
from backend.rag.vector_store import ChromaVectorStore
from backend.evaluation.metrics import (
    GroundTruthQueryItem,
    evaluate_retrieval_recall_at_k,
    evaluate_vision_classifier
)


class TestChromaDBRetrieval:
    """Test suite for ChromaDB spatiotemporal indexing and retrieval evaluation."""

    def test_chromadb_insert_and_query(self, tmp_path):
        """Verify vector insertion and top-k retrieval in ChromaDB."""
        persist_dir = str(tmp_path / "test_chroma")
        store = ChromaVectorStore(persist_dir=persist_dir, collection_name="test_chunks")

        # Create mock chunks
        rng = np.random.default_rng(42)
        vec1 = rng.normal(0, 1.0, 512).astype(np.float32)
        vec1 = (vec1 / np.linalg.norm(vec1)).tolist()

        vec2 = rng.normal(0, 1.0, 512).astype(np.float32)
        vec2 = (vec2 / np.linalg.norm(vec2)).tolist()

        chunks = [
            SpatiotemporalChunk(
                video_id="lecture_01",
                timestamp_start=0.0,
                timestamp_end=2.0,
                transcript="Welcome to the lecture.",
                visual_label="slide",
                visual_labels=["slide"],
                frame_reference="/data/frames/lecture_01/f0.jpg",
                multimodal_embedding=vec1
            ),
            SpatiotemporalChunk(
                video_id="lecture_01",
                timestamp_start=2.0,
                timestamp_end=4.0,
                transcript="This is the microservice architecture diagram.",
                visual_label="architecture diagram",
                visual_labels=["architecture diagram"],
                frame_reference="/data/frames/lecture_01/f1.jpg",
                multimodal_embedding=vec2
            )
        ]

        count = store.insert_chunks(chunks)
        assert count == 2

        # Search with exact vec2
        results = store.search_similar(query_embedding=vec2, video_id="lecture_01", top_k=2)
        assert len(results) == 2
        # Top match should be chunk 2 (architecture diagram)
        assert results[0]["timestamp_start"] == 2.0
        assert results[0]["visual_labels"] == ["architecture diagram"]
        assert results[0]["similarity_score"] > 0.95

    def test_retrieval_evaluation_metrics(self):
        """Verify Recall@k and MRR computation."""
        ground_truth = [
            GroundTruthQueryItem(
                query_id="q1",
                query_text="When is microservice architecture explained?",
                video_id="lecture_01",
                target_timestamp_start=10.0,
                target_timestamp_end=12.0
            ),
            GroundTruthQueryItem(
                query_id="q2",
                query_text="Where is the database consistency discussed?",
                video_id="lecture_01",
                target_timestamp_start=24.0,
                target_timestamp_end=26.0
            )
        ]

        retrieval_map = {
            "q1": [{"video_id": "lecture_01", "timestamp_start": 10.0}],  # Rank 1 hit
            "q2": [{"video_id": "lecture_01", "timestamp_start": 2.0}, {"video_id": "lecture_01", "timestamp_start": 24.0}] # Rank 2 hit
        }

        eval_result = evaluate_retrieval_recall_at_k(ground_truth, retrieval_map, tolerance_seconds=2.0)
        assert eval_result.total_queries == 2
        assert eval_result.recall_at_1 == 0.5  # 1/2
        assert eval_result.recall_at_3 == 1.0  # 2/2
        assert eval_result.mrr == (1.0 + 0.5) / 2.0  # 0.75

    def test_empty_dataset_evaluation_handling(self):
        """Verify evaluation gracefully handles missing benchmarks without fabricating data."""
        eval_result = evaluate_retrieval_recall_at_k([], {})
        assert eval_result.total_queries == 0
        assert eval_result.dataset_status == "Evaluation dataset not yet provided."
