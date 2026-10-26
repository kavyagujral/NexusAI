"""
Nexus Spatiotemporal RAG Engine - Evaluation Framework
======================================================
Provides empirical evaluation metrics for:
1. Spatiotemporal Grounding Retrieval (Recall@k, Mean Reciprocal Rank)
2. CNN Visual Classification (Top-1 Accuracy, Weighted/Macro F1-Score)
3. Attention Distribution Validity (Row-sum stochasticity & bounds)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field
from backend.utils.logger import logger


class GroundTruthQueryItem(BaseModel):
    """Benchmark query with ground-truth temporal window."""
    query_id: str
    query_text: str
    video_id: str
    target_timestamp_start: float
    target_timestamp_end: float
    target_visual_label: Optional[str] = None


class RetrievalEvaluationResult(BaseModel):
    """Empirical evaluation result container for retrieval."""
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float = Field(..., description="Mean Reciprocal Rank")
    dataset_status: str = "Evaluated on provided benchmark set"


def evaluate_retrieval_recall_at_k(
    ground_truth_items: List[GroundTruthQueryItem],
    retrieval_results_map: Dict[str, List[Dict[str, Any]]],
    tolerance_seconds: float = 3.0
) -> RetrievalEvaluationResult:
    """
    Evaluates Recall@k and MRR for spatiotemporal retrieval results.

    A retrieved temporal window is considered a hit if:
        abs(retrieved_start - target_start) <= tolerance_seconds

    Args:
        ground_truth_items: List of labeled benchmark queries.
        retrieval_results_map: Dictionary mapping query_id to list of retrieved result dictionaries.
        tolerance_seconds: Temporal boundary tolerance in seconds.

    Returns:
        RetrievalEvaluationResult with exact Recall@1, Recall@3, Recall@5, and MRR.
    """
    if not ground_truth_items:
        logger.info("Evaluation dataset not yet provided.")
        return RetrievalEvaluationResult(
            total_queries=0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            dataset_status="Evaluation dataset not yet provided."
        )

    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    reciprocal_ranks = []

    for item in ground_truth_items:
        retrieved_list = retrieval_results_map.get(item.query_id, [])
        hit_rank = None

        for rank_idx, match in enumerate(retrieved_list):
            ret_start = match.get("timestamp_start", 0.0)
            ret_vid = match.get("video_id", "")

            # Check matching video and timestamp window proximity
            if (not item.video_id or ret_vid == item.video_id) and abs(ret_start - item.target_timestamp_start) <= tolerance_seconds:
                hit_rank = rank_idx + 1
                break

        if hit_rank is not None:
            if hit_rank <= 1:
                r1_hits += 1
            if hit_rank <= 3:
                r3_hits += 1
            if hit_rank <= 5:
                r5_hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth_items)
    return RetrievalEvaluationResult(
        total_queries=n,
        recall_at_1=round(r1_hits / n, 4) if n > 0 else 0.0,
        recall_at_3=round(r3_hits / n, 4) if n > 0 else 0.0,
        recall_at_5=round(r5_hits / n, 4) if n > 0 else 0.0,
        mrr=round(float(np.mean(reciprocal_ranks)), 4) if n > 0 else 0.0,
        dataset_status=f"Evaluated on {n} benchmark queries"
    )


class VisionClassifierEvaluationResult(BaseModel):
    """Empirical evaluation result container for CNN vision classification."""
    total_samples: int
    top1_accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    dataset_status: str


def evaluate_vision_classifier(
    y_true: List[str],
    y_pred: List[str],
    categories: List[str]
) -> VisionClassifierEvaluationResult:
    """
    Computes Top-1 Accuracy and Macro/Per-class F1-scores for the CNN visual encoder.

    Args:
        y_true: Ground truth category labels.
        y_pred: Predicted category labels.
        categories: List of target classes.

    Returns:
        VisionClassifierEvaluationResult container.
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return VisionClassifierEvaluationResult(
            total_samples=0,
            top1_accuracy=0.0,
            macro_f1=0.0,
            per_class_f1={},
            dataset_status="Evaluation dataset not yet provided."
        )

    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    per_class = f1_score(y_true, y_pred, average=None, labels=categories, zero_division=0)
    per_class_dict = {
        cat: round(float(score), 4)
        for cat, score in zip(categories, per_class)
    }

    return VisionClassifierEvaluationResult(
        total_samples=len(y_true),
        top1_accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        per_class_f1=per_class_dict,
        dataset_status=f"Evaluated on {len(y_true)} labeled validation frames"
    )


class RetrievalEvaluationBenchmark:
    """Benchmark suite for testing and evaluating spatiotemporal grounding retrieval."""
    
    def __init__(self, items: Optional[List[GroundTruthQueryItem]] = None):
        self.items = items or []

    def add_benchmark_item(self, item: GroundTruthQueryItem):
        self.items.append(item)
