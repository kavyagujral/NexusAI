"""
Nexus Spatiotemporal RAG Engine - Multimodal Attention & Alignment
===================================================================
Fuses spoken audio transcript features (from GRU), CNN visual frame features (from ResNet-50),
and sinusoidal temporal positional encodings into unified 512-dimensional spatiotemporal embeddings
using the custom pure-NumPy single-head cross-modal attention layer.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel
from backend.models.numpy_attention import NumPyCrossModalAttentionSingleHead, scaled_dot_product_attention_np
from backend.config import settings
from backend.utils.logger import logger


class SpatiotemporalChunk(BaseModel):
    """Unified container for an aligned spatiotemporal video chunk."""
    video_id: str
    timestamp_start: float
    timestamp_end: float
    transcript: str
    visual_label: str = "slide"
    visual_labels: List[str] = ["slide"]
    frame_reference: str
    audio_embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None
    multimodal_embedding: Optional[List[float]] = None


def generate_temporal_positional_encoding(timestamps: np.ndarray, d_model: int) -> np.ndarray:
    """
    Computes continuous sinusoidal temporal positional encodings for timestamps in seconds.

    Math:
        PE(t, 2i)   = sin(t / 10000^(2i / d_model))
        PE(t, 2i+1) = cos(t / 10000^(2i / d_model))
    """
    if timestamps.ndim == 1:
        timestamps = timestamps[:, np.newaxis]  # (N, 1)

    pe = np.zeros((timestamps.shape[0], d_model), dtype=np.float32)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)).astype(np.float32)

    pe[:, 0::2] = np.sin(timestamps * div_term)
    pe[:, 1::2] = np.cos(timestamps * div_term)
    return pe


class SpatiotemporalAligner:
    """
    Aligns multimodal video streams:
    - Audio representations (from GRU sequence encoder)
    - Visual frame representations (from ResNet-50 CNN)
    - Sinusoidal temporal positional encodings
    """

    def __init__(self, embedding_dim: int = settings.EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        
        # Primary Single-Head Cross-Modal Attention Engine (from report)
        self.cross_attention = NumPyCrossModalAttentionSingleHead(d_model=embedding_dim)
        logger.info(f"Initialized SpatiotemporalAligner with Single-Head NumPy Attention (dim={embedding_dim})")

    def fuse_features(
        self,
        audio_embed: np.ndarray,
        visual_embed: np.ndarray,
        timestamp_seconds: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuses audio and visual representations with continuous temporal positional encodings.

        Formulation:
            A_mod = Audio + PE(t)
            V_mod = Visual + PE(t)
            Z_hat, alpha = CrossModalAttention(A_mod, V_mod)

        Returns:
            normalized_fused_vector: 512-dimensional unit vector
            attention_weights: Attention scores
        """
        a = np.asarray(audio_embed, dtype=np.float32)
        v = np.asarray(visual_embed, dtype=np.float32)

        # Compute continuous sinusoidal temporal encoding with proportional magnitude
        pe = generate_temporal_positional_encoding(np.array([timestamp_seconds]), self.embedding_dim)[0]
        pe_norm = np.linalg.norm(pe)
        if pe_norm > 1e-9:
            pe = (pe / pe_norm) * 0.1  # Subtle 10% temporal modulation

        # Modulate representations with temporal position
        a_temporal = a + pe
        v_temporal = v + pe


        # Execute pure NumPy cross-modal attention forward pass
        fused_vec, attn_weights = self.cross_attention.forward(
            audio_features=a_temporal,
            visual_features=v_temporal
        )

        # L2-normalize for ChromaDB cosine similarity indexing
        norm = float(np.linalg.norm(fused_vec))
        if norm > 1e-12:
            normalized_embedding = fused_vec / norm
        else:
            normalized_embedding = fused_vec

        return normalized_embedding, attn_weights
