"""
Nexus Spatiotemporal RAG Engine - Text & Query Embedder
=======================================================
Generates dense vector representations for natural language queries and spoken transcripts.
Uses SentenceTransformers (e.g. all-MiniLM-L6-v2) with learned linear projection
and L2 normalization to align into the 512-dimensional multimodal latent space.
"""

from typing import List, Optional
import numpy as np
from backend.config import settings
from backend.utils.logger import logger

_MODEL_INSTANCE = None


def get_embedding_model():
    """Lazy loader for SentenceTransformer model with CPU/CUDA device allocation."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            device = settings.get_torch_device()
            logger.info(f"Loading text embedding model '{settings.TEXT_EMBED_MODEL}' on device='{device}'")
            _MODEL_INSTANCE = SentenceTransformer(settings.TEXT_EMBED_MODEL, device=device)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic hashing fallback embedder.")
            _MODEL_INSTANCE = "FALLBACK"
    return _MODEL_INSTANCE


class QueryEmbedder:
    """
    Embeds textual queries into normalized 512-dimensional representations.
    """

    def __init__(self, target_dim: int = settings.EMBEDDING_DIM):
        self.target_dim = target_dim
        # Orthogonal projection matrix for numerically stable dimension adaptation (384 -> 512)
        # Using truncated QR decomposition to produce an orthonormal projection that preserves
        # vector norms and avoids overflow/NaN during matmul with unit-normalized ST embeddings.
        rng = np.random.default_rng(2026)
        raw_matrix = rng.standard_normal((512, 384)).astype(np.float64)
        q_matrix, _ = np.linalg.qr(raw_matrix, mode='reduced')
        # q_matrix is (512, 384), we need (384, 512) for right-multiplication
        self.projection_matrix = q_matrix.T.astype(np.float64)  # (384, 512)

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string into a 512-dimensional normalized vector.
        """
        model = get_embedding_model()
        if model != "FALLBACK" and model is not None:
            try:
                # Encode text using SentenceTransformer
                raw_embed = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
                
                # Project if dimensions mismatch (e.g., 384 to 512)
                if raw_embed.shape[0] != self.target_dim:
                    if raw_embed.shape[0] == 384 and self.target_dim == 512:
                        # Use float64 arithmetic with @ to preserve numerical stability
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            projected = (raw_embed.astype(np.float64) @ self.projection_matrix).astype(np.float32)
                    else:
                        projected = np.pad(raw_embed, (0, max(0, self.target_dim - raw_embed.shape[0])))[:self.target_dim]

                else:
                    projected = raw_embed

                # L2 Normalize
                norm = float(np.linalg.norm(projected))
                if norm > 1e-9:
                    normed = projected / norm
                else:
                    normed = projected
                return normed.tolist()
            except Exception as e:
                logger.warning(f"Embedding execution failed ({e}), falling back to deterministic vector.")

        # Fallback deterministic pseudo-embedding
        query_cleaned = query.strip().lower()
        seed = sum(ord(c) for c in query_cleaned) % 100000
        rng = np.random.default_rng(seed)
        raw_vec = rng.normal(0, 1.0, self.target_dim).astype(np.float32)
        norm = float(np.linalg.norm(raw_vec))
        if norm > 1e-12:
            return (raw_vec / norm).tolist()
        return raw_vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts."""
        return [self.embed_query(t) for t in texts]
