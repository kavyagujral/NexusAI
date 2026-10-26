"""
Unit Tests for Custom Pure NumPy Attention Mechanism (CLO01 / CLO02)
=====================================================================
Validates report-compliant Single-Head Cross-Modal Attention from first principles:
- Softmax numerical stability and row probability sum == 1.0
- LayerNorm properties
- Scaled Dot-Product Attention: Audio (Query) attending to Visual (Key/Value)
- Residual connection and output shapes
- Sinusoidal continuous temporal positional encodings
"""

import numpy as np
import pytest
from backend.models.numpy_attention import (
    softmax_np,
    layer_norm_np,
    scaled_dot_product_attention_np,
    NumPyCrossModalAttentionSingleHead,
    NumPyMultiHeadAttention
)
from backend.models.multimodal_attention import (
    generate_temporal_positional_encoding,
    SpatiotemporalAligner
)


class TestNumPyAttentionFormulations:
    """Test suite verifying mathematical attention computations in pure NumPy."""

    def test_softmax_stochasticity_and_stability(self):
        """Verify softmax sums to 1.0 along the designated axis and handles large numbers without overflow."""
        # Extreme values test
        x = np.array([[1000.0, 1001.0, 1002.0], [-500.0, 0.0, 500.0]], dtype=np.float64)
        sm = softmax_np(x, axis=-1)

        # Shape preservation
        assert sm.shape == x.shape
        # Probability sum = 1.0 along axis -1
        assert np.allclose(np.sum(sm, axis=-1), np.array([1.0, 1.0]))
        # Non-negative
        assert np.all(sm >= 0.0)
        # Shift invariance: softmax(x) == softmax(x + 1000)
        assert np.allclose(sm[0], softmax_np(x[0] + 500.0), atol=1e-6)

    def test_layer_norm_from_first_principles(self):
        """Verify LayerNorm produces mean ~0.0 and unit variance ~1.0 along last axis."""
        x = np.random.randn(4, 16, 512).astype(np.float32) * 5.0 + 10.0
        normed = layer_norm_np(x)

        # Mean should be 0.0 along feature axis
        means = np.mean(normed, axis=-1)
        assert np.allclose(means, 0.0, atol=1e-4)

        # Variance should be 1.0 along feature axis
        vars_ = np.var(normed, axis=-1)
        assert np.allclose(vars_, 1.0, atol=1e-3)

    def test_single_head_cross_modal_attention(self):
        """
        Verify Single-Head Cross-Modal formulation:
        Audio (Query), Visual (Key/Value), Residual connection + LayerNorm.
        """
        d_model = 512
        attention = NumPyCrossModalAttentionSingleHead(d_model=d_model, seed=2026)

        audio_vec = np.random.randn(d_model).astype(np.float32)
        visual_vec = np.random.randn(d_model).astype(np.float32)

        fused_out, alpha = attention.forward(audio_vec, visual_vec)

        # Shape checks
        assert fused_out.shape == (d_model,)
        # Attention score for single item must be 1.0
        assert np.isclose(alpha, 1.0, atol=1e-5)

    def test_scaled_dot_product_attention_masking(self):
        """Verify that masked positions receive approximately 0.0 attention weight."""
        q = np.array([[[1.0, 0.0]]], dtype=np.float32)       # (1, 1, 2)
        k = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32) # (1, 2, 2)
        v = np.array([[[10.0], [20.0]]], dtype=np.float32)   # (1, 2, 1)
        
        # Mask out second key position
        mask = np.array([[[1, 0]]], dtype=np.float32)  # (1, 1, 2)

        output, attn_weights = scaled_dot_product_attention_np(q, k, v, mask=mask)

        # Attn weight on masked element must be ~0.0
        assert attn_weights[0, 0, 1] < 1e-6
        # Attn weight on unmasked element must be ~1.0
        assert np.isclose(attn_weights[0, 0, 0], 1.0, atol=1e-5)
        # Output should be purely the first value (10.0)
        assert np.isclose(output[0, 0, 0], 10.0, atol=1e-4)

    def test_temporal_positional_encoding(self):
        """Verify continuous sinusoidal temporal positional encodings."""
        timestamps = np.array([0.0, 2.0, 4.0, 6.0])
        d_model = 512
        pe = generate_temporal_positional_encoding(timestamps, d_model)

        assert pe.shape == (4, d_model)
        # Values must be bounded in [-1.0, 1.0]
        assert np.all(pe >= -1.0) and np.all(pe <= 1.0)

    def test_spatiotemporal_aligner_fusion(self):
        """Verify end-to-end spatiotemporal fusion yields normalized 512-dim unit vector."""
        aligner = SpatiotemporalAligner(embedding_dim=512)
        
        audio_embed = np.random.randn(512).astype(np.float32)
        visual_embed = np.random.randn(512).astype(np.float32)
        timestamp = 14.0

        fused_vec, _ = aligner.fuse_features(audio_embed, visual_embed, timestamp)

        assert fused_vec.shape == (512,)
        # L2 norm must be 1.0
        norm = np.linalg.norm(fused_vec)
        assert np.isclose(norm, 1.0, atol=1e-5)
