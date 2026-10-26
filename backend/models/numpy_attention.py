"""
Nexus Spatiotemporal RAG Engine - Custom NumPy Cross-Modal Attention
====================================================================
Academic Requirements Mapping:
CLO01 / CLO02: Applied Mathematics & Vectorized Tensor Operations in Pure NumPy.

Report-Compliant Mathematical Formulation:
------------------------------------------
Given Spoken Audio representation A and Visual Frame representation V:
    Audio  -> Query (Q = A * W_Q)
    Visual -> Key   (K = V * W_K)
    Visual -> Value (V_l = V * W_V)

Scaled Dot-Product Similarity:
    S = (Q * K^T) / sqrt(d_k)

Attention Weights (Softmax Normalization):
    alpha = softmax(S) = exp(S - max(S)) / sum(exp(S - max(S)))

Weighted Feature Aggregation with Residual Connection:
    Z = alpha * V_l + A

Layer Normalization (from first principles in NumPy):
    Z_hat = LayerNorm(Z) = ((Z - mu) / sqrt(sigma^2 + eps)) * gamma + beta

Where:
- A in R^(Batch x Seq_A x d_model): Spoken Audio representation (from GRU/text encoder)
- V in R^(Batch x Seq_V x d_model): Visual Frame representation (from ResNet-50 CNN)
- W_Q in R^(d_model x d_k): Query projection matrix
- W_K in R^(d_model x d_k): Key projection matrix
- W_V in R^(d_model x d_v): Value projection matrix
- gamma, beta: Learnable/calibrated LayerNorm scale and shift parameters

Note on Multi-Head Attention:
The report's primary implemented methodology is Single-Head Cross-Modal Attention.
Multi-Head Attention is preserved as an optional architectural extension (future work).
"""

from typing import Optional, Tuple
import numpy as np


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable Softmax implementation using pure NumPy.

    Math:
        softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Subtracting max(x) across the target axis prevents floating-point overflow,
    while maintaining exact mathematical equivalence.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / (sum_exp_x + 1e-12)


def layer_norm_np(
    x: np.ndarray,
    gamma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Layer Normalization implemented from first principles in pure NumPy.

    Math:
        mu = (1 / d) * sum(x)
        sigma^2 = (1 / d) * sum((x - mu)^2)
        x_hat = ((x - mu) / sqrt(sigma^2 + eps)) * gamma + beta

    Args:
        x: Input tensor of shape (..., d_model)
        gamma: Scale parameter (broadcastable to d_model)
        beta: Shift parameter (broadcastable to d_model)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with identical shape as x.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)

    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta

    return x_norm


def scaled_dot_product_attention_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Attention using pure NumPy matrix operations.

    Math:
        Scores = (Q @ K^T) / sqrt(d_k)
        Attention_Weights = softmax(Scores + Mask)
        Output = Attention_Weights @ V

    Shapes:
        q: (..., Seq_Q, d_k)
        k: (..., Seq_K, d_k)
        v: (..., Seq_K, d_v)
        mask: Optional mask (..., Seq_Q, Seq_K)
    """
    d_k = q.shape[-1]
    
    # Transpose last two dimensions of K: (..., d_k, Seq_K)
    k_t = np.ascontiguousarray(np.swapaxes(k, -1, -2))
    
    # Dot product similarity: (..., Seq_Q, Seq_K)
    scores = np.matmul(q, k_t) / float(np.sqrt(d_k))

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax probabilities summing to 1.0 along the Key axis
    attention_weights = softmax_np(scores, axis=-1)

    # Weighted Value aggregation: (..., Seq_Q, d_v)
    output = np.matmul(attention_weights, v)

    return output, attention_weights


class NumPyCrossModalAttentionSingleHead:
    """
    Primary Report-Compliant Single-Head Cross-Modal Attention Engine.
    
    Formulation:
        Q = A @ W_Q          (Audio -> Query)
        K = V @ W_K          (Visual -> Key)
        V_l = V @ W_V        (Visual -> Value)
        S = (Q @ K^T) / sqrt(d_k)
        alpha = softmax(S)
        Z = alpha @ V_l + A  (Residual aggregation)
        Z_hat = LayerNorm(Z)
    """

    def __init__(self, d_model: int = 512, seed: int = 2026):
        self.d_model = d_model
        
        # Numerically stable orthogonal initialization for projection matrices
        # Orthogonal init preserves vector norms during matmul and prevents
        # the overflow/NaN issues that occur with Xavier uniform on d=512.
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        
        def _ortho_init(rng, shape, scale):
            """Generate orthogonal matrix via QR decomposition, scaled."""
            raw = rng.standard_normal(shape).astype(np.float64)
            q, _ = np.linalg.qr(raw)
            return np.ascontiguousarray(q * scale, dtype=np.float32)
        
        self.W_q = _ortho_init(rng, (d_model, d_model), scale)
        self.W_k = _ortho_init(rng, (d_model, d_model), scale)
        self.W_v = _ortho_init(rng, (d_model, d_model), scale)
        
        # LayerNorm parameters
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def forward(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes single-head cross-modal attention forward pass.

        Args:
            audio_features: (Batch, Seq_A, d_model) or (Seq_A, d_model) or (d_model,)
            visual_features: (Batch, Seq_V, d_model) or (Seq_V, d_model) or (d_model,)
            mask: Optional attention mask

        Returns:
            fused_representation (Z_hat): Normalized fused features
            attention_weights (alpha): Cross-modal attention weights
        """
        # Use float64 for all matmul operations to avoid Apple Silicon Accelerate
        # BLAS subnormal float32 overflow warnings, then cast back to float32.
        a = np.ascontiguousarray(audio_features, dtype=np.float64)
        v = np.ascontiguousarray(visual_features, dtype=np.float64)

        # Handle dimensional shapes: expand to 3D (Batch, Seq, d_model)
        orig_ndim = a.ndim
        if a.ndim == 1:
            a = a[np.newaxis, np.newaxis, :]  # (1, 1, d)
        elif a.ndim == 2:
            a = a[np.newaxis, :, :]            # (1, Seq_A, d)

        if v.ndim == 1:
            v = v[np.newaxis, np.newaxis, :]  # (1, 1, d)
        elif v.ndim == 2:
            v = v[np.newaxis, :, :]            # (1, Seq_V, d)

        # 1. Project Audio to Queries: Q = A @ W_Q, K = V @ W_K, V_l = V @ W_V
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Q = np.matmul(a, self.W_q.astype(np.float64))
            K = np.matmul(v, self.W_k.astype(np.float64))
            V_l = np.matmul(v, self.W_v.astype(np.float64))


        # Cast back to float32 for softmax and downstream ops
        Q = Q.astype(np.float32)
        K = K.astype(np.float32)
        V_l = V_l.astype(np.float32)
        a = a.astype(np.float32)

        # 3. Scaled Dot-Product Attention: alpha = softmax(Q @ K^T / sqrt(d_k))
        attn_out, alpha = scaled_dot_product_attention_np(Q, K, V_l, mask=mask)

        # 4. Residual Connection: Z = alpha @ V_l + A
        Z = attn_out + a

        # 5. Layer Normalization: Z_hat = LayerNorm(Z)
        Z_hat = layer_norm_np(Z, gamma=self.gamma, beta=self.beta)

        # Squeeze back to original rank if needed
        if orig_ndim == 1:
            Z_hat = Z_hat[0, 0]
            alpha = alpha[0, 0]
        elif orig_ndim == 2:
            Z_hat = Z_hat[0]
            alpha = alpha[0]

        return Z_hat, alpha


# Alias for report primary formulation
NumPyCrossModalAttention = NumPyCrossModalAttentionSingleHead


class NumPyMultiHeadAttention:
    """
    Multi-Head Attention Extension (Preserved as architectural extension / future work).
    """

    def __init__(self, d_model: int = 512, num_heads: int = 4, seed: int = 2026):
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        limit = float(np.sqrt(6.0 / (2.0 * d_model)))
        
        self.W_q = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return np.ascontiguousarray(np.swapaxes(reshaped, 1, 2))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, num_heads, seq_len, d_k = x.shape
        transposed = np.ascontiguousarray(np.swapaxes(x, 1, 2))
        return transposed.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        batch_size = q.shape[0]

        proj_q = np.matmul(q, self.W_q)
        proj_k = np.matmul(k, self.W_k)
        proj_v = np.matmul(v, self.W_v)

        q_heads = self._split_heads(proj_q)
        k_heads = self._split_heads(proj_k)
        v_heads = self._split_heads(proj_v)

        out_heads, attn_weights = scaled_dot_product_attention_np(q_heads, k_heads, v_heads, mask=mask)
        concat_out = self._combine_heads(out_heads)
        output = np.matmul(concat_out, self.W_o)

        return output, attn_weights
