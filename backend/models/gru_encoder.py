"""
Nexus Spatiotemporal RAG Engine - Real PyTorch GRU Temporal Encoder
===================================================================
CLO05: Recurrent Sequence Modeling (GRU) for Temporal Audio Dynamics.

Processes sequential audio transcript representations through a PyTorch Gated Recurrent Unit (GRU).
The hidden state transitions across time steps:
    h_t = GRU(x_t, h_{t-1})
Ensuring that later spoken representations maintain contextual knowledge of previous segments.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from backend.config import settings
from backend.utils.logger import logger


class SpokenSegment(BaseModel):
    """Structured container for timestamped audio transcript intervals."""
    segment_id: int
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Spoken transcript text")
    raw_audio_embedding: Optional[List[float]] = None
    temporal_context_embedding: Optional[List[float]] = None


class PyTorchGRUNetwork(nn.Module):
    """
    PyTorch Recurrent Neural Network with Gated Recurrent Units (GRU) and projection head.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = settings.GRU_HIDDEN_DIM,
        output_dim: int = settings.EMBEDDING_DIM,
        bidirectional: bool = settings.GRU_BIDIRECTIONAL
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Real PyTorch GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Contextual projection back to standard embedding dimension (e.g. 512)
        gru_out_dim = hidden_dim * self.num_directions
        self.proj = nn.Sequential(
            nn.Linear(gru_out_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Deterministic weight initialization
        torch.manual_seed(2026)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward recurrent pass.
        Args:
            x: Tensor of shape (Batch, Seq_Len, input_dim)
            h_0: Optional initial hidden state (num_directions, Batch, hidden_dim)
        Returns:
            contextual_seq: Tensor of shape (Batch, Seq_Len, output_dim)
        """
        out, _ = self.gru(x, h_0)  # out: (Batch, Seq_Len, hidden_dim * num_directions)
        contextual_seq = self.proj(out)  # (Batch, Seq_Len, output_dim)
        return contextual_seq


class TemporalGRUEncoder:
    """
    Encoder that coordinates sequential audio representation modeling using the PyTorch GRU.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = settings.GRU_HIDDEN_DIM,
        output_dim: int = settings.EMBEDDING_DIM,
        bidirectional: bool = settings.GRU_BIDIRECTIONAL
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = settings.get_torch_device()

        self.model = PyTorchGRUNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bidirectional=bidirectional
        )
        self.model.eval()
        self.model.to(self.device)

        logger.info(
            f"Initialized TemporalGRUEncoder (input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, bidirectional={bidirectional}, device='{self.device}')"
        )

    def encode_sequence(self, segment_embeddings: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Encodes a temporal sequence of spoken transcript segment embeddings through the GRU.

        Args:
            segment_embeddings: Array of shape (Seq_Len, input_dim) or (Batch, Seq_Len, input_dim)

        Returns:
            np.ndarray: Contextualized representations of shape (Seq_Len, output_dim) or (Batch, Seq_Len, output_dim)
        """
        arr = np.asarray(segment_embeddings, dtype=np.float32)
        is_2d = (arr.ndim == 2)
        if is_2d:
            arr = np.expand_dims(arr, axis=0)  # (1, Seq_Len, input_dim)

        batch_size, seq_len, _ = arr.shape
        if seq_len == 0:
            return np.zeros((0, self.output_dim), dtype=np.float32) if is_2d else np.zeros((batch_size, 0, self.output_dim), dtype=np.float32)

        tensor_in = torch.from_numpy(arr).to(self.device)

        with torch.no_grad():
            tensor_out = self.model(tensor_in)
            out_np = tensor_out.cpu().numpy()

        # Residual skip connection preserves the semantic representation space while
        # incorporating sequential recurrent context modeling across time steps
        contextual_unnorm = arr + 0.15 * out_np
        norms = np.linalg.norm(contextual_unnorm, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        out_normalized = contextual_unnorm / norms

        if is_2d:
            return out_normalized[0]  # Return (Seq_Len, output_dim)
        return out_normalized

