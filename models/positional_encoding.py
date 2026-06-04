"""
Positional Encoding for C-JEPA Temporal Transformer

Provides:
1. Temporal positional encoding (per timestep)
2. Object ID encoding (per slot)
3. Combined spatio-temporal embeddings
"""

import math
import torch
import torch.nn as nn
from typing import Literal


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create sinusoidal encoding matrix [max_len, d_model]
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model//2]

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D] or [B, T, N, D]

        Returns:
            x with positional encoding added [same shape as input]
        """
        seq_len = x.shape[1]
        if x.dim() == 3:  # [B, T, D]
            x = x + self.pe[:seq_len, :]
        elif x.dim() == 4:  # [B, T, N, D]
            x = x + self.pe[:seq_len, :].unsqueeze(1)  # Broadcast over N
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    Each position gets a learned embedding vector.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D] or [B, T, N, D]

        Returns:
            x with positional encoding added [same shape as input]
        """
        seq_len = x.shape[1]
        if x.dim() == 3:  # [B, T, D]
            x = x + self.pe[:seq_len, :]
        elif x.dim() == 4:  # [B, T, N, D]
            x = x + self.pe[:seq_len, :].unsqueeze(1)  # Broadcast over N
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        return self.dropout(x)


class ObjectIDEncoding(nn.Module):
    """
    Learnable per-object (per-slot) encoding.
    Allows the model to distinguish between different slot indices.
    """

    def __init__(self, num_slots: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.object_embedding = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, N, D] where N = num_slots

        Returns:
            x with object encoding added [B, T, N, D]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, T, N, D], got {x.dim()}D")

        # Add object embedding: [N, D] broadcast to [B, T, N, D]
        x = x + self.object_embedding.unsqueeze(0).unsqueeze(0)

        return self.dropout(x)


class SpatioTemporalEncoding(nn.Module):
    """
    Combined spatio-temporal encoding for object-centric sequences.

    Adds both:
    - Temporal positional encoding (which frame)
    - Object ID encoding (which slot/object)
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        max_temporal_len: int = 50,
        temporal_encoding: Literal['learnable', 'sinusoidal'] = 'learnable',
        dropout: float = 0.1
    ):
        super().__init__()

        # Temporal encoding
        if temporal_encoding == 'sinusoidal':
            self.temporal_encoding = SinusoidalPositionalEncoding(
                d_model, max_len=max_temporal_len, dropout=0.0
            )
        elif temporal_encoding == 'learnable':
            self.temporal_encoding = LearnablePositionalEncoding(
                d_model, max_len=max_temporal_len, dropout=0.0
            )
        else:
            raise ValueError(f"Unknown temporal encoding: {temporal_encoding}")

        # Object ID encoding
        self.object_encoding = ObjectIDEncoding(num_slots, d_model, dropout=0.0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, N, D]

        Returns:
            x with spatio-temporal encodings [B, T, N, D]
        """
        # Add temporal encoding
        x = self.temporal_encoding(x)  # [B, T, N, D]

        # Add object encoding
        x = self.object_encoding(x)    # [B, T, N, D]

        return self.dropout(x)


class FutureQueryEncoding(nn.Module):
    """
    Learnable query embeddings for future prediction.

    Creates query tokens for each (future_timestep, slot) pair.
    These queries will attend to the encoded history context.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        future_length: int
    ):
        super().__init__()
        self.num_slots = num_slots
        self.future_length = future_length

        # Learnable query embeddings [future_length, num_slots, d_model]
        self.query_embeddings = nn.Parameter(
            torch.randn(future_length, num_slots, d_model) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size B

        Returns:
            Future query tokens [B, T_future * N, D]
        """
        # Expand to batch: [T_future, N, D] -> [B, T_future, N, D]
        queries = self.query_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # [B, T_future, N, D]

        # Flatten temporal and slot dimensions
        B, T, N, D = queries.shape
        queries = queries.reshape(B, T * N, D)  # [B, T_future * N, D]

        return queries


if __name__ == "__main__":
    # Test positional encodings
    print("Testing Positional Encodings...")

    B, T, N, D = 4, 10, 11, 128
    x = torch.randn(B, T, N, D)

    # Test sinusoidal
    sin_pe = SinusoidalPositionalEncoding(D, max_len=50)
    out1 = sin_pe(x)
    print(f"Sinusoidal PE: {x.shape} -> {out1.shape}")

    # Test learnable
    learn_pe = LearnablePositionalEncoding(D, max_len=50)
    out2 = learn_pe(x)
    print(f"Learnable PE: {x.shape} -> {out2.shape}")

    # Test object encoding
    obj_enc = ObjectIDEncoding(num_slots=N, d_model=D)
    out3 = obj_enc(x)
    print(f"Object Encoding: {x.shape} -> {out3.shape}")

    # Test spatio-temporal
    st_enc = SpatioTemporalEncoding(
        d_model=D, num_slots=N, max_temporal_len=50
    )
    out4 = st_enc(x)
    print(f"Spatio-Temporal Encoding: {x.shape} -> {out4.shape}")

    # Test future queries
    future_queries = FutureQueryEncoding(
        d_model=D, num_slots=N, future_length=6
    )
    queries = future_queries(batch_size=B)
    print(f"Future Queries: batch_size={B} -> {queries.shape}")

    print("\n✓ All positional encoding tests passed!")
