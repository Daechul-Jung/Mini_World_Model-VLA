from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Straight-through Vector Quantization (van den Oord et al., 2017).

    Args:
        num_embeddings: codebook size K
        embedding_dim: code vector dimension D
        beta: commitment loss weight (typically 0.25)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, C, H, W) continuous encoder output
        Returns:
            z_q:    (B, C, H, W)  quantized tensor (straight-through gradient)
            loss:   scalar        VQ + commitment loss
            indices:(B, H, W)     nearest codebook entry per spatial position
        """
        z_bhwc = z.permute(0, 2, 3, 1).contiguous()       # (B, H, W, C)
        z_flat = z_bhwc.view(-1, self.embedding_dim)       # (B*H*W, C)

        # Squared Euclidean distances to codebook
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2.0 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = d.argmin(dim=1)                          # (B*H*W,)
        z_q = self.embedding(indices).view(z_bhwc.shape)  # (B, H, W, C)

        # Codebook loss + commitment loss
        loss = (
            F.mse_loss(z_q.detach(), z_bhwc)
            + self.beta * F.mse_loss(z_q, z_bhwc.detach())
        )

        # Straight-through estimator: gradients flow through z, not z_q
        z_q = z_bhwc + (z_q - z_bhwc).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()        # (B, C, H, W)
        indices = indices.view(z.shape[0], z.shape[2], z.shape[3])  # (B, H, W)

        return z_q, loss, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map discrete indices back to continuous embeddings.

        Args:
            indices: (B, H, W) or (B, N)
        Returns:
            z_q: (B, C, H, W) if 3-D input, else (B, N, C)
        """
        flat = indices.view(-1)
        z_q = self.embedding(flat)

        if indices.ndim == 3:
            B, H, W = indices.shape
            z_q = z_q.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)
        elif indices.ndim == 2:
            z_q = z_q.view(*indices.shape, self.embedding_dim)

        return z_q
