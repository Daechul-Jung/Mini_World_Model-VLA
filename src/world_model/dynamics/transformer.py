from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention and Transformer blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (autoregressive) self-attention.
    Uses PyTorch's scaled_dot_product_attention with a causal mask for efficiency.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each: (B, H, T, head_dim)

        # is_causal=True applies the causal mask automatically (fused kernel)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Dynamics Transformer  (Genie / ST-Transformer style)
# ---------------------------------------------------------------------------

class DynamicsTransformer(nn.Module):
    """
    Causal transformer that models p(z_{t+1} | z_{<=t}, a_{<=t}).

    Inspired by Genie's Latent Action Model (Bruce et al., 2024) and
    the ST-Transformer (Micheli et al., 2023 / IRIS).

    Sequence layout per timestep:  [frame_tok_0, ..., frame_tok_{N-1}, action_tok]
    The full sequence is the concatenation over all T timesteps.

    During training: teacher-force, predict all N frame tokens of t+1
                     from all tokens of t_0 .. t.
    During generation: autoregressively sample N tokens for the next frame.

    Args:
        vocab_size:       VQ-VAE codebook size (K)
        tokens_per_frame: spatial tokens per frame (h * w after VQ-VAE)
        action_dim:       continuous action dimension; 0 = no action conditioning
        n_layers:         transformer depth
        dim:              model (embedding) dimension
        num_heads:        number of attention heads
        max_frames:       maximum context length in frames
        dropout:          dropout rate throughout
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        tokens_per_frame: int = 256,
        action_dim: int = 0,
        n_layers: int = 12,
        dim: int = 512,
        num_heads: int = 8,
        max_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.action_dim = action_dim
        self.dim = dim
        self.has_action = action_dim > 0

        # How many positions per timestep in the flat sequence
        self._stride = tokens_per_frame + (1 if self.has_action else 0)
        max_seq = max_frames * self._stride

        self.tok_emb = nn.Embedding(vocab_size, dim)
        if self.has_action:
            self.act_proj = nn.Linear(action_dim, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

        # Prediction head: only predicts frame tokens, not action positions
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights: output embedding shares with input token embedding
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _build_sequence(
        self,
        frame_tokens: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Interleave frame token embeddings and (optional) action embeddings
        into a single flat sequence.

        Args:
            frame_tokens: (B, T, N) discrete indices
            actions:      (B, T, action_dim) or None
        Returns:
            seq: (B, T * stride, dim)
        """
        B, T, N = frame_tokens.shape
        x_frames = self.tok_emb(frame_tokens)  # (B, T, N, dim)

        if self.has_action and actions is not None:
            x_acts = self.act_proj(actions).unsqueeze(2)   # (B, T, 1, dim)
            x = torch.cat([x_frames, x_acts], dim=2)      # (B, T, N+1, dim)
        else:
            x = x_frames                                   # (B, T, N, dim)

        return x.reshape(B, T * self._stride, self.dim)

    def _frame_positions(self, T: int, device: torch.device) -> torch.Tensor:
        """Flat sequence indices that correspond to frame tokens (not action tokens)."""
        offsets = torch.arange(self.tokens_per_frame, device=device)
        starts = torch.arange(T, device=device) * self._stride
        return (starts[:, None] + offsets[None, :]).reshape(-1)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            frame_tokens: (B, T, N)   — frame token indices for T timesteps
            actions:      (B, T, D_a) — (optional) actions at each timestep
        Returns:
            logits: (B, T, N, vocab_size) — per-token next-token predictions
        """
        B, T, N = frame_tokens.shape
        seq = self._build_sequence(frame_tokens, actions)   # (B, T*stride, dim)

        pos = torch.arange(seq.shape[1], device=seq.device)
        seq = self.drop(seq + self.pos_emb(pos))

        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)                                # (B, T*stride, dim)

        # Extract only the frame-token positions for logit prediction
        frame_pos = self._frame_positions(T, seq.device)   # (T*N,)
        frame_out = seq[:, frame_pos, :]                   # (B, T*N, dim)
        logits = self.head(frame_out.reshape(B, T, N, self.dim))  # (B, T, N, vocab)
        return logits

    @torch.no_grad()
    def generate_next_frame(
        self,
        frame_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 256,
    ) -> torch.Tensor:
        """
        Sample the next frame's tokens given history.

        Args:
            frame_tokens: (B, T, N)   — observed frame tokens (context)
            actions:      (B, 1, D_a) — action for the next step or None
        Returns:
            next_tokens: (B, N) sampled token indices
        """
        self.eval()
        # Append a dummy action slot for the next step if needed
        if self.has_action and actions is not None:
            combined_actions = torch.cat([
                actions[:, :frame_tokens.shape[1]],  # history actions
                actions[:, frame_tokens.shape[1]:frame_tokens.shape[1]+1],
            ], dim=1) if actions.shape[1] > frame_tokens.shape[1] else actions
        else:
            combined_actions = None

        logits = self(frame_tokens, combined_actions)   # (B, T, N, vocab)
        next_logits = logits[:, -1, :, :] / temperature  # (B, N, vocab)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
            next_logits[next_logits < v[..., -1:]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)             # (B, N, vocab)
        next_tokens = torch.multinomial(
            probs.reshape(-1, self.vocab_size), 1
        ).reshape(frame_tokens.shape[0], -1)               # (B, N)
        return next_tokens
