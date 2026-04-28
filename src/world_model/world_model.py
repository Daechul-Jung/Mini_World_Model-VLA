from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer.vqvae import VQVAE
from .dynamics.transformer import DynamicsTransformer
from .decoder.unet import UNet
from .decoder.ddpm import DDPMScheduler


class GenieWorldModel(nn.Module):
    """
    Generative world model inspired by Genie (Bruce et al., 2024) and DIAMOND
    (Micheli et al., 2024).

    Architecture pipeline
    ---------------------
    1. **VQ-VAE tokenizer**: encode video frames → discrete spatial token maps.
       Each frame becomes an h×w grid of integer indices from a codebook of size K.

    2. **Dynamics Transformer**: causal GPT-style model that predicts the next
       frame's token map given the history of frame tokens (and optionally actions).
       This is where the world model "learns physics / room layout."

    3. **Diffusion decoder**: a UNet denoiser that generates a pixel-level frame
       conditioned on the VQ-VAE latent of the predicted tokens.
       This produces high-quality, sharp imagined frames.

    Usage as an RL environment
    --------------------------
    Call `imagine()` to roll out n future frames given context frames + actions.
    The imagined frames can serve as a generative environment for training VLA
    or RL agents without real-world interaction.

    Args:
        vqvae:      VQVAE tokenizer
        dynamics:   DynamicsTransformer next-token predictor
        decoder:    UNet denoiser
        scheduler:  DDPMScheduler for noise schedule
    """

    def __init__(
        self,
        vqvae: VQVAE,
        dynamics: DynamicsTransformer,
        decoder: UNet,
        scheduler: DDPMScheduler,
    ):
        super().__init__()
        self.vqvae = vqvae
        self.dynamics = dynamics
        self.decoder = decoder
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize_frames(
        self, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of video frames.

        Args:
            frames: (B, T, C, H, W) in [-1, 1]
        Returns:
            z_q:    (B, T, D, h, w) quantized latents
            vq_loss: scalar VQ commitment loss
            indices:(B, T, h, w)   discrete code indices
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        z_q, vq_loss, indices = self.vqvae.encode(flat)
        _, D, h, w = z_q.shape
        return z_q.reshape(B, T, D, h, w), vq_loss, indices.reshape(B, T, h, w)

    def latent_to_context(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Reshape spatial latent map into sequence of context tokens for UNet cross-attention.

        Args:
            z_q: (B, D, h, w)
        Returns:
            ctx: (B, h*w, D)
        """
        B, D, h, w = z_q.shape
        return z_q.reshape(B, D, h * w).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Loss computation (for training)
    # ------------------------------------------------------------------

    def vqvae_loss(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        VQ-VAE reconstruction loss over a batch of single frames.

        Args:
            frames: (B, C, H, W) in [-1, 1]
        Returns:
            loss, metrics dict
        """
        _, loss, metrics = self.vqvae(frames)
        return loss, metrics

    def dynamics_loss(
        self,
        indices: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Next-token prediction cross-entropy loss.

        Args:
            indices: (B, T, h, w)
            actions: (B, T, D_a) or None
        Returns:
            scalar cross-entropy loss
        """
        B, T, h, w = indices.shape
        tokens = indices.reshape(B, T, h * w)            # (B, T, N)

        # Predict t+1 from t_0..t, so slice input and target
        logits = self.dynamics(tokens[:, :-1], actions[:, :-1] if actions is not None else None)
        targets = tokens[:, 1:].reshape(-1)              # (B*(T-1)*N,)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)

    def diffusion_loss(
        self,
        frames: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion denoising score-matching loss (predict added noise).

        Args:
            frames: (B, C, H, W)  clean target frames in [-1, 1]
            z_q:   (B, D, h, w)  VQ quantized latent of the same frames
        Returns:
            scalar MSE loss between predicted and actual noise
        """
        B = frames.shape[0]
        t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=frames.device)
        x_t, noise = self.scheduler.add_noise(frames, t)
        ctx = self.latent_to_context(z_q)                # (B, h*w, D)
        noise_pred = self.decoder(x_t, t, ctx)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Imagination / rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def imagine(
        self,
        context_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        n_steps: int = 1,
        ddim_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 256,
    ) -> torch.Tensor:
        """
        Roll out imagined frames autoregressively.

        Each step:
          1. DynamicsTransformer predicts next frame's token map.
          2. Diffusion UNet decodes the token map to a pixel-level image.

        Args:
            context_frames: (B, T_ctx, C, H, W) observed frames in [-1, 1]
            actions:        (B, T_ctx + n_steps, D_a) all actions or None
            n_steps:        number of future frames to generate
            ddim_steps:     DDIM denoising steps (50 is fast, 200 is high quality)
            temperature:    sampling temperature for dynamics model
            top_k:          top-k filtering for dynamics model
        Returns:
            imagined: (B, n_steps, C, H, W) generated frames
        """
        B, T_ctx, C, H, W = context_frames.shape

        _, _, ctx_indices = self.tokenize_frames(context_frames)
        h, w = ctx_indices.shape[2], ctx_indices.shape[3]
        N = h * w

        token_history = ctx_indices.reshape(B, T_ctx, N)
        imagined: List[torch.Tensor] = []

        for step in range(n_steps):
            act = (
                actions[:, step : step + 1]
                if actions is not None and step < actions.shape[1]
                else None
            )
            next_tokens = self.dynamics.generate_next_frame(
                token_history, act, temperature=temperature, top_k=top_k
            )  # (B, N)

            # Decode predicted tokens through VQ codebook → latent map
            next_indices = next_tokens.reshape(B, h, w)
            z_q = self.vqvae.quantizer.decode_indices(next_indices)  # (B, D, h, w)
            ctx = self.latent_to_context(z_q)                        # (B, N, D)

            frame = self._ddim_sample(ctx, (B, C, H, W), ddim_steps)
            imagined.append(frame)

            token_history = torch.cat([token_history, next_tokens.unsqueeze(1)], dim=1)

        return torch.stack(imagined, dim=1)   # (B, n_steps, C, H, W)

    def _ddim_sample(
        self,
        context: torch.Tensor,
        shape: Tuple[int, ...],
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM deterministic (or stochastic with eta>0) reverse sampling."""
        device = context.device
        x = torch.randn(shape, device=device)
        T = self.scheduler.num_train_timesteps

        ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
        for i, t_cur in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < steps else torch.tensor(-1, device=device)
            t_batch = t_cur.expand(shape[0])
            noise_pred = self.decoder(x, t_batch, context)
            x = self.scheduler.ddim_step(noise_pred, t_cur.item(), t_prev.item(), x, eta)

        return x

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_medium(
        cls,
        action_dim: int = 0,
        image_size: int = 256,
        max_frames: int = 16,
    ) -> "GenieWorldModel":
        """
        Build a medium-sized world model (~275M params, fits ~16GB VRAM with bf16).

        Component breakdown:
          VQ-VAE:   ~40M  — 3 stages, 8× downsample, codebook 1024×256
          Dynamics: ~85M  — 12L / 512d / 8h GPT-small
          Decoder:  ~150M — UNet 128/256/512/1024 channels
        """
        from .tokenizer.vqvae import VQVAE
        from .dynamics.transformer import DynamicsTransformer
        from .decoder.unet import UNet
        from .decoder.ddpm import DDPMScheduler

        spatial_tokens = (image_size // 8) ** 2   # 32*32=1024 for 256px with 3 downsample stages

        vqvae = VQVAE(
            in_channels=3,
            base_channels=128,
            channel_mults=(1, 2, 4),
            num_embeddings=1024,
            latent_dim=256,
            n_res_blocks=2,
        )
        dynamics = DynamicsTransformer(
            vocab_size=1024,
            tokens_per_frame=spatial_tokens,
            action_dim=action_dim,
            n_layers=12,
            dim=512,
            num_heads=8,
            max_frames=max_frames,
        )
        decoder = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=128,
            channel_mults=(1, 2, 4, 8),
            attn_at_levels=(False, False, True, True),
            context_dim=256,
            n_res_blocks=2,
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
        return cls(vqvae, dynamics, decoder, scheduler)

    @classmethod
    def create_small(
        cls,
        action_dim: int = 0,
        image_size: int = 128,
    ) -> "GenieWorldModel":
        """
        Smaller variant (~80M params) for quick experiments on constrained hardware.
        Use 128px images (spatial_tokens = 16*16 = 256).
        """
        spatial_tokens = (image_size // 8) ** 2

        vqvae = VQVAE(
            in_channels=3,
            base_channels=64,
            channel_mults=(1, 2, 4),
            num_embeddings=512,
            latent_dim=128,
            n_res_blocks=1,
        )
        dynamics = DynamicsTransformer(
            vocab_size=512,
            tokens_per_frame=spatial_tokens,
            action_dim=action_dim,
            n_layers=6,
            dim=256,
            num_heads=4,
            max_frames=8,
        )
        decoder = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            channel_mults=(1, 2, 4),
            attn_at_levels=(False, True, True),
            context_dim=128,
            n_res_blocks=1,
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
        return cls(vqvae, dynamics, decoder, scheduler)
