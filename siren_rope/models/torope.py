"""
Generative Recommender: Time-and-Order RoPE variant (torope).

Uses ToRoPE — Rotary Position Embedding whose rotation angle is the sum of a
fixed order-based frequency (like standard RoPE) and a scaled timestamp offset.
Both signals share the same frequency basis (early fusion), so the timestamp
simply shifts the effective position of each token by its elapsed time.
Compare with ordinal_rope.py (order only) and siren_rope.py (order + learned SIREN angle).

Reference: "Rotate Both Ways: Time-and-Order RoPE for Generative Recommendation"
           arXiv:2510.20455
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ToRoPE(nn.Module):
    """
    Time-and-Order Rotary Position Embedding.

    The rotation angle for dimension pair i is:
        θ_i = (position + time_scale × Δt_days) × ω_i

    where:
    - position is the forward sequence index (0=oldest token, S-1=newest).
    - Δt_days is the wall-clock time elapsed since the oldest token in the
      sequence, measured in days (so the oldest token always contributes 0).
    - ω_i = 1 / base^(2i / d_k) is the standard RoPE frequency for pair i.
    - time_scale controls how much a one-day gap shifts the effective position.
      At time_scale=0.01, two tokens 100 days apart rotate by the same amount
      as two tokens 1 position apart.

    Using relative time (Δt) rather than absolute timestamps makes the encoding
    invariant to when the sequence was recorded, while still capturing the
    temporal spacing between events.

    Setting time_scale=0.0 recovers vanilla RoPE.
    Uses real-valued cos/sin (not complex numbers) for torch.compile compatibility.
    """

    def __init__(self, d_k: int, base: float = 10000.0, time_scale: float = 0.01):
        """
        Args:
            d_k: Per-head key/query dimension. Must be even.
            base: Frequency base for the rotation ladder. Higher base → slower
                  rotation at high-frequency dimensions, useful for long sequences.
            time_scale: How much one day of elapsed time shifts the effective
                        position. Typical range [0.0, 0.1]; 0.01 means a 100-day
                        gap equals a 1-position shift in rotation angle.
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for ToRoPE, but got d_k={d_k}.")
        self.d_k = d_k
        self.time_scale = time_scale

        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_angles(
        self, positions: torch.Tensor, timestamp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cos/sin rotation tensors from sequence positions and timestamps.

        The effective position for token (b, s) is:
            pos_eff = positions[s] + time_scale × (timestamp[b,s] − min_t[b]) / 86400

        Args:
            positions: [S]     — forward sequence indices 0, 1, ..., S-1.
            timestamp: [B, S]  — unix timestamps in seconds, ordered oldest→newest.
        Returns:
            cos, sin: [B, 1, S, d_k//2]  — broadcast-ready over attention heads.
        """
        # Position contribution: [S, d_k//2]
        pos_angles = torch.outer(positions, self.inv_freq)

        # Relative elapsed time in days (oldest token in each sequence = 0).
        t_days = timestamp.float() / 86400.0                          # [B, S]
        t_rel = t_days - t_days.min(dim=-1, keepdim=True)[0]         # [B, S], ≥0
        time_angles = torch.einsum("bs,f->bsf", t_rel * self.time_scale, self.inv_freq)  # [B, S, d_k//2]

        angles = pos_angles.unsqueeze(0) + time_angles  # [B, S, d_k//2]
        angles = angles.unsqueeze(1)                    # [B, 1, S, d_k//2]
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    @torch.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    def _apply_rotation_compiled(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotation to each pair of dimensions in q and k.

        Treats the last dimension as pairs (even, odd) and rotates each pair:
        (e, o) → (e·cos − o·sin, e·sin + o·cos).

        Args:
            q, k: [B, H, S, d_k]
            cos, sin: [B, 1, S, d_k//2]  (broadcast over heads)
        Returns:
            Rotated q and k with the same shapes.
        """
        q_r = q.view(*q.shape[:-1], -1, 2)
        k_r = k.view(*k.shape[:-1], -1, 2)
        q_even, q_odd = q_r[..., 0], q_r[..., 1]
        k_even, k_odd = k_r[..., 0], k_r[..., 1]
        q_rot = torch.stack([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1).flatten(-2)
        k_rot = torch.stack([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1).flatten(-2)
        return q_rot, k_rot

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, timestamp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k:      [B, H, S, d_k]
            timestamp: [B, S]           unix timestamps in seconds, ordered oldest→newest.
        Returns:
            Rotated q and k with the same shapes.
        """
        S = q.size(-2)
        positions = torch.arange(S, device=q.device, dtype=torch.float)  # [S]
        cos, sin = self._build_angles(positions, timestamp)
        dtype = q.dtype
        q_rot, k_rot = self._apply_rotation_compiled(q.float(), k.float(), cos, sin)
        return q_rot.to(dtype), k_rot.to(dtype)


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Strictly causal multi-head self-attention with ToRoPE.

    Causality is enforced by dropping the query at position 0 before computing
    attention (so token i can only attend to tokens 0..i-1), then prepending a
    learned constant embedding as the output for position 0.  That constant is
    the model's representation of "no prior context", which it optimizes during
    training rather than hard-coding to zero.
    """

    def __init__(self, d_model: int, num_heads: int, rope: ToRoPE, scale: float = 0.025):
        """
        Args:
            d_model: Model dimension. Must be divisible by num_heads.
            num_heads: Number of attention heads.
            rope: Shared ToRoPE instance (typically shared across all layers).
            scale: Attention logit scale (replaces the usual 1/√d_k).
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = scale
        self.rope = rope
        # Learned output for position 0, which has no prior context to attend to.
        self.padding_embedding = nn.Parameter(torch.zeros(d_model))

    def _split_heads(self, x: torch.Tensor, d: int) -> torch.Tensor:
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, d).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, S, d_model]
            timestamp: [B, S]  — unix timestamps in seconds, oldest→newest.
        Returns:
            [B, S, d_model]  — causally attended output at every position.
        """
        B = query.size(0)
        Q = self._split_heads(self.W_q(query), self.d_k)
        K = self._split_heads(self.W_k(key), self.d_k)
        V = self._split_heads(self.W_v(value), self.d_v)

        Q, K = self.rope(Q, K, timestamp)

        # Drop Q[0]: token i attends only to keys at positions 0..i-1.
        # is_causal=True applies a lower-triangular mask over the [S-1, S] grid,
        # so Q[i] (sequence pos i+1) is masked against K[i+1..S-1].
        context = F.scaled_dot_product_attention(
            Q[:, :, 1:, :], K, V,
            dropout_p=0.05 if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )  # [B, H, S-1, d_v]

        # Prepend learned no-context embedding as the output for position 0.
        pad = self.padding_embedding.view(1, self.num_heads, 1, self.d_v).expand(B, -1, -1, -1)
        context = torch.cat([pad, context], dim=2)  # [B, H, S, d_v]

        return self.W_o(context.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_v))


class TransformerLayer(nn.Module):
    """
    One transformer block: causal self-attention followed by a feed-forward network,
    both with pre-LayerNorm and a small skip_scale on the residual branch.

    skip_scale < 1 keeps early-training residuals small, stabilizing deep stacks.
    """

    def __init__(self, d_model: int, d_ff: int, num_heads: int, rope: ToRoPE,
                 skip_scale: float = 0.1, attn_scale: float = 0.025):
        super().__init__()
        self.mha = MultiHeadAttentionWithRoPE(d_model, num_heads, rope, scale=attn_scale)
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(inplace=True),
            nn.Linear(d_ff, d_model),
        )
        nn.init.kaiming_uniform_(self.ffwd[0].weight, nonlinearity="relu")
        self.prenorm_1 = nn.LayerNorm(d_model)
        self.prenorm_2 = nn.LayerNorm(d_model)
        self.skip_scale = skip_scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, timestamp: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q, k, v:   [B, S, d_model]  — typically q=k=item reps, v=item reps + action embs.
            timestamp: [B, S]           — unix timestamps, passed through to ToRoPE.
        Returns:
            [B, S, d_model]  — updated item representations.
        """
        q = q + self.skip_scale * self.mha(self.prenorm_1(q), self.prenorm_1(k), self.prenorm_1(v), timestamp)
        return q + self.skip_scale * self.ffwd(self.prenorm_2(q))


class GenerativeRecommender(nn.Module):
    """
    Generative Recommender with Time-and-Order RoPE (ToRoPE).

    The model processes a user's item history and predicts engagement labels
    (e.g. click, like, dwell) for each position in a strictly causal manner:
    the prediction at position i uses only items 0..i-1.

    Architecture:
      1. Project item and action features independently into d_model space.
      2. Run num_layers transformer blocks. Each block attends over item
         representations (Q=K=items) but aggregates item+action values (V),
         so action signals inform the sequence representation without being
         directly attended to as queries.  ToRoPE uses both sequence order and
         item timestamps to rotate Q and K.
      3. Run a single action-attention pass: Q=K=final item reps, V=action
         embeddings. This produces a separate action-aware vector per position
         by using item similarity to weight which past actions are relevant.
      4. Concatenate item rep, action-aware rep, and per-item dynamic context,
         then project to output labels via an MLP head.
    """

    def __init__(
        self,
        d_seq: int,
        d_action: int,
        d_dyn_context: int,
        d_model: int = 512,
        num_heads: int = 4,
        num_layers: int = 12,
        d_labels: int = 2,
        dff_multiplier: float = 4.0,
        time_scale: float = 0.01,
        skip_scale: float = 0.1,
        attn_scale: float = 0.025,
    ):
        """
        Args:
            d_seq: Dimension of raw item (sequence) features.
            d_action: Dimension of raw user action features per item.
            d_dyn_context: Dimension of dynamic context features fed to the head.
            d_model: Internal model width.
            num_heads: Number of attention heads (d_model must be divisible by num_heads).
            num_layers: Number of transformer layers.
            d_labels: Number of output labels per position.
            dff_multiplier: Feed-forward hidden size = d_model × dff_multiplier.
            time_scale: Passed to ToRoPE; controls how much a one-day gap between
                        events shifts their relative rotation angle.
            skip_scale: Residual branch scale; small values stabilize training.
            attn_scale: Attention logit scale (replaces 1/√d_k).
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model

        self.seq_projection = nn.Sequential(
            nn.Linear(d_seq, d_model),
            nn.ReLU(),
        )
        self.seq_norm = nn.LayerNorm(d_model)

        self.action_embedding = nn.Sequential(
            nn.Linear(d_action, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, d_model),
            nn.ReLU(),
        )

        d_k = d_model // num_heads
        self.rope = ToRoPE(d_k=d_k, time_scale=time_scale)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                d_ff=int(d_model * dff_multiplier),
                num_heads=num_heads,
                rope=self.rope,
                skip_scale=skip_scale,
                attn_scale=attn_scale,
            )
            for _ in range(num_layers)
        ])

        self.action_attention = MultiHeadAttentionWithRoPE(
            d_model=d_model, num_heads=num_heads, rope=self.rope, scale=attn_scale
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + d_model + d_dyn_context, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, d_labels),
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        action_features: torch.Tensor,
        timestamp: torch.Tensor,
        dynamic_context_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_features:        [B, S, d_seq]        item content features per position,
                                                           ordered oldest→newest.
            action_features:          [B, S, d_action]     observed user actions per item
                                                           (e.g. click=1/0, dwell time).
            timestamp:                [B, S]               unix timestamps in seconds,
                                                           ordered oldest→newest (matches
                                                           sequence_features ordering).
            dynamic_context_features: [B, S, d_dyn_context] per-item context for the head
                                                           (e.g. request-time signals).
        Returns:
            {"logits": [B, S, d_labels]}  — raw (pre-sigmoid) label scores per position.
        """
        x = self.seq_norm(self.seq_projection(sequence_features))   # [B, S, d_model]
        action_emb = self.action_embedding(action_features)          # [B, S, d_model]

        for layer in self.transformer_layers:
            x = layer(q=x, k=x, v=x + action_emb, timestamp=timestamp)

        action_emb = self.action_attention(query=x, key=x, value=action_emb, timestamp=timestamp)

        logits = self.head(torch.cat([x, action_emb, dynamic_context_features], dim=-1))
        return {"logits": logits}
