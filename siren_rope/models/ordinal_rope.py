"""
Generative Recommender: position-only RoPE variant (ordinal_rope).

Uses standard Rotary Position Embedding (RoPE) based on sequence order alone.
Compare with siren_rope.py (order + learned temporal angle via SIREN) and
torope.py (order + wall-clock timestamp via early fusion).
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) based on sequence order.

    Each token is assigned a position index counting down from the most-recent
    token (position 0) to the oldest (position S-1). This reverse ordering means
    the model always sees the newest item at a fixed position regardless of
    how long the sequence is, which helps generalize to variable-length histories.

    When is_interleaving=True, adjacent pairs share the same position index,
    e.g. [item1, action1, item2, action2, ...] → positions [C-1, C-1, C-2, C-2, ...].

    Rotation uses real-valued cos/sin (not complex numbers) for torch.compile
    compatibility: (x_even, x_odd) → (x_even·cos − x_odd·sin, x_even·sin + x_odd·cos).
    """

    def __init__(
        self,
        d_k: int,
        base: float = 100000.0,
        seq_length: int = 1024,
        is_interleaving: bool = False,
    ):
        """
        Args:
            d_k: Per-head key/query dimension. Must be even.
            base: Controls the range of rotation frequencies. Higher base → slower
                  rotation in high-frequency dimensions, useful for long sequences.
            seq_length: Maximum sequence length; pre-computes cos/sin up to this size.
            is_interleaving: If True, consecutive pairs share a position index (for
                             interleaved item/action sequences).
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires d_k to be even, but got d_k={d_k}.")

        self.d_k = d_k
        self.is_interleaving = is_interleaving

        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Position indices: [seq_length-1, seq_length-2, ..., 0] (newest = 0)
        t = seq_length - 1 - torch.arange(seq_length, dtype=torch.float)
        if is_interleaving:
            t = torch.repeat_interleave(t, 2)  # each index appears twice: [2*seq_length]

        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs).unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs).unsqueeze(0).unsqueeze(0), persistent=False)

    @staticmethod
    @torch.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    def _apply_rotation_compiled(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotation to each pair of dimensions in q and k.

        Treats the last dimension as pairs (even, odd) and rotates each pair by the
        corresponding angle: (e, o) → (e·cos − o·sin, e·sin + o·cos).

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

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate q and k by their positional angles.

        Slices the last seq_len entries from the pre-computed buffers, so the model
        always sees the most-recent items at the lowest position indices regardless
        of the actual sequence length passed in (up to seq_length).

        Args:
            q, k: [B, H, S, d_k]
        Returns:
            Rotated q and k with the same shapes.
        """
        seq_len = q.size(-2)
        cos = self.freqs_cos[..., -seq_len:, :]
        sin = self.freqs_sin[..., -seq_len:, :]
        dtype = q.dtype
        q_rot, k_rot = self._apply_rotation_compiled(q.float(), k.float(), cos, sin)
        return q_rot.to(dtype), k_rot.to(dtype)


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Strictly causal multi-head self-attention with RoPE.

    Causality is enforced by dropping the query at position 0 before computing
    attention (so token i can only attend to tokens 0..i-1), then prepending a
    learned constant embedding as the output for position 0.  That constant is
    the model's representation of "no prior context", which it optimizes during
    training rather than hard-coding to zero.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE, scale: float = 0.025):
        """
        Args:
            d_model: Model dimension. Must be divisible by num_heads.
            num_heads: Number of attention heads.
            rope: Shared RoPE instance (typically shared across all layers).
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
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, S, d_model]
        Returns:
            [B, S, d_model]  — causally attended output at every position.
        """
        B = query.size(0)
        Q = self._split_heads(self.W_q(query), self.d_k)
        K = self._split_heads(self.W_k(key), self.d_k)
        V = self._split_heads(self.W_v(value), self.d_v)

        Q, K = self.rope(Q, K)

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

    def __init__(self, d_model: int, d_ff: int, num_heads: int, rope: RoPE,
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q, k, v: [B, S, d_model]  — typically q=k=item reps, v=item reps + action embs.
        Returns:
            [B, S, d_model]  — updated item representations.
        """
        q = q + self.skip_scale * self.mha(self.prenorm_1(q), self.prenorm_1(k), self.prenorm_1(v))
        return q + self.skip_scale * self.ffwd(self.prenorm_2(q))


class GenerativeRecommender(nn.Module):
    """
    Generative Recommender with position-based RoPE.

    The model processes a user's item history and predicts engagement labels
    (e.g. click, like, dwell) for each position in a strictly causal manner:
    the prediction at position i uses only items 0..i-1.

    Architecture:
      1. Project item and action features independently into d_model space.
      2. Run num_layers transformer blocks. Each block attends over item
         representations (Q=K=items) but aggregates item+action values (V),
         so action signals inform the sequence representation without being
         directly attended to as queries.
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
        sequence_length: int = 1024,
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
            sequence_length: Maximum supported sequence length (pre-computes RoPE buffers).
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
        self.rope = RoPE(d_k=d_k, seq_length=sequence_length)

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
        dynamic_context_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_features:        [B, S, d_seq]        item content features per position.
            action_features:          [B, S, d_action]     observed user actions per item
                                                           (e.g. click=1/0, dwell time).
            dynamic_context_features: [B, S, d_dyn_context] per-item context for the head
                                                           (e.g. request-time signals).
        Returns:
            {"logits": [B, S, d_labels]}  — raw (pre-sigmoid) label scores per position.
        """
        x = self.seq_norm(self.seq_projection(sequence_features))   # [B, S, d_model]
        action_emb = self.action_embedding(action_features)          # [B, S, d_model]

        for layer in self.transformer_layers:
            x = layer(q=x, k=x, v=x + action_emb)

        action_emb = self.action_attention(query=x, key=x, value=action_emb)

        logits = self.head(torch.cat([x, action_emb, dynamic_context_features], dim=-1))
        return {"logits": logits}
