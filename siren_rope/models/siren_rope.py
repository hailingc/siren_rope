"""
Generative Recommender: SIREN-enhanced RoPE variant (siren_rope).

Uses Rotary Position Embedding (RoPE) whose rotation angle is the sum of a fixed
order-based frequency (like standard RoPE) and a learned temporal angle produced by
a SIREN network conditioned on wall-clock timestamps and reverse-chronological order.
Compare with ordinal_rope.py (order only) and torope.py (order + timestamp, no SIREN).
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class RoPE(nn.Module):
    """
    RoPE with a learned temporal angle fused from a SIREN network.

    The final rotation angle per dimension pair is:
        θ_i = order_freq_i × order_scale  +  angle_i × freq_i

    where:
    - order_freq_i is the standard RoPE frequency for dimension i (pre-computed
      from sequence position, newest item = position 0).
    - angle is a d_angle-dimensional vector output by a SIREN network, encoding
      wall-clock time and reverse-chronological order.
    - freq_i is a learnable per-dimension scale applied to angle (initialized to π).
    - order_scale is a learnable scalar that controls the overall strength of the
      position signal relative to the temporal signal.

    Uses real-valued cos/sin (not complex numbers) for torch.compile compatibility.
    """

    def __init__(self, d_k: int, d_angle: int, base: float = 100000, seq_length: int = 1024):
        """
        Args:
            d_k: Per-head key/query dimension. Must be even, and d_k//2 must be
                 divisible by d_angle.
            d_angle: Dimensionality of the SIREN angle vector. Each angle dimension
                     is tiled (d_k//2 // d_angle) times to cover all rotation pairs.
            base: Base for the order-based frequency ladder (same as standard RoPE).
            seq_length: Maximum sequence length; pre-computes order frequencies up to
                        this size so forward passes avoid recomputation.
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")
        if (d_k // 2) % d_angle != 0:
            raise ValueError("d_k // 2 must be divisible by d_angle")

        self.d_k = d_k
        self.half_d_k = d_k // 2
        self.d_angle = d_angle
        self._angle_repeat = self.half_d_k // d_angle

        # Learnable scale on the order-based contribution (initialized to 1).
        self.order_scale = nn.Parameter(torch.tensor(1.0))
        # Learnable per-dimension scale on the SIREN angle (initialized to π).
        self.freq = nn.Parameter(torch.ones(d_angle) * math.pi)

        order_inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("order_inv_freq", order_inv_freq, persistent=False)
        # Pre-computed order frequencies: positions [seq_length-1, ..., 0] (newest = 0).
        t = seq_length - 1 - torch.arange(seq_length, dtype=torch.float)
        self.register_buffer("order_freqs", torch.outer(t, order_inv_freq).contiguous(), persistent=False)

    def _build_freqs_cos_sin(self, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the SIREN angle with order-based frequencies to produce cos/sin tensors.

        Steps:
          1. Scale angle by the learnable freq vector.
          2. Tile angle along the last dim to match d_k//2 (if d_angle < d_k//2).
          3. Add the pre-computed order frequencies scaled by order_scale.
          4. Return cos and sin of the combined angles.

        Args:
            angle: [B, S, d_angle]  — SIREN output for each token.
        Returns:
            cos, sin: [B, 1, S, d_k//2]  — broadcast-ready over attention heads.
        """
        angle = angle.view(angle.size(0), 1, angle.size(1), angle.size(2))
        rotation = angle.float() * self.freq.float()
        if self._angle_repeat > 1:
            rotation = rotation.repeat_interleave(self._angle_repeat, dim=-1)

        seq_len = angle.size(-2)
        if seq_len <= self.order_freqs.size(0):
            order_freqs = self.order_freqs[-seq_len:]
        else:
            positions = seq_len - 1 - torch.arange(seq_len, device=angle.device, dtype=torch.float)
            order_freqs = torch.outer(positions, self.order_inv_freq.to(angle.device))
        rotation = rotation + order_freqs * self.order_scale
        return torch.cos(rotation), torch.sin(rotation)

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
        self, q: torch.Tensor, k: torch.Tensor, angle: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k:  [B, H, S, d_k]
            angle: [B, S, d_angle]  — SIREN output encoding time and order.
        Returns:
            Rotated q and k with the same shapes.
        """
        cos, sin = self._build_freqs_cos_sin(angle)
        dtype = q.dtype
        q_rot, k_rot = self._apply_rotation_compiled(q.float(), k.float(), cos, sin)
        return q_rot.to(dtype), k_rot.to(dtype)


class SineLayer(nn.Module):
    """
    One layer of a SIREN network: sin(ω₀ · (Wx + b)).

    Initialized so the distribution of pre-activations is uniform on [-π, π],
    which keeps sine outputs well-distributed across the full cycle.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network that encodes temporal features into RoPE angles.

    The main path is a sine-activated MLP (SIREN), which excels at representing
    smooth periodic signals like time-of-day and time-of-week.

    A parallel ReLU residual path is added for training stability: pure SIREN
    networks can be hard to optimize from random initialization because sine
    activations produce oscillating gradients. The ReLU path provides a smoother
    gradient signal early in training without hurting the expressiveness of the
    sine path once training settles.
    """

    def __init__(self, in_features: int = 1, hidden_features: int = 8, hidden_layers: int = 2,
                 out_features: int = 1, outermost_linear: bool = True,
                 first_omega_0: float = 30, hidden_omega_0: float = 30):
        """
        Args:
            in_features: Input dimension (e.g. 6 for [order, year, sin_day, cos_day, sin_week, cos_week]).
            hidden_features: Width of each hidden layer in both paths.
            hidden_layers: Number of hidden layers (same for both paths).
            out_features: Output dimension = d_angle passed to RoPE.
            outermost_linear: If True, the final layer of the sine path is a plain
                              linear (no sine activation), which gives unbounded output
                              range needed for rotation angles.
            first_omega_0: Frequency scale for the first sine layer (controls input
                           sensitivity; higher → captures higher-frequency patterns).
            hidden_omega_0: Frequency scale for subsequent sine layers.
        """
        super().__init__()

        # Sine path
        sine_layers: list = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            sine_layers.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        if outermost_linear:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6 / hidden_features) / hidden_omega_0
                final.weight.uniform_(-bound, bound)
            sine_layers.append(final)
        else:
            sine_layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*sine_layers)

        # ReLU residual path for training stability
        relu_layers: list = [nn.Linear(in_features, hidden_features)]
        for _ in range(hidden_layers):
            relu_layers += [nn.ReLU(), nn.Linear(hidden_features, hidden_features)]
        relu_layers += [nn.ReLU(), nn.Linear(hidden_features, out_features)]
        self.residual = nn.Sequential(*relu_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Strictly causal multi-head self-attention with SIREN-enhanced RoPE.

    Causality is enforced by dropping the query at position 0 before computing
    attention (so token i can only attend to tokens 0..i-1), then prepending a
    learned constant embedding as the output for position 0.  That constant is
    the model's representation of "no prior context", which it optimizes during
    training rather than hard-coding to zero.

    The SIREN angle is passed in from outside so it can be pre-computed once
    per forward pass and reused across all layers.
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
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, S, d_model]
            angle: [B, S, d_angle]  — SIREN output for each token.
        Returns:
            [B, S, d_model]  — causally attended output at every position.
        """
        B = query.size(0)
        Q = self._split_heads(self.W_q(query), self.d_k)  # [B, H, S, d_k]
        K = self._split_heads(self.W_k(key), self.d_k)
        V = self._split_heads(self.W_v(value), self.d_v)

        Q, K = self.rope(Q, K, angle=angle)

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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: [B, S, d_model]  — typically q=k=item reps, v=item reps + action embs.
            angle: [B, S, d_angle]    — SIREN output, passed through to RoPE.
        Returns:
            [B, S, d_model]  — updated item representations.
        """
        q = q + self.skip_scale * self.mha(self.prenorm_1(q), self.prenorm_1(k), self.prenorm_1(v), angle=angle)
        return q + self.skip_scale * self.ffwd(self.prenorm_2(q))


class GenerativeRecommender(nn.Module):
    """
    Generative Recommender with SIREN-enhanced RoPE for temporal sequence modeling.

    The model processes a user's item history and predicts engagement labels
    (e.g. click, like, dwell) for each position in a strictly causal manner:
    the prediction at position i uses only items 0..i-1.

    Architecture:
      1. Project item and action features independently into d_model space.
      2. Encode timestamps + order into a d_angle-dimensional SIREN angle for
         each token. Two separate SIREN networks produce angles for the main
         transformer stack (angle_nn) and the action attention step (action_angle_nn).
      3. Run num_layers transformer blocks. Each block attends over item
         representations (Q=K=items) but aggregates item+action values (V),
         so action signals inform the sequence representation without being
         directly attended to as queries.
      4. Run a single action-attention pass: Q=K=final item reps, V=action
         embeddings. This produces a separate action-aware vector per position
         by using item similarity to weight which past actions are relevant.
      5. Concatenate item rep, action-aware rep, and per-item dynamic context,
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
        rope_d_angle: int = 64,
        angle_nn_width: int = 16,
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
            rope_d_angle: SIREN output dimension (= number of angle components per token).
                          Must evenly divide d_model // num_heads // 2.
            angle_nn_width: Hidden width of each SIREN network.
            skip_scale: Residual branch scale; small values stabilize training.
            attn_scale: Attention logit scale (replaces 1/√d_k).
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.sequence_length = sequence_length

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
        self.rope = RoPE(d_k=d_k, d_angle=rope_d_angle, seq_length=sequence_length)

        # SIREN input: [order, time_in_year, sin_day, cos_day, sin_week, cos_week]
        siren_kwargs = dict(
            in_features=6,
            hidden_features=angle_nn_width,
            hidden_layers=2,
            out_features=rope_d_angle,
            outermost_linear=True,
            first_omega_0=10,
            hidden_omega_0=10,
        )
        self.angle_nn = SIREN(**siren_kwargs)
        self.action_angle_nn = SIREN(**siren_kwargs)

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

        self._seconds_per_day = 86400
        self._seconds_per_week = 604800
        self._seconds_per_year = 31536000

    def _compute_angle_features(
        self, timestamp: torch.Tensor, order_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Build a 6-dimensional SIREN input from timestamps and position order.

        The six features are:
          [0] reverse-chronological order (0.0=newest, ~1.0=oldest)
          [1] fractional year (centered near 0 for recent dates)
          [2] sin of time-of-day (captures daily periodicity)
          [3] cos of time-of-day
          [4] sin of day-of-week (captures weekly periodicity)
          [5] cos of day-of-week

        Args:
            timestamp: [B, S]  — unix timestamps in seconds.
            order_features: [B, S]  — normalized reverse-chronological position in [0, 1].
        Returns:
            [B, S, 6]
        """
        t = timestamp.unsqueeze(-1).float()
        time_in_day = t / self._seconds_per_day - 20474
        time_in_week = t / self._seconds_per_week - 2924
        time_in_year = t / self._seconds_per_year - 55
        order = order_features.unsqueeze(-1).float()
        return torch.cat(
            [
                order,
                time_in_year,
                torch.sin(time_in_day * 2 * math.pi),
                torch.cos(time_in_day * 2 * math.pi),
                torch.sin(time_in_week * 2 * math.pi),
                torch.cos(time_in_week * 2 * math.pi),
            ],
            dim=-1,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        action_features: torch.Tensor,
        timestamp: torch.Tensor,
        order_features: torch.Tensor,
        dynamic_context_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_features:        [B, S, d_seq]        item content features per position.
            action_features:          [B, S, d_action]     observed user actions per item
                                                           (e.g. click=1/0, dwell time).
            timestamp:                [B, S]               unix timestamps in seconds.
            order_features:           [B, S]               normalized reverse position in [0,1]
                                                           (0.0 = newest, 1.0 = oldest).
            dynamic_context_features: [B, S, d_dyn_context] per-item context for the head
                                                           (e.g. request-time signals).
        Returns:
            {"logits": [B, S, d_labels]}  — raw (pre-sigmoid) label scores per position.
        """
        x = self.seq_norm(self.seq_projection(sequence_features))   # [B, S, d_model]
        action_emb = self.action_embedding(action_features)          # [B, S, d_model]

        angle_feat = self._compute_angle_features(timestamp, order_features)  # [B, S, 6]
        angle = self.angle_nn(angle_feat)             # [B, S, rope_d_angle]
        action_angle = self.action_angle_nn(angle_feat)

        for layer in self.transformer_layers:
            x = layer(q=x, k=x, v=x + action_emb, angle=angle)

        action_emb = self.action_attention(query=x, key=x, value=action_emb, angle=action_angle)

        logits = self.head(torch.cat([x, action_emb, dynamic_context_features], dim=-1))
        return {"logits": logits}
