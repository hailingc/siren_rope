# SIREN-RoPE: Learning to Rotate

**Temporal and Semantic Rotary Encoding for Sequential Modeling**

*Hailing Cheng, Daqi Sun, Xinyu Lu — LinkedIn Inc.*

> **NeurIPS 2026** · [Paper (PDF)](paper/siren_rope.tex)

---

## The Core Idea

Every Transformer dedicates enormous capacity to learning rich representations in **semantic embedding space** — yet the *rotation manifold* acted upon by Rotary Positional Embeddings (RoPE) has been treated as a fixed, hand-crafted structure populated only by discrete ordinal indices.

We argue this **rotation space** is a largely overlooked second dimension of expressivity in the attention mechanism.

### The Complex-Number Analogy

The analogy to complex numbers is precise:

| Domain | "Real" axis | "Imaginary" axis |
|--------|-------------|------------------|
| Complex numbers | Real line (algebra everyone knew) | Imaginary axis — unlocked entirely new structure |
| Attention | **Token embedding** — *what* a token means (semantic) | **Rotation manifold** — *how* it relates to every other token (dynamic) |

Just as introducing the imaginary axis opened algebraic structure once believed impossible, treating the rotation manifold as a **learnable, signal-conditioned space** opens an orthogonal degree of freedom in attention.

```
Standard attention:
  ┌──────────────┐
  │  Embedding   │  ← Semantic dimension (well-studied)
  │  Space       │
  └──────────────┘

SIREN-RoPE:
  ┌──────────────┐   ┌──────────────────────────┐
  │  Embedding   │ ⊕ │   Rotation Manifold       │
  │  Space       │   │   (temporal + semantic)   │
  └──────────────┘   └──────────────────────────┘
   "what"              "how / when / relative to what"
```

---

## Method

### Rotation Angle Formulation

SIREN-RoPE replaces the fixed ordinal angle `p·θ_j` with:

```
Θ_j(T_i, p_i) = f_φ(T_i)_j · ω^s_j   +   p_i · θ_j · λ
                └──── Temporal (SIREN) ────┘   └─ Ordinal ─┘
```

- `f_φ` — dual-branch SIREN network mapping timestamps → rotation angles
- `ω^s_j` — learnable per-dimension frequency scales (init: π)
- `θ_j = base^{−2j/d_k}` — standard RoPE inverse frequencies
- `λ` — learnable scalar gate balancing temporal vs. ordinal (init: 1.0)

Setting `λ→0` eliminates the ordinal component; `ω^s_j→0` recovers standard RoPE. Both remain non-trivial after training, confirming each signal carries independent information.

### Dual-Branch SIREN Architecture

```
Timestamp T  ──→  [cos(2πT/τ_d), sin(2πT/τ_d),      5-dim
                   cos(2πT/τ_w), sin(2πT/τ_w), T̃]  ──┐
                                                        │
                        ┌───────────────────────────────┤
                        │                               │
                   ┌────▼─────┐              ┌──────────▼──────┐
                   │  SIREN   │              │  DNN (ReLU)     │
                   │  branch  │              │  branch         │
                   │ sin(ω₀Wx)│              │  (aperiodic)    │
                   └────┬─────┘              └──────────┬──────┘
                        │  Periodic patterns             │  Monotone trends
                        │  (daily/weekly cycles)         │  (recency decay)
                        └──────────┬─────────────────────┘
                                   │ additive fusion
                                   ▼
                           f_φ(T) ∈ ℝ^{d_k/2}   rotation angles
```

The `(cos, sin)` encoding of each temporal cycle ensures continuity across period boundaries — midnight and end-of-week are no different from any other transition.

### Why Temporal Signals Belong in Rotation, Not Embedding

A key ablation finding: adding temporal features **directly to token embeddings** is *worse* than the ordinal RoPE baseline on every metric. The same signals routed through the rotation dimension yield consistent improvements. Temporal context is an inherently *relational* property — it describes how a token relates to others in time — not an intrinsic property of the token's content.

---

## Results

Evaluated on a production-scale news feed dataset (one year of logs, three engagement tasks):

| Model | Contribution NE↓ | Contribution AUC↑ | Like NE↓ | Like AUC↑ | LongDwell NE↓ | LongDwell AUC↑ |
|---|---|---|---|---|---|---|
| Ordinal RoPE (production) | 0.6206 | 0.9102 | 0.5985 | 0.9238 | 0.8362 | 0.7597 |
| TO-RoPE (scalar time) | 0.6218 | 0.9095 | 0.5999 | 0.9231 | 0.8349 | 0.7613 |
| **SIREN-RoPE (ours)** | **0.6182** | **0.9115** | **0.5963** | **0.9249** | **0.8334** | **0.7633** |

NE = Normalized Entropy (calibration); AUC = ranking quality.

**Overhead:** +0.3% parameters, ~2.4% wall-clock training time, ~1.8% inference time.

---

## Repository Structure

```
siren_rope/
├── paper/
│   └── siren_rope.tex          # Full NeurIPS paper source
├── siren_rope/
│   ├── __init__.py             # Package exports
│   └── models/
│       ├── ordinal_rope.py     # Baseline: standard RoPE (ordinal position only)
│       ├── torope.py           # Baseline: TO-RoPE (ordinal + scalar timestamp)
│       └── siren_rope.py       # Ours: SIREN-RoPE (dual-branch temporal rotation)
└── tests/
    └── test_models.py          # Shape, correctness, and causal invariance checks
```

---

## Quick Start

```python
import torch
from siren_rope import SIRENRoPERecommender

model = SIRENRoPERecommender(
    d_seq=128,          # item feature dimension
    d_action=3,         # action feature dimension (e.g., contribution/like/dwell flags)
    d_dyn_context=64,   # dynamic context dimension
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_labels=3,         # number of engagement tasks
    sequence_length=50,
    rope_d_angle=16,    # SIREN output angle dimension
)

B, S = 4, 50
out = model(
    sequence_features=torch.randn(B, S, 128),
    action_features=torch.zeros(B, S, 3),
    timestamp=torch.tensor([[1_700_000_000.0 - i * 3600 for i in range(S)]] * B),
    order_features=torch.arange(S).flip(0).float().unsqueeze(0).expand(B, -1) / S,
    dynamic_context_features=torch.randn(B, S, 64),
)
# out["logits"]: [B, S, 3]  — raw scores per task per position
```

### Run Tests

```bash
cd /path/to/siren_rope
python -m pytest tests/
# or directly:
python tests/test_models.py
```

### Compare All Three Variants

```python
from siren_rope import SIRENRoPERecommender, OrdinalRoPERecommender, ToRoPERecommender

# Ordinal RoPE — position indices only
baseline = OrdinalRoPERecommender(d_seq=128, d_action=3, d_dyn_context=64, d_labels=3)

# TO-RoPE — ordinal + scalar timestamp offset
torope = ToRoPERecommender(d_seq=128, d_action=3, d_dyn_context=64, d_labels=3, time_scale=0.01)

# SIREN-RoPE — full dual-branch temporal rotation (ours)
siren = SIRENRoPERecommender(d_seq=128, d_action=3, d_dyn_context=64, d_labels=3)
```

---

## Key Ablation Findings

| Finding | Result |
|---|---|
| Temporal signals in embedding space | Hurts vs. baseline (−NE on all tasks) |
| Temporal signals in rotation space | Consistent gains (SIREN-RoPE) |
| SIREN-only (no DNN branch) | Worst variant — oscillatory gradients impede training |
| DNN-only (no SIREN branch) | Strong on calibration, weaker on ranking AUC |
| Full dual-branch | Best overall: DNN stabilizes, SIREN adds spectral richness |
| Semantic rotation (static features) | No gain — static signals belong in embeddings |

---

## Citation

```bibtex
@inproceedings{cheng2026sirenrope,
  title     = {Learning to Rotate: Temporal and Semantic Rotary Encoding for Sequential Modeling},
  author    = {Cheng, Hailing and Sun, Daqi and Lu, Xinyu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

---

## License

[Apache 2.0](LICENSE)
