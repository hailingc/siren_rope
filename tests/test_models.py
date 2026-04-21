"""Simple tests for GenerativeRecommender (SIREN, RoPE, and ToRoPE variants)."""

import torch
from siren_rope.models.siren_rope import GenerativeRecommender as GenerativeRecommenderSiren
from siren_rope.models.ordinal_rope import GenerativeRecommender as GenerativeRecommenderRoPE
from siren_rope.models.torope import GenerativeRecommender as GenerativeRecommenderToRoPE

B, S = 2, 16
D_SEQ = 128       # sequence (item) feature dim
D_ACTION = 3      # action features: contribution, like, longdwell
D_DYN = 64        # dynamic context feature dim
D_LABELS = 3      # contribution, like, longdwell

# ── Shared inputs ─────────────────────────────────────────────────────────────

sequence_features = torch.randn(B, S, D_SEQ)
action_features = torch.zeros(B, S, D_ACTION)
action_features[:, :S // 2, 0] = 1.0              # first half: some contributions
dynamic_context_features = torch.randn(B, S, D_DYN)

# ── SIREN model ───────────────────────────────────────────────────────────────

model_siren = GenerativeRecommenderSiren(
    d_seq=D_SEQ,
    d_action=D_ACTION,
    d_dyn_context=D_DYN,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_labels=D_LABELS,
    sequence_length=S,
    rope_d_angle=16,
)
model_siren.eval()

now = 1_700_000_000.0
timestamp = torch.tensor([[now - i * 3600 for i in range(S)]] * B)
order_features = torch.arange(S).flip(0).float().unsqueeze(0).expand(B, -1) / S

with torch.no_grad():
    out_siren = model_siren(
        sequence_features=sequence_features,
        action_features=action_features,
        timestamp=timestamp,
        order_features=order_features,
        dynamic_context_features=dynamic_context_features,
    )

logits_siren = out_siren["logits"]
labels = ["contribution", "like", "longdwell"]

print(f"[siren] logits shape: {logits_siren.shape}")
assert logits_siren.shape == (B, S, D_LABELS), f"unexpected shape {logits_siren.shape}"
probs = torch.sigmoid(logits_siren)
for i, label in enumerate(labels):
    print(f"  {label:15s} mean prob: {probs[..., i].mean():.4f}")
print("[siren] Test passed.")

# ── RoPE model ────────────────────────────────────────────────────────────────

model_rope = GenerativeRecommenderRoPE(
    d_seq=D_SEQ,
    d_action=D_ACTION,
    d_dyn_context=D_DYN,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_labels=D_LABELS,
    sequence_length=S,
)
model_rope.eval()

with torch.no_grad():
    out_rope = model_rope(
        sequence_features=sequence_features,
        action_features=action_features,
        dynamic_context_features=dynamic_context_features,
    )

logits_rope = out_rope["logits"]

print(f"\n[rope] logits shape: {logits_rope.shape}")
assert logits_rope.shape == (B, S, D_LABELS), f"unexpected shape {logits_rope.shape}"
probs = torch.sigmoid(logits_rope)
for i, label in enumerate(labels):
    print(f"  {label:15s} mean prob: {probs[..., i].mean():.4f}")
print("[rope] Test passed.")

# ── RoPE: strict causality check ──────────────────────────────────────────────
# Perturbing input at position p should not affect outputs at positions < p.

model_rope.train()  # need grads, but we use no_grad trick via clone comparison
model_rope.eval()

seq2 = sequence_features.clone()
seq2[:, S // 2, :] += 1.0   # perturb position S//2

with torch.no_grad():
    logits_orig = model_rope(sequence_features, action_features, dynamic_context_features)["logits"]
    logits_pert = model_rope(seq2, action_features, dynamic_context_features)["logits"]

# Positions strictly before S//2 must be unchanged (causal invariance)
unchanged = torch.allclose(logits_orig[:, : S // 2, :], logits_pert[:, : S // 2, :], atol=1e-5)
assert unchanged, "Causal invariance violated: earlier positions changed after later perturbation"

# Position S//2 and after should differ
changed = not torch.allclose(logits_orig[:, S // 2 :, :], logits_pert[:, S // 2 :, :], atol=1e-5)
assert changed, "Expected outputs at and after perturbed position to differ"

print("[rope] Causal invariance check passed.")

# ── ToRoPE model ──────────────────────────────────────────────────────────────

# Timestamps ordered oldest→newest (torope expects this direction)
now = 1_700_000_000.0
timestamp_fwd = torch.tensor([[now - (S - 1 - i) * 3600 for i in range(S)]] * B)

model_torope = GenerativeRecommenderToRoPE(
    d_seq=D_SEQ,
    d_action=D_ACTION,
    d_dyn_context=D_DYN,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_labels=D_LABELS,
    time_scale=0.01,
)
model_torope.eval()

with torch.no_grad():
    out_torope = model_torope(
        sequence_features=sequence_features,
        action_features=action_features,
        timestamp=timestamp_fwd,
        dynamic_context_features=dynamic_context_features,
    )

logits_torope = out_torope["logits"]

print(f"\n[torope] logits shape: {logits_torope.shape}")
assert logits_torope.shape == (B, S, D_LABELS), f"unexpected shape {logits_torope.shape}"
probs = torch.sigmoid(logits_torope)
for i, label in enumerate(labels):
    print(f"  {label:15s} mean prob: {probs[..., i].mean():.4f}")
print("[torope] Test passed.")

# ── ToRoPE: strict causality check ────────────────────────────────────────────

seq3 = sequence_features.clone()
seq3[:, S // 2, :] += 1.0

with torch.no_grad():
    logits_orig_t = model_torope(sequence_features, action_features, timestamp_fwd, dynamic_context_features)["logits"]
    logits_pert_t = model_torope(seq3, action_features, timestamp_fwd, dynamic_context_features)["logits"]

unchanged_t = torch.allclose(logits_orig_t[:, : S // 2, :], logits_pert_t[:, : S // 2, :], atol=1e-5)
assert unchanged_t, "ToRoPE causal invariance violated: earlier positions changed after later perturbation"

changed_t = not torch.allclose(logits_orig_t[:, S // 2 :, :], logits_pert_t[:, S // 2 :, :], atol=1e-5)
assert changed_t, "ToRoPE: expected outputs at and after perturbed position to differ"

print("[torope] Causal invariance check passed.")
