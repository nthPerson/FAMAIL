# Weight-decision memo — α-Pareto sweep complete (Q0 hard checkpoint)

**Date:** 2026-07-11 · **Status:** ⏸ AWAITING ROBERT'S KEEP-VS-RE-ANCHOR DECISION — no campaign stage
launches before it. · **Artifacts:** [`alpha_sweep_summary.md`](alpha_sweep_summary.md) (table),
[`alpha_pareto.png`](alpha_pareto.png) (scatter), [`alpha_sweep_summary.json`](alpha_sweep_summary.json).
Sweep provenance: 5 points × (k=10,000 SZ PRIMARY, trim+lift editor, infeasible-trim filter) + the
shipped headline as anchor; ledger row Q0.

## The frontier (6 points, single run each)

| α (sp, ca, fi) | ΔF_causal | ΔF_spatial | Pareto |
|---|---:|---:|:---:|
| (0.0, 0.9, 0.1) | +0.0221 | +0.0057 | — |
| (0.1, 0.8, 0.1) | +0.0226 | +0.0061 | — |
| **(0.2, 0.7, 0.1) ★ shipped** | **+0.0222** | **+0.0064** | **—  (dominated)** |
| (0.35, 0.55, 0.1) | +0.0217 | +0.0076 | — |
| **(0.55, 0.35, 0.1)** | **+0.0227** | **+0.0094** | ✓ |
| (0.8, 0.1, 0.1) | +0.0185 | +0.0117 | ✓ |

## Findings (what the completed sweep actually says)

1. **The primary axis is FLAT across α_spatial ∈ [0, 0.55].** ΔF_causal spans +0.0217..+0.0227 — a
   0.0010 band. Sweep points are single runs with documented tie-nondeterministic edit ordering
   (LIFT_ALGORITHM_REFERENCE §9), so differences at this scale are plausibly run noise, not signal.
   Only at α_spatial = 0.8 does the causal gain finally drop (+0.0185).
2. **The secondary axis is MONOTONE and material.** ΔF_spatial rises steadily +0.0057 → +0.0117 as
   α_spatial grows — that is a real dose-response, not noise.
3. **Consequence: the shipped (0.2, 0.7, 0.1) is weakly dominated by (0.55, 0.35, 0.1)** (+0.0005
   causal — noise-scale; +0.0030 spatial — ~47% relative). The documented selection criterion
   (max ΔF_causal s.t. ΔF_spatial ≥ 0) now selects (0.55, 0.35, 0.1). Note the criterion's constraint
   no longer binds under trim+lift (every point has ΔF_spatial > 0), so it degenerates to
   "max ΔF_causal" — i.e., it is now selecting on what is plausibly noise.

## The decision

### Option A — KEEP (0.2, 0.7, 0.1) as the experimental basis *(my recommendation)*
- **Reading:** the objective's primary gain is weight-insensitive over a wide range — arguably the
  best possible "why these weights" story; the shipped point sits within noise of the frontier's
  causal maximum. Heavier spatial weightings buy secondary-axis gains without costing the primary
  axis until the extreme.
- **Paper handling (required for honesty):** report the full table including the domination;
  state plainly that (0.55, 0.35, 0.1) offers a larger secondary-axis gain at statistically
  indistinguishable primary-axis performance — a deployment-preference knob, not a different method
  — and that we retain the original configuration for continuity with the full experimental suite.
  MOTIVATION.md's "selected under an explicit criterion" sentence must be reworded (the criterion was
  applied at design time over the demand-only editor's behavior; under trim+lift it no longer
  discriminates).
- **Cost:** zero GPU. Campaign launches immediately as planned.
- **Optional hedge:** queue a post-campaign replication of the s55 point (one ~8h idle-GPU run) to
  test whether its +0.0227 is stable under a different tie-ordering, before camera-ready.

### Option B — RE-ANCHOR to (0.55, 0.35, 0.1)
- **Reading:** run the paper on the Pareto point; a sharp reviewer can never ask "why not the
  dominating configuration?"
- **True cost (larger than the 9h headline-rerun estimate):** SZ + SF headline edits (~8h + ~40min)
  PLUS re-running everything already computed on the 0.2-corpora that the campaign was NOT going to
  redo — the SZ weighted-BC sweep (~10h), rollout-allocation eval, G5 fidelity battery, channel
  decompositions, external metrics — call it **+1.5–2 GPU-days before the planned 3–5-day campaign
  even starts**, ~5 days before the abstract deadline. Every drafted number in 5.2/5.3/5.5/5.7, the
  abstract, and methodology §3.2 re-slots.
- **Risks:** the causal margin motivating the move is noise-scale and unreplicated; the α transfer
  to SF is unverified (the sweep is SZ-only), and SF already carries the mean(Y|D) framing tension;
  the trim-vs-trim+lift ablation would compare against a differently-weighted trim baseline unless
  the trim-only reference is also re-derived.

### Option C — KEEP + REPLICATE (A plus the hedge made explicit)
Keep for all current reporting; run the s55 replication as a scheduled low-priority campaign
addendum; revisit only if the replication confirms a stable, materially-better point — a
camera-ready decision, not an abstract-week one.

## Recommendation

**A (or equivalently C).** The dominating point's causal edge is inside the noise band of single
runs; the real, monotone finding is the F_spatial dose-response, which the paper can report as a
tunable trade-off without re-anchoring its entire experimental basis days before the abstract. The
honest-disclosure cost of A is one plainly-worded paragraph; the cost of B is measured in GPU-days
and re-verification risk.

*This memo is the Q0 gate artifact. The campaign driver stays un-launched until Robert answers.*
