# iFGSM arm — iterative signed-gradient attack (PENDING GPU)

**Status: ⏳ built + reviewed + merged (2026-07-09); NOT yet run — the GPU is held by the α-Pareto sweep.**
Results, tables, and figures land here when the run-book executes.

**Method.** Batched iterative signed-gradient attack on the frozen HuMID driver-identity discriminator
over continuous float-grid seeking trajectories: perturb every seeking state's (x, y) within a
per-coordinate L∞ budget of **ε = 2 grid cells** (the suite's shared budget), descending the
discriminator's same-driver probability, best-iterate kept, patience-stopped; then discretize back onto
the grid and score with the standard harness (fairness rescore + external metrics + Fidelity-A/B).
Attacks the SAME trajectory set the FAMAIL trim+lift headline edited, so the comparison isolates edit
*direction* at equal budget.

**Paper-facing naming caveat (load-bearing).** This arm is **"iFGSM with random restart" (PGD-style), NOT
vanilla ST-iFGSM**: the frozen Siamese identity head (|emb₁ − emb₂|) has zero subgradient at an identical
(original, original) pair, so a textbook vanilla attack starting at δ=0 is a stationary no-op. The
gradient arms therefore start from a random point inside the ε-ball by necessity; the runner's
`--no-random-start` flag is retained precisely to demonstrate the vanilla no-op empirically (a legitimate
ablation row). The paper must label the arm accordingly.

**Framing (Meeting-41, canonical):** a FIDELITY/editing-quality baseline, not a fairness method —
fairness is expected NOT to improve.

**Run-book:** `famail_temporal/baselines/STATUS.md` (Mission-3 section). Code:
`famail_temporal/baselines/{stifgsm_baseline,run_stifgsm_baseline}.py` (mode `ifgsm`). Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-09-mission3-data-aug-baselines*`.
