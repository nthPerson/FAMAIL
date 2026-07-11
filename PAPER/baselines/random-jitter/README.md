# Random-jitter arm — direction placebo (PENDING GPU)

**Status: ⏳ built + reviewed + merged (2026-07-09); NOT yet run — the GPU is held by the α-Pareto sweep.**
Results, tables, and figures land here when the run-book executes.

**Method.** Seeded full-budget random perturbation: `delta ~ Uniform{−1,+1}^(S,2) × ε` (ε = 2 grid cells,
the suite's shared L∞ budget) applied to every seeking state's continuous (x, y), no gradient and no
discriminator calls during the "attack" (scored after), then discretize and score with the standard
harness. Same trajectory set as the FAMAIL headline.

**Role.** The direction placebo for the gradient arms: it separates "what bounded perturbation of this
magnitude does by itself" from "what the discriminator's gradient direction adds," exactly as the
random-oversampling placebo separates corpus inflation from demographic targeting in
`../demographic-oversampling/`.

**Framing (Meeting-41, canonical):** a FIDELITY/editing-quality baseline, not a fairness method —
fairness is expected NOT to improve.

**Run-book:** `famail_temporal/baselines/STATUS.md` (Mission-3 section). Code:
`famail_temporal/baselines/{stifgsm_baseline,run_stifgsm_baseline}.py` (mode `random`).
