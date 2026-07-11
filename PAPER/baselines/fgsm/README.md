# FGSM arm — single-step signed-gradient attack (PENDING GPU)

**Status: ⏳ built + reviewed + merged (2026-07-09); NOT yet run — the GPU is held by the α-Pareto sweep.**
Results, tables, and figures land here when the run-book executes.

**Method.** The iFGSM arm's engine with `max_iterations = 1` and step = ε: one full-budget signed-gradient
step from a PGD-style random start inside the ε = 2-cell L∞ ball (see `../ifgsm/README.md` for why the
random restart is necessary — the identity head is stationary at δ = 0), then discretize and score with
the standard harness. Same trajectory set as the FAMAIL headline.

**Provenance note (methodology rigor, must survive into the paper's repro record).** The original attack
loop scored-then-stepped, so FGSM's single step was discarded and the arm silently returned its
initialization — a real bug caught ONLY by the whole-branch final review (per-task reviews missed it
because the test stub had zero gradient at the origin). Fixed in commit `6da3d27` (post-step scoring pass
+ an asymmetric-stub gradient-path test). **Any FGSM numbers must come from the corrected engine
(`6da3d27`+).**

**Framing (Meeting-41, canonical):** a FIDELITY/editing-quality baseline, not a fairness method —
fairness is expected NOT to improve.

**Run-book:** `famail_temporal/baselines/STATUS.md` (Mission-3 section). Code:
`famail_temporal/baselines/{stifgsm_baseline,run_stifgsm_baseline}.py` (mode `fgsm`).
