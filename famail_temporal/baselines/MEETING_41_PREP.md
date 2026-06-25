# Meeting 41 Prep — Weighted-BC Recovers Level-2 Transfer: Results + Paper-Framing Options

**Date:** 2026-06-20
**Scope:** Reports the weighted-BC significance sweep (the experiment recommended in `MEETING_40_PREP.md`)
and lays out four pressure-tested paper-framing options to open the discussion with Dr. Zhang. No editor
change; the lever lives entirely in the BC trainer.
**Artifacts:** `famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30/sweep.json` (headline, 6 seeds);
`.../full_5seed_w10_w30/` (the preceding 5-seed run). Code on branch `level-2-usability` (uncommitted):
`gan/train_mle.py` (`sample_weights`), `gan/tests/test_train_mle_weights.py`, `run_weighted_bc_smoke.py`.

---

## TL;DR

1. **Level-2 is recovered.** Vanilla BC averages the edited fairness away (the locked L2 negative). **Upweighting
   the edited demonstrations' loss during BC turns the transfer positive, significantly and unanimously:**
   paired edited−raw F_causal goes **−0.0019 (w=1) → +0.0186 (w=10) → +0.0242 (w=20) → +0.0274 (w=30)**, all
   arms 6/6 seeds same-sign, **Wilcoxon p = 0.031**.
2. **No identity-fidelity cost; a small tunable distributional cost.** Fidelity-A (HuMID identity, gate PASSED)
   is flat (~0.8406–0.8410). Fidelity-B rises gently (0.0121 → 0.0184), ~20× below the GAN-collapse (0.32)
   that disqualified GAN-gen. **w is a fairness↔realism knob**: w=10 is most efficient (~7.9× fairness gain per
   unit Fid-B), w=30 is max fairness.
3. **Mechanism, not magic.** The w=1 arm reproduces the locked L2 negative within noise (−0.0019 vs −0.0022),
   so the *only* changed variable across the sweep is the weight. F_spatial rises in tandem (rules out a
   degenerate-generator metric artifact). This points to BC's **averaging of the ~3.6% edited minority** as the
   L2 bottleneck — not a fundamental transfer wall and not the 1/N metric insensitivity.
4. **One load-bearing experiment is still missing (the placebo).** Every framing below is rated *viable, not
   yet bulletproof* for the same reason: we have **not** yet shown that upweighting a *random* 3.6% subset does
   **not** produce the gain. Until that control is run, a reviewer can attribute the effect to "oversampling any
   minority moves a global aggregate" rather than "the edited trajectories specifically carry transferable
   fairness." **Recommend running it before committing the framing.**

---

## 1. The significance table (6 seeds, 20 epochs, gate PASSED → Fidelity-A trusted)

| arm | F_causal | ΔF_causal vs raw (paired) | Wilcoxon p | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.8083 ± 0.0025 | — | — | 0.0830 | 0.8410 | 0.0121 |
| edited (w=1) | 0.8064 ± 0.0023 | **−0.0019 ± 0.0016** (6/6 neg) | 0.031 | 0.0841 | 0.8409 | 0.0119 |
| edited_w10 | 0.8269 ± 0.0018 | **+0.0186 ± 0.0027** (6/6 pos) | 0.031 | 0.0860 | 0.8406 | 0.0145 |
| edited_w20 | 0.8325 ± 0.0019 | **+0.0242 ± 0.0026** (6/6 pos) | 0.031 | 0.0868 | 0.8409 | 0.0166 |
| edited_w30 | 0.8357 ± 0.0020 | **+0.0274 ± 0.0021** (6/6 pos) | 0.031 | 0.0871 | 0.8406 | 0.0184 |

- **p = 0.031** is the n=6 unanimous-sign floor (the smallest achievable at n=6); all four arms hit it because
  each is 6/6 same-sign. It is significance, but not high-powered — see §4.
- **Dose-response is concave** (diminishing returns): marginal gain +0.0205 (w1→10), +0.0056 (w10→20), +0.0032
  (w20→30). **Efficiency** ΔF_causal / ΔFidelity-B: w10 ≈ 7.9× · w20 ≈ 5.4× · w30 ≈ 4.4×.
- **Effective edited fraction after reweighting** (by gradient mass): 3.6% (w1) → ~27% (w10) → ~43% (w20) →
  ~53% (w30). This is *why* the w=30 transfer (+0.0274) exceeds the unweighted data-level gap (+0.0128):
  weighting reshapes the effective training distribution — it amplifies, it does not "recover a fixed quantity."

---

## 2. What this means

The Level-2 negative was a property of *vanilla* BC, not of the data. A one-line, editor-agnostic training
change (importance-weight the edited demonstrations) realizes the data-level fairness in the trained policy,
with identity realism free and distributional realism only mildly and controllably traded. This is the
"fairness-aware training procedure that inherits the data-level fairness" the L2 doc left as an open question.
It re-opens a **model-level** version of the umbrella claim — the subject of the framing discussion below.

---

## 3. Four paper-framing options (to discuss)

All four use the *same* numbers and preserve the vanilla-BC negative honestly (as the w=1 arm). They differ in
where the weight of the contribution sits. Each was adversarially reviewed; all four came back **viable**.

### Frame 1 — "Negative-then-resolved" (tension-and-release)
- **One-liner:** Data-level fairness doesn't survive vanilla BC — but a single training-side knob recovers it,
  restoring the umbrella claim at the policy level.
- **L2 becomes:** the SETUP/diagnosis act of a single arc; the w=1 negative is the puzzle whose resolution is
  the payoff (and the internal control).
- **Licenses:** "Within importance-weighted BC, the editor's data-level fairness transfers to the policy; the
  vanilla-BC failure was BC averaging a small edited minority."
- **Must not claim:** transfer through cGAIL/full IL; that the editor improved; that the gain is bounded by the
  data-level gap.
- **Main reviewer risk:** the payoff *depends on* the placebo landing right. Best read, highest variance.

### Frame 2 — "Fairness-Aware Behavior Cloning (FA-BC)" (edit-then-reweight recipe)
- **One-liner:** FAMAIL is a two-stage recipe — edit the data, then upweight the edits during BC — and stage 2
  is what makes fairness reach the policy.
- **L2 becomes:** the ablation that motivates and validates stage 2 (editing alone is necessary but not
  sufficient).
- **Licenses:** "A two-stage edit-then-reweight procedure trains measurably fairer BC policies, with a tunable
  fairness knob, unanimous over 6 seeds."
- **Must not claim:** that it's "fairness-aware IL" (it's BC); that it works without edit labels; novelty of
  importance weighting itself.
- **Main reviewer risk:** novelty ("importance weighting is known") + the oracle-labels assumption.

### Frame 3 — "The data is the asset" (editing as reusable fair-data production; trainer is a scoping variable)
- **One-liner:** The editor produces the fairest faithful dataset (L1, the durable contribution); whether that
  fairness shows up in a policy is a property of the *trainer*, and one fairness-aware trainer realizes it.
- **L2 becomes:** preserved in full as an honest negative for fairness-*blind* BC + a load-bearing diagnostic;
  weighted-BC is a focused proof-of-concept, NOT promoted to co-headline.
- **Licenses:** L1 data-quality win (unchanged) + "data-level fairness is realizable by at least one
  fairness-aware trainer; vanilla BC's failure is the trainer's, not the data's."
- **Must not claim:** general IL transfer; "zero cost" (say "no detectable Fid-A change at n=6").
- **Main reviewer risk:** scoping overreach ("trainer is a scoping variable" is stronger than n=6 IW-BC shows).
  **Most robust option: L1 stands even if the placebo kills the recovery story.**

### Frame 4 — "The fairness-inheritance knob" (tunable F_causal ↔ Fidelity-B trade-off)
- **One-liner:** A single importance-weight traces a controllable fairness↔realism trade-off at fixed identity
  fidelity — turning "does fairness transfer?" into a tunable, characterized curve.
- **L2 becomes:** the zero-inheritance left endpoint (w=1) of the trade-off curve + calibration anchor.
- **Licenses:** "w is a monotone, diminishing-returns control on inherited fairness, bought only with small
  distributional drift, identity fidelity held fixed."
- **Must not claim:** a dense/optimal "Pareto frontier" (only 4 points); "inheritance" (w=30 exceeds the data
  gap → use "amplify"); "free/decoupled" identity fidelity (say "not significantly changed at n=6").
- **Main reviewer risk:** the trade-off figure is the contribution, so it needs a denser sweep; language must
  de-escalate from "Pareto/inheritance/free."

### Synthesis lean (a suggestion — this is your and Dr. Zhang's call)
**Frame 3 as the spine, Frame 1 as the L2 arc.** Frame 3 keeps the paper safe (the L1 data-quality contribution
stands regardless of how the recovery story fares), while Frame 1 gives the L2 section its compelling
negative-then-resolved narrative. Decide the exact blend **after** the placebo result, since Frame 1's payoff is
placebo-dependent and Frame 3 is not.

---

## 4. Shared must-haves before submission (needed under ANY framing)

1. **Random-subset placebo (decisive, ~½–1 day, reuses the harness).** Upweight a *random* 3,773-trajectory
   subset at w=10/30 over the same 6 paired seeds. The contribution as a *fairness* method survives only if
   this does **not** raise F_causal. Currently not run — this is the #1 priority.
2. **Oversampling vs loss-weighting control.** Duplicate edited rows vs the per-sequence loss weight, to
   substantiate the "effective-distribution reshaping" explanation for why w=30 > the data gap.
3. **Policy-collapse check at high dose.** `n_empty` (degenerate rollouts) rises with w; report terminal-cell
   entropy / trip-length distribution per arm to show large-w policies aren't silently degrading (flat
   Fidelity-A alone won't catch this).
4. **Honest statistics.** State p = 0.031 is the n=6 floor; if cheap, extend to n≈9–10 to move off it.
5. **Language discipline.** "reproduces within noise" not "exactly"; "no detectable Fid-A change at n=6" not
   "zero cost"; "amplify" not "inherit/recover"; "dose-response curve" not "Pareto frontier."
6. **Limitations up front.** IW-BC (not cGAIL/full IL); requires a *labeled* edited subset; single corpus;
   w=30 exceeds the data gap by design (distribution reshaping).
7. **Preserve the vanilla-BC negative** as a first-class result (it's true regardless of the recovery).

---

## 5. Decisions & open questions for Dr. Zhang

1. **Headline placement:** should the paper's headline *depend on* the weighted-BC recovery (Frame 1/2/4), or
   stay anchored on the L1 data asset with recovery as support (Frame 3, robust to a bad placebo)?
2. **Novelty positioning:** is importance-weighted BC of a labeled fair subset novel enough for the UIC venue
   on its own, or should it be framed as a diagnosis-and-fix rather than a new algorithm?
3. **Scope:** re-test the umbrella claim head-to-head at the *policy* level (weighted-BC on bcgen/gangen too),
   or explicitly scope the recovery to "vs raw-trained policy"?
4. **Labeled-subset assumption:** acceptable as-is (we edited the data, so we know the subset), or do we need a
   noisy-label / inferred-subset robustness point?
5. **Keep the demographic-granularity smoothing direction out of this paper** (it's a separate effort) — agree?
6. **Contingency:** if the placebo comes back ambiguous/positive, do we fall back to the plain-negative L2 +
   the granularity diagnosis (the pre-weighted-BC story)?

---

## Reproduction

```bash
# Headline 6-seed significance sweep:
python -m famail_temporal.baselines.run_weighted_bc_smoke \
  --seeds 0,1,2,3,4,5 --weights 10,20,30 --mle-epochs 20 \
  --out-dir famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30
# The lever: optional `sample_weights` in gan/train_mle.py (no-op when None → locked L2 numerics intact);
# verified by gan/tests/test_train_mle_weights.py (uniform==unweighted, upweighting biases, bad-length rejected).
```
