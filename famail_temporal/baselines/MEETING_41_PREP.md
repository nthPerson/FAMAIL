# Meeting 41 Prep — Weighted-BC Recovers Level-2 Transfer: Results + Paper-Framing Options

**Date:** 2026-06-20 · **Updated 2026-06-25 — the load-bearing placebo control has been run and PASSED (§6).**
**Scope:** Reports the weighted-BC significance sweep (the experiment recommended in `MEETING_40_PREP.md`)
and lays out four pressure-tested paper-framing options to open the discussion with Dr. Zhang. No editor
change; the lever lives entirely in the BC trainer. **The §4 #1 must-have — the random-subset placebo —
is now done: the F_causal gain is edit-specific, not a generic oversampling artifact (§6).**
**Artifacts:** `famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30/sweep.json` (headline, 6 seeds);
`.../full_5seed_w10_w30/` (the preceding 5-seed run); `.../placebo_6seed_w10_w30/sweep.json` (the placebo
control, 6 seeds, 2026-06-25). Code on `main` (committed): `gan/train_mle.py` (`sample_weights`, `8a22caf`);
`gan/tests/test_train_mle_weights.py`; `run_weighted_bc_smoke.py` (now with `random_subset_weight_vector` +
`--placebo`/`--placebo-seed`, `4b0ddd0`); placebo selector tests `tests/test_run_weighted_bc_smoke.py`.

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
4. **The load-bearing placebo is now RUN and PASSED (2026-06-25 — see §6).** Upweighting a *random*,
   size-matched (3,773-traj) NON-edited subset over the same 6 paired seeds leaves both fairness axes within
   noise — random ΔF_causal **−0.0012 (w10) / −0.0015 (w30)**, non-significant (p=0.22), ~⅛ of the 0.012-bit
   floor — while the *edited* arms reproduce their **+0.0186 / +0.0274** gains exactly (6/6, p=0.031). The gain
   is **edit-specific, not a generic oversampling artifact.** Sharper still: the placebo *does* significantly
   reshape the distribution (Fidelity-B +0.0034, p=0.031) and applies the same dose-driven degeneracy pressure
   (matched n_empty) yet moves *no* fairness axis — so the "any oversampling moves a 1/N aggregate" objection is
   refuted by a *present-but-insufficient* confound, not left untested. **The framings below are no longer
   "viable pending the placebo"; the edit-specificity gate is cleared.**

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
- **Main reviewer risk:** the payoff depended on the placebo landing right — **and it did (§6)**, so this
  frame's central risk is now retired. Residual: it is recovery through *importance-weighted BC*, not full IL.

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
  **Was the most robust option (L1 stands regardless); with the placebo now passed (§6), the recovery leg it
  leans on is also load-bearing-verified, so this frame is strictly stronger than before.**

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
negative-then-resolved narrative. The "decide the blend after the placebo" caveat is now **resolved**: the
placebo passed (§6), so Frame 1's payoff is verified, not pending. Frame 3 + Frame 1 remains the recommended
blend — now both legs are load-bearing-verified rather than one being contingent.

---

## 4. Shared must-haves before submission (needed under ANY framing)

1. **Random-subset placebo — ✅ DONE (2026-06-25), PASSED.** Upweighted a *random* 3,773-traj NON-edited
   subset at w=10/30 over the same 6 paired seeds; F_causal did **not** rise (random Δ −0.0012/−0.0015,
   non-significant), while edited reproduced +0.0186/+0.0274. Specificity confirmed — full numbers in §6.
2. **Oversampling vs loss-weighting control.** Duplicate edited rows vs the per-sequence loss weight, to
   substantiate the "effective-distribution reshaping" explanation for why w=30 > the data gap. *(Still open,
   but the placebo already shows the reshaping confound alone does not produce fairness.)*
3. **Policy-collapse check at high dose.** `n_empty` (degenerate rollouts) rises with w (to ~2/seed at w30;
   one w10 seed hit 12). The placebo gives a partial answer: at matched dose, **random_w30 n_empty
   [0,4,2,2,2,2] ≈ edited_w30 [4,0,2,2,2,2]**, so degeneracy pressure is dose-driven and generic — yet only
   edited gains fairness, so the gain is not a degeneracy artifact. *Still worth* reporting terminal-cell
   entropy / trip-length per arm to fully close it (flat Fidelity-A alone won't catch silent degradation).
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
6. **Contingency:** ~~if the placebo comes back ambiguous/positive, fall back to plain-negative L2 + the
   granularity diagnosis~~ — **moot: the placebo passed (§6).** The recovery story is the live story; the
   plain-negative L2 + granularity diagnosis is now a *limitations/scoping* note, not a fallback headline.

---

## 6. The random-subset placebo control (2026-06-25) — PASSED

**Question it answers.** A reviewer can object that upweighting *any* small minority reshapes the effective
training distribution and can move a 1/N global metric, so the weighted-BC F_causal gain might be an
oversampling artifact rather than something the *edited* trajectories specifically carry. The placebo settles
it: upweight a **random, size-matched (3,773-traj) NON-edited** subset of the **raw** corpus at the same doses
and seeds, and ask whether F_causal still rises.

**Design.** One fixed random subset (`placebo_seed=12345`, drawn with an independent RNG so it cannot perturb
the per-seed training determinism), applied to the raw corpus; arms `random_w10`/`random_w30` compared against
the in-process `raw`. Run alongside the reproduced `raw`/`edited`/`edited_w10`/`edited_w30` arms (6 paired
seeds, 20 epochs, gate PASSED 0.841/0.174). The selector is TDD'd (`tests/test_run_weighted_bc_smoke.py`).

**Result (paired Δ vs raw, F_causal — fairness, ↑ better):**

| arm | what is upweighted | ΔF_causal vs raw | Wilcoxon p | signs | ΔF_spatial | ΔFidelity-B |
|---|---|---:|---:|---:|---:|---:|
| edited_w10 | the 3,773 **edited** trajs | **+0.0186 ± 0.0027** | 0.031 | 6/6 + | +0.0030 (p=.03) | +0.0024 |
| edited_w30 | the 3,773 **edited** trajs | **+0.0274 ± 0.0021** | 0.031 | 6/6 + | +0.0041 (p=.03) | +0.0063 |
| **random_w10** | a **random** 3,773 non-edited | **−0.0012 ± 0.0018** | **0.219** | 4−/1·/1+ | +0.0000 (p=.69) | +0.0012 |
| **random_w30** | a **random** 3,773 non-edited | **−0.0015 ± 0.0028** | **0.219** | 4−/1·/1+ | −0.0004 (p=.16) | +0.0034 (p=.03) |

**Verdict: PASS — the F_causal gain is edit-specific.** Adversarially verified (4 independent lenses +
synthesis, all PASS, high confidence):

1. **Random upweighting moves no fairness axis.** Both random arms sit at ~−0.001, *negative* and ~⅛ of the
   0.012-bit seed-noise floor, mixed-sign and non-significant (p=0.22 ≠ the 0.031 unanimous-sign floor). The
   edited arms clear the floor 6/6. **Edited − random gap: +0.0198 (w10), +0.0290 (w30)** — about an order of
   magnitude.
2. **A second fairness axis corroborates.** F_spatial moves 6/6-significantly *only* for edited; random leaves
   it flat (p=0.69 / 0.16). Two independent fairness metrics respond only to the edited subset.
3. **The placebo is *not* inert — and that strengthens it.** `random_w30` genuinely reshapes the distribution
   (Fidelity-B +0.0034, 6/6, p=0.031) and carries the same dose-driven degeneracy pressure as edited (matched
   `n_empty`: random_w30 [0,4,2,2,2,2] vs edited_w30 [4,0,2,2,2,2]) — yet produces **zero** fairness gain. The
   "any reshaping moves a 1/N aggregate" confound is therefore *present and demonstrably insufficient*, not
   merely untested. Distributional reshaping is generic to upweighting; converting it to fairness is
   edit-specific.
4. **Pipeline integrity confirmed.** The common arms reproduce the headline sweep *bit-identically* on
   F_causal / F_spatial / Fidelity-B; the in-process `edited(w1)` reproduces the locked L2 negative (−0.0019,
   p=0.031); the gate passed. The independent-RNG guard held: adding the placebo arms did not disturb the
   paired design. (Fidelity-A *means* drift ~6e-4 between runs — within `raw`'s own ~4e-4 std, from the shared
   eval-RNG being consumed by a different arm set; the load-bearing fairness/Fid-B metrics are exact.)

**Residual caveats (honest).** (a) Single fixed random draw — multi-subset repetition remains an *optional*
robustness extension, not run (mitigated by how flat, even slightly negative, the result is); (b) this is
recovery through *importance-weighted BC*, not cGAIL/full IL; (c) terminal-cell entropy / trip-length per arm
not yet reported to fully close the high-dose degeneracy question (matched-dose `n_empty` parity argues
degeneracy is generic, not the fairness source); (d) n=6 caps significance at p=0.031 — extend to n≈8–10 for a
stronger p. None of these change the verdict.

---

## Reproduction

```bash
# Headline 6-seed significance sweep:
python -m famail_temporal.baselines.run_weighted_bc_smoke \
  --seeds 0,1,2,3,4,5 --weights 10,20,30 --mle-epochs 20 \
  --out-dir famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30
# Placebo control (§6): adds random_w10/random_w30 (random size-matched non-edited subset of raw):
python -m famail_temporal.baselines.run_weighted_bc_smoke \
  --seeds 0,1,2,3,4,5 --weights 10,30 --placebo 10,30 --placebo-seed 12345 --mle-epochs 20 \
  --out-dir famail_temporal/results/weighted_bc_sweep/placebo_6seed_w10_w30
# The lever: optional `sample_weights` in gan/train_mle.py (no-op when None → locked L2 numerics intact);
# verified by gan/tests/test_train_mle_weights.py (uniform==unweighted, upweighting biases, bad-length rejected).
# The placebo selector: random_subset_weight_vector in run_weighted_bc_smoke.py (independent RNG, built once
# before the seed loop); verified by tests/test_run_weighted_bc_smoke.py (size-match, disjoint-from-edited,
# reproducible, global-RNG-isolation).
```
