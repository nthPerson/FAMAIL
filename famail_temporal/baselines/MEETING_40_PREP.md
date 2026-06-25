# Meeting 40 Prep — Consolidated Progress Since Meeting 39

**Date:** 2026-06-25 · **Covers:** everything since Meeting 39 (2026-06-18).
**Purpose:** one briefing for the upcoming meeting with Dr. Zhang. It absorbs and supersedes the two
interim prep notes — the Meeting-39 relocation feasibility analysis (formerly this file) and the
weighted-BC results note (`MEETING_41_PREP.md`, kept for full detail) — and adds the placebo control and a
new data-quality finding (stuck-GPS "sinks").
**Bottom line:** the relocation idea from Meeting 39 was pressure-tested and set aside as likely-marginal;
the cheap experiment it recommended instead (weighted BC) **worked, is significant, and is now
placebo-confirmed edit-specific** — which re-opens a *model-level* version of the umbrella claim and turns
the meeting into a **paper-framing decision**. Separately, a data-quality audit found 6 stuck-GPS pickup
artifacts that **depress** our (secondary) spatial-fairness number by ~23% — i.e. we *under*-claim spatial
fairness — but **leave the headline causal results untouched.**

---

## TL;DR — what changed since Meeting 39

1. **Relocation idea: pressure-tested → likely marginal (don't build it).** F_causal is genuinely
   *unreachable* by the current gradient editor (demographics are district-constant; the ε=2-cell edit
   radius ≪ a district's 86–544 cells, so settling never crosses a boundary). Only a cross-district
   *teleport* can move F_causal — but "teleport far" and "stay realistic" are a geometry-forced
   contradiction, and relocation doesn't touch the thing Level-2 identified as the real bottleneck. **§A.**

2. **The cheap experiment relocation pointed to instead — weighted BC — is the headline win.** Upweighting
   the ~3.6% edited demonstrations during BC turns the Level-2 transfer **positive, significant, and
   unanimous**: paired edited−raw F_causal goes **−0.0019 (w=1) → +0.0186 (w10) → +0.0242 (w20) → +0.0274
   (w30)**, 6/6 seeds each, Wilcoxon p=0.031, **no detectable identity-fidelity change at n=6**, a small tunable distributional
   cost. The Level-2 negative was a property of *vanilla* BC, not of the data. **§B.**

3. **The load-bearing placebo PASSED — the gain is edit-specific, not an oversampling artifact.** Upweighting
   a *random*, size-matched non-edited subset moves F_causal essentially to zero (−0.0012/−0.0015,
   non-significant, ~⅛ of the noise floor) while the edited arms reproduce their gains exactly. The placebo
   *does* perturb the distribution (it isn't inert) yet moves no fairness axis — so "any oversampling moves a
   global metric" is refuted, not merely untested. Adversarially verified (one caveat: the placebo rests on a
   single fixed random draw; multi-subset robustness is optional and not yet run). **§C.**

4. **New data-quality finding: 6 stuck-GPS pickup "sinks" in the raw data.** Six drivers each have a frozen
   pickup coordinate generating ~10–12k phantom pickups (~20% of all raw pickups combined). They do **not**
   match the real cGAIL points-of-interest (real hubs look normal). **Only one cell, S1=(28,52), materially
   distorts F_spatial (removing it *raises* F_spatial +23%, so the artifact deflates it); F_causal is
   essentially unaffected (Δ+0.0004).** Because the headline config is causal-emphasis, the main results are
   robust — and since the editor targets the most-unfair cells, S1 (DSR 67) *may* have been a top edit target
   (an unverified hypothesis, flagged as the next check — not established). Worth cleaning. PI's call. **§D.**

5. **Net effect on the paper.** We now have a *model-level* recovery story on top of the data-level L1 win,
   stress-tested from three angles (relocation ruled out as a distraction, weighted-BC positive, placebo
   confirms specificity). The meeting's main job is **choosing the framing** (4 options drafted; lean =
   "data-is-the-asset" spine + "negative-then-resolved" arc) and **deciding the sinks cleanup**. **§E–F.**

**Where each level stands now:** L1 data-quality = **POSITIVE** (edited is the fairest faithful source);
L2 transfer = **NEGATIVE under vanilla BC, RECOVERED under weighted BC** (significant + placebo-confirmed).

---

## §A. The Meeting-39 relocation ideas — pressure-tested (no code changed)

Meeting 39 asked: *can we make the trajectory edits move fairness more, and would that survive training?*
The three ideas were checked against the committed code before spending any of the ~1 month on them.

### Idea scoreboard

| # | Meeting-39 idea | By | Code-grounded verdict |
|---|---|---|---|
| 1 | **Physical relocation** (teleport worst-offenders out of concentration, then let the editor settle) | Robert | **Likely marginal.** Valid only via a cross-district F_causal move; stage-2 settling adds ≈0; collides with realism + 1/N dilution. |
| 2 | **Target-locations-first** scan (O(AB) over the grid → top-K POIs) vs O(N) over trajectories | Dr. Zhang | **Real but non-binding.** Genuine constant-factor speedup (infra exists), but the binding limit is 1/N metric sensitivity, not selection runtime. |
| 3 | **Select by gradient magnitude** instead of negative-contribution | Dr. Zhang | **Half-validated.** "Contribution ≈ gradient" holds for the *spatial* term, **fails for the causal** term; and gradient-selection + teleport is self-defeating (the teleport destroys the gradient that justified selection). |
| — | Randomize **location, not time** | both | Sound and supported by the existing active-cell mask + per-(cell,t) grids. |
| — | Gradient heatmap as the verification gate | Robert (done) | Correct gate; tool lives at `visualization/gradient_heatmap/`. |

### Why relocation is mechanically narrow

- **F_causal is unreachable by gradient settling.** `F_causal = 1 − r²_demo` regresses residual demand on
  **district-level** demographics; demographics are constant within each of 10 districts (sizes 86–544
  cells), so `(I − H_demo)` is flat in the district interior — the causal gradient only changes when a pickup
  **crosses a district boundary**. The editor's edit radius is an **L∞ ε-ball of 2 cells** ≪ district size, so
  it essentially never crosses one. ➡️ **Only a physical cross-boundary teleport can move F_causal at all.**
- **Built-in ceiling.** A teleport that stays in-district (the "realistic/nearby" case) moves F_causal by ≈0,
  and both metrics are **global aggregates over N=34,524 active (cell,time) units**, so a single relocation
  has **O(1/N) ≈ 2.9e-5** leverage; ~50 realistic relocations touch ~0.3% of units — below the ~0.012-bit
  seed-noise floor. *Distance ≠ leverage:* a committed default-config run moved **1000** trajectories yet
  pushed F_causal **−0.0058 (the wrong way)** while piling **320** of them into a single destination cell.
- **Wrong bottleneck for transfer.** Level-2 isolated *why* the signal dies: BC's teacher-forced MLE averages
  over the unedited ~96.4% and has no fairness term. Relocation makes individual edits *bigger* but changes
  neither the **edited fraction** nor the **averaging operator** — and teleported terminals are *off-manifold*
  w.r.t. the (driver, start-context) conditioning BC transfers through, so BC is *more* likely to regularize
  them away.

### Several Meeting-39 premises didn't match the committed code

Default α is **(0.33, 0.33, 0.34)**, not (0.2, 0.7, 0.1) — the latter is the causal-emphasis runtime override
used for the headline results; edit radius is **ε=2**, not 3; committed movement is **mean ~1.2 / max 2.83**
(= 2√2), not 0.81 / 2.2 (those figures weren't found in any committed run — worth confirming which run
produced them); and the contribution score is **not** [−1,+1]-normalized.

### Verdict and the experiment it pointed to

**Plausibility: likely marginal.** Relocation targets edit *magnitude*; the transfer bottleneck is edit
*fraction* + the averaging operator. The recommended cheap, decoupled, decisive experiment was a
**weighted-BC ablation (no relocation)** — upweight the existing ~3,773 edited trajectories 10–30× in the MLE
loop and re-check transfer. *That experiment is §B, and it worked.* A defensible alternative finding remains
on the table: *the binding limitation is district-level demographic resolution, not the editor* — a clean,
publishable diagnosis that motivates cell-level demographics as future work (being explored separately).

---

## §B. The weighted-BC lever — Level-2 transfer recovered

The §A-recommended ablation, run as a paired-seed sweep that reuses the locked Level-2 evaluator (same
driver-conditioned BC policy, same identity gate + fidelity axes). The lever lives entirely in the BC trainer
(an optional per-sequence `sample_weights` in the MLE loss); **the editor and the locked L1/L2 tables are
untouched.**

### Significance table (6 seeds, 20 epochs, identity gate PASSED → Fidelity-A trusted)

| arm | F_causal | ΔF_causal vs raw (paired) | Wilcoxon p | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.8083 ± 0.0025 | — | — | 0.0830 | 0.8410 | 0.0121 |
| edited (w=1) | 0.8064 ± 0.0023 | **−0.0019 ± 0.0016** (6/6 neg) | 0.031 | 0.0841 | 0.8409 | 0.0119 |
| edited_w10 | 0.8269 ± 0.0018 | **+0.0186 ± 0.0027** (6/6 pos) | 0.031 | 0.0860 | 0.8406 | 0.0145 |
| edited_w20 | 0.8325 ± 0.0019 | **+0.0242 ± 0.0026** (6/6 pos) | 0.031 | 0.0868 | 0.8409 | 0.0166 |
| edited_w30 | 0.8357 ± 0.0020 | **+0.0274 ± 0.0021** (6/6 pos) | 0.031 | 0.0871 | 0.8406 | 0.0184 |

- **Mechanism, not magic.** The w=1 arm reproduces the locked Level-2 negative within noise (−0.0019 vs the
  documented −0.0022), so the *only* variable changing across the sweep is the weight. **Both** fairness axes
  rise together (F_spatial +0.0041 at w30), which rules out a degenerate-generator metric artifact.
- **No detectable identity-fidelity change at n=6; a small, tunable distributional cost.** Fidelity-A is flat
  (~0.8407, all paired p≥0.09); Fidelity-B rises gently 0.0121→0.0184 (>15× below the GAN-collapse 0.32 across
  the sweep — ~17× at the w=30 endpoint — that disqualified GAN-gen). **w is a fairness↔realism knob** — w=10
  most efficient (~7.9× fairness per unit Fid-B), w=30 max fairness.
- **Why w=30 (+0.0274) exceeds the unweighted data-level gap (+0.0128):** weighting reshapes the *effective*
  training distribution (edited fraction by gradient mass 3.6% → ~27% → ~43% → ~53%). It **amplifies**; it
  does not "recover a fixed quantity." This is the honest framing (not "inherit"/"recover").

**What it means:** the Level-2 negative was a property of *vanilla* BC, not of the data. A one-line,
editor-agnostic training change realizes the data-level fairness in the trained policy — the "fairness-aware
training procedure" the L2 write-up left as an open question — re-opening a **model-level** umbrella claim.

---

## §C. The placebo control — the fairness gain is edit-specific (PASSED)

**The objection it answers:** upweighting *any* small minority reshapes the effective training distribution and
could move a 1/N global metric, so the §B gain might be a generic oversampling artifact rather than something
the *edited* trajectories specifically carry. **The control:** upweight a *random*, size-matched (3,773-traj)
**non-edited** subset of the raw corpus at the same doses and seeds, and ask whether F_causal still rises.

| arm | upweighted | ΔF_causal vs raw | Wilcoxon p | signs | ΔF_spatial | ΔFidelity-B |
|---|---|---:|---:|---:|---:|---:|
| edited_w10 | the 3,773 **edited** | **+0.0186 ± 0.0027** | 0.031 | 6/6 + | +0.0030 (p=.03) | +0.0024 |
| edited_w30 | the 3,773 **edited** | **+0.0274 ± 0.0021** | 0.031 | 6/6 + | +0.0041 (p=.03) | +0.0063 |
| **random_w10** | a **random** 3,773 non-edited | **−0.0012 ± 0.0018** | **0.219** | 4−/1·/1+ | +0.0000 (p=.69) | +0.0012 |
| **random_w30** | a **random** 3,773 non-edited | **−0.0015 ± 0.0028** | **0.219** | 4−/1·/1+ | −0.0004 (p=.16) | +0.0034 (p=.03) |

**Verdict: PASS — edit-specific.** (Adversarially verified: 4 independent lenses + synthesis, unanimous PASS,
high confidence.)

1. Random upweighting moves *no* fairness axis: both arms sit at ~−0.001 (negative, mixed-sign,
   non-significant), ~⅛ of the 0.012-bit seed-noise floor. The **edited − random gap is +0.0198 (w10) /
   +0.0290 (w30)** — about an order of magnitude. A second fairness axis (F_spatial) corroborates: it moves
   6/6-significantly only for edited, flat for random.
2. **The placebo is *not* inert — and that strengthens it.** random_w30 genuinely reshapes the distribution
   (Fidelity-B +0.0034, 6/6, p=0.031) and carries the same dose-driven degeneracy pressure as edited (matched
   total `n_empty`, 12 vs 12) — yet produces zero fairness gain. So the "any reshaping moves a 1/N aggregate" confound is
   *present and demonstrably insufficient*, refuted rather than left untested.
3. **Pipeline integrity confirmed:** the common arms (raw, edited, edited_w10/w30) reproduce the headline
   sweep bit-identically on F_causal/F_spatial/Fidelity-B, the in-process w=1 arm reproduces the locked L2
   negative, and the identity gate passed — so adding the placebo arms didn't disturb the paired design.

**Residual caveats (honest):** single fixed random draw (multi-subset robustness optional, not run); recovery
is through *importance-weighted BC*, not full IL/cGAIL; terminal-cell entropy / trip-length per arm not yet
reported to fully close high-dose degeneracy (matched-dose `n_empty` parity argues degeneracy is generic, not
the fairness source); n=6 caps significance at p=0.031 (extend to n≈8–10 for a stronger p). None change the
verdict.

---

## §D. Data-quality finding — 6 stuck-GPS pickup "sinks" (surfaced via the gradient heatmap)

Investigating the gradient-heatmap concentration-contour overlay surfaced a raw-data artifact: **6 per-driver
"stuck-GPS" pickup sinks.** Six specific drivers have a meter-on (pickup) coordinate frozen at one exact
lat/lon, each generating ~10–12k phantom pickups; together **~20% of all raw pickup events**. Each sink cell's
pickups are 100% from a single driver (1/50). Dropoffs geocode normally, so this is a **pickup-only** spatial
skew.

- **Sink cells (0-indexed):** S1 (28,52), S2 (20,28), S3 (28,28), S4 (24,5), S5 (22,46), S6 (17,38).
- **Not the real hubs.** The sinks do **not** match the cGAIL points-of-interest — the real downtown PoIs
  (深圳站, 福田站, Coco Park, MixC, CBDs, hospitals) appear as **normal balanced cells** (pickup≈dropoff ~1.0–1.5,
  like the global median 1.06). The sinks are a separate per-driver artifact, not real demand.
- **Why it mostly doesn't matter for our results.** F_spatial keys on DSR = pickup/active_taxis, so only a sink
  in a *low-supply* cell distorts it. In the metric data **only S1 (28,52) is an extreme DSR outlier** (peak
  DSR 67 vs p99=0.4):

  | F_spatial scenario | value | vs baseline |
  |---|---:|---:|
  | baseline (as-reported) | 0.0822 | — |
  | remove S1 only | 0.1013 | **+23%** (S1 = 88% of the distortion) |
  | remove all 6 sinks | 0.1039 | +26% |
  | cap at p99 | 0.0968 | +18% |

  Because F_spatial is a *fairness* score (higher = fairer), the artifact makes us look **less** spatially fair
  than we are — we are **under-claiming** spatial fairness, not over-claiming. **F_causal is essentially
  unaffected (Δ+0.0004).** Since the headline config is **causal-emphasis α=(0.2,0.7,0.1)**, the main results
  are robust; only the secondary (α=0.2) spatial number is contaminated, and almost entirely by one cell.
  *(The 0.0822 baseline here is the data-level audit value — the §B policy-level table's raw F_spatial 0.0830
  is the BC-trained number; different evaluators, not a discrepancy.)*
- **The connection to §A/editing (a hypothesis, not yet verified).** The editor targets the most-unfair cells,
  so S1 (DSR 67) *may* have been a top edit target — a phantom. **We have not yet confirmed this against the
  editor's actual selection log; it is the next check, not an established finding.** If true, it is the
  concrete version of the §A worry about edits piling into high-count cells (cf. the committed run that piled
  320 trajectories into one cell): some high-count cells are *data artifacts*, not demand. F_causal results are
  unaffected either way, but a clean re-baseline would remove a phantom from the spatial story (and from
  whatever the editor did to it). The pipeline (`data/source_generation/removal.py`) does only
  trajectory-invariant checks — there is **no stuck-GPS/sink detection** — so the artifacts propagate into
  `pickup_counts` → metrics + viz.

**Status: PI's call, not yet actioned.** Options: exclude/cap the 6 coordinates (minimally just S1) before
metrics/editing; and/or add stuck-GPS detection to `source_generation`; then optionally re-baseline F_spatial
(and re-check whether the editor's behavior changes once S1 is no longer a target).

---

## §E. Where this leaves the paper

The chain of logic since Meeting 39 is clean and mutually reinforcing:

- **L1 (data quality) — POSITIVE, unchanged.** The editor produces the fairest *faithful* dataset (edited
  F_causal 0.8180 > raw 0.8052 among faithful sources; identity-faithful, gate passed; GAN-gen disqualified by
  distributional collapse). This is the durable contribution and stands regardless of everything below. *(These
  are data-level values; §B's raw F_causal 0.8083 is the policy-level BC metric — different evaluators, not an
  inconsistency.)*
- **L2 (transfer) — NEGATIVE under vanilla BC, RECOVERED under weighted BC.** §B turns the honest negative into
  a diagnosis-and-fix; §C proves the fix is edit-specific. The umbrella claim now has a *model-level* form, not
  only a data-level one.
- **Relocation (§A) is no longer the "if I have time" target** — the highest-leverage lever turned out to be
  the *trainer* (weighted BC), not the editor.
- **The sinks (§D)** are a data-hygiene item that strengthens the work (we found and characterized an artifact)
  without threatening the headline (F_causal robust).

**Highest-leverage use of the remaining time** is therefore likely the framing + a couple of cheap robustness
points (§F), not a relocation build.

---

## §F. Decisions & open questions for Dr. Zhang

**1 — Paper framing (the main decision).** Four options were drafted and adversarially reviewed; full detail in
`MEETING_41_PREP.md §3`. All use the *same* numbers and keep the vanilla-BC negative honest (as the w=1 arm):

| Frame | One-liner | L2 becomes | Main residual risk |
|---|---|---|---|
| **1 — Negative-then-resolved** | Data fairness dies in vanilla BC; one training knob recovers it | the setup/diagnosis act of a single arc | recovery is via IW-BC, not full IL (placebo risk now retired) |
| **2 — Fairness-Aware BC recipe** | FAMAIL = edit the data, then upweight the edits during BC | the ablation that validates stage 2 | novelty ("IW is known") + oracle-labels assumption |
| **3 — The data is the asset** | The editor makes the fairest faithful dataset; realizing it in a policy is a trainer property | an honest negative for fairness-*blind* BC + diagnostic | scoping overreach; **most robust — L1 stands regardless** |
| **4 — Fairness-amplification knob** | A single weight traces a fairness↔realism trade-off at fixed identity | the zero-amplification left endpoint (w=1) | only 4 points; language must avoid "Pareto/inherit/free" |

**Lean (your call):** **Frame 3 spine + Frame 1 arc** — Frame 3 keeps the paper safe (L1 stands no matter what),
Frame 1 gives L2 its compelling narrative; with the placebo passed, *both* legs are now load-bearing-verified.

- **1a. Headline placement:** should the headline *depend on* the weighted-BC recovery (F1/2/4), or stay
  anchored on the L1 data asset with recovery as support (F3)?
- **1b. Novelty positioning:** is IW-BC of a labeled fair subset novel enough for the UIC venue on its own, or
  do we frame it as diagnosis-and-fix rather than a new algorithm?
- **1c. Scope:** re-test the umbrella claim head-to-head at the *policy* level (weighted-BC on bc-gen/gan-gen
  too), or scope the recovery to "vs raw-trained policy"?
- **1d. Labeled-subset assumption:** acceptable as-is (we edited the data, so we know the subset), or do we need
  a noisy/inferred-subset robustness point?

**2 — Stuck-GPS sinks (§D):** do we (a) exclude/cap at least S1=(28,52) before metrics/editing, (b) add
stuck-GPS detection to `source_generation`, and (c) re-baseline F_spatial (and re-run editing to see if its #1
target changes)? F_causal is unaffected either way, so this is about cleaning the secondary spatial story, not
rescuing the headline. *(Per protocol, any change to the editing pipeline's inputs needs your sign-off.)*

**3 — Cheap robustness before submission (optional, none change the verdict):** multi-subset placebo;
oversampling-vs-loss-weighting control; terminal-cell entropy / trip-length per arm at high dose; extend the
sweep to n≈8–10 for a stronger p than the n=6 floor (0.031).

**4 — Granularity direction:** keep the demographic-granularity / cell-level-demographics exploration as
separate future work (it's a parallel effort), or fold its diagnosis ("the limit is data resolution, not the
editor") into this paper's limitations?

---

## Appendix — artifacts & where claims are grounded

| Thread | Artifact / source |
|---|---|
| Weighted-BC sweep (§B) | `famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30/sweep.json`; lever `baselines/gan/train_mle.py` (`sample_weights`, `8a22caf`) |
| Placebo control (§C) | `famail_temporal/results/weighted_bc_sweep/placebo_6seed_w10_w30/sweep.json`; selector `baselines/run_weighted_bc_smoke.py::random_subset_weight_vector` + `--placebo` (`4b0ddd0`); tests `baselines/tests/test_run_weighted_bc_smoke.py` |
| Framings (§B/F) | `baselines/MEETING_41_PREP.md` (full detail, incl. §6 placebo writeup) |
| Relocation analysis (§A) | grounded in `algorithm/modifier.py`, `fairness/causal.py`, `fairness/hat_matrices.py`, `config.py`, `evaluation/diagnostics.py`; Level-2 negative `baselines/LEVEL2_RESULTS.md` |
| Stuck-GPS sinks (§D) | **full writeup: `MEETING_41_PREP.md §7`** + memory `project_pickup_gps_sinks`; audit of `raw_data/taxi_record_0{7,8,9}_50drivers.pkl` → `source_data/pickup_dropoff_counts.pkl`; pipeline gap in `data/source_generation/removal.py`; surfaced via `visualization/gradient_heatmap/` |
| L1 / L2 results | `baselines/LEVEL1_V2_RESULTS.md`, `baselines/LEVEL2_RESULTS.md` |

*Reproduction (weighted-BC + placebo):*
```bash
python -m famail_temporal.baselines.run_weighted_bc_smoke \
  --seeds 0,1,2,3,4,5 --weights 10,20,30 --mle-epochs 20 \
  --out-dir famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30
python -m famail_temporal.baselines.run_weighted_bc_smoke \
  --seeds 0,1,2,3,4,5 --weights 10,30 --placebo 10,30 --placebo-seed 12345 --mle-epochs 20 \
  --out-dir famail_temporal/results/weighted_bc_sweep/placebo_6seed_w10_w30
```
