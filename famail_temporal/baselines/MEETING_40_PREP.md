# Meeting 40 Prep — Pressure-Testing the Meeting-39 Trajectory-Editing Improvement Ideas

**Date:** 2026-06-19 (analysis follows the Meeting-39 discussion on 2026-06-18)
**Scope:** No code was changed. This is a code-grounded feasibility analysis of the three
improvement ideas discussed in Meeting 39, run before committing any of the remaining ~1 month
to implementation. Claims below are cited to the files that back them.
**Context:** Level-2 came back **negative** (`baselines/LEVEL2_RESULTS.md`): the data-level
fairness edge does **not** survive BC training (paired edited−raw F_causal **−0.0022 ± 0.0016**,
5/5 seeds). Meeting 39 asked: *can we make the trajectory edits move fairness more, and would
that survive training?*

---

## TL;DR

1. **The relocation idea has one mechanically valid path — and it's narrow.** F_causal is
   genuinely *unreachable* by the current gradient editor because demographics are
   district-constant and the edit radius (ε=2 cells) is far smaller than a district (86–544
   cells), so gradient settling can never cross a district boundary. A **physical teleport across
   a boundary** is the only operation that can move F_causal at all. This confirms the granularity
   intuition mechanically. **But** the same fact means a "nearby/realistic" teleport (stays
   in-district) moves F_causal by ≈0 — so *teleport-far* and *stay-realistic* are a
   **geometry-forced contradiction**, not a tunable trade-off.

2. **Relocation attacks the wrong bottleneck for the actual goal (transfer).** Level-2 already
   isolated *why* the signal dies: BC's teacher-forced MLE averages over the unedited ~96.4% of
   trajectories and has no fairness term. Relocation makes individual edits *bigger*, but changes
   neither the **edited fraction** nor the **averaging operator**. Teleported pickups are
   *off-manifold* w.r.t. the (driver, start-context) conditioning the policy transfers through, so
   BC is *more* likely to regularize them away, not less. **No mechanism here makes a bigger
   data-level gap survive training.**

3. **Both fairness metrics are global aggregates with O(1/N) per-edit leverage** (N = 34,524
   active cell-time units). ~50 relocations (10 POIs × 5) touch ~0.3% of units. Teleport
   *distance* and *global-metric leverage* are **decoupled** — we can make edits look dramatic
   while the scalar barely moves (the committed default-config run already moved 1000 trajectories
   and pushed F_causal **−0.0058**, the wrong way, while piling up to **320** trajectories into a
   single destination cell).

4. **Several Meeting-39 premises don't match the committed code** (see §2): default α is
   **(0.33, 0.33, 0.34)** not (0.2, 0.7, 0.1); the edit radius is **ε=2** not 3; committed
   movement is **mean ~1.2 / max 2.83** not 0.81 / 2.2; and the contribution score is **not**
   normalized to [−1, +1]. These change the baseline we'd be "improving on."

5. **Recommended next step (cheap, decoupled, decisive):** a **weighted-BC ablation with no
   relocation** — upweight/oversample the existing ~3,773 edited trajectories 10–30× in the MLE
   loop and re-check transfer. ~1 day, no algorithm change. It directly answers *is the bottleneck
   BC-averaging or edit-magnitude?* and tells us whether any data-side edit (relocation included)
   can transfer. **Holding for Dr. Zhang's go-ahead before running.**

---

## Idea scoreboard (what was discussed → feasibility)

| # | Meeting-39 idea | Proposed by | Code-grounded verdict |
|---|---|---|---|
| 1 | **Physical relocation** (teleport worst-offenders out of concentration, then let the editor settle) | Robert | **Likely marginal.** Valid only via cross-district F_causal; stage-2 settling contributes ≈0; collides with realism + 1/N dilution. |
| 2 | **Target-locations-first** scan (O(AB) over the grid → top-K POIs → worst trajectories) vs O(N) over trajectories | Dr. Zhang | **Real but non-binding.** Genuine constant-factor speedup (infra already exists in `diagnostics.py`), but the binding limiter is 1/N metric sensitivity, not selection runtime — "marginal," as the meeting concluded. |
| 3 | **Select by gradient magnitude** instead of negative-contribution | Dr. Zhang | **Half-validated.** The "contribution ≈ gradient" proxy holds for the *spatial* term, **fails for the causal** term (a high-\|α_causal\| district-interior cell can have ~0 local gradient). But gradient-selection + teleport is *self-defeating* (the teleport destroys the gradient that justified selection). |
| — | Randomize **location not time** | both | Sound and supported — the active-cell mask + per-(cell,t) count grids exist to do this. |
| — | Gradient heatmap as the verification gate | Robert (done) | Correct gate; tool built at `famail_temporal/visualization/gradient_heatmap/`. The proxy claim in #3 should be read off it before changing any rule. |

---

## 1. The one valid mechanism — why relocation *could* help (and its built-in ceiling)

F_causal = `1 − r²_demo` = `R'(I − H_demo)R / R'MR`, an R² of regressing residual demand on
**district-level** demographics (`fairness/causal.py`, `fairness/hat_matrices.py`). Verified facts:

- Demographics are **constant within each of 10 districts** (distinct value count = 1 per
  district); district sizes range **86–544 cells**. The design matrix uses **3 features** → H_demo
  has **rank 4** (`config.py:45-49`, `hat_matrices.py:106-121`).
- Therefore `(I − H_demo)` is identical for every cell in a district → **the causal gradient is
  flat in the district interior** and only changes sharply when a pickup **crosses a district
  boundary** into a different demographic level.
- The editor's edit radius is an **L∞ ε-ball of 2.0 cells** (`config.py:58`,
  `modifier.py:476-499`). 2 cells ≪ 86–544 → **the gradient editor essentially never crosses a
  boundary**, so F_causal's demographic-explained variance is unreachable by gradient settling
  alone.

➡️ **A physical teleport across a district boundary is the only operation that can move F_causal.**
This is the granularity argument, confirmed mechanically — and the genuine merit of the idea.

**The built-in ceiling:** a teleport that lands in the *same* district (or a "nearby" cell, per the
realism constraint) changes F_causal by ≈0. And because both metrics are **global aggregates over
N = 34,524 active (cell, time-block) units** (`spatial.py`, `hat_matrices.py:227-237`), a single
relocation changes only 2 of N entries → marginal effect **O(1/N) ≈ 2.9e-5**. F_spatial responds
through the **pickup→DSR** half only (the dropoff→ASR half is untouched), and Gini is **sub-linear**
in concentration, so gains shrink as destinations fill toward the mean.

---

## 2. Meeting-39 premises vs. committed code

| Meeting-39 premise | Committed code |
|---|---|
| α = (0.2, 0.7, 0.1); fidelity "tuned down" | **Default (0.33, 0.33, 0.34)** — fidelity is the *largest* weight (`config.py:52-54`). (0.2,0.7,0.1) exists only as a runtime override. |
| Edit limit ≈ 3 cells; trajectories move **0.81 avg / 2.2 max** | Radius is **ε = 2.0 (L∞)** (`config.py:58`). Committed artifacts show **mean ~1.15–1.35, max 2.828** (= 2√2, the ε=2 corner). **The 0.81/2.2 figures were not found in any committed run** — need to confirm which run/config produced them. |
| Contribution score normalized to [−1, +1] | **No [−1,+1] normalization exists** (`algorithm/attribution.py`). Score is the raw signed α_i (≈[−1, 1/N]); ranking uses only sign + order. |
| (implicit) larger gradient → larger edit | Step is `0.1·sign(grad)` (`modifier.py`) — **magnitude doesn't affect step size**, only whether it clears a 1e-8 deadband. Weakens the gradient-magnitude-selection rationale: its only real benefit is skipping fully-stalled (≈0-gradient) targets. |
| Editor places into active cells | Editor does **not** constrain the destination to active cells (only `int()`-truncates to the grid; pre-checks the *origin's* 5×5). A teleport-to-active-cell sampler is **new code** (the active mask exists: `data/active_mask.py`). |

---

## 3. Limitations not (fully) raised in Meeting 39

Ordered by how much they bear on the decision.

1. **BC-transfer futility (the strategic one).** Relocation changes per-edit magnitude, not the
   edited fraction (~3.6%) or the MLE averaging that Level-2 identified as the killer. Teleported
   terminals are off-manifold relative to the (driver, start-context) conditioning, so BC treats
   them as **high-loss outliers to regularize away**, and random destinations inject **label noise**
   into exactly the conditional being fit (raising terminal-cell entropy back toward the raw mean).
   *Implication: improving the editor may not improve Level-2 at all.*

2. **1/N aggregate dilution.** ~50 edits at O(1/N) leverage cannot move a 34,524-unit aggregate
   past the **~0.012-bit seed-noise floor** that already swallowed the Level-1→Level-2 signal.
   Distance ≠ leverage.

3. **Geometry-forced contradiction.** *Teleport far to change F_causal* (must cross a boundary)
   vs *stay nearby for realism* (stays in-district, ΔF_causal ≈ 0) cannot both hold for the
   interior trajectories the heuristic targets.

4. **Fidelity is not blind to teleports.** The discriminator ingests the full seeking sequence with
   only the terminal swapped (`modifier.py:438-450`), so a teleport is a **kinematically
   discontinuous trip** it can flag as fake — while `grad_fidelity ≈ 0` means the editor **cannot
   correct** it. This likely degrades the Fidelity-A/B parity that makes the current Level-2 and
   Level-1 results clean → risks converting "no transfer, no cost" into "no transfer, **with cost**."

5. **Manu's motivation points the opposite way.** "High-demand areas are more fair" is
   correlational; relocation **de-concentrates** the very cells found to be fair, and dense cores
   differ from peripheries on the *same demographic covariates the metric uses* (confounding).

6. **Self-defeating selection.** Gradient-magnitude selection + teleport: the teleport destroys the
   local gradient used to select. The criterion the two-stage design *actually* needs is
   **expected post-relocation ΔF** (a counterfactual source→destination pair score) — never named in
   the meeting.

7. **Top-K-POI misses diffuse causal unfairness.** POIs are a *spatial/concentration* target;
   causal drag is spread across district *interiors* — the scan optimizes the easy (spatial)
   fairness and under-samples the hard (causal) fairness we care about.

8. **Silent-undo implementation hazard.** The ε-cap re-anchors to `true_original`, captured from
   the *pre-teleport* pickup (`editing_loop.py`, `modifier.py:493-498`). If stage-0 doesn't rewrite
   `states[-1]` **and** reset `orig_pos`/`cum_disp` to the destination, every gradient step gets
   clipped **back to a 5×5 box around the original cell** — the relocation becomes a no-op masked by
   the cap. (Flagged so we don't waste a run on it.)

---

## 4. Smoke-test verdict

**Plausibility: likely marginal.** Relocation can make edits visibly larger and the cross-boundary
F_causal mechanism is real, but at a realistic dosage it cannot move the global scalars past the
seed-noise floor, and **nothing in it addresses why BC averaged the signal away.** It targets edit
*magnitude*; the transfer bottleneck is edit *fraction* + the averaging operator.

### Recommended cheapest experiment (decoupled from the whole relocation build)

**Weighted-BC ablation, no relocation.** Re-run the Level-2 paired pipeline but **upweight /
oversample the existing ~3,773 edited trajectories 10–30×** in the MLE loop.

- **If transfer turns positive** → the bottleneck is BC-averaging; redirect the spare month to the
  *actual* transfer levers (importance-weighted / oversampled BC, or a fairness regularizer in the
  policy loss) — all independent of edit magnitude.
- **If transfer stays ≈0 even when edits dominate the loss** → no data-side edit (relocation
  included) can rescue transfer; stop and lock the negative result.
- ~1 day, no algorithm change, and a **new** experiment (not a re-run of the locked Level-2 table).

*(Optional bound on relocation itself: an oracle-ceiling test — force the worst-K terminal cells to
their fairest cross-district active destination in numpy, no editor, recompute data-level F_causal.
Prediction: at the ~50-edit dosage the gain is < ~0.04, i.e. under 3× the noise floor.)*

---

## 5. What this means for the month / the paper

- The paper plan is unchanged: **submit with current results** (data-level Level-1 positive +
  honest Level-2 negative); the editing framework is the contribution.
- The **highest-leverage** use of the remaining time may **not** be improving the trajectory editor
  but **improving the transfer mechanism** (weighted BC / fairness-regularized policy) — same paper
  structure, but a different "if I have time" target than Meeting 39 assumed.
- A defensible **alternative finding**: *the binding limitation is the district-level demographic
  resolution, not the editor.* If realistic local edits provably can't move F_causal, that is itself
  a clean, publishable diagnosis (and motivates cell-level demographic data as future work).

---

## 6. Discussion points for Dr. Zhang

1. Given Level-2 isolated BC-averaging as the bottleneck, do we pivot the spare month from
   relocation to the transfer levers (weighted BC / fairness regularizer)? Or is improving the
   editor's fairness worth pursuing independently of transfer?
2. Is "the limitation is demographic-data granularity, not the editor" a finding we want to frame
   and pursue (cell-level demographics)?
3. Confirm the experimental baseline: which run produced the **0.81 / 2.2** movement figures, and do
   we run any relocation work at committed **(0.33,0.33,0.34)** or the meeting's **(0.2,0.7,0.1)**?
   (Per protocol, changing the selection rule contribution→gradient-magnitude needs sign-off.)
4. Pre-register the success metric **before** any build: relocation "works" only if the **global F
   delta** (ideally the BC-transfer delta) clears the seed-noise floor — **displacement statistics
   must not count as success**.
5. If we do build relocation, do we add hard guardrails — score teleported sequences with the
   existing discriminator; cap destination pile-up — given the fidelity risk?

---

## Appendix — where the claims are grounded

| Area | Files |
|---|---|
| Optimizer / ε-ball / sign-step | `algorithm/modifier.py`, `algorithm/soft_cell_assignment.py`, `config.py:52-58,87,107` |
| Objective (weighted sum) | `algorithm/objective.py:131-160` |
| Selection / attribution / active mask | `algorithm/attribution.py`, `algorithm/editing_loop.py`, `data/active_mask.py`, `data/loader.py:41-44` |
| F_spatial (Gini on DSR/ASR) | `fairness/spatial.py:66-178` |
| F_causal (district-level R²) | `fairness/causal.py:25-158`, `fairness/hat_matrices.py:106-315` |
| Per-cell gradient grid (for the O(AB) scan / proxy check) | `evaluation/diagnostics.py:14-58` |
| Movement instrumentation | `evaluation/cell_histogram_analysis.py:82-143` |
| Level-2 negative result | `baselines/LEVEL2_RESULTS.md` |
| Gradient heatmap explorer | `visualization/gradient_heatmap/` |
