# Why the editor levels down — mechanism analysis + lifting-up options

> **⚠️ TERMINOLOGY NOTE (Meeting 43, 2026-07-16, Dr. Kash):** the manuscript no longer
> describes trim-only as "leveling down" (relocation under conservation ≠ classic leveling
> down); the term survives in the paper only as a cited analogy. This doc predates the ruling
> — its analysis (2,455/2,455 flow, 32× leverage, 93% at floor) remains the provenance for
> the reworded §3.4/§4 prose.

**Status:** analysis DONE (2026-07-07); Option A rollout evaluation DONE (§6.4) — **negative: the
trained policies do not lift up either; the leveling-down propagates through training with a small
dose-dependent perverse drain (poor-area pickup share −10% at w30, 0/6 seeds, p=.031). Strengthens
the supply-side roadmap (§5 B/C).**
**Context:** deep-dive on [`FINDINGS.md`](FINDINGS.md) §4.1 — *"the improvement is leveling-down: the
over-served group is reduced, the under-served group is (nearly) untouched."* Question asked: **why**,
and can we lift up the under-served group instead (or do both)?
**Analysis target:** headline Shenzhen PRIMARY edit
(`famail_temporal/results/2026-06-29T12-06-55_k-10000_causal_emphasis_no-dedup_cleaned_hcm`),
migrant axis, district-extremes grouping. Script: [`scripts/leveling_analysis.py`](scripts/leveling_analysis.py).

---

## 0. TL;DR

Leveling-down is **structural, not an optimizer quirk** — it is the *only* fairness-improving move the
current design can make. Three compounding causes, all verified empirically:

1. **The selection never sees the poor group.** All **2,455 / 2,455** edited pickups originated *and*
   landed in advantaged (low-migrant) cells; **zero** edits touched a disadvantaged cell in either
   direction. The α-attribution is residual-*variance*-based, and only over-served cells carry big
   residuals.
2. **The demand lever is ~inert on the poor side.** `∂Y/∂D = −S/D²` leverage is **~32× larger** for
   adding demand to rich cells than removing it from poor cells, and **93% of poor-group units sit
   at/below `DEMAND_FLOOR`** where removal changes nothing.
3. **The actual inequity is supply-side, and supply is frozen.** Median taxi presence: poor cells
   **1.8** vs rich cells **17.6** (~10×). The editor's only mutable quantity is the pickup location
   (demand); it has no supply channel at all.

**Consequence:** under `Y = supply/demand` with frozen supply, the only way a demand-only editor can
*raise* the under-served group's ratio is to **delete/displace recorded pickups out of poor areas** —
perverse (it would teach downstream policies to serve poor areas *less*). Leveling-down (padding demand
into over-served rich cells) is the constrained optimum, not a failure of optimization. Lifting up
requires a **supply-side lever** (§5).

---

## 1. Setup

- `Y = supply/demand = active_taxis / max(pickups, DEMAND_FLOOR)`, `DEMAND_FLOOR = 0.5`. Higher Y =
  better served. The editor moves only pickups (demand); `active_taxis_3d` (supply) is a frozen
  environmental artifact.
- Groups: migrant axis, district-extremes (D = high-migrant/poor, A = low-migrant/rich, middle
  excluded) — the FINDINGS.md headline cell.
- Code facts (verified):
  - Candidate filter = trajectories with **strictly negative** per-unit attribution αᵢ, ranked
    most-negative first — `famail_temporal/algorithm/attribution.py:121` (`rank_trajectories`,
    ascending) and `:166-168` (`select_top_k` breaks at `score >= 0`).
  - αᵢ is the exact variance decomposition `αᵢ = 1/N − [(MR)ᵢ² − ((I−H)R)ᵢ²]/R'MR` — a *residual
    magnitude* score with no over-/under-served distinction.
  - The modifier mutates **only** `_base_pickup_3d` (unclamped subtract-at-origin/add-at-destination,
    `famail_temporal/algorithm/modifier.py:569-570`); the knowledge graph shows no
    modifier→active_taxis dependency. Y-flooring happens at evaluation
    (`Y = S/max(D, 0.5)`), so the external-metrics reconstruction is Y-equivalent to the modifier's
    own accounting (ΔY_A ties out to 4 decimals: −0.6033).

---

## 2. Evidence

### 2.1 Flow matrix — the selection never touches the poor group

Of the k=10,000 edit budget, only **2,455** trajectories passed the strictly-negative-α filter
(consistent with the post-cleanup edited counts). Their pickup-cell group, origin → destination:

| origin \ destination | D (poor/hi-migrant) | A (rich/lo-migrant) | middle/excluded |
|---|---:|---:|---:|
| **D (poor/hi-migrant)** | 0 | 0 | 0 |
| **A (rich/lo-migrant)** | 0 | **2,450** | 5 |
| **middle/excluded** | 0 | 0 | 0 |

1,495 of 2,455 edits moved to a different cell; all movement is *within* the advantaged group.
Resulting group means (ties out with the official tables):

| group | mean Y before | after | Δ |
|---|---:|---:|---:|
| D | 7.0734 | 7.0734 | **+0.0000** (untouched — literally zero flows) |
| A | 21.2723 | 20.6690 | **−0.6033** |

**Why D is never selected:** under-served cells have high demand and Y ≈ 7 close to the demand-adjusted
prediction g₀(D) → small residuals → αᵢ ≥ 0 → excluded by the strictly-negative filter. Over-served
rich cells (Y ≈ 21 ≫ prediction) carry the variance → most-negative αᵢ → they are the *entire*
candidate set.

### 2.2 Leverage asymmetry — the gradient goes where the response is

`∂Y/∂D = −S/D²` (zero below the floor):

| group | units | mean \|∂Y/∂D\| **add** | mean \|∂Y/∂D\| **remove** | median D | median S | % units ≤ floor (removal inert) |
|---|---:|---:|---:|---:|---:|---:|
| D (poor) | 6,950 | 13.2 | **1.18** | 0.000 | **1.8** | **93.2%** |
| A (rich) | 9,389 | **37.6** | 4.40 | 0.000 | **17.6** | 65.5% |

Adding demand to rich cells is **~32×** more Y-effective than removing it from poor cells. Gradient
ascent therefore pads demand into over-served rich cells — which is exactly the 2,450 within-A moves.
Note also the supply column: **the disparity FAMAIL measures lives in S (1.8 vs 17.6), the one
quantity the editor cannot move.**

### 2.3 Oracle bound — what could a *perfect* demand-only editor do?

Greedy upper bound: remove every removable pickup from D cells (only 4,700 pickups exist in D cells;
3,138 are above-floor/effective) →

- **max Δ mean(Y|D) = +1.54** (vs the observed −0.60 achieved on the A side).

So metric-space lifting-up is not impossible — but it is **perverse**: it means deleting/displacing
~3k recorded pickups out of poor neighborhoods. Pickups are simultaneously the metric's *demand* and
the data's *service delivered*; in the data-augmentation story this trains downstream policies to pick
up in poor areas *less*. **There is no non-perverse lifting-up move available to a demand-only editor.**

### 2.4 A nuance worth stating: the over-service being trimmed is idle slack

The group disparity is a *mean-of-ratios* (7.07 vs 21.27, ~3×), dominated by rich cells where many
taxis idle against near-zero demand (65% of A units below the demand floor). The *ratio-of-sums*
(aggregate S / aggregate D) is far closer: 7.38 vs 9.50. The editor's leveling-down concentrates on
those idle-slack cells — supporting a "slack-trimming; no group's absolute recorded service is
reduced" framing for the current result.

---

## 3. Interpretation for the paper

- **Report precisely:** 100% of edits originate and land in the over-served group; the under-served
  group's absolute recorded service is untouched; between-region inequality (Theil) strictly falls.
  On Shenzhen this is *slack reduction*, and no group's absolute service falls.
- **Engage the leveling-down objection (Parfit) head-on**, then answer with the constrained-optimality
  result: frozen supply + conserved demand + demand floor ⇒ no non-perverse lifting-up direction
  exists for *any* demand-only editor (quantified by the oracle bound). A reviewer attack becomes a
  demonstrated property of the problem — and the motivation for the supply-side roadmap.
- **Elevate the disadvantaged-group level row** (already in the tables) as the explicit "lifting-up
  test": currently Δ = 0, honestly reported.
- **SF footnote:** on SF both groups' levels dipped — different structure (many tract regions,
  boundaries everywhere, so ≤2-cell moves cross group lines); the same flow analysis under
  `FAMAIL_CITY=sf12` would pin it (follow-up).

---

## 4. The lever/metric mismatch (the core insight)

Pickups play two roles at once: in the fairness metric they are **demand** (the denominator); in the
data they are **service delivered**. Any edit that raises a poor area's supply/demand *ratio* by
touching pickups must reduce its recorded *service*. Only **supply** — taxi seeking presence — can
raise the ratio while also increasing real service. The current editor has no supply lever.

---

## 5. Options to lift up (ranked)

- **A. Supply-endogenous rollout evaluation** *(no algorithm change; run now — §6)*. FAMAIL is a
  data-augmentation method; the deliverable is the *policy trained on the edited data*. Rollouts
  generate whole seeking trajectories, so at rollout time **supply is endogenous**. Measure whether
  the policy trained on edited+upweighted data allocates more seeking presence / pickups to
  under-served cells than the raw-trained policy. If yes → material, system-level lifting-up (the
  right answer to the objection). If null → motivates B/C. Either outcome is publishable evidence.
  *Honest prior:* the edited data itself moves nothing into D (§2.1), so a positive result would be
  emergent from upweighting + generalization — this is a genuine hypothesis test, not a confirmation.
- **B. Supply-aware editing ("seeking-tail rerouting")** *(the principled fix; algorithm extension;
  PI + protocol gate; likely post-KDD)*. Extend the editor to move the last few *seeking* states with
  the pickup and make S endogenous via a differentiable ΔS channel (the same soft-assignment trick,
  applied to the 5×5 active-taxis aggregation). Gives the objective a non-perverse lever: seeking time
  routed into under-served cells raises their S → Y↑ **and** adds poor-area service demonstrations for
  downstream learning — genuine lifting-up, combinable with the current demand-side trimming ("both at
  once"). Bonuses: the fidelity discriminator actually scores the seeking stream (the fidelity term
  becomes load-bearing), and it cleanly differentiates FAMAIL from ST-iFGSM.
- **C. Supply augmentation** *(augmentation-native; cheaper than B)*. *Add* fidelity-screened synthetic
  seeking trajectories routed into under-served areas (copy-and-reroute real ones or BC-generated),
  accounting their ΔS at evaluation. Propose → fidelity-score → fairness-gain → greedy-accept; no
  differentiable supply channel needed. Narrative: *"edit to trim over-service; augment to lift
  under-service."*
- **Anti-recommendations** (documented so they are not relitigated):
  - *Both-sided demand transfer* (stratified selection moving poor-boundary pickups into rich cells):
    statistically lifts Y|D (up to the +1.54 oracle) by displacing recorded service out of poor
    areas — normatively indefensible.
  - *Asymmetric/one-sided F_causal alone*: redirects the objective, but the only lever it can pull is
    still deleting poor-area demand. Objective asymmetry helps only *paired with* a supply lever
    (where it would steer B/C toward lifting up rather than more trimming).

---

## 6. Option A experiment — supply-endogenous rollout evaluation

**Question:** do policies trained on edited+upweighted data reposition *seeking supply* and *pickups*
toward under-served areas, relative to raw-trained policies?

### 6.1 Design (protocol-faithful to the published weighted-BC sweep)

- **Arms:** `raw`, `edited` (w=1 control), `edited_w10`, `edited_w30` × **seeds 0–5** — identical
  protocol to `weighted_bc_sweep/cleaned_hcm_6seed` (driver-conditioned `TrajectoryLSTM`,
  `train_mle` 20 epochs, lr 1e-3, batch 32, max_batch_tokens 8192; upweighting = loss weights on the
  2,455 edited trajectories; edit dir = the same hcm headline edit). No checkpoints exist from the
  original sweep (policies were trained → evaluated → discarded), so each policy is **retrained
  deterministically** (`set_all_seeds(seed)`).
- **Rollouts:** `generate_trajectories` (full seeking sequences, flat cell ids, terminal = pickup),
  **corpus-matched contexts identical across arms** (one rollout per real trajectory, same driver +
  start cell + start t_block) — so allocation differences are policy-driven, not seed-context-driven.
- **Metrics** per (arm, seed), per axis (migrant/comp/housing, district-extremes cell grouping):
  **share of generated state-visits** (endogenous supply allocation) and **share of terminal pickups**
  (service allocation) landing in D / A / middle cells; supply-per-pickup ratio-of-sums per group as a
  secondary. Paired per-seed Δ(arm − raw), Wilcoxon across the 6 seeds.
- **Script:** [`scripts/option_a_rollout_eval.py`](scripts/option_a_rollout_eval.py). Outputs
  (gitignored): `famail_temporal/baselines/external_fairness/results/option_a_rollout/`
  (per-policy JSON, crash-resumable — re-running skips completed policies; `summary.json`).

### 6.2 Reference rows — the training data itself (computed, real numbers)

Allocation shares of the **training corpora** (migrant axis, district-extremes; 95,297 trajectories,
2.14M states):

| corpus | states share_D | states share_A | pickups share_D | pickups share_A |
|---|---:|---:|---:|---:|
| raw | 0.0696 | 0.6589 | 0.0494 | 0.7455 |
| edited | 0.0696 | 0.6589 | 0.0494 | 0.7454 |

Two implications: (i) the corpus is heavily rich-area-concentrated to begin with (**4.9% of pickups /
7.0% of seeking states** in disadvantaged cells vs ~66–75% in advantaged); (ii) **the edit signal
contains ≈zero D-allocation change** (identical to 4 decimals — the §2.1 within-A moves). So any
policy-level shift toward D would be *emergent* from upweighting + generalization, and a null is the
structurally expected outcome. Either result is informative: positive → material system-level
lifting-up exists; null → quantifies that the current pipeline nowhere lifts up, strengthening the
case for the supply-side extension (§5 B/C).

### 6.3 Status — pipeline validated; full run blocked on GPU availability

- Plumbing **validated end-to-end** via CPU smoke (2 arms × 1 seed × 1 epoch × 1,500-trajectory
  slice: train → rollout → shares → JSON all pass; smoke artifacts are marked `smoke` and excluded
  from aggregation).
- The full 24-policy run needs the GPU. **CUDA is currently unavailable in this WSL environment**
  (torch: "Found no NVIDIA driver"; NVML init fails; `/dev/dxg` + `libcuda.so` present — the classic
  stale-WSL-VM state; the original sweeps ran on this same box/RTX 3070). CPU is infeasible at
  protocol fidelity (measured ≈34 h/policy). Fix is Windows-side: `wsl --shutdown` from
  PowerShell, reopen, confirm `python -c "import torch; print(torch.cuda.is_available())"` → True.
- **Launch command once GPU is back** (~2.5–3.5 h; resumable; daemonized):

```bash
cd /home/robert/FAMAIL && setsid nohup python PAPER/external-metrics/scripts/option_a_rollout_eval.py \
  > famail_temporal/baselines/external_fairness/results/option_a_rollout/nohup.log 2>&1 &
```

### 6.4 RESULTS (2026-07-07, 24 policies, run complete) — **answer: NO lifting-up; the leveling-down propagates through training, with a small perverse drain**

**Absolute allocation levels** (migrant axis, mean ± std over 6 seeds; corpus reference below).
Right column = the published rollout ΔF_causal for the *same arms/protocol/edit dir*
(`weighted_bc_sweep/cleaned_hcm_6seed/paired_stats.json`):

| arm | states share_D (supply) | pickups share_D (service) | pickups share_A | published rollout ΔF_causal vs raw |
|---|---:|---:|---:|---:|
| raw | 0.0699 ± 0.0040 | 0.0500 ± 0.0004 | 0.7458 ± 0.0036 | — |
| edited (w=1) | 0.0694 ± 0.0035 | 0.0503 ± 0.0006 | 0.7448 ± 0.0031 | −0.0012 (n.s.) |
| edited_w10 | 0.0675 ± 0.0024 | **0.0467 ± 0.0003** | 0.7514 ± 0.0041 | **+0.0205** (p=.031) |
| edited_w30 | 0.0688 ± 0.0026 | **0.0452 ± 0.0011** | **0.7580 ± 0.0029** | **+0.0311** (p=.031) |
| *training corpus* | *0.0696* | *0.0494* | *0.7455* | — |

**Paired per-seed deltas vs raw** (migrant axis):

| arm | Δ pickups share_D | seeds positive | Wilcoxon p | Δ states share_D | p |
|---|---:|---:|---:|---:|---:|
| edited (w=1) | +0.0003 | 3/6 | 0.56 | −0.0005 | 0.84 |
| edited_w10 | **−0.0033** | **0/6** | **0.031** | −0.0024 | 0.094 |
| edited_w30 | **−0.0048** | **0/6** | **0.031** | −0.0011 | 0.56 |

**Findings:**

1. **No lifting-up anywhere.** Seeking-supply allocation to disadvantaged areas is flat
   (Δ n.s. in all arms). The policy does not reposition supply toward under-served areas.
2. **The upweighted policies serve poor areas *less*, dose-dependently.** Pickup share to
   disadvantaged cells falls ~7% (w10) → ~10% (w30) *relative*, 0/6 seeds positive, p = 0.031 (the
   n=6 minimum) — while pickup share to advantaged cells **rises** (+1.2 pp at w30). Since rollouts
   are a fixed budget (95,297, corpus-matched contexts identical across arms), allocation is
   zero-sum: the fair-trained policy redirects trips toward rich areas at the expense of poor and
   middle areas. The vanilla-edited (w=1) arm moves nothing — the effect is specifically the
   **upweighting** amplifying the edit signal, which (§2.1) consists entirely of rich-area pickups
   relocated within rich areas.
3. **The published policy-level F_causal gain is system-level leveling-down.** The same policies
   whose rollout F_causal improves +0.021/+0.031 achieve it by padding pickups into over-served rich
   cells (their Y ↓ toward parity) and *reducing* poor-area pickup share. The endogenous
   supply-per-pickup ratio in D rises 24.7 → 27.9 (w30) — but decomposition shows the rise is
   **denominator shrinkage** (fewer poor-area pickups), not supply moving in: the §2.3 "perverse
   route" materialized at the policy level. The ratio metric improves while generated *service* to
   poor areas declines.
4. **Corroborating wrinkle (housing axis):** pickup share to *low-housing* cells rises with dose
   (+0.0054 at w30, 6/6, p=.031). On Shenzhen the low-housing group is **over-served** (DI 2.5 > 1,
   §4.2 of FINDINGS.md) — so this is the *same* mechanism (demand padded into over-served cells),
   not a lift of a deprived group. The comp axis mirrors migrant exactly (coincident districts).

**Implication.** Option A returns a clean negative: **no stage of the current pipeline — data edit or
trained policy — lifts up the under-served group, and the policy stage adds a small perverse drain.**
This (i) is a load-bearing caveat on the Pillar-2 interpretation (the rollout F_causal gain should be
described as system-level over-service trimming, not increased service to under-served areas → PI
discussion), and (ii) decisively strengthens the case for a **supply-side lever** (§5 B: seeking-tail
rerouting; §5 C: supply augmentation) as the roadmap for genuine lifting-up. Negative-but-informative;
n=6 with the same protocol/error-bar standard as the published sweep.

**Artifacts:** per-policy JSON + `summary.json` + `corpus_refs.json` under
`famail_temporal/baselines/external_fairness/results/option_a_rollout/` (gitignored); run log
`run.log` (~3h55m on the RTX 3070, 20 epochs × 24 policies, ~575 s each).

---

## 7. Provenance

- Analysis script: [`scripts/leveling_analysis.py`](scripts/leveling_analysis.py) (flow matrix,
  leverage, oracle; reproducible against the edit dir named at top).
- Code facts: `famail_temporal/algorithm/attribution.py` (selection),
  `famail_temporal/algorithm/modifier.py:569-570` (demand-only mutation),
  `famail_temporal/fairness/causal.py` (α decomposition).
- Numbers in §2 produced 2026-07-07 on `main` (post `1d534c3`), Shenzhen PRIMARY cleaned data.
