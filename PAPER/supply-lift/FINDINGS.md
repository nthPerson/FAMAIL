# Supply-lift editing — FINDINGS

**Status:** built + validated on branch `supply-lift-editing` (2026-07-07 → 07-09). Headline datasets are
the **filtered** Shenzhen PRIMARY and SF sf12 supply-lift runs. **One eval is still in flight:** the
downstream rollout-allocation eval (§9) is re-running as of this writing — that section is a marked stub.

**Motivation:** the trajectory editor's published mechanism ("trim" — relocate a recorded pickup) improves
fairness only by **leveling down** (it removes service from the over-served group and never adds service to
the under-served group; see [`PAPER/external-metrics/`](../external-metrics/) FINDINGS §4.1 and
[`LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md)). Supply-lift adds a **supply
lever**: it reroutes a trajectory's *seeking tail* toward under-served cells, with the supply grid made
**endogenous** to the objective. This bundle is the curated result.

**One-line result:** On **Shenzhen PRIMARY (filtered)** F_causal rises **+0.0222** (vs **+0.0144** trim-only)
and — for the first time in this line of work — the **under-served (migrant) group's service ratio rises**
(`mean(Y|D)` +0.047, CI excludes 0), with the **supply component of that rise statistically significant on
both cities** (SZ +0.009 tier-1 / +0.024 tier-2; SF +0.020). It is **no longer pure leveling-down.** On
**SF** the same supply channel replicates and every *external* metric improves (incl. the migrant axis, which
trim-only could not move), but the **total** `mean(Y|D)` moves **negative** because lift also reroutes pickups
into under-served cells (demand there rises); §5.2 presents both readings without resolving them.

---

## 1. Motivation chain (why this workstream exists)

Three findings from the preceding week set the target; full detail in
[`LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md).

1. **Leveling-down is structural, not an optimizer quirk.** On the published Shenzhen PRIMARY trim edit, all
   **2,455 / 2,455** edited pickups originated *and* landed in advantaged (low-migrant) cells — **zero** edits
   touched a disadvantaged cell. Under `Y = supply/demand` with **supply frozen**, `dY/dD = -S/D^2` leverage is
   ~**32x** larger for padding demand into rich cells than removing it from poor cells, and **93%** of poor-group
   units sit at the `DEMAND_FLOOR` where removal changes nothing. Median taxi presence: poor cells **1.8** vs
   rich **17.6** (~10x). A demand-only editor's only way to *raise* the poor ratio is to delete recorded pickups
   from poor areas — perverse. Leveling-down (padding demand into over-served cells) is the constrained optimum.

2. **The leveling-down propagates through training — negative rollout eval.** BC policies trained on trim-edited
   data allocated *fewer* pickups to poor areas, dose-dependently: raw poor-area pickup share **0.0500 -> 0.0452**
   at w30 (**-10%**, **0/6** seeds positive, Wilcoxon **p = .031**);
   [`data/rollout_trimonly_prior_summary.json`](data/rollout_trimonly_prior_summary.json). Even the arm whose
   rollout F_causal *improves* (+0.031) achieves it by padding pickups into over-served cells.

3. **Stage-0 oracle gate cleared the build (G0).** A greedy upper bound on `delta mean(Y|D)` from tail rerouting
   ([`data/oracle.json`](data/oracle.json), Task 1): baseline `mean(Y|D)` = 7.0734,
   **ceiling_fraction = +0.882**, **ceiling_distinct-seeking = +0.827**, vs a go/no-go threshold of **+0.3**
   (N_D = 6,950; 5,538 candidates; 2,571 applied; ~16 s). A supply-channel-**only** greedy variant clears +0.3
   by 2.6x (**+0.786**, controller decomposition). The oracle is a generous ceiling: a realism-constrained editor
   lands well below it (see §3 headline vs §1 ceiling).

---

## 2. Method (summary; full spec + code in §11)

- **Two edit modes.** `trim` = the existing mechanism (relocate the pickup); its optimization path is
  byte-identical to the published editor (gate **G1**). `lift` = new: candidate trajectories are scored by the
  objective's **supply gradient** `dL/dS`, and the optimizer translates the **whole seeking tail** (last
  `TAIL_LEN=4` states + pickup) with **linearly tapered offsets** `[0.25, 0.5, 0.75, 1.0]`, so the trajectory
  physically cruises through the under-served area before pickup.
- **Endogenous supply (delta-S).** Each tail state contributes presence mass (1 state = 1/12 hourly presence, over
  a 5x5 neighborhood) to a **differentiable delta-S grid** that feeds `F_spatial` and `F_causal` *during*
  optimization — the editor sees the supply consequences of its own moves.
- **Adjacency repair (new data-quality property).** The legacy editor silently broke the king-move rule
  (`max(|dx|,|dy|) <= 1`) on every >=2-cell move. Tail translation includes a **provably-exact repair** (reviewer
  brute-forced 3,000 cases: returns "infeasible" *exactly* when no compliant assignment exists).
- **Skip-on-infeasible rule.** An edit is applied **only when a king-compliant repair exists**; otherwise it is
  skipped. Lift skips natively; trim was made symmetric by a post-process filter (§8).
- **Budget split.** Shenzhen k = 10,000: trim keeps its published selection (2,455, -> 2,340 after filtering),
  lift fills the rest (7,545). SF k = 2,000: trim 1,371 -> 1,324, lift 629. Trim's pickups/demand grid reproduce
  legacy exactly (verified mid-run: the combined run reproduced the published trim numbers before lift began).

---

## 3. Shenzhen PRIMARY (filtered) — headline

`data/shz_primary_filtered_metrics.json`, `data/external_fairness_shz_primary_filtered.json`,
[`tables/shenzhen-primary-filtered.md`](tables/shenzhen-primary-filtered.md). Baseline is the same cleaned
Shenzhen data as the trim-only headline in `PAPER/external-metrics/` (identical "before" values).

| Metric | Before | After | Delta | vs trim-only |
|---|---:|---:|---:|---:|
| **F_causal** (optimization label, 1 = fairest) | 0.79880 | **0.82101** | **+0.02222** | +0.0144 |
| F_spatial | 0.10343 | 0.10978 | **+0.00636** | ~0 (trim-only -0.0002) |
| Theil (between-region inequality, lower better) | 0.1550 | **0.1468** | **-0.0082** · CI [-0.0092, -0.0072] | — |
| gini_dsr (lower better) | 0.89809 | 0.88556 | **-0.01254** | — |
| `mean(Y|D)` migrant, district-extremes | 7.0734 | **7.1203** | **+0.0468** · CI [+0.0022, +0.0932] | ~0 (untouched under trim-only) |
| SDR gap (migrant, adv - dis) | 14.1989 | **13.3412** | -0.8576 · CI [-0.9603, -0.7573] | narrowed by leveling-down only |
| Disparate impact (migrant) | 0.3325 | 0.3480 | +0.0155 · CI [+0.0128, +0.0182] | +0.0097 |

**The key qualitative change.** Under trim-only the disadvantaged (high-migrant) group's absolute service was
*flat* (7.0734 -> 7.0734). Under supply-lift it **rises** (7.0734 -> 7.1203, +0.047, CI excludes 0). Both groups
still move toward each other — the advantaged group also falls (21.2723 -> 20.4615, -0.811) — so **numerically
the gap still closes mostly by leveling-down**, but the disadvantaged group is **no longer untouched**: there is
now a small, statistically robust lift-up component. That component is the supply channel (§4). Magnitudes are
small by design (the editor moves each pickup at most eps = 2 cells); the value is **direction + significance +
the first non-flat under-served level**, not magnitude.

---

## 4. The channel decomposition (the framing result)

`data/shz_primary_filtered_channel_decomposition.json`. `delta mean(Y|D)` is one metric evaluated at three points —
`Y(S,D) -> Y(S,D') -> Y(S',D')` — so the edit's demand move (pickups relocate) and supply move (tails relocate)
split **exactly**. Migrant axis, district-extremes, N_D = 6,950, paired bootstrap B = 2,000, seed 0:

| Channel | Delta | 95% CI | Significant? |
|---|---:|---:|:--:|
| **Supply** (tails add presence to D cells) | **+0.00910** | [+0.00535, +0.01296] | **YES** |
| **Supply, tier-2 distinct-taxi recount** | **+0.02421** | [+0.02076, +0.02788] | **YES** |
| Demand (pickup relocation) | +0.03775 | [-0.00651, +0.08363] | no |
| Total (tier-1) | +0.04685 | [+0.00222, +0.09318] | yes |
| Total (tier-2) | +0.06195 | [+0.01671, +0.10755] | yes |
| Supply-first (robustness ordering) | +0.01049 | [+0.00677, +0.01428] | YES |
| Demand-second (robustness ordering) | +0.03635 | [-0.00744, +0.08205] | no |

**Lead with the supply channel.** It is significant under **both** the internal fractional-presence convention
(tier-1) *and* the honest distinct-taxi recount (tier-2), while the **demand** channel — which rides the
`DEMAND_FLOOR` relocation and carries huge variance — is **not** significant. The defensible "lifting-up" claim
is therefore: *rerouting seeking behavior adds real, distinct-taxi-verified supply to under-served areas,
producing a small but statistically robust rise in their service ratio.*

**What tier-2 means.** Tier-2 re-counts supply from raw GPS as **distinct taxis per neighborhood-hour** with the
edited tails substituted (the honest semantics; the editor's internal convention is a fractional-presence
approximation). The recount machinery (`data/shz_primary_filtered_supply_recount.json`, gate **G2**) reproduces
the production supply grid **exactly** (MAE **0.0**, corr 1.0 over 34,524 cells) and matched **100%** of the
9,885 edit histories to raw pings (9,885/9,885, 0 unmatched). Under the honest count the supply effect is
**~2.7x larger** than the internal convention credits (+0.0242 vs +0.0091). *Tier convention caveat:* when citing
a single supply number, state which tier — +0.009 (tier-1, internal) vs +0.024 (tier-2, distinct-taxi).

---

## 5. SF / Cabspotting (filtered) — second city

`data/sf12_filtered_metrics.json`, `data/external_fairness_sf12_filtered.json`,
[`tables/sf12-filtered.md`](tables/sf12-filtered.md).

### 5.1 Every external metric improves; the supply channel replicates

| Metric | Before | After | Delta | note |
|---|---:|---:|---:|---|
| **F_causal** | 0.87515 | **0.90792** | **+0.03277** | trim-only published +0.0139 -> **~2.4x** |
| F_spatial | 0.18463 | 0.20265 | +0.01802 | — |
| gini_dsr (lower better) | 0.82657 | 0.78949 | -0.03708 | — |
| Theil (lower better) | 0.2137 | 0.2056 | **-0.0081** · CI [-0.0095, -0.0067] | significant |
| Migrant DP gap (district-extremes) | 2.1466 | 2.0708 | **-0.0758** · CI [-0.1141, -0.0317] | **was n.s. under trim-only** |
| Migrant DI (district-extremes) | 0.7076 | 0.7137 | **+0.0061** · CI [+0.0012, +0.0105] | new, toward parity |

(Under the `median_split` convention the SF migrant axis remains **n.s.** — DP Delta -0.0341, CI [-0.0693, +0.0021]
— matching the previously-published SF caveat; district-extremes is the primary convention.)

**Supply channel (SF, migrant, N_D = 1,350, B = 2,000):** `data/sf12_filtered_channel_decomposition.json`

| Channel | Delta | 95% CI | Significant? |
|---|---:|---:|:--:|
| **Supply** | **+0.01947** | [+0.01115, +0.02786] | **YES (positive — core claim replicates)** |
| Demand | -0.05248 | [-0.07768, -0.03020] | **YES (negative)** |
| **Total** | **-0.03302** | [-0.05990, -0.00905] | **YES (negative)** |

### 5.2 The SF cross-metric tension (demand endogeneity — presented, not resolved)

On SF the two readings genuinely diverge, and we surface both without adjudicating:

- **Reading A (ratio metric).** `mean(Y|D)` for the migrant group moves **net negative** (-0.0330, CI excludes 0):
  the supply channel is positive as designed (+0.0195), but lift *also* reroutes pickups **into** under-served
  cells, so demand there **rises** and the demand channel is a larger negative (-0.0525). The S/D **ratio** for the
  poor group falls. Under this reading the SF `mean(Y|D)` lifting-up claim does **not** hold and should be withheld.
- **Reading B (external / demand-side fairness).** Every established metric — DP, DI, **and** Theil — improves
  significantly, **including the migrant axis that trim-only editing could not move**. More rides *served* in poor
  areas is a demand-side parity gain (DP measures pickup-share parity) even as it lowers the ratio. This connects
  directly to the **demand-endogeneity** argument in the objective-motivation write-up: recorded demand is
  suppressed by under-supply, so raising served demand in poor areas is the intended effect.

This is **not** a violator artifact — the net-negative total was present on the raw (pre-filter) SF dir (-0.0363)
and **persists after filtering** (-0.0330). **Decision needed (PI):** which SF fairness story goes in the paper —
external metrics (uniformly better, ratio caveated) or hold the SF `mean(Y|D)` claim entirely. **Shenzhen needs no
such caveat** (its demand channel is n.s., not negative).

---

## 6. Fidelity

Gate **G5** (`.superpowers/sdd/g5-fidelity-report.md`, unfiltered mode split) + the filtered-corpus re-check
(`.superpowers/sdd/task-11a-report.md`). Shenzhen; SF supply-lift fidelity was not separately evaluated (§10 deferral).

**Fidelity-A (HuMID driver-identity, higher better) — STABLE.** Real-anchored validation gate **PASSED** (matched
0.849 vs mismatched 0.192, margin >> 0.20). Filtered corpus: edited-combined **0.8457** vs raw 0.8489 (Delta
**-0.0033**), consistent with the published trim-only delta of -0.002. Mode split (G5, unfiltered): **trim-mode
-0.0059** vs **lift-mode -0.0031** — lift edits, despite moving the whole seeking tail, are **no more**
identity-damaging than trim (if anything slightly *less*).

**Fidelity-B (discriminator-free distributional, JS bits, lower better) — a by-design trade-off.** Lift's
whole-tail relocation shifts trajectory-shape statistics more than trim's pickup-only move: lift-mode **0.2645** vs
trim-mode **0.1601** (~1.65x), both above the published trim-only reference 0.1689. This is the expected
distributional cost of moving seeking tails and should be **disclosed as a trade-off**, not treated as
disqualifying — G5's stability criterion is on the identity axis (Fidelity-A), which holds.

---

## 7. Weighted-BC sweep — Pillar 2 on the filtered supply-lift corpus

`data/weighted_bc_paired_stats.json`, `data/weighted_bc_dose_response.json`, `data/weighted_bc_manifest.json`.
Full published protocol: 10 arms x 6 seeds, paired edited - raw, Wilcoxon (identity gate passed: matched 0.8475 /
mismatched 0.1931).

| Arm | Delta F_causal | Wilcoxon p | verdict |
|---|---:|---:|---|
| edited, w = 1 (vanilla BC) | +0.0023 | .156 | **n.s.** — vanilla BC averages the edit away |
| edited, w = 10 | **+0.0232** | .031 (6/6) | significant |
| edited, w = 20 | **+0.0280** | .031 (6/6) | significant |
| edited, w = 30 | **+0.0310** | .031 (6/6) | significant |
| random placebo, w10 / w30 | -0.0011 / -0.0027 | .56 / .094 | **null** — gain is edit-specific, not oversampling |
| most-fair select, w10 / w20 / w30 | +0.0034 / +0.0014 / +0.0007 | .094 / .56 / .84 | **null** — not a selection artifact |

**New vs trim-only: F_spatial now ALSO propagates.** Delta F_spatial is positive and significant at every weight —
**+0.0042** (w10) -> **+0.0048** (w20) -> **+0.0057** (w30), all p = .031, 6/6 — which the trim-only corpus never
achieved. Fidelity-A is essentially unchanged (paired diffs +0.0001 -> +0.0006; magnitudes within noise).
Fidelity-B rises modestly with weight (paired diff **+0.0157** at w30, p = .031) — the same trade-off family as the
data-level Fidelity-B note in §6.

---

## 8. The skip-on-infeasible rule (disclosure)

`data/shz_primary_filtered_PROVENANCE.md`, `data/sf12_filtered_PROVENANCE.md`.

- **Rule-first.** We adopted the uniform rule *"an edit is applied only when a king-compliant repair exists"* by
  **user decision on 2026-07-08**, symmetric with lift (which already skips on infeasible). The decision **precedes**
  the filtered metric numbers (T11a). ~5% of trim edits fall back to the legacy adjacency-violating move when their
  tapered-tail repair is infeasible (**115 / 2,455** on Shenzhen; **47 / 1,371** on SF); those trajectories are
  reverted to their originals.
- **Favorable-direction disclosure.** The rule was adopted before its effects were computed, but it moved **every**
  metric favorably (Shenzhen F_causal +0.0209 -> **+0.0222**; SF +0.0223 -> **+0.0328** — the infeasible edits were
  actively counterproductive, more so on the small 32x30 SF grid). We disclose this rather than bury it: the rule is
  principled, and the decision timestamp precedes the numbers.
- **Replay-exact identification.** Violators are identified by **replaying the modifier's own fallback decision**
  (`apply_tail_perturbation(delta_int, TAIL_LEN, GRID_DIMS)` on the original returns `None`) — the exact condition
  under which the editor fell back, exact by construction and city-robust. Shenzhen: exactly the **same 115** on
  re-derivation (locked as a regression test); **100%** absolute king-compliance after filtering. SF: 47 reverted;
  **edit-relative 100%** (no edit introduces a new violation), absolute **87.40%** (raw-corpus baseline 84.95% — the
  edited corpus is *more* compliant than raw) because **14.9%** of SF's raw trajectories pre-violate adjacency
  (Cabspotting GPS gaps up to 18.6 cells — a source-data property, §10).
- **delta-S integrity.** The filtered delta-S grid is rebuilt **from scratch** from the surviving histories and
  matches the persisted `delta_supply_3d.npz` **bit-for-bit** (max abs diff **0.0**, both cities). **Survivors were
  not re-optimized** after removing the reverted trims (approved trade-off vs a multi-hour GPU re-run; the reverts
  are pickup-only single-cell-mass moves, coupling negligible) — the filtered grids are exact for "these surviving
  edits applied to base," which is not byte-identical to a from-scratch skip-on-infeasible run.

---

## 9. Downstream rollout-allocation eval — **PENDING (running now)**

> **TODO — RESULTS NOT YET IN.** The rollout-allocation eval (renamed from "Option A"; policy-rollout allocation
> shares) is **re-running as of 2026-07-09** on the filtered supply-lift corpus — the smoke passed and the full
> 6-seed run launched (`famail_temporal/baselines/external_fairness/results/option_a_rollout_supplylift/`,
> `run.log`). It must be re-run because the training data changed under supply-lift.
>
> **Prior trim-only result was NEGATIVE** and is the bar to beat (§1.2,
> [`data/rollout_trimonly_prior_summary.json`](data/rollout_trimonly_prior_summary.json)): raw poor-area pickup
> share **0.0500 -> 0.0452** at w30 (**-0.0048**, ~-10%, **0/6** seeds positive, Wilcoxon **p = .031**) —
> trim-edited policies allocated *fewer* pickups to poor areas, dose-dependently. The open question this eval
> answers: do policies trained on **supply-lift** data serve or drain poor areas? **Fill this section when the run
> completes.**

---

## 10. Limitations / deferrals (for the paper's limitations section)

- **Demand channel is n.s. (Shenzhen) / negative (SF).** The defensible lifting-up claim rides the **supply**
  channel only. On SF the total `mean(Y|D)` is net-negative (§5.2) — a genuine, un-spun result requiring a PI
  framing decision before any SF ratio-metric claim.
- **SF tier-2 recount plumbing deferred.** SF's supply channel is the internal-convention **tier-1** number only;
  the distinct-taxi raw-GPS recount (which on Shenzhen made the supply effect ~2.7x larger and significant) is not
  yet wired for SF's pipeline. So the SF supply number is a lower bound of the honest count, not the honest count.
- **SF raw adjacency data-quality.** 14.9% of SF's raw Cabspotting trajectories violate king-move adjacency (GPS
  gaps up to 18.6 cells) — a source-data property. Absolute king-compliance is therefore unattainable on SF by
  construction; the cross-city gate is **edit-relative** compliance (100% post-filter). One caveat sentence in the paper.
- **Survivors not re-optimized** after the skip-on-infeasible revert (§8) — approved, negligible-coupling trade-off,
  not byte-identical to a from-scratch run.
- **2 alternate Shenzhen feature-set runs deferred** (`{housing,gdp,comp}`, `{housing,comp,migrant,logpopdensity}`)
  — robustness parity with the trim-only 3-feature-set sweep; idle-GPU work, not yet run.
- **SF supply-lift fidelity not separately evaluated.** Fidelity (§6) is Shenzhen-only for supply-lift; the SF
  Fidelity-A 0.958 reported in `PAPER/second-dataset/` is for the trim-only dual-claim edit, a different corpus.
- **First-order bootstrap.** Unit-level paired bootstrap; units are spatially correlated and demographics are
  district-constant, so CIs are first-order (inherited caveat from `PAPER/external-metrics/` FINDINGS §5).
- **Oracle is a generous ceiling** (§1.3): a realism-constrained editor lands well below the +0.88 upper bound.

---

## 11. Provenance

- **Design / methodology:** `docs/superpowers/specs/2026-07-08-supply-lift-editing-design.md`
- **Implementation recipe (reproducible):** `docs/superpowers/plans/2026-07-08-supply-lift-editing.md`
- **Full execution ledger (every number's origin, incidents, reviews):** `.superpowers/sdd/progress.md`;
  task reports `task-1-report.md` (oracle), `task-11a-report.md` (filter + batteries + SF),
  `task-11e-sf-eval-report.md` (SF diagnosis), `g5-fidelity-report.md`, `perf-lift-investigation.md`.
- **Presentation summary (numbers verified):** `docs/presentations/meeting_42_update/SUPPLY_LIFT_UPDATE.md`.
- **Per-claim -> source-file map:** [`data_provenance.md`](data_provenance.md).
- **Motivation deep-dive:** [`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md)
  (incl. §6.4 rollout-eval result).
- **Process:** superpowers brainstorm -> spec -> plan -> subagent-driven execution; two hard human checkpoints
  (oracle gate G0; full gate package G2/G4/G5/G6); fresh-agent review of every task; three float32-residual
  production incidents found and fixed with reviewed regression tests.
