# External Fairness Metrics — Results Overview (Meeting 42)

> Source document for slide assembly. Sections map roughly to slides; tables and bolded
> numbers are the load-bearing content. This doc covers the **completed external-metrics
> results** (the "prove it on metrics we didn't optimize" deliverable). The forward-looking
> fix it motivates — supply-lift editing — is a separate briefing:
> [`supply_lift_briefing.md`](supply_lift_briefing.md). Full detail + numbers:
> `PAPER/external-metrics/` (`FINDINGS.md`, `LEVELING_DOWN_MECHANISM.md`).

---

## 1. The ask (Meeting 41 P0): fairness on metrics we never optimized

Reporting a fairness gain on the metric you *optimize* is circular — "you can gradient-ascend it."
So we computed **four established fairness metrics that are NOT in the objective**, measured
**before-edit → after-edit** on the same edited datasets:

- **Outcome:** per active `(cell, hour)` unit, `Y = supply/demand = active_taxis / max(pickups, 0.5)`
  (higher = better served).
- **Metrics:** **supply/demand ratio** (group levels), **demographic parity** (the group gap),
  **disparate impact** (group ratio, the 0.8 rule), **Theil index** (between-region inequality).
- **Groups:** two ways × three equity axes — *district-extremes* (bottom vs top third of
  neighborhoods) and *median-split*, over **housing price, compensation, migrant share**.
- **Rigor:** paired bootstrap 95% CIs (B=1000). **Four datasets:** Shenzhen ×3 demographic
  feature sets + San Francisco (sf12).

**The point:** improvement here can't be an artifact of what the editor optimized.

## 2. Headline: Shenzhen improves **unanimously**, and it's **robust**

Every axis × both groupings × the Theil index moves **toward fairness**, and **every Δ 95% CI
excludes zero**. The headline cell (migrant axis, district-extremes, PRIMARY feature set):

| Metric | Before | After | Δ (95% CI) | reading |
|---|---:|---:|---:|---|
| **Disparate impact** | 0.3325 | 0.3422 | **+0.0097** [0.0086, 0.0108] | ratio → 1 (fairer) |
| **Demographic parity** (gap) | 14.199 | 13.596 | **−0.603** [−0.667, −0.536] | gap → 0 (fairer) |
| **Theil** (between-region) | 0.1550 | 0.1491 | **−0.0059** [−0.0065, −0.0052] | inequality ↓ (fairer) |

**Robust across all three demographic feature sets** — the gain does **not** depend on which
features the editor optimized (migrant axis, district-extremes):

| Feature set optimized | Theil Δ | DP gap Δ | DI Δ |
|---|---:|---:|---:|
| **{housing, comp, migrant}** (PRIMARY) | −0.0059 | −0.603 | **+0.0097** |
| {housing, gdp, comp} | −0.0056 | −0.571 | +0.0092 |
| {housing, comp, migrant, logpopdensity} | −0.0055 | −0.536 | +0.0086 |

## 3. External validity: SF reproduces the **direction**, weakly

San Francisco (sf12) points the same way but smaller and mixed — an honest, not a clean, replication:

| SF axis / grouping | DI Δ (95% CI) | verdict |
|---|---|---|
| Compensation / district-extremes | +0.0182 [0.0143, 0.0224] | **significant** |
| Compensation / median-split | +0.0160 [0.0125, 0.0198] | **significant** |
| Theil (between-region) | −0.0045 [−0.0058, −0.0035] | **significant** |
| Housing / district-extremes | +0.0056 [0.0001, 0.0109] | barely significant |
| **Migrant / district-extremes** | +0.0034 [−0.0013, 0.0076] | **not significant** |
| **Migrant / median-split** | −0.0007 [−0.0052, 0.0039] | **not significant (≈0)** |

**Say it plainly:** external validity **holds for compensation + Theil**, is **weak for housing**,
and **does not hold for the migrant axis on SF** (partly the 12-driver sample → wide CIs).

## 4. The key finding: the improvement is **leveling-down**

Reporting the supply/demand **group levels** (not just the gap) exposes *how* the gap closes.
Shenzhen PRIMARY, district-extremes — before → after of each group's mean `Y`:

| Axis | Disadvantaged level | Advantaged level | who moves |
|---|---|---|---|
| **Migrant** | 7.0734 → 7.0734 (**+0.000**) | 21.272 → 20.669 (−0.603) | **advantaged ↓; disadvantaged flat** |
| **Compensation** | 7.0734 → 7.0734 (**+0.000**) | 19.046 → 18.563 (−0.482) | **advantaged ↓; disadvantaged flat** |
| **Housing** (over-served group is low-price) | 23.048 → 22.358 (−0.691) | 9.208 → 9.208 (**−0.000**) | **over-served ↓; other flat** |

**General statement:** the editor equalizes by **reducing whichever group is over-served — never
by raising the under-served group.** On Shenzhen the disadvantaged (poor, high-migrant) group's
absolute service is **unchanged** (7.0734 before *and* after). The gap closes **from the top**.
This is the weak form of fairness — **Parfit's leveling-down objection** — and a reviewer *will*
raise it. We raise it first.

## 5. Why it levels down — **structural, proven** (not an optimizer quirk)

Three compounding causes (full analysis + an oracle bound in `LEVELING_DOWN_MECHANISM.md`):

1. **Selection never sees the poor group.** All **2,455 / 2,455** edits originate *and* land in
   advantaged cells — **zero** touch a disadvantaged cell. (Attribution is residual-*variance*-based;
   only over-served cells carry big residuals.)
2. **The demand lever is ~inert on the poor side.** Adding demand to rich cells is **~32×** more
   effective on `Y` than removing it from poor cells, and **93%** of poor units sit at/below the
   demand floor.
3. **The real inequity is supply-side, and supply is frozen.** Median taxi presence: poor **1.8** vs
   rich **17.6** (~10×). The editor moves only demand (pickup location); it has **no supply channel**.

**Oracle bound:** even a *perfect* demand-only editor could raise the poor group only by **deleting
~3k of its recorded pickups** — perverse (it would teach downstream policies to serve poor areas
*less*). So leveling-down is the **constrained optimum**, not a failure. **Option A** (24 trained
policies) confirms the downstream stage doesn't lift up either — upweighted policies serve poor areas
**~7–10% less** (0/6 seeds, p=.031); the policy-level F_causal gain is system-level over-service
trimming. → This is exactly what the **supply-lift** build sets out to fix
([`supply_lift_briefing.md`](supply_lift_briefing.md)).

## 6. Method honesty (say these before a reviewer does)

- **Housing-axis direction is city-dependent.** Shenzhen: low-housing areas are *over*-served
  (DI **2.50 > 1**). SF: *under*-served (DI **0.80 < 1**). So "low housing price = disadvantaged" is
  not a city-invariant proxy for "under-served" — lead with **migrant + compensation**.
- **Demographic parity ≡ the supply/demand gap** by construction (both are `mean_A − mean_D` on
  continuous `Y`). The honest count of distinct external metrics is **{disparate impact, DP/gap,
  Theil} + the group levels** — not four independent statistics.
- **The bootstrap is first-order.** Units aren't iid (spatial correlation; district-constant
  demographics), so CIs understate true uncertainty; a clean driver-level bootstrap isn't available
  (the demand grid is an independent mean-hourly artifact, not a per-driver sum). Disclosed, not hidden.
- **Associational, not causal**; small SF sample (12 drivers).

## 7. The framing this earns

Present the current result as a principled **over-service-reduction ("slack-trimming") fairness
editor** — on Shenzhen it trims idle over-service (no group's absolute recorded service falls) and
improves **metrics it never optimized**, unanimously and robustly. The leveling-down limit is a
**demonstrated property of the demand-only / frozen-supply problem** (that's the contribution, and
the answer to the Parfit objection) — which **motivates the supply-side uplift lever** as the next
step / future work. *"Edit to trim over-service; reroute/augment to lift under-service."*

---

**Provenance:** results + numbers → `PAPER/external-metrics/FINDINGS.md`; mechanism + Option-A →
`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`; figures → `PAPER/external-metrics/figures/`;
the fix it motivates → [`supply_lift_briefing.md`](supply_lift_briefing.md).
