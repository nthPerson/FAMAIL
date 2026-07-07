# External fairness metrics (before → after edit) — FINDINGS

**Status:** DONE (2026-07-06). Built + run on branch `external-fairness-metrics`.
**Motivation:** Meeting 41 (2026-07-02) P0 — *"the big one."* Prove the editor improves fairness on
**established metrics that are NOT in its objective**, so the claim is not self-certifying
("you can gradient-ascend the metric you optimize"). See [[meeting41-plan]].

**One-line result:** On **Shenzhen** the external metrics improve **unanimously** (every axis × both
groupings × Theil, all 95% CIs exclude 0) and **robustly across all 3 demographic feature sets**. On
**SF sf12** the direction reproduces (Theil + compensation-axis significant) but is **weaker**, and the
**migrant axis is not significant**. **Caveat (load-bearing):** the gains are achieved by
**reducing the over-served group's service, not by raising the under-served group's** (see §4.1).

---

## 1. Method (summary; full spec + code linked in §7)

- **Outcome variable:** per active `(cell,t)` unit, `Y = supply/demand = active_taxis / max(pickups, 0.5)`.
  Higher `Y` = better served. Same convention as `F_causal`.
- **Four metrics**, all on continuous `Y`, computed **before-edit → after-edit**:
  - **Supply/demand ratio** — the two group **levels** `mean(Y|disadvantaged)`, `mean(Y|advantaged)`.
  - **Demographic parity (DP)** — signed gap `mean(Y|A) − mean(Y|D)` (0 = parity).
  - **Disparate impact (DI)** — ratio `mean(Y|D) / mean(Y|A)` (1 = parity; the 0.8 rule).
  - **Theil index** — between-region inequality of `Y` (0 = equal); grouping-independent, one per dataset.
- **Groups** (per equity axis ∈ {housing, comp, migrant}, two strategies, as a robustness check):
  - **district_extremes** — bottom-third vs top-third of the distinct demographic *regions* by that axis
    (Shenzhen: ~3-vs-3 of 10 districts; SF: thirds of ACS tracts); middle third excluded.
  - **median_split** — units below vs above the axis median (no exclusions).
  - Disadvantaged pole: **low housing price, low compensation, high migrant ratio** (a labeling choice;
    group *levels* are always reported so the direction is transparent — see §4.2).
- **Regions** (for Theil) are derived from the demographic *values themselves* (cells sharing a
  profile = a region) → city-agnostic (SF has no district-id file).
- **Uncertainty:** paired unit-level bootstrap, `B=1000`, seed 0 → 95 % percentile CIs on before, after,
  and the Δ. Honest caveat: units are not iid (spatial correlation; district-constant demographics), so
  these are first-order CIs (§5).
- **Datasets (4 edited runs):** Shenzhen PRIMARY `{housing,comp,migrant}`, Shenzhen `{housing,gdp,comp}`,
  Shenzhen `{housing,comp,migrant,logpopdensity}`, SF sf12 causal-emphasis. **Grouping axes are always
  the 3 equity axes regardless of which feature set the editor optimized** — this is what makes the
  metric "external."

Full tables per dataset: [`tables/`](tables/). Figures: [`figures/`](figures/).

---

## 2. Headline result — every Shenzhen metric improves, significantly and robustly

**Shenzhen PRIMARY `{housing,comp,migrant}`** ([`tables/shenzhen-primary.md`](tables/shenzhen-primary.md)) —
all 12 group-comparison cells + Theil move toward fairness, **all Δ CIs exclude 0**. Migrant axis,
district_extremes (the headline cell):

| Metric | Before | After | Δ | Δ 95% CI |
|---|---:|---:|---:|---:|
| Disparate impact | 0.3325 | 0.3422 | **+0.0097** | [0.0086, 0.0108] |
| Demographic parity (gap) | 14.1989 | 13.5956 | **−0.6033** | [−0.6668, −0.5355] |
| Theil (between-region) | 0.1550 | 0.1491 | **−0.0059** | [−0.0065, −0.0052] |

**Robust across all 3 Shenzhen feature sets** (migrant / district_extremes) — the external gain does
**not** depend on which features the editor optimized:

| Feature set | Theil Δ | DP Δ | DI Δ |
|---|---:|---:|---:|
| PRIMARY {housing,comp,migrant} | −0.0059 | −0.6033 | +0.0097 |
| {housing,gdp,comp} | −0.0056 | −0.5711 | +0.0092 |
| {housing,comp,migrant,logpopdensity} | −0.0055 | −0.5360 | +0.0086 |

(from [`tables/combined.md`](tables/combined.md); before-edit values are identical across the three —
same cleaned Shenzhen data — only the edit differs.)

---

## 3. External validity — SF sf12 reproduces the direction, weakly

**SF sf12** ([`tables/sf12.md`](tables/sf12.md)) — same direction as Shenzhen, but smaller and mixed:

| Axis / grouping | DI Δ (CI) | verdict |
|---|---|---|
| CompPerCapita / district_extremes | +0.0182 [0.0143, 0.0224] | **significant** |
| CompPerCapita / median_split | +0.0160 [0.0125, 0.0198] | **significant** |
| AvgHousingPricePerSqM / district_extremes | +0.0056 [0.0001, 0.0109] | barely significant |
| AvgHousingPricePerSqM / median_split | +0.0030 [−0.0016, 0.0080] | n.s. |
| **MigrantRatio / district_extremes** | +0.0034 [−0.0013, 0.0076] | **n.s.** |
| **MigrantRatio / median_split** | −0.0007 [−0.0052, 0.0039] | **n.s. (≈0)** |
| Theil (between-region) | Δ −0.0045 [−0.0058, −0.0035] | **significant** |

**Takeaway:** external validity **holds for compensation + Theil**, is **weak/borderline for housing**,
and **does not hold for the migrant axis on SF**. This is an honest asymmetry vs. Shenzhen (where migrant
was the strongest axis) and must be stated plainly in the paper.

---

## 4. Key findings for PI discussion (surfaced, not "fixed")

### 4.1 The improvement is *leveling-down*: the over-served group is reduced, the under-served group is (nearly) untouched

The supply/demand **group levels** (exposed after the final review) reveal the mechanism. Shenzhen
PRIMARY, district_extremes — before → after of each group's `mean(Y)`:

| Axis | Disadvantaged level | Advantaged level | who moves |
|---|---|---|---|
| MigrantRatio | 7.0734 → 7.0734 (**+0.000**) | 21.2723 → 20.6690 (−0.603) | **advantaged (over-served) ↓; disadvantaged flat** |
| CompPerCapita | 7.0734 → 7.0734 (**+0.000**) | 19.0456 → 18.5634 (−0.482) | **advantaged (over-served) ↓; disadvantaged flat** |
| AvgHousingPricePerSqM | 23.0484 → 22.3577 (−0.691) | 9.2082 → 9.2079 (**−0.000**) | **disadvantaged (over-served, DI>1) ↓; advantaged flat** |

**General statement:** the editor equalizes by **reducing whichever group is over-served, never by
augmenting the under-served group.** On Shenzhen the disadvantaged (poor, high-migrant) group's absolute
service is *unchanged* (7.0734 both before and after — the same units for comp & migrant, since low-comp
and high-migrant districts coincide). A reviewer will ask whether "fairness improvement" that only removes
service from advantaged areas — and never adds service to disadvantaged areas — is the intended, desirable
notion. **This needs a PI decision on framing** (and possibly a mechanism deep-dive: why the edited
pickups never raise disadvantaged-area `Y`).

On **SF** it is slightly worse: for several axes **both** groups lose service (advantaged more), so some
gains partly come from an overall service reduction — e.g. SF migrant/extremes disadvantaged 5.1945 → 5.1456
(−0.049) and advantaged 7.3411 → 7.2375 (−0.104).

### 4.2 The housing-axis disparity direction is city-dependent

- **Shenzhen:** low-housing areas are **over**-served — DI **2.50 > 1** (they get ~2.5× the service of
  high-housing areas). Editing pulls DI *down* toward 1.
- **SF:** low-housing areas are **under**-served — DI **0.80 < 1**. Editing pushes DI *up* toward 1.

So "disadvantaged = low housing price" is **not** a city-invariant proxy for "under-served." Migrant and
compensation behave consistently (disadvantaged = under-served, DI < 1) in both cities; housing does not.
Recommend leading the paper with **migrant + compensation** and treating housing as a sensitivity axis
with this caveat noted.

### 4.3 Demographic parity ≡ the supply/demand gap (by construction)

On a continuous outcome, `DP = mean(Y|A) − mean(Y|D)` **is** the supply/demand-ratio *gap* — the same
number. They are not independent metrics. The tables therefore report the supply/demand row as the group
**levels** (which *are* independent information — see §4.1), and DP as the gap. When counting "distinct
established metrics not in the objective," the honest set is **{DI, DP/gap, Theil}** plus the descriptive
levels — not four independent statistics. State this so a reviewer doesn't catch a duplicated row.

### 4.4 Magnitudes are small — consistent with the ε-bounded edit

Baseline disparities are large (migrant DI ≈ 0.33 → poor/high-migrant areas get ~⅓ the service). The edit
closes only ≈ +0.01 of DI (~1 % of the distance to parity). This is expected: the editor moves each pickup
at most ε = 2 cells. The value is **direction + significance + robustness**, not magnitude.

---

## 5. Caveats / limitations (for the paper's limitations section)

- **Leveling-down** (§4.1) — the central caveat.
- **Unit-level bootstrap is first-order.** Units are spatially correlated and demographics are
  district-constant, so the CIs understate true uncertainty. A clean driver-level bootstrap is *not*
  available: the demand grid is an independent mean-hourly artifact, not a per-driver sum, and supply is
  environmental (see spec §7). Disclosed, not hidden.
- **Associational, not causal** — like `F_causal`, these compare group means of an observational outcome.
- **Ecological / district-constant demographics** — grouping resolves to ~10 Shenzhen districts / SF tracts
  (few effective DOF; ecological-fallacy exposure).
- **Disadvantaged-pole labeling** is a choice (§4.2); group levels are always reported so it is flippable.
- **Small SF sample** — sf12 = 12 drivers → wider/lumpier CIs; the migrant-axis nulls partly reflect this.

---

## 6. Reproduce

Data must be present (Shenzhen `source_data/`; SF `source_data/second_dataset/sf_source_12` + `cache/sf_12`).
Raw run outputs land under `famail_temporal/baselines/external_fairness/results/` (**gitignored**); the
curated copies in [`tables/`](tables/) + [`figures/`](figures/) are the committed record.

```bash
# Shenzhen (default env = shenzhen)
python -m famail_temporal.baselines.run_external_fairness \
  --edit-dir famail_temporal/results/2026-06-29T12-06-55_k-10000_causal_emphasis_no-dedup_cleaned_hcm \
  --dataset shenzhen-primary --bootstrap 1000 --seed 0
python -m famail_temporal.baselines.run_external_fairness \
  --edit-dir famail_temporal/results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned \
  --dataset shenzhen-gdp-comp
python -m famail_temporal.baselines.run_external_fairness \
  --edit-dir famail_temporal/results/2026-06-28T11-46-12_k-10000_causal_emphasis_no-dedup_cleaned_4feat \
  --dataset shenzhen-logpop

# SF — city selected by the env var (DataBundle.load() has no city arg)
FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_external_fairness \
  --edit-dir famail_temporal/results/2026-06-30T23-13-33_sf12-fair-ce \
  --dataset sf12 --bootstrap 1000 --seed 0

# Cross-dataset table
python -m famail_temporal.baselines.run_external_fairness --combine \
  famail_temporal/baselines/external_fairness/results/*/external_fairness.json \
  --out-dir famail_temporal/baselines/external_fairness/results
```

**Cross-check:** our DI on the migrant axis matches the pre-existing `district_metrics.compute_di`
in direction (both < 1; magnitudes differ by grouping definition — ours = migrant terciles, theirs =
hukou top/bottom-3). Not a bug.

---

## 7. Provenance

- **Design / methodology (the *why*):** [`docs/superpowers/specs/2026-07-02-external-fairness-metrics-design.md`](../../docs/superpowers/specs/2026-07-02-external-fairness-metrics-design.md)
- **Full implementation recipe (the *how*, reproducible):** [`docs/superpowers/plans/2026-07-02-external-fairness-metrics.md`](../../docs/superpowers/plans/2026-07-02-external-fairness-metrics.md)
- **Code** (branch `external-fairness-metrics`): `famail_temporal/baselines/external_fairness.py`
  (metrics/grouping/bootstrap), `external_fairness_io.py` (Y, demographics, edited-grid),
  `run_external_fairness.py` (orchestration + JSON/table/figure + CLI). Tests:
  `famail_temporal/baselines/tests/test_external_fairness*.py` (24 tests).
- **Process:** built via superpowers brainstorm → spec → plan → subagent-driven execution (10 TDD tasks,
  each spec+quality reviewed; opus whole-branch review; fixes applied in `d4730e5`).

---

## 8. Deferred (manuscript-polish; NOT done — decide before the figures ship)

- Figure legibility: terse `axis[:6]/g[:4]` labels; add a "good direction" cue (DP/Theil improve leftward,
  DI rightward on the shared zero axis).
- `json.dumps` emits bare `NaN` tokens (fine for Python `json`, rejected by `jq`/browsers) — sanitize to
  `null` if a dashboard consumes the JSON.
- Output-dir/module name adjacency (`external_fairness.py` vs `external_fairness/results/`) — harmless now,
  a footgun if an `__init__.py` ever lands under the results tree; consider renaming the output dir.
