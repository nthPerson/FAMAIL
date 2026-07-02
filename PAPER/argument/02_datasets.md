# Datasets

Two cities. **Shenzhen is primary; San Francisco is external validity.** The two are deliberately
matched on fleet *density* (not fleet *size*) so the fairness signal is comparable; see §2.

---

## 1. Shenzhen (primary)

- **Sample:** a 50-driver sample of real Shenzhen taxi GPS trajectories (seeking + driving segments).
- **Grid:** 0.01° cells → a **48×90** spatial grid.
- **Time:** editor grid `T = 24` hourly buckets.
- **Demographics:** cell demographics resolve to **10 district-level profiles**, so `F_causal` is a
  partial R² over 10 district DOF (an ecological-fallacy caveat applies — see
  [`03_fairness_theory.md`](03_fairness_theory.md)).

### Demographic feature sets

`F_causal` is feature-set-specific, so the two-pillar story was measured under three demographic
feature sets. The **PRIMARY** set is `{housing, comp, migrant}` (neighborhood housing price,
per-capita compensation, migrant/hukou population share):

| feature set | before-edit F_causal | role |
|---|---|---|
| **`{housing, comp, migrant}`** | **0.799** | ★ PRIMARY — all-equity axes; higher baseline |
| `{housing, gdp, comp}` | 0.807 | sensitivity |
| `{housing, comp, migrant, logpopdensity}` | 0.725 | density sensitivity |

The PRIMARY set was chosen for construct validity (every axis is an equity / SES / population-structure
variable) and because its before-edit F_causal is **not the lowest** of the three — so the headline
metric is *not* the one that maximizes apparent baseline unfairness. All three sets reproduce every
directional conclusion (see [`05_results_shenzhen.md`](05_results_shenzhen.md) §6).

### Stuck-GPS data cleanup

Raw Shenzhen data contained per-driver **stuck-GPS pickup sinks**: single driver plates parked at one
cell emitting thousands of phantom "pickups" with almost no matching drop-offs (a GPS/meter artifact,
not real demand). A signature rule (n_pickups ≥ 1000 ∧ dropoff_ratio < 0.02 per (plate, rounded-coord))
flagged **10 calibrated sink cells across 9 driver plates**; filtering them removed **106,677** phantom
pickups. The full pipeline was re-run on the cleaned data. The cleanup is **demographic-independent**
(a filter on raw GPS sinks, not on any F_causal feature set), so it is valid for all three feature sets.

The headline sink at grid **(29,53)** recovers **+0.0885 locally** (its per-cell F_spatial
contribution), but the **net global** F_spatial recovery is only **+0.0213** — the difference is a
redistribution residual spread across non-sink cells, not an inconsistency. The cleanup changed **no
conclusion** (F_causal-emphasis results are robust to it). Figure:
`PAPER/shared_cleanup/figures/sink_spatial_attr_before_after.png`.

---

## 2. San Francisco Cabspotting (external validity)

- **Dataset:** 536 SF Yellow-Cab taxis, ~11.2M GPS pings, 2008-05-17 → 06-10, format
  `[lat lon occupancy time]` (occupancy 1 = driving/fare, 0 = seeking/free — a **native occupancy
  flag** that splits seeking vs. driving for free).
- **Grid:** faithful constant 0.01° cells → SF footprint **32×30** (not forced to Shenzhen's 48×90;
  forcing it would fold/distort the trajectories and change the ε-ball edit scale).
- **Demographics:** majority-overlap of **ACS 2006–2010** tracts onto cells (matching Shenzhen's
  district-mapping method), reusing the Shenzhen feature *names* filled with ACS values:
  `housing` = median home value (B25077), `comp` = per-capita income (B19301),
  `migrant` = foreign-born share (B05002). This keeps the PRIMARY equity set city-independent.
- **Time:** editor grid `T = 24` hourly; trajectory `time_bucket` 1–288 (5-min); `days_in_week = 7`
  (SF taxis run 7 days; Shenzhen data was Mon–Fri).

### The fleet-density regime discovery → the sf12 subsample

A fairness-only smoke on the **full 536-taxi fleet** returned baseline F_causal ≈ 0.982 with the
editor a near-no-op. This is not a bug but a **fleet-density regime mismatch**: SF's full fleet has
~0.56 drivers/cell vs Shenzhen's ~0.012 (~47× denser), so the 5×5 supply measure saturates, the
service-inequity gradient vanishes, and F_causal → 1 with nothing to edit. The fix is **fleet
subsampling** to restore Shenzhen's density:

| subsample | drivers | n_active | baseline F_causal | Δ (causal-emphasis) | verdict |
|---|---|---|---|---|---|
| full fleet | 536 | 11,596 | 0.982 | — | rejected (saturated) |
| sf50 (count-matched) | 50 | 7,854 | 0.956 | +0.0041 | rejected (still saturated) |
| **sf12 (density-matched)** | 12 | 4,230 | **0.870** | **+0.0199** | **CHOSEN** |

`sf12` matches Shenzhen's density (~0.012 drivers/cell) and is the only subsample that produces a
publishable fairness gain. Config: causal-emphasis (α_spatial = 0.2, α_causal = 0.7),
DEMAND_FLOOR = 0.5. Figure: `PAPER/second-dataset/figures/sf_supply_demand.png`.

---

## 3. Compatibility rationale — why these two cities

The dual claim (fairer **and** realistic) is what constrains the dataset choice. Realism is enforced
by a pre-trained, **driver-conditioned, 3-stream Siamese discriminator** over **dense per-driver
trajectory sequences**; it **cannot score origin–destination (OD) pairs** and **must be retrained per
city**. Consequently:

- **OD-only US trip records (NYC TLC, Chicago, DC) are incompatible with the dual claim** — they
  publish pickup→dropoff rows with no dense traces and weak/no persistent driver IDs, so they can
  carry the *fairness* half but never the *realism* half. (This is why the earlier "NYC + Census" idea
  was set aside.)
- **Dense-trace + persistent-driver-ID data is compatible**, with per-city discriminator retraining.

**SF Cabspotting is the only US dense-trace taxi set** with (i) a native occupancy flag, (ii)
persistent per-taxi IDs, and (iii) a native US-Census/ACS demographic join — so it drops into the
existing pipeline with **zero algorithm change**. Fallbacks were Porto (#2, non-US) and Rome (#3);
DiDi was excluded.

---

## 4. Shenzhen vs SF at a glance

| property | Shenzhen (primary) | SF (sf12, external validity) |
|---|---|---|
| trajectories | 50-driver sample | 536-taxi fleet → 12-driver density-matched subsample |
| grid (0.01° cells) | 48×90 | 32×30 |
| drivers/cell | ~0.012 | ~0.012 (matched by subsampling) |
| demographics | 10 district profiles | ACS 2006–2010 tracts → cells |
| PRIMARY features | {housing, comp, migrant} | same names, ACS-filled |
| before-edit F_causal | 0.799 (PRIMARY) | 0.870 (sf12) |
| discriminator | pre-trained per city | retrained for SF (val-AUC 0.998) |

Because F_causal is city-specific and associational, absolute baselines (0.799 vs 0.870) are **not**
cross-city comparable; SF establishes that the *conclusions* reproduce, not that the magnitudes match.

---

## Sources / provenance

- Shenzhen cleanup: `PAPER/shared_cleanup/README.md`, `PAPER/shared_cleanup/tables/dataset_summary.md`,
  `.../tables/sink_f_spatial_decomposition.md`; raw counts:
  `famail_temporal/source_data/processing_metadata.json`.
- Shenzhen feature sets: `PAPER/feature_selection/tables/comparison_across_sets.md`;
  `PAPER/by_feature_set/housing-comp-migrant/README.md`.
- SF dataset, regime discovery, sf12 selection: `PAPER/second-dataset/FINDINGS.md` §1–3;
  `PAPER/second-dataset/tables/subsample_selection.csv`.
- Figures (referenced, not regenerated): `PAPER/shared_cleanup/figures/sink_spatial_attr_before_after.png`,
  `PAPER/second-dataset/figures/sf_supply_demand.png`.
