# SF Second Dataset — Phase 2 Decisions (D1–D4)

> **Status:** Adopted from the recommended options in
> `docs/superpowers/plans/2026-06-29-sf-second-dataset.md`, on the user's
> authorization to proceed in the isolated worktree **pending final PI
> confirmation at the next meeting** (a few days out). These fix the
> protocol-governed F_causal/F_fidelity intermediate calculations; Phase 3 reads
> the exact constants from here. If the PI revises any choice, update this file
> and re-run the affected Phase-3 tasks.

## D1 — Grid extent & normalizer reconciliation (R7)

- **Cell size:** `GRID_SIZE_DEG = 0.01` (faithful; matches Shenzhen).
- **Operational bbox:** the 0.5–99.5 percentile core of the SF pings —
  `lat ∈ [37.532, 37.846]`, `lon ∈ [-122.497, -122.201]`.
- **Grid dims:** `X_GRID_MAX = 32` (latitude axis), `Y_GRID_MAX = 30` (longitude axis).
  Cell `(x, y)` ← `x = floor((lat − lat_min)/0.01)`, `y = floor((lon − lon_min)/0.01)`,
  clipped to `[0, 31] × [0, 29]` (same `gps_to_grid` convention as Shenzhen).
- **Normalizer extent (R7 fix):** pin ONE consistent extent. Editor cells `0..31 / 0..29`;
  discriminator coords `+1 → 1..32 / 1..30`; `FeatureNormalizer(x_max=32, y_max=30)`.
  **Do not** copy Shenzhen's 49/89. (Range becomes `[1/32 .. 1.0]` / `[1/30 .. 1.0]`.)
- *Rejected:* forcing 48×90 (folds out-of-extent SF data via the clip; distorts the
  ε-ball physical scale). Widening to ~40×40 (SFO/East-Bay edges) deferred unless the
  active set proves too thin — the core 32×30 already clears `n_active ≈ 10–12k`.

## D2 — Demographic aggregation method (R4)

- **Method:** areal interpolation of ACS 2006–2010 tracts onto the 32×30 grid using the
  TIGER 2010 polygons (`demographics/tiger_2010_tracts_06_CA.zip`).
  - **Intensive** (area-weighted mean) for the rate/level features: `housing`
    (median value), `comp` (per-capita income), `migrant` (foreign-born share).
  - **Extensive** (area-apportioned sum) for `population`, then
    `logdensity = log(pop / cell_land_km²)`.
- **Non-residential cells:** weight by each tract's residential land overlap; a cell whose
  overlapping tracts have ~zero residential population is marked **inactive for F_causal**
  (excluded from the demographic regression), mirroring how Shenzhen handles non-residential
  CBD/water cells via the active-mask finite-demographics filter.
- **Dependency / fallback:** areal interpolation needs `geopandas`+`shapely` (new deps,
  confined to the SF *build* — never imported by `algorithm/`/`fairness/`/`fidelity/`). If the
  environment cannot install them, fall back to **population-weighted centroid + nearest-tract**
  (the R4-probe method) and record the substitution here. *(R4 probe showed both give
  non-degenerate, well-conditioned signal — max VIF < 1.9 — so the fallback is acceptable.)*

## D3 — Vintage & construct

- **ACS vintage:** 2006–2010 5-year (centered on the May–Jun 2008 trajectories). Confirmed.
- **Geometry vintage:** 2010 tracts (consistent with 2006–2010 ACS).
- **Migrant construct:** US **foreign-born share** (`B05002_013/_001`) as the analog of
  Shenzhen's rural-migrant/hukou axis. **Documented caveat** for the paper: analog, not identical.

## D4 — Temporal discretization

- **Editor grid time axis:** `T = 24` hourly blocks (unchanged from Shenzhen `config.T`).
- **Trajectory-state `time_bucket`:** `0..287` (5-min), matching the discriminator's
  `time_buckets = 288` temporal encoding.
- **Week cycle:** `days_in_week = 7` for SF (vs Shenzhen Mon–Fri = 5); the discriminator's
  cyclic day encoding follows. `day_index` is 1-indexed per the discriminator convention.

## Config impact (Phase 3)

A **city-switchable config** is required (default = `shenzhen`, numerically identical to today):
`GRID_DIMS`, `SOURCE_DATA_DIR`, `DEMOGRAPHIC_FEATURES`, `DISCRIMINATOR_CHECKPOINT_FILENAME`,
and the normalizer/`days_in_week` extent select on a `FAMAIL_CITY` env var (or equivalent). This
is a data-layer parameterization, **not** an algorithm change; the `shenzhen` path must remain
bit-identical (regression-tested).
