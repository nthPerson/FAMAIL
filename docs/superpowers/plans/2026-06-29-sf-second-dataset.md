# SF Second Dataset (Cabspotting + ACS) — Build Plan (Phases 2–5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce FAMAIL's dual claim — *edited trajectories stay realistic (F_fidelity) while improving fairness (F_spatial, F_causal)* — on the San Francisco Cabspotting + US-Census second dataset, **with zero change to the editing algorithm or the discriminator architecture** (only weights, config, and data change).

**Architecture:** Four gated phases. **Phase 2** resolves the two protocol-governed decisions (grid-extent/normalizer reconciliation R7, demographic aggregation method R4) via `AskUserQuestion` + PI sign-off — no code lands first. **Phase 3** builds an SF data pipeline that emits `source_data` artifacts in the *same schema* the Shenzhen `loader.py` already consumes (SF raw loader → occupancy-split trajectories + pickup/dropoff grid → per-cell ACS demographics), plus the per-driver multi-stream corpus the discriminator retrain needs. **Phase 4** retrains the Multi-Stream Siamese discriminator on SF in the parent monorepo (GPU). **Phase 5** runs the unchanged FAMAIL editor end-to-end on SF and documents results. Phases 4–5 PAUSE for explicit go-ahead before each GPU run.

**Tech Stack:** Python 3, NumPy, pandas, PyTorch, pytest. Reuses `famail_temporal.data.source_generation`, `famail_temporal.data.loader`, `famail_temporal.algorithm`, `famail_temporal.fairness`, `famail_temporal.fidelity` (all read-only for the algorithm/fairness/fidelity packages). Demographic build reuses `famail_temporal/docs/build_sf_demographics.py`.

## Global Constraints

- **Branch / worktree:** `worktree-second-dataset-compat` (off `main`). Stage **named files only** — never `git add -A`. Everything under `famail_temporal/source_data/` is gitignored (data + the `.census_api_key`); never commit data or the key.
- **Zero algorithm change (hard requirement).** No edits to `famail_temporal/algorithm/`, `famail_temporal/fairness/`, or `famail_temporal/fidelity/`. The editor (5×5 pickup relocation, ST-iFGSM, soft assignment), the objective composition, and the discriminator *architecture* stay byte-for-byte identical. Only **weights (a new checkpoint), config constants, and data** change. If any task appears to require editing those packages, STOP and escalate — it means the dataset is being contorted (see `docs/SECOND_DATASET_COMPATIBILITY.md`).
- **Algorithm-change protocol.** The grid-extent reconciliation (R7) and the demographic aggregation method (R4) are F_causal/F_fidelity *intermediate calculations*. They MUST be resolved via `AskUserQuestion` + PI sign-off (Phase 2) **before** any Phase-3 code that depends on them. Do not silently pick a method.
- **Faithful gridding (locked finding).** Constant `GRID_SIZE_DEG = 0.01` square cells (matches Shenzhen `source_generation/config.py:10`). SF core footprint = **~32×30** cells over bbox ≈ lat 37.532–37.846, lon −122.497 to −122.201 (0.5–99.5 pct of pings). Do **not** force 48×90 — it would fold out-of-extent data (`quantization.py:57-58` clip) and distort the ε-ball scale. Final bbox/extent is a Phase-2 output.
- **Data facts (measured, see `docs/SECOND_DATASET_COMPATIBILITY.md` §10):** 536 cabs, 11.2M pings, 2008-05-17→06-10. Occupancy flag: `1 = driving` (fare), `0 = seeking`. Seeking 441k / driving 461k trajectories at 5-min gap split; 96.3% of steps cGAIL-legal at 0.01°. `n_active` ≈ 10–12k (vs Shenzhen 34,524). R4 demographics PASS (housing CV 0.31, income 0.41, migrant 0.39; max VIF < 1.9).
- **Demographic source layer (assembled):** `source_data/second_dataset/demographics/` — `acs_2006_2010_tracts.csv` (716 tracts, SF+San Mateo+Alameda), `tract_geometry_2010.csv`, `tiger_2010_tracts_06_CA.zip` (polygons), `MANIFEST.json`. Vintage ACS 2006–2010 (centered on 2008). Census key in `source_data/second_dataset/.census_api_key` or `$CENSUS_API_KEY`.
- **Coordinate convention:** the discriminator is 1-indexed `[1..X, 1..Y]`; the editor is 0-indexed. The +1 injection in `fidelity/context.py` and the normalizer denominators must be reconciled to the SF extent (R7). Never copy Shenzhen's 49/89.
- **Discriminator training code lives in the parent monorepo**, NOT in `famail_temporal/` (only inference is ported — `fidelity/README.md:6-9`). Phase 4 runs there.
- **GPU discipline:** implement and unit-test all non-GPU code first; **PAUSE for explicit user go-ahead before each GPU run** (Phase 4 retrain, Phase 5 edit run).

---

## Decisions to resolve at Phase 2 (do NOT pre-decide)

These are the only open intermediate-calculation choices; each is an `AskUserQuestion` + PI sign-off gate.

- **D1 — Grid extent & bbox (R7).** Final `(X_GRID_MAX, Y_GRID_MAX)`, the lat/lon origin, and whether the operational bbox is the 0.5–99.5 pct core (~32×30) or a widened box that includes the SFO/East-Bay edges (~up to 40×40). Plus the reconciled normalizer denominators + 1-index offset for the retrained discriminator. *Recommendation to present:* core 32×30, denominators = `(X_GRID_MAX, Y_GRID_MAX)` exactly, +1 offset retained.
- **D2 — Demographic aggregation method (R4).** Areal interpolation (TIGER polygons; extensive vs intensive per variable) vs population-weighted centroid vs nearest-tract. Plus the non-residential-cell rule (bay/SFO/commercial cells: NaN-and-exclude vs nearest-residential). *Recommendation to present:* areal interpolation via TIGER polygons, intensive (area-weighted mean) for medians/shares, extensive for counts→density; exclude cells with <X% residential land.
- **D3 — Vintage confirmation.** Confirm ACS 2006–2010 (centered on 2008) as the production vintage, and the migrant construct (`foreign-born share` as the hukou analog). *Recommendation:* confirm; document the construct caveat in the paper.
- **D4 — `days_in_week` & time discretization.** SF is 7-day (not Shenzhen Mon–Fri); confirm `days_in_week=7` and `N_TIME_BUCKETS` (288 5-min, matching Shenzhen, or 24 hourly). *Recommendation:* 7-day; keep 288 to match the discriminator's temporal encoding.

---

## File Structure

New SF-specific modules (parallel to the Shenzhen pipeline; Shenzhen code untouched):

- **Create** `famail_temporal/data/source_generation/sf_raw_loader.py` — parse `cabspottingdata/new_*.txt` → tidy DataFrame `[driver_id, lat, lon, occupancy, time_utc]`.
- **Create** `famail_temporal/data/source_generation/sf_config.py` — SF grid constants (extent/bbox/`days_in_week`) from D1/D4; mirrors `config.py` field names.
- **Create** `famail_temporal/data/source_generation/sf_segmentation.py` — occupancy + gap split into seeking/driving trajectories; pickup/dropoff (occ transitions).
- **Create** `famail_temporal/data/source_generation/sf_demographics.py` — ACS tracts (`acs_2006_2010_tracts.csv`) → per-cell `{housing, comp, migrant, logdensity}` via the D2 method; writes the SF demographics artifact.
- **Create** `famail_temporal/data/source_generation/sf_build.py` — orchestrator: raw → grid pickup/dropoff/active_taxis tensors + multi-stream corpus + demographics → `source_data/second_dataset/sf_source/` in the loader's schema.
- **Create (maybe)** thin SF loader path — either a `DataBundle.load(city="sf")` parameter or `famail_temporal/data/sf_loader.py` — decided after Task 3.0 reads `loader.py`'s schema.
- **Create** tests in `famail_temporal/data/source_generation/tests/`: `test_sf_raw_loader.py`, `test_sf_segmentation.py`, `test_sf_demographics.py`, `test_sf_build_schema.py`.
- **Create (deliverables)** `famail_temporal/baselines/SF_SECOND_DATASET_RESULTS.md` (Phase 5), and update `famail_temporal/docs/SECOND_DATASET_COMPATIBILITY.md` (Phase-2 decisions + final results).

---

## Phase 2 — Resolve protocol-governed decisions (GATE, no code)

### Task 2.1: Pose decisions D1–D4 and record sign-off

**Files:** Create `famail_temporal/docs/SF_PHASE2_DECISIONS.md`

- [ ] **Step 1:** Re-read `docs/SECOND_DATASET_COMPATIBILITY.md` §10 + `demographics/README.md` caveats so the options are framed with current numbers.
- [ ] **Step 2:** Pose D1–D4 to the user via `AskUserQuestion` (one question each, recommendation first). Do **not** proceed to Phase 3 until answered.
- [ ] **Step 3:** Record the chosen options + rationale in `SF_PHASE2_DECISIONS.md` (this becomes the source of truth Phase-3 tasks read for exact constants).
- [ ] **Step 4:** Email/notify the PI with the decisions for sign-off (per the two-pillar workflow); note "awaiting PI confirmation" in the doc if async.
- [ ] **Step 5: Commit** `git add famail_temporal/docs/SF_PHASE2_DECISIONS.md && git commit -m "docs(sf): record Phase-2 grid/demographic decisions"`

**Acceptance:** `SF_PHASE2_DECISIONS.md` contains concrete values for `X_GRID_MAX, Y_GRID_MAX`, bbox, normalizer denominators, `days_in_week`, `N_TIME_BUCKETS`, the demographic aggregation method, and the non-residential-cell rule.

### Task 2.2: Author the Phase-3 detailed TDD sub-plan

**Files:** Create `docs/superpowers/plans/2026-XX-XX-sf-phase3-pipeline.md`

> Phase 3's exact code depends on the D1–D4 values AND a close read of `loader.py`/`event_stream.py`/`raw_loader.py` internals. Per the project's brainstorm→spec→plan workflow, generate the fine-grained Phase-3 TDD plan **after** Phase 2 resolves, using the writing-plans skill. The tasks below (Phase 3) are the roadmap that sub-plan expands.

- [ ] **Step 1:** Using `superpowers:writing-plans`, expand Tasks 3.0–3.6 into per-step TDD tasks with the concrete constants from `SF_PHASE2_DECISIONS.md`.
- [ ] **Step 2: Commit** the sub-plan.

---

## Phase 3 — SF data pipeline (TDD; non-GPU; the bulk of the build)

> Roadmap. Each task below becomes a full TDD task in the Task-2.2 sub-plan. Representative tests shown; finalize against the real `loader.py` schema (Task 3.0).

### Task 3.0: Document the `source_data` schema loader.py consumes

**Files:** Read `famail_temporal/data/loader.py`, `famail_temporal/data/source_generation/{event_stream,writer,raw_loader}.py`; Create `famail_temporal/docs/SF_SOURCE_SCHEMA.md`

- [ ] **Step 1:** Trace what `DataBundle.load()` reads from `source_data/` (file names, tensor shapes, dtypes: `pickup_3d`, `dropoff_3d`, `active_taxis_3d`, the `(N,)` active-unit ordering, `multi_stream` seeking/driving/profile dicts, demographics table columns).
- [ ] **Step 2:** Write `SF_SOURCE_SCHEMA.md` enumerating every artifact + schema the SF build must emit to be loadable unchanged.
- [ ] **Step 3: Commit.**

**Acceptance:** schema doc lists each `source_data` artifact with exact shape/dtype/column names; this is the contract for Tasks 3.4–3.5.

### Task 3.1: SF raw loader

**Files:** Create `sf_raw_loader.py`; Test `tests/test_sf_raw_loader.py`

- [ ] **Step 1: Write failing test** — `load_sf_raw(dir)` returns a DataFrame with columns `[driver_id, lat, lon, occupancy, time_utc]`, sorted by `(driver_id, time_utc)`, driver_id integer-encoded, invalid coords dropped.
```python
def test_load_sf_raw_basic(tmp_path):
    d = tmp_path / "cabspottingdata"; d.mkdir()
    (d / "new_x.txt").write_text("37.75 -122.41 0 1213084687\n37.76 -122.42 1 1213084650\n")
    df = load_sf_raw(str(d))
    assert list(df.columns) == ["driver_id", "lat", "lon", "occupancy", "time_utc"]
    assert df["time_utc"].is_monotonic_increasing  # within the single driver
    assert df["occupancy"].isin([0, 1]).all()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (reuse the fast `raw.split()` parse from `docs/sf_cabspotting_derisk.py`). **Step 4:** PASS. **Step 5: Commit.**

### Task 3.2: Occupancy + gap segmentation → seeking/driving + pickups/dropoffs

**Files:** Create `sf_segmentation.py`; Test `tests/test_sf_segmentation.py`

- [ ] **Step 1: Write failing test** — `segment(df, gap_sec=300)` yields `seeking` (occ=0) and `driving` (occ=1) trajectory lists (each a sequence of `[x,y,t,d]` *after gridding*), splitting on occupancy change or `dt>gap_sec`; `pickups`/`dropoffs` from 0→1 / 1→0 transitions.
```python
def test_segmentation_splits_on_occupancy_and_gap():
    # one driver: seek, pickup(0->1), drive, dropoff(1->0); plus a >gap jump
    df = make_df(occ=[0,0,1,1,0], dt=[0,60,60,60,9999])
    out = segment(df, gap_sec=300)
    assert len(out.seeking) >= 1 and len(out.driving) == 1
    assert out.pickups == 1 and out.dropoffs == 1
```
- [ ] Steps 2–5 as standard TDD. **Acceptance:** counts reproduce the de-risk totals (~441k seeking / ~461k driving / ~441k pickups) on the full dataset within tolerance.

### Task 3.3: SF demographics → per-cell features (D2 method)

**Files:** Create `sf_demographics.py`; Test `tests/test_sf_demographics.py`. Inputs: `demographics/acs_2006_2010_tracts.csv`, `tiger_2010_tracts_06_CA.zip` (if areal interpolation chosen in D2).

- [ ] **Step 1: Write failing test** — `build_cell_demographics(grid, method=...)` returns an array shaped `(X_GRID_MAX, Y_GRID_MAX, n_features)` for `{housing, comp, migrant}` (+logdensity), non-NaN over active cells, with the D2 non-residential-cell rule applied; cross-cell std>0 for each feature.
```python
def test_cell_demographics_nondegenerate(sf_grid):
    feats = build_cell_demographics(sf_grid, method=PHASE2_METHOD)
    assert feats.shape[-1] == 3
    for j in range(3):
        col = feats[..., j][np.isfinite(feats[..., j])]
        assert col.std() > 0            # non-degenerate (R4)
```
- [ ] Steps 2–5. If D2 = areal interpolation, this task **may add `geopandas`/`shapely`** to `requirements.txt` (the only new dep; confine to the SF build, not the algorithm). **Acceptance:** per-cell feature variance + max VIF < ~5 over active cells (matches the R4 probe).

### Task 3.4: Grid tensors (pickup/dropoff/active_taxis) in loader schema

**Files:** Modify/extend `sf_config.py`, add `sf_build.py` grid step; Test `tests/test_sf_build_schema.py`

- [ ] **Step 1: Write failing test** — gridded `pickup_3d/dropoff_3d/active_taxis_3d` have shape `(X_GRID_MAX, Y_GRID_MAX, T)` per `SF_SOURCE_SCHEMA.md`, non-negative, and `active_taxis` (distinct cabs per cell-time) yields `n_active` in the measured 10–12k range under `ACTIVE_SUPPLY_THRESHOLD`.
- [ ] Steps 2–5. **Acceptance:** `n_active` within [9k, 13k]; tensors match the documented schema exactly.

### Task 3.5: Multi-stream corpus (seeking/driving/profile per driver) for the retrain

**Files:** `sf_build.py` multi-stream step; Test in `test_sf_build_schema.py`

- [ ] **Step 1: Write failing test** — emits `seeking_trajs`, `driving_trajs`, `profile_features` dicts keyed by `driver_id`, 1-indexed coords, matching the structure `fidelity/context.py::MultiStreamData` expects; profile is 11-dim per driver (home cell, shift percentiles, modal pickup cell, avg seek/drive distance+duration, trips/day).
- [ ] Steps 2–5. **Acceptance:** ≥11,000 `(driver,day)` cells with ≥5 seeking trajs (matches de-risk pair-feasibility); profile vectors finite for ≥95% of drivers.

### Task 3.6: Assemble `source_data/second_dataset/sf_source/` + SF DataBundle load

**Files:** `sf_build.py` writer; SF loader path (per Task 3.0 decision); Test `tests/test_sf_build_schema.py::test_databundle_loads`

- [ ] **Step 1: Write failing test** — after `sf_build.main()`, `DataBundle.load(<sf path>)` returns a bundle with the canonical `(N,)` active-unit ordering asserted, `bundle.unit_map.n_units == n_active`, and `FAMAILObjective(bundle).forward(base_pickup)` runs and returns finite `F_spatial`, `F_causal` (with `ALPHA_FIDELITY=0`, no checkpoint needed).
- [ ] Steps 2–5. **Acceptance:** baseline F_spatial/F_causal computed on SF with the unchanged objective; **no edits to `algorithm/`/`fairness/`**.

---

## Phase 4 — Retrain the discriminator on SF (GPU; parent monorepo; PAUSE)

### Task 4.1: Build the retrain corpus + pairs

**Files:** parent monorepo `discriminator/` training pipeline; inputs = SF multi-stream corpus from Task 3.5.

- [ ] **Step 1:** Export the SF seeking/driving/profile corpus to the parent monorepo's training input format.
- [ ] **Step 2:** Form ~5k same-driver/different-day positive + ~5k different-driver negative pairs (labels from driver_id + date).
- [ ] **Step 3:** Set normalizer `x_max/y_max/time_buckets/days_in_week` to the D1/D4 SF values (the only architecture *constructor* changes; weights are re-fit).
- [ ] **Acceptance:** corpus + pair counts logged; constructor args match `SF_PHASE2_DECISIONS.md`.

### Task 4.2: Train to a new `best.pt` — **PAUSE before launching GPU**

- [ ] **Step 1:** PAUSE — confirm go-ahead with the user before the GPU run.
- [ ] **Step 2:** Train end-to-end; target val-AUC comparable to Shenzhen's 0.982 (report whatever it reaches).
- [ ] **Step 3:** Drop the new checkpoint at `famail_temporal/discriminator_checkpoints/sf/best.pt`; set `config.DISCRIMINATOR_CHECKPOINT_FILENAME` accordingly (config-only).
- [ ] **Step 4:** Write `discriminator_checkpoints/sf/README.md` provenance (date, corpus, epochs, val-AUC, grid extent).
- [ ] **Acceptance:** `fidelity.checkpoint.load_discriminator()` loads it; a smoke `compute_ffidelity` on one SF trajectory returns a finite [0,1] score (no `nn.Identity` stub).

---

## Phase 5 — Run FAMAIL end-to-end on SF + document (GPU; PAUSE)

### Task 5.1: Smoke run (k small) — **PAUSE before GPU**

- [ ] **Step 1:** PAUSE — confirm go-ahead.
- [ ] **Step 2:** `python -m famail_temporal.evaluation.runner --name sf-smoke --data <sf path> --k 50` with `ALPHA_FIDELITY>0` and the SF checkpoint.
- [ ] **Step 3:** Verify the editor runs unchanged (5×5 relocation), F_fidelity is active (non-stub), and per-iteration F_causal/F_spatial/F_fidelity are finite.
- [ ] **Acceptance:** smoke completes; no edits to `algorithm/`/`fidelity/`; objective improves over iterations on the edited cells.

### Task 5.2: Full run + paired comparison — **PAUSE before GPU**

- [ ] **Step 1:** PAUSE. **Step 2:** Full run (matched protocol to the Shenzhen headline config). Capture the dual-claim result: edited−raw `F_causal`/`F_spatial` (improved) with `F_fidelity` preserved.
- [ ] **Step 3:** Write `famail_temporal/baselines/SF_SECOND_DATASET_RESULTS.md` (L1/L2 house style): scorecard, the two honest caveats (2008/24-day vintage; ~⅓ active-unit footprint), and the construct note.
- [ ] **Step 4:** Update `docs/SECOND_DATASET_COMPATIBILITY.md` with the realized result. **Commit** (results docs only; run artifacts under `results/` stay gitignored).
- [ ] **Acceptance:** a reproducible SF result demonstrating realistic + fairer edits with the unchanged algorithm, ready for the paper's robustness section.

---

## Self-Review

- **Spec coverage:** Phases map to the `docs/SECOND_DATASET_COMPATIBILITY.md` §5 "what changes" (config/engineering/algorithm/R7) — D1/R7 → Task 2.1+4.1; R4 demographics → Task 3.3; source-gen re-grid → Tasks 3.1–3.4; retrain → Phase 4; un-contorted run → Phase 5. ✓
- **Protocol:** the two intermediate-calc decisions (R7, R4) are gated in Phase 2 before dependent code. ✓
- **Zero-algorithm-change:** asserted in Global Constraints and re-checked in Tasks 3.6/5.1 acceptance. ✓
- **Known dependency:** Phase 3's exact code depends on D1–D4 + a `loader.py` schema read (Task 3.0); the fine-grained TDD steps are authored in the Task-2.2 sub-plan once Phase 2 lands — this is the project's brainstorm→spec→plan workflow, not a placeholder.

---

## Execution Handoff

Phases 2 (decisions) and 3 (pipeline) are non-GPU and can proceed once D1–D4 are answered. Phases 4–5 are GPU and PAUSE-gated. Recommended: **subagent-driven** execution for Phase 3 (fresh subagent per task, review between), after Phase 2's `AskUserQuestion` gate.
