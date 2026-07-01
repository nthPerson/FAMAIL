# Second-Dataset Compatibility for FAMAIL — A Grounded Assessment

| | |
|---|---|
| **Status** | Investigation complete; recommendation pending PI review |
| **Date** | 2026-06-29 |
| **Scope** | Which second taxi/for-hire-vehicle dataset can carry FAMAIL's central claim — *"edited trajectories are realistic (F_fidelity) while improving fairness (F_spatial, F_causal)"* — **without contorting the algorithm** |
| **Method** | (1) deep-research survey of candidate datasets; (2) code-grounded analysis of `famail_temporal` (esp. the F_fidelity discriminator); (3) independent verification of load-bearing claims against source |
| **Headline** | The realism term is the binding constraint. It requires **dense, per-driver trajectory sequences**, which **rules out all OD-only US data** (NYC TLC, Chicago, DC) for the *dual* claim. **Rank: #1 SF Cabspotting, #2 Porto, #3 Rome.** |

---

## 1. Motivation

FAMAIL's results are currently established on a single dataset: Shenzhen, China taxi GPS trajectories binned onto a 48×90×24 spatio-temporal grid, with per-cell demographic features. A KDD-class submission wants a **second dataset** to demonstrate robustness. The non-negotiable requirement, set by the research lead, is that the second dataset must let us make the **same two claims** as on Shenzhen — *realistic edits* **and** *improved fairness* — **without re-engineering the algorithm to fit the data**. A dataset that forces us to redefine or drop a metric is a contortion, not a robustness check.

This document records the investigation that answers: *which candidate datasets actually satisfy that bar, and how should they be ranked?*

## 2. Method

1. **Dataset survey (deep research).** A fan-out web search + adversarial-verification pass over candidate taxi/TNC datasets and their demographic-join feasibility (US Census/ACS in particular). Produced the candidate set in §5 and the granularity/access/Census-join ratings.
2. **Code-grounded compatibility analysis.** A multi-agent read of `famail_temporal` — the editor (`algorithm/`), the F_fidelity discriminator (`fidelity/`), the fairness metrics (`fairness/`), the data layer (`data/`), and the source-generation pipeline (`data/source_generation/`) — to extract exactly what the algorithm requires from input data.
3. **Independent verification.** The load-bearing claims (what the discriminator consumes; what the editor edits; the grid-extent convention) were re-checked by hand against source. Citations below are `file:line` into `famail_temporal/`.

> **Provenance note.** Two of the five code-reading agents returned malformed structured output and were discarded; the synthesis and a hand audit re-derived their facts directly from source. The conclusions in §3–§4 were each confirmed against at least two independent reads.

## 3. How FAMAIL works, and what it requires from data

### 3.1 The editor relocates a *pickup cell*, not a path

`TrajectoryModifier` runs ST-iFGSM, and the **only** quantity it perturbs is a trip's **pickup cell**, moved within a bounded neighborhood. It reads the trip's time block `t*` from its pickup `time_bucket`, subtracts the trip's mass from `_base_pickup_3d[orig_cell, t*]`, gradient-ascends the pickup location, and clips to grid bounds (`algorithm/modifier.py:1-22`). Movement is confined by `EPSILON_BALL = 2.0` — explicitly "the cGAIL 5×5 window" (`config.py:65, 92`) — with soft mass spread over a 5×5 neighborhood (`config.py:114`). Driving paths, intermediate states, and dropoffs are untouched.

So **the editor's own data primitive is small**: per-trip pickup cell + time bucket, aggregated into a 48×90×T occupancy tensor (`config.GRID_DIMS = (48, 90)`, `config.py:20`). It needs no GPS polylines *at modification time*.

This is what makes the dataset question subtle: the *editor* would be content with origin points, but the *realism claim* is enforced by a separate component with much heavier requirements.

### 3.2 F_fidelity is a driver-conditioned, sequence-based, same-driver classifier

Realism is scored by `compute_ffidelity`, which forwards a **`MultiStreamSiameseDiscriminator`** (`fidelity/model.py:285`; `fidelity/compute.py:41-42`). Three facts about it dominate the entire analysis:

1. **It is a same-driver classifier.** Both Siamese branches represent the *same* driver; only the seeking-stream slot 0 differs (original vs. modified). The score is `P(the modified trajectory still looks like it came from the same driver as the original)` (`fidelity/context.py:10-15, 62-71`; `algorithm/objective.py:41`).

2. **It consumes dense, variable-length *sequences*, in three streams — and only three.** (`fidelity/context.py:48-159`, `fidelity/model.py:285-426`)
   - **Seeking** — a bidirectional stacked LSTM over the empty-taxi cruising path, `[B, N=5, L, 4]` of `[x_grid, y_grid, time_bucket, day_index]` states.
   - **Driving** — an independent LSTM over the occupied (passenger-carrying) path, `[B, N, L_d, 4]`.
   - **Profile** — an 11-dim per-driver feature vector through an FCN.
   There are **no exogenous streams** (no weather, POI, or external feeds). Consequence: a candidate dataset's lack of POI/weather side-channels is **irrelevant** — the model never consumes them.

3. **It is pre-trained and frozen, on Shenzhen.** `load_discriminator` loads the checkpoint, sets eval mode, and freezes parameters (`fidelity/checkpoint.py`); the module is an "opaque inference-only" port with **no training code** (`fidelity/README.md:6-9`). With no checkpoint, the loader silently substitutes an `nn.Identity` stub and **fidelity is disabled** (`data/loader.py:247-252`). The installed weights were trained on **50 specific Shenzhen drivers** over a fixed weekday window. The learned identity manifold **does not transfer** to other drivers/cities.

**The decisive implication:** the realism claim reduces to *"the edited trajectory, encoded as a dense movement sequence, is still classified as the same driver."* **You cannot compute F_fidelity from origin–destination pairs** — there is no dense sequence to encode and no driver-identity anchor. And because the discriminator is Shenzhen-specific, **every** second city requires a **full discriminator retrain** to produce its own checkpoint.

### 3.3 What retraining the discriminator requires

To produce a new city's checkpoint (parametrically identical architecture, new weights):

- A corpus of **per-driver, multi-day, dense traces**, split into **seeking** and **driving** streams (Shenzhen retained ~10⁵ of each).
- ~10⁴ **labeled pairs** (same-driver/different-day positives vs. different-driver negatives) — labels derived automatically from driver IDs + dates, not hand-annotated.
- An **11-dim per-driver profile** computed from raw GPS (home cell, shift percentiles, modal pickup cell, average seek/drive distance + duration, trips/day). The profile stream is **degradable-optional**: the model has a zero-default path that runs without it at some accuracy cost (`fidelity/model.py:423-426`).

This is the gate: *a dataset that cannot supply per-driver, multi-day, dense traces cannot retrain the discriminator, and therefore cannot support the realism half of the claim without redefining F_fidelity.*

### 3.4 The fairness half needs much less — but still a grid + demographics

- **F_spatial** consumes gridded `pickup / dropoff / active_taxis` counts and a Gini over active cells.
- **F_causal** regresses a residual against per-cell demographics — the PRIMARY equity set `{AvgHousingPricePerSqM, CompPerCapita, MigrantRatio}` (`config.py:52-55`).
- The **active-unit set** is `(cell, t)` units above a supply threshold (`data/active_mask.py`); the cleaned Shenzhen run has `n_active = 34,524`.
- The **source-generation pipeline itself is trace-native**: it grids raw GPS and enforces a cGAIL action-space invariant (`max(|dx|,|dy|) ≤ 1` between consecutive states). So even the editor's *data preparation*, not just the discriminator, presupposes dense traces.

## 4. The compatibility test

A second dataset must satisfy these to support **both** claims un-contorted:

| ID | Requirement | Code grounding |
|----|-------------|----------------|
| **R1** | Per-trip **pickup cell + time bucket**, griddable to ~48×90 × hourly. Needs real coordinates, not opaque zone IDs. | `config.py:20`; `modifier.py:1-22` |
| **R2** | A real **dense-trace corpus** large enough to retrain the discriminator (~10⁵ seeking + 10⁵ driving → ~10⁴ pairs). | `fidelity/checkpoint.py`; `loader.py:247-252` |
| **R3** | Per-driver, **multi-day** dense traces under a **stable driver ID**, splittable into seeking vs. driving (occupancy flag ideal). 11-dim profile degradable-optional. | `fidelity/model.py:285-426`; `fidelity/context.py:48-159` |
| **R4** | Per-cell demographics analogous to **{housing, compensation, migrant}**, joinable to the grid, non-zero variance. | `config.py:52-55`; `data/demographics.py` |
| **R5** | Enough **trip volume** for stable active-unit fairness metrics (thousands of active `(cell,t)` units). | `data/active_mask.py`; `n_active = 34,524` |
| **R6** | **cGAIL action-space compatibility**: traces dense enough that gridded consecutive states satisfy `max(|dx|,|dy|) ≤ 1`. | `data/source_generation/invariants.py` |
| **R7** | **Grid-extent consistency (pre-retrain blocker).** Pin a single x/y extent across the editor grid, the +1 1-indexing convention, and the discriminator's normalizer denominators; retrain on data gridded to *that* extent. | `fidelity/model.py:33,46-47` (÷49) vs `config.py:20` (48); `fidelity/context.py` (+1 offset) |

**R2 + R3 are the gate.** They demand a *dense GPS-trace dataset with persistent multi-day driver IDs*. Any OD/zone candidate fails them.

## 5. Candidate datasets and verdicts

Every dataset requires the discriminator to be retrained (§3.2). The operative question is therefore **"can it be retrained?"** — i.e., does it satisfy R2+R3?

### 5.1 OD / zone-level candidates — all INCOMPATIBLE for the dual claim

These are origin–destination records (a pickup and a dropoff point/zone, *no intermediate pings*).

| Dataset | Geography | R1 | R2 traces | R3 driver+streams | R4 demographics | Verdict |
|---|---|---|---|---|---|---|
| NYC TLC pre-2016 lat/lon | US | ✅ real lat/lon | ❌ no pings | ❌ no driver ID | ✅ clean ACS tract | **INCOMPATIBLE** |
| NYC TLC 2016+ taxi-zone | US | ❌ ~263 zone IDs | ❌ | ❌ | ⚠️ coarse (zones) | **INCOMPATIBLE** |
| Washington DC DFHV | US | ⚠️ block-snapped ~250 m | ❌ | ❌ no driver ID | ✅ clean ACS | **INCOMPATIBLE** |
| Chicago Taxi Trips | US | ⚠️ tract/centroid | ❌ | ❌ | ⚠️ ~27% tracts nulled | **INCOMPATIBLE** |
| Chicago TNP (rideshare) | US | ⚠️ tract/centroid | ❌ | ❌ **no persistent driver id** | ⚠️ ~44% tracts nulled | **INCOMPATIBLE** |

These satisfy the **fairness** half well (NYC pre-2016 and DC have clean Census/ACS joins), but **none can retrain the discriminator**. Using any of them forces dropping or redefining F_fidelity — abandoning the paper's central "realistic" claim. That is exactly the contortion to avoid.

### 5.2 Dense GPS-trace candidates — COMPATIBLE-WITH-RETRAINING

| Dataset | Geography | R2 corpus | R3 streams + driver ID | R4 demographics | Verdict |
|---|---|---|---|---|---|
| **SF Cabspotting** (CRAWDAD epfl/mobility) | US | ⚠️ ~500 taxis, ~30 days, 2008 | ✅ driver ID + **native occupancy flag** (seeking/driving split), multi-day | ✅ **native US Census/ACS** (~2008 vintage) | **COMPATIBLE-W/-RETRAIN** |
| **Porto** (ECML/PKDD 2015, UCI) | Portugal | ✅ 1.7M trips, 442 taxis, 1 yr | ⚠️ taxi ID; driving = trip records; **seeking reconstructed** from inter-trip gaps | ⚠️ Portugal **INE** (non-US) | **COMPATIBLE-W/-RETRAIN** |
| **Rome** (CRAWDAD roma/taxi) | Italy | ⚠️ ~320 taxis, ~30 days | ⚠️ driver ID, multi-day; **no occupancy** → seeking/driving inferred | ⚠️ **ISTAT** (non-US) | **COMPATIBLE-W/-RETRAIN** |
| **DiDi GAIA** (Chengdu/Xi'an) | China | ✅ very large | ⚠️ order-level; cross-day driver continuity release-dependent | ⚠️ **NBS** (coarse) | **COMPATIBLE (gated on DUA + continuity)** |

"COMPATIBLE-WITH-RETRAINING" is the **best achievable verdict** — no dataset lets you skip retraining — and it is the correct one for these four, because they actually *can* be retrained (they supply R2+R3).

## 6. Ranked recommendation

Weighting **un-contorted fit** (F_fidelity survives unchanged) highest, then demographic-join cleanliness (R4), then volume (R2/R5):

### #1 — SF Cabspotting (CRAWDAD epfl/mobility)
The **only** candidate that is simultaneously (a) a **dense trace** dataset, (b) carries a **native occupancy flag** so the seeking-vs-driving split the architecture demands comes *for free* (rather than being inferred), (c) has **persistent per-taxi IDs across ~30 days** so same-driver/different-day pairs form directly, and (d) has a **native US Census/ACS join** (~2008 vintage). It keeps the **entire pipeline un-contorted** — editor, source-generation filter, F_causal demographics, and a *retrained-but-architecturally-identical* discriminator. **The claim is reproduced with zero algorithm change.**

- **Biggest risk: trajectory/active-cell volume, *not* driver count.** Shenzhen trained on only **50** drivers, so SF's ~500 taxis is a surplus on driver count. The binding figures are Shenzhen's ~10⁵ seeking trajectories and `n_active = 34,524`. SF's ~30-day window yields fewer trajectories-per-driver-day and a thinner active-cell set — where forming ~10⁴ well-separated pairs and clearing the active-cell threshold for stable Gini/residual regression could strain. *Mitigations:* smaller grid, coarser T, more day-pairs per driver — while watching that this doesn't thin the active set below significance.
- **Secondary:** the data is old (2008) and a single ~1-month snapshot — a relevance/scale caveat the paper must state honestly.

### #2 — Porto ECML/PKDD 2015 (UCI)
Best **volume and licensing** (1.7M trips, 442 taxis, a full year, CC BY 4.0). R2/R5 are comfortably satisfied; 15 s polylines grid cleanly (R1/R6); a full year gives abundant same-driver/different-day pairs. The safest choice for the *statistical* stability of both the retrain and the fairness metrics.

- **Biggest risk: seeking reconstruction + non-US demographics.** Each Porto record **is one occupied trip by construction**, so the *driving* stream is given directly; only the **seeking (inter-trip) stream must be reconstructed** from gaps between consecutive trips — a bounded, largely deterministic step, but the reconstruction choices feed the very signal F_fidelity validates, so they must be made explicit and sanity-checked. Demographics come from Portugal **INE** (non-US), a coarser, less-tooled join than US ACS — added F_causal-mapping risk. (The occupancy-flag advantage of SF is partly offset by this being a deterministic derivation, so the SF–Porto gap is narrower than it first appears.)

### #3 — Rome (CRAWDAD roma/taxi)
A viable European fallback (dense ~7 s pings, multi-day driver IDs, ISTAT demographics) but it carries the **union** of the top-two risks — small scale like SF *and* no occupancy flag (fully inferred seeking/driving split, without Porto's trip-record structure to anchor it) *and* non-US demographics — with none of their compensating strengths.

*(DiDi GAIA is excluded from the top 3 despite its volume: the DUA is China-gated and not guaranteed, order-level traces may not preserve the cross-day driver continuity R3 needs, and NBS demographics are coarse — too much access/structure uncertainty for a ~1-month deadline. Profile-stream thinness is **not** part of this exclusion, since R3's profile is degradable-optional.)*

## 7. What changes in the code for the top pick (SF Cabspotting)

| Layer | Change | Coupling |
|---|---|---|
| **Config knobs** | Normalizer `x_max / y_max / time_buckets / days_in_week` (→ **7-day** cycle for SF, not Mon–Fri); `GRID_DIMS`; `DEMOGRAPHIC_FEATURES` → ACS analogues; checkpoint path; re-tune `ACTIVE_SUPPLY_THRESHOLD` for the smaller fleet. | configurable |
| **Real engineering** (no algorithm change) | Retrain the discriminator end-to-end on SF (build seeking/driving corpora via the occupancy flag, form ~10⁴ pairs, recompute the 11-dim profiles, train a new checkpoint — training code lives in the **parent monorepo**, not `famail_temporal`); re-run source-generation (`gps_to_grid` + action-space filter) at a resolution keeping the rejection rate tolerable; rebuild the demographics grid + hat matrices from ACS; re-fit all standardization stats. | retrainable per city |
| **Algorithm** | **None required** — editor (5×5 pickup relocation, ST-iFGSM, soft assignment), objective composition, and discriminator *architecture* stay identical. Only **weights, config, and data** change. (Choosing an OD dataset instead would have forced redefining F_fidelity — a genuine algorithm change and a weakening of the claim.) | — |
| **Pre-retrain blocker (R7)** | Resolve the **grid-extent / +1 mismatch first**: the normalizer divides x by 49 ("grid is 50 wide", `fidelity/model.py:33,46-47`) while `config.GRID_DIMS` is 48 and `context.py` adds +1 to 1-index coords. Pin one consistent extent across editor grid, +1 convention, and normalizer denominators, and retrain on data gridded to it — do **not** copy Shenzhen's 49/89 constants. | hardcoded → must resolve |

## 8. Open questions & caveats

1. **R7 grid-extent mismatch** (above) — an unresolved numerics convention, not a copyable constant. An off-by-one would silently propagate into the retrained weights.
2. **Inference-padding discrepancy (dataset-independent).** F_fidelity pads sequences to the per-call max length at inference, vs. training's fixed 256/128 (`fidelity/context.py:239-265`); the score impact has not been assessed. Worth checking regardless of dataset choice.
3. **SF volume/recency.** Quantify SF's likely `n_active` and trajectories-per-driver before committing, to confirm R2/R5 hold; state the 2008/30-day limitation in the paper.
4. **R4 demographic remap** ({housing, comp, migrant} → ACS columns) changes an intermediate calculation and should be designed and signed off, not improvised.

## 9. Implication for the paper

This **reverses the naive "US Census data is richest, so use NYC"** intuition — including the second-dataset option (NYC + US Census) previously mooted at the planning level. NYC TLC (and all US open OD data) can reproduce the **fairness** results but **cannot reproduce the realism half** of the claim without redefining F_fidelity, which would mean publishing a *different, weaker* fidelity term on the second dataset than on Shenzhen.

The two honestly-available robustness claims split cleanly:

- **SF Cabspotting** → *"the full method — including the same Multi-Stream Siamese F_fidelity — transfers to a US city."* Strongest and un-contorted; the cost is small/old data.
- **NYC / DC / Chicago** → only *"the fairness editing transfers,"* with realism dropped or surrogate'd — the contortion we are trying to avoid.

**Recommended next step:** raise this with the PI before committing, since it contradicts the previously-mooted NYC choice; then, if SF is approved, scope the integration (seeking/driving segmentation, SF→grid, ACS join, discriminator retrain) with the R4/R7 decisions surfaced for explicit sign-off.

---

## 10. Phase 1 de-risk — measured on the SF Cabspotting data (2026-06-29)

The dataset was obtained (`source_data/second_dataset/cabspottingdata/`: 536 cabs, 11.2M GPS pings, `[lat lon occupancy time]`, May 17–Jun 10 2008) and measured directly. **Assumptions:** trajectories segmented on occupancy-change or >5 min gap; grid origin = the 0.5–99.5 percentile ("core") bbox; `n_active` proxied by distinct cabs per `(cell, hour)`.

### 10.1 Gridding decision — keep constant 0.01°, do NOT force 48×90

Confirmed against `source_generation/quantization.py`: Shenzhen grids by **0.01° bins then clips to 48×90**, so 0.01° (not the cell count) is the invariant.

| | SF (faithful 0.01°) | Shenzhen | Forced 48×90 on SF |
|---|---|---|---|
| Core bbox | 0.315° × 0.295° | ~0.48° × 0.90° | (same core) |
| Grid dims | **32 × 30** (960 cells) | 48 × 90 (4,320) | 48 × 90 |
| Physical cell | 1.106 km × 0.880 km | 1.106 km × 1.028 km | **0.725 km × 0.289 km** |
| ε-ball=2 (5×5) span | 2.21 km | 2.21 km | distorted (non-square) |

**Decision: keep `GRID_SIZE_DEG = 0.01`; set the SF grid to its own binned extent (~32×30 on the core; up to ~40×40 if the operational bbox includes SFO/East Bay edges). Do not reuse 48×90.** The faithful grid preserves both the ~1 km cell and the physical meaning of the ε-ball edit window; forcing 48×90 would shrink/skew cells (0.73×0.29 km) and corrupt the edit scale. This makes recomputing the discriminator normalizer denominators to the SF grid a required part of the R7 fix.

### 10.2 Risk profile vs Shenzhen

| Requirement | SF measured | Shenzhen | Verdict |
|---|---|---|---|
| **R2 discriminator corpus** | seeking **441k** / driving **461k** trajectories | 105k / 92k | ✅ **abundant (~4–5×)** |
| **R3 pair feasibility** | 533/536 cabs ≥2 days; **11,722** (cab,day) with ≥5 seeking trajs; 441k trajs vs ~10k pairs needed | 50 drivers / 66 days | ✅ **abundant** |
| **R5 trip volume** | **441,710** fares (pickups) | ~95k corpus | ✅ abundant |
| **R6 action-space** | **96.3%** of steps ≤1 cell (99.2% ≤2) at 0.01°; median ping 60 s | pipeline drops ~38–50% | ✅ **cleaner than Shenzhen** |
| **R5 active-unit footprint** | ~**10–12k** active `(cell,hour)` over 32×30×24=23,040 | n_active **34,524** | ⚠️ **~⅓ of Shenzhen** (the one genuine weak point) |

### 10.3 Revised verdict

The pre-data worry — "discriminator-corpus volume" — is **retired**: SF's seeking/driving corpora and fare count are *4–5× Shenzhen's*, pair formation is trivial, and action-space legality (96.3%) is actually *better* than Shenzhen. The remaining risks narrow to:

1. **Smaller fairness-metric footprint** — ~10–12k active units vs 34.5k. This is a direct consequence of SF's compact geography at a faithful ~1 km cell, not a data-volume problem. Still thousands of units — ample for a 3-feature F_causal residual regression and a stable Gini — but ~3× fewer than Shenzhen, so report it honestly.
2. **Recency / window** — 2008, 24 days (vs Shenzhen's 3 months). A relevance caveat for the paper, not a blocker.
3. **Untested (deferred to Phase 2/3)** — the R7 grid-extent/normalizer reconciliation (now SF-specific at 32×30). *(R4 demographic join was probed and PASSED — see §10.4.)*

**Net:** SF's *structural* fit is **stronger** than the pre-data assessment assumed; it remains the #1 recommendation, and the honest residual risk to put before the PI is the **smaller active-unit footprint + 2008/24-day vintage**, not the discriminator corpus.

### 10.4 R4 demographic-join probe — PASS (strong, well-conditioned signal)

The last untested requirement (R4): does the faithful 32×30 SF grid carry usable, non-degenerate demographic signal for F_causal? Joined ACS 5-year (Census Reporter, keyless) for SF + San Mateo + Alameda counties (797 tracts) to the active taxi cells via tract centroids (Gazetteer internal points; population-weighted, nearest-tract fallback). Tested variance + collinearity over the 776 active footprint cells (85% with all features finite).

| Feature (ACS analog) | mean | std | CV | range | VIF |
|---|---|---|---|---|---|
| **housing** = median home value `B25077` | $1.32M | $402k | 0.31 | $315k – $2.0M | 1.68 |
| **comp** = per-capita income `B19301` | $85.1k | $35.0k | 0.41 | $12.6k – $202k | 1.82 |
| **migrant** = foreign-born share `B05002` | 0.33 | 0.128 | 0.39 | 0.08 – 0.95 | 1.18 |
| *(logdensity, sensitivity)* `B01003`/ALAND | 7.63 | 1.36 | 0.18 | 2.18 – 10.38 | 1.04 |

**Verdict: PASS.** All three primary features show substantial cross-cell variance (CV 0.31–0.41; foreign-born share spans 8%→95%, matching SF's real enclave geography), and the design matrix is **well-conditioned — max VIF 1.82, *better* than Shenzhen's primary-set max of 4.45** (`config.py:48`). F_causal's hat-matrix `(I − H_demo)` would be non-degenerate. The smaller active footprint (§10.3) does not starve the demographic signal.

**Caveats (probe-grade):** (1) **Vintage** — ACS 2020–2024, ~14 yr after the 2008 taxi data; absolute values are 2020s-inflated, but the *spatial structure* (and hence the variance R4 needs) is stable. Production must use **2008–2012** ACS (keyed Census API / NHGIS). (2) **Aggregation** — centroid assignment + nearest-tract fallback (60% of active cells fell back, as the taxi footprint extends over bay/SFO/commercial beyond residential tracts); production uses proper areal interpolation, which would also flag non-residential cells. (3) **Construct** — US "foreign-born share" is an *analog* of Shenzhen's rural-migrant/hukou axis, not identical; state this in the paper.

*Reproduce: `python famail_temporal/second_dataset/docs/sf_cabspotting_derisk.py` (structural de-risk) and `python famail_temporal/second_dataset/docs/sf_cabspotting_r4_probe.py` (R4 join; needs network for Census Reporter + Gazetteer). Both standalone; read `source_data/second_dataset/cabspottingdata/` or `$SF_CAB_DIR`.*

---

## References

### Code (`famail_temporal/`)
- `config.py:20, 52-55, 65, 92, 114` — grid dims, demographic features, ε-ball / cGAIL window, soft neighborhood.
- `algorithm/modifier.py:1-22` — pickup-cell relocation; sole mutator of `_base_pickup_3d`.
- `algorithm/objective.py:41` — F_fidelity invocation in the objective.
- `fidelity/README.md:6-9` — inference-only port, no training code.
- `fidelity/model.py:33, 46-47, 285-426` — normalizer extent; three-stream Siamese architecture.
- `fidelity/context.py:10-15, 48-159, 239-265` — same-driver branches, five context tensors, inference padding.
- `fidelity/checkpoint.py`, `fidelity/compute.py:41-42` — freeze/load; similarity score.
- `data/loader.py:247-252` — `nn.Identity` fallback when no checkpoint.
- `data/active_mask.py`, `data/demographics.py`, `data/source_generation/invariants.py` — active units, demographics join, action-space invariant.

### Datasets
- NYC TLC Trip Record Data — https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Chicago Taxi Trips — https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew
- Chicago TNP (rideshare) — https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018-2022-/m6dm-c72p
- Washington DC DFHV Taxicab & FHV Trips — https://opendata.dc.gov/
- US Census ACS 5-year + TIGER/Line — https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-data.html
- **SF Cabspotting** (CRAWDAD epfl/mobility) — https://ieee-dataport.org/open-access/crawdad-epflmobility
- **Porto** (ECML/PKDD 2015, UCI) — https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015
- Rome (CRAWDAD roma/taxi) — https://ieee-dataport.org/open-access/crawdad-romataxi
- DiDi GAIA — https://outreach.didichuxing.com/research/opendata/en/

*Investigation artifacts (deep-research + code-analysis workflow runs) are recorded in the project memory entry `second-dataset-compat`.*
