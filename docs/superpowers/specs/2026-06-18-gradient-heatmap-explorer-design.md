# Gradient Heatmap Explorer — Design

- **Date:** 2026-06-18
- **Status:** Approved (brainstorming complete; ready for implementation planning)
- **Branch context:** created off `level-2-usability` work; tool lives under `famail_temporal/visualization/`
- **Author:** Robert Ashe (with Claude)
- **Origin:** FAMAIL Meeting 39 action item — *"Robert to generate gradient heat maps across all 24 hourly timesteps to visualize gradient magnitudes across the spatial state space."*

---

## 1. Motivation & context

In Meeting 39, the PI (Dr. Xin Zhang) and Robert discussed switching the trajectory-selection
criterion of the `famail_temporal` editing algorithm from **per-trajectory contribution score**
to **gradient magnitude**. The conceptual claims that motivated this — and that the heat map is
meant to **verify** — are:

1. **Spatial gradient ∝ concentration** — `∂F_spatial/∂pickup` is larger in cells where rides
   concentrate (spatial fairness is roughly linear in cell concentration).
2. **Causal gradient peaks at district boundaries** — `∂F_causal/∂pickup` has higher magnitude near
   district borders, a consequence of demographic-data granularity (a double regression on
   demographics), even where the raw causal-fairness *value* is unremarkable.
3. **|attribution| ≈ gradient magnitude** — the current negative-attribution selection criterion is
   *probably* a good proxy for high gradient magnitude, but a heat map would confirm it and guide
   whether to change the selection rule.

The tool is a **diagnostic** to inspect these fields visually, one hourly slice at a time, with
Shenzhen district boundaries overlaid. It is not a paper-figure pipeline, though it exports
publication-quality PNGs.

### Why hourly slices

The state space is 3D (48 × 90 × 24). As agreed in the meeting, we visualize 2D spatial slices and
**cycle through the 24 hourly time blocks** (each block is the Mon–Fri weekday average for that
hour). This is the natural, lossless way to present the third dimension.

---

## 2. Findings from codebase exploration (ground truth)

These were verified directly against the code and saved artifacts and are the foundation of the design.

- **The gradient field already exists.** `famail_temporal/evaluation/diagnostics.py :: compute_gradient_sensitivity(bundle, pickup_3d)` returns a `(48, 90, 24, 2)` float32 array:
  - channel 0 = `∂F_spatial/∂pickup[x,y,t]`
  - channel 1 = `∂F_causal/∂pickup[x,y,t]`
  - inactive cells are `NaN`.
  22 such arrays are already saved as `famail_temporal/results/*/gradient_sensitivity_{before,after}.pkl` (dict with keys `grid`, `channel_names`, `time_blocks`, `active_mask`).

- **Grid is `48 × 90 × 24`, not 49×90×24.** Verified: `config.GRID_DIMS == (48, 90)`, `T == 24`; `district_id_grid` and `valid_mask` are `(48, 90)`; all saved grids are `(48, 90, 24, ·)`. The "49" in the original request is an off-by-one from a 0-indexed range — exactly the class of error behind prior axis-flip bugs.

- **Per-cell attribution field.** `famail_temporal/evaluation/grid.py :: build_fairness_grid(bundle)` returns `(48, 90, 24, 4)`: ch0 = per-cell spatial contribution αᵢ (Σ = F_spatial), ch1 = per-cell causal contribution αᵢ (Σ = F_causal), ch2/ch3 = Gini DSR/ASR diagnostics. Sign convention: positive = contributes above the 1/N baseline (fair), negative = drags below (priority for editing).

- **F_fidelity has no per-cell spatial gradient.** F_fidelity is a per-trajectory discriminator score; its gradient flows through LSTM context tensors, not through pickup-cell counts, so `∂F_fidelity/∂pickup_cell ≈ 0`. This is why the saved gradient field has only 2 channels. Confirmed empirically at the *per-trajectory* level too: across all 25 saved runs, `grad_fidelity_norm` is exactly 0 in every recent/production run and ≤ 4.7e-6 in a handful of early diagnostic runs. **There is no meaningful fidelity gradient to visualize.**

- **The per-term gradient/attribution fields are α-weight-independent.** `∂F_spatial` and `∂F_causal` (and their attributions) do not depend on `ALPHA_*`; the weights only appear when *combining* terms. Therefore the **Combined** and **Spatial+Causal** views can be computed in-app as α-weighted sums of cached per-term arrays, with α as live display knobs — no torch recompute on α change.

- **Everything needed is dataset-level ("before-edit").** Gradient, attribution, and concentration all derive from the raw dataset via `DataBundle.load()`. The tool needs **no specific experiment results directory** (the only layer that did — the fidelity proxy — is ≈0 and was dropped).

- **Orientation (verified against geography AND the user's ArcGIS screenshot).** Data is indexed `array[x_grid (row, 0=South → 47=North), y_grid (col, 0=West → 89=East)]`. District centroids land correctly (Nanshan SW, Bao'an far-W, Guangming NW, Dapeng far-E, Pingshan/Longgang NE, Yantian SE coast), and a rendered district map matches the screenshot with no left-right mirror.

  **Canonical display recipe:** horizontal axis = `y_grid` (West→East, left→right); vertical axis = `x_grid` (South→North) with **South at the bottom** (`origin='lower'` for Matplotlib; for Plotly `go.Heatmap`, map `z[row=x_grid][col=y_grid]` with the row axis increasing **upward**, no y-reversal). Cells must be square.

---

## 3. Goals & non-goals

### Goals
- View any 1 of the 24 hourly slices of the 48×90 grid as a heatmap of the selected quantity/term.
- 5 fairness **filters**: `F_spatial`, `F_causal`, `F_fidelity`, `Combined`, `Spatial+Causal`.
- 3 **quantities**: `Gradient`, `Attribution`, `Concentration` (the two comparison layers chosen by the user, switchable against the gradient).
- Always-available **Shenzhen district boundary** overlay; non-Shenzhen cells masked out.
- **Square cells** and **correct orientation** (per §2 recipe), robust to the historical flip/mirror bugs.
- Controls to cycle hours (slider + prev/next + optional play) and to choose quantity/term.
- Display controls: signed vs |magnitude|; shared-vs-per-slice color scaling; percentile clip.
- **Concentration as a switchable panel**, with an optional concentration-contour overlay on the gradient/attribution panel.
- Export a publication-quality PNG of the current view.

### Non-goals (YAGNI — explicitly deferred)
- Per-experiment before/after comparison; loading arbitrary `results/*` runs.
- Live algorithm re-runs from the UI.
- The per-trajectory fidelity proxy (≈0 everywhere; dropped after evidence review).
- 3D / temporal-aggregate views; cross-hour animation export.
- Multi-dataset selection.

---

## 4. Architecture — precompute + thin app

Three components, chosen over "compute live in the app" so the UI never imports torch and stays
instant; the heavy `DataBundle.load()` + autograd cost is paid once.

1. **`precompute.py` (CLI).** Loads `DataBundle`, computes the gradient field
   (`compute_gradient_sensitivity`), attribution field (`build_fairness_grid`), and pickup
   concentration; assembles district geometry + boundary segments; writes one compact cache file.
   CPU-only; run once (or after a dataset change). Accepts `--out` and is idempotent.

2. **Bundle cache** (`cache/gradient_viz_bundle.npz`). Compact numeric arrays + district geometry +
   metadata. Readable with numpy alone (no torch).

3. **`app.py` (Streamlit + Plotly).** Loads the cached bundle via `st.cache_data`, renders the
   selected slice interactively. Pure numpy/pandas/plotly at render time.

Supporting modules: `loader.py` (read/validate bundle), `geometry.py` (district boundary segments,
orientation assertions), `render.py` (Plotly figure builders + Matplotlib PNG export).

---

## 5. Data model — cached bundle contents

| Key | Shape | Source | Meaning |
|---|---|---|---|
| `grad_spatial` | (48,90,24) f32 | `compute_gradient_sensitivity` ch0 | ∂F_spatial/∂pickup (signed) |
| `grad_causal`  | (48,90,24) f32 | ch1 | ∂F_causal/∂pickup (signed) |
| `attr_spatial` | (48,90,24) f32 | `build_fairness_grid` ch0 | per-cell spatial contribution αᵢ (signed) |
| `attr_causal`  | (48,90,24) f32 | ch1 | per-cell causal contribution αᵢ (signed) |
| `pickup`       | (48,90,24) f32 | `bundle.pickup_3d` | ride concentration (counts) |
| `active_mask`  | (48,90,24) bool | `bundle.mask_3d` | active cells; inactive → NaN/transparent |
| `district_id_grid` | (48,90) int8 | mapping pkl | district id per cell; -1 = non-Shenzhen |
| `valid_mask`   | (48,90) bool | mapping pkl | True = inside Shenzhen |
| `district_names` | (10,) str | mapping pkl | id → name |
| `boundary_segments` | list | derived | line segments between differing district ids + Shenzhen outer edge |
| `meta` | dict | config | default α's, grid dims, source provenance, precompute timestamp |

Inactive cells across the spatial/attribution arrays are stored as `NaN` and rendered transparent.

---

## 6. UI model

Two orthogonal selectors plus time and display controls.

- **Quantity:** `Gradient` · `Attribution` · `Concentration`.
- **Term / filter:** `F_spatial` · `F_causal` · `F_fidelity` · `Combined` · `Spatial+Causal`.
- **Time:** hour slider `0–23`, ◀/▶ buttons, optional "play" auto-advance.
- **Display options:**
  - signed (diverging, centered at 0) vs **|magnitude|** (sequential);
  - color scale **shared across all 24 hours** (default — magnitudes comparable hour-to-hour) vs per-slice autoscale;
  - robust **percentile clip** (e.g. clip color range to the 99th percentile of |value|) to tame outliers;
  - toggle **district boundaries** (on by default);
  - toggle **concentration contour overlay** on the main panel.
- **Concentration panel:** a switchable second panel (chosen over an always-on overlay) showing the
  `pickup` layer for the current hour, for side-by-side comparison with the gradient. The optional
  contour overlay draws concentration iso-lines on top of the gradient/attribution panel.

### Filter → array mapping & combination math

Let `gsp, gca` be the per-term arrays for the selected **Quantity** (gradient or attribution), and
`αsp, αca, αfi` the (live, slider-adjustable) weights defaulting to `meta` config values.

- `F_spatial`     → `gsp`
- `F_causal`      → `gca`
- `Spatial+Causal`→ `αsp·gsp + αca·gca`
- `Combined`      → `αsp·gsp + αca·gca + αfi·0` (≡ Spatial+Causal at the per-cell level, because the
  fidelity term contributes no per-cell field; the app annotates this equivalence).
- `F_fidelity`    → flat field; render with a prominent banner: **"F_fidelity has no per-cell spatial
  gradient (≈0 by construction); it acts as a per-trajectory realism constraint, not a spatial steering force."**
- **Concentration** quantity is term-agnostic — it ignores the term selector and always shows `pickup`.

---

## 7. Rendering

- **Library:** Plotly `go.Heatmap` for the interactive app (square cells via `yaxis.scaleanchor='x', scaleratio=1`; hover shows `x_grid, y_grid, district name, value`; zoom/pan). Matplotlib for a **"Download publication PNG"** button (crisp boundaries, paper-ready).
- **Orientation:** per the verified §2 recipe. `geometry.py` includes an assertion that recomputes a few district centroids from `district_id_grid` and checks the expected compass quadrants, so a future data/orientation regression fails loudly.
- **Color maps:** signed quantities → `RdBu_r` centered at 0; magnitude/concentration → sequential (`Viridis`; concentration offers a log scale because counts are skewed).
- **District boundaries:** drawn from precomputed `boundary_segments` (edges between cells of
  differing district id, plus the Shenzhen/non-Shenzhen edge) as line traces over the heatmap —
  giving crisp ArcGIS-style outlines rather than relying on color blocks.
- **Masking:** inactive/non-Shenzhen cells transparent (NaN), so the basemap/background shows through
  and the district outline reads cleanly.

---

## 8. File layout & run

```
famail_temporal/visualization/gradient_heatmap/
  precompute.py     # CLI: DataBundle -> cache/gradient_viz_bundle.npz
  app.py            # Streamlit entrypoint
  loader.py         # read + validate cached bundle
  geometry.py       # district boundary segments + orientation assertions
  render.py         # Plotly figure builders + Matplotlib PNG export
  cache/            # gradient_viz_bundle.npz (gitignored; a derived artifact, regenerated by precompute)
  README.md
  tests/
    test_geometry.py        # orientation invariants, boundary segments
    test_loader.py          # bundle round-trip: shapes, NaN, attribution sums-to-F
    test_combination.py     # α-reweighting math (Combined ≡ Spatial+Causal at cell level)
```

Run: `streamlit run famail_temporal/visualization/gradient_heatmap/app.py`
Precompute: `python -m famail_temporal.visualization.gradient_heatmap.precompute`

Dependencies: `streamlit`, `plotly`, `numpy`, `pandas`, `matplotlib` (all already in use); `torch`
only inside `precompute.py` (already a dependency).

---

## 9. Testing strategy

- **Orientation invariants** (`test_geometry.py`): district centroids fall in expected compass
  quadrants (Nanshan SW, Dapeng far-E, Bao'an far-W, Guangming NW); the canonical-orientation render
  places South at the bottom and West at the left.
- **Bundle round-trip** (`test_loader.py`): precompute → load preserves shapes/dtypes; inactive cells
  are NaN; `nansum(attr_spatial)` ≈ F_spatial and `nansum(attr_causal)` ≈ F_causal per the
  attribution partition property.
- **Combination math** (`test_combination.py`): `Combined` equals `Spatial+Causal` at the per-cell
  level; α reweighting is exact and linear.
- **Boundary segments** (`test_geometry.py`): every segment separates two differing district ids (or
  Shenzhen/non-Shenzhen); count is stable.
- **Manual:** render the in-app district map, confirm against the ArcGIS screenshot.

---

## 10. Risks & open questions

- **Orientation regressions** — mitigated by the centroid assertion in `geometry.py` and the
  Plotly-vs-Matplotlib parity (both must produce South-at-bottom, West-at-left).
- **`DataBundle.load()` cost / dependencies in precompute** — it may pull the discriminator even
  though the gradient field only needs spatial+causal. Acceptable for a one-time CPU precompute;
  the plan should confirm precompute runs without a GPU and without a trained discriminator
  checkpoint (fidelity is not needed for any cached layer).
- **Cache provenance** — `meta` records the dataset/config snapshot so a stale cache is detectable;
  the `.npz` is **gitignored and regenerated** by `precompute.py` (a derived artifact, not committed).
- **Combined ≡ Spatial+Causal** — intentional and annotated; this equivalence is itself a useful
  visual confirmation that fidelity exerts no per-cell steering.
