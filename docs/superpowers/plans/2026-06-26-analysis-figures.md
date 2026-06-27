# Analysis & Figures (Plan 5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the cleaned-data artifacts into paper-grade tables and figures — the dataset-cleanup summary (E31), the editor dirty-vs-clean delta (E22 editor-level), the per-sink F_spatial decomposition (E23), the before/after gradient heatmaps (E16), and the dual-Pareto + CSV mirrors (E15/E17) — plus the experiment-level cleanup_delta + headline-table figures once the experiment sequence finishes (Wave 2).

**Architecture:** Each artifact is a small, mostly-pure analysis function in a new `famail_temporal/analysis/` package (no torch where avoidable), with a thin CLI. Pure transforms (delta computation, decomposition, CSV mirroring) are unit-tested with synthetic inputs; the I/O-heavy drivers read existing on-disk artifacts and are validated against the real files. Figures use matplotlib (Agg, headless).

**Tech Stack:** Python 3.12, numpy, pandas, matplotlib (Agg), pytest. CPU only.

**Spec:** `docs/superpowers/specs/2026-06-25-data-cleanup-rerun-design.md` §6/§6.2 (E15, E16, E17, E22, E23, E31). Recon: workflow `wg0n3zsri` (figure/analysis code map). Branch: `data-cleanup-rerun`.

## Global Constraints
- **DO NOT modify `source_data/` or `cache/` while the experiment sequence (`baaigffdf`) is running** — the later runners reload them at startup. Tasks that need the DIRTY bundle loaded (E23 dirty export, E16 dirty heatmap) WRITE code now but DEFER execution to after the sequence (a safe swap: back up cleaned → restore dirty → export → restore cleaned, all post-sequence).
- **Read-only over existing artifacts.** Analysis reads editor `metrics.json`/grids, `processing_metadata.json`, experiment result JSONs, and the enrichment `.npz`/`.csv` — it never recomputes the editor or the experiments.
- **Grid channel convention** (fixed): channel `0 = spatial αᵢ` (sums to F_spatial), `1 = causal αᵢ` (sums to F_causal). Sink cells are 1-indexed in the pipeline grid; the headline sink (29,53) = `grid[28,52]` 0-indexed.
- **Pin to confirmed schemas** (from recon + the smoke): editor `metrics.json` keys `metrics_before/metrics_after/deltas{f_spatial,f_causal,gini_dsr,gini_asr}`, `dataset{n_trajectories,n_drivers,n_active_units}`; `processing_metadata.json` keys `removal_summary{n_removed,removal_rate,total_seeking_extracted,...}` (+ clean-only `stuck_gps_sinks`); L1-v2 `level1_v2_metrics.json` `sources.{raw,edited,bc,gan}.{f_causal,f_spatial,fidelity_a,fidelity_b,...}` + `level1_v2_multiseed.json` `per_source.{src}.{metric}.{mean,std,values,t_ci}`.
- **TDD** every pure transform; frequent commits.

## File Structure
- **Create** `famail_temporal/analysis/__init__.py`.
- **Create** `famail_temporal/analysis/_io.py` — tiny shared readers (`read_json`, `editor_metrics(dir)`, `processing_metadata(dir)`).
- **Create** `famail_temporal/analysis/dataset_summary.py` (E31), `cleanup_delta.py` (E22), `sink_decomposition.py` (E23), `heatmap_pair.py` (E16), and extend `baselines/pareto.py`/`figure.py`/`run_data_pareto.py` (E15/E17).
- **Create** `famail_temporal/analysis/tests/` with a test module per task.

---

### Task 1: shared analysis I/O (`_io.py`)

**Files:** Create `famail_temporal/analysis/__init__.py`, `famail_temporal/analysis/_io.py`, `famail_temporal/analysis/tests/__init__.py`, `famail_temporal/analysis/tests/test_io.py`.

**Interfaces:**
- `read_json(path) -> dict`
- `editor_metrics(run_dir) -> dict` (loads `<run_dir>/metrics.json`)
- `processing_metadata(source_dir) -> dict` (loads `<source_dir>/processing_metadata.json`)

- [ ] **Step 1: failing test** (`tests/test_io.py`)
```python
import json
from famail_temporal.analysis import _io


def test_read_json_and_editor_metrics(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"deltas": {"f_causal": 0.012}}))
    assert _io.read_json(d / "metrics.json")["deltas"]["f_causal"] == 0.012
    assert _io.editor_metrics(d)["deltas"]["f_causal"] == 0.012
```

- [ ] **Step 2: run → FAIL** (`python -m pytest famail_temporal/analysis/tests/test_io.py -q`).

- [ ] **Step 3: implement** (`_io.py`)
```python
from __future__ import annotations
import json
from pathlib import Path


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def editor_metrics(run_dir) -> dict:
    return read_json(Path(run_dir) / "metrics.json")


def processing_metadata(source_dir) -> dict:
    return read_json(Path(source_dir) / "processing_metadata.json")
```
Also create empty `analysis/__init__.py` and `analysis/tests/__init__.py`.

- [ ] **Step 4: run → PASS.**

- [ ] **Step 5: commit** `feat(analysis): shared artifact readers for the figures/analysis layer`

---

### Task 2: E31 — dataset cleanup summary (`dataset_summary.py`)  [READY-NOW: runnable immediately]

**Files:** Create `famail_temporal/analysis/dataset_summary.py` + `tests/test_dataset_summary.py`.

**Interfaces:**
- `dataset_summary(dirty_meta: dict, clean_meta: dict) -> dict` (pure) → rows comparing dirty vs clean: `n_removed`, `removal_rate`, `total_seeking_extracted`, plus `stuck_gps_sinks.n_pickups_removed` / number of sink cells (clean only), and the derived `phantom_pickups_removed` + `seeking_corpus_dirty/clean`.
- `write_dataset_summary(dirty_source_dir, clean_source_dir, out_dir) -> Path` (CLI) → writes `dataset_summary.json` + `dataset_summary.md`.

- [ ] **Step 1: failing test** (pure function, synthetic metadata)
```python
from famail_temporal.analysis.dataset_summary import dataset_summary


def test_dataset_summary_pairs_dirty_and_clean():
    dirty = {"removal_summary": {"n_removed": 195840, "removal_rate": 0.4975,
                                  "total_seeking_extracted": 214286}}
    clean = {"removal_summary": {"n_removed": 119290, "removal_rate": 0.3895,
                                  "total_seeking_extracted": 133091},
             "stuck_gps_sinks": {"n_pickups_removed": 106677,
                                  "flagged_cells": [[17, 39], [29, 53]]}}
    s = dataset_summary(dirty, clean)
    assert s["dirty"]["removal_rate"] == 0.4975
    assert s["clean"]["removal_rate"] == 0.3895
    assert s["clean"]["n_sink_cells"] == 2
    assert s["clean"]["phantom_pickups_removed"] == 106677
    assert s["delta"]["removal_rate"] == round(0.3895 - 0.4975, 4)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement** `dataset_summary` (pull the `removal_summary` scalars from each, read `stuck_gps_sinks` from clean only with `.get(...)` defaults, compute the deltas + `n_sink_cells = len(flagged_cells)`), then `write_dataset_summary` (calls `_io.processing_metadata` on each dir, renders a small markdown table, writes both files; CLI via `argparse` with `--dirty-source`, `--clean-source`, `--out-dir`).

- [ ] **Step 4: run → PASS.**

- [ ] **Step 5: RUN IT (safe now — reads only `processing_metadata.json`):**
`python -m famail_temporal.analysis.dataset_summary --dirty-source famail_temporal/source_data_dirty --clean-source famail_temporal/source_data --out-dir famail_temporal/results/analysis/dataset_summary`
Confirm `dataset_summary.{json,md}` written with the real numbers (dirty 0.4975 / clean 0.3895 / 106,677 phantom).

- [ ] **Step 6: commit** `feat(analysis): dataset cleanup summary dirty-vs-clean (E31)`

---

### Task 3: E22 (editor-level) — editor dirty-vs-clean fairness delta (`cleanup_delta.py`)  [READY-NOW]

**Files:** Create `famail_temporal/analysis/cleanup_delta.py` + `tests/test_cleanup_delta.py`.

**Interfaces:**
- `editor_delta(dirty_metrics: dict, clean_metrics: dict) -> dict` (pure) → for each of `f_spatial,f_causal` (and gini): the dirty `metrics_before/after/delta`, the clean `metrics_before/after/delta`, and the dirty→clean shift in the BEFORE baseline (the sink-removal effect) + in the edit delta (edit-signal robustness).
- `write_editor_cleanup_delta(dirty_run_dir, clean_run_dir, out_dir) -> Path` → `cleanup_delta_editor.csv` + `.json`.

- [ ] **Step 1: failing test** (synthetic editor metrics)
```python
from famail_temporal.analysis.cleanup_delta import editor_delta


def test_editor_delta_isolates_baseline_shift_and_edit_robustness():
    dirty = {"metrics_before": {"f_spatial": 0.0822, "f_causal": 0.8052},
             "metrics_after":  {"f_spatial": 0.0825, "f_causal": 0.8180},
             "deltas": {"f_spatial": 0.0003, "f_causal": 0.0128}}
    clean = {"metrics_before": {"f_spatial": 0.1034, "f_causal": 0.8069},
             "metrics_after":  {"f_spatial": 0.1025, "f_causal": 0.8193},
             "deltas": {"f_spatial": -0.0009, "f_causal": 0.0124}}
    d = editor_delta(dirty, clean)
    # F_spatial baseline rose by the sink removal (~+0.021)
    assert round(d["f_spatial"]["baseline_shift_dirty_to_clean"], 4) == round(0.1034 - 0.0822, 4)
    # F_causal edit delta is ~unchanged (robust): |Δ_clean - Δ_dirty| small
    assert abs(d["f_causal"]["edit_delta_shift"]) < 0.001
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement** `editor_delta` (per metric: `baseline_shift = clean.before - dirty.before`; `after_shift = clean.after - dirty.after`; `edit_delta_dirty = dirty.deltas`; `edit_delta_clean = clean.deltas`; `edit_delta_shift = clean.deltas - dirty.deltas`), then `write_editor_cleanup_delta` (uses `_io.editor_metrics`, writes a flat CSV one-row-per-metric + JSON; CLI `--dirty-run`, `--clean-run`, `--out-dir`).

- [ ] **Step 4: run → PASS.**

- [ ] **Step 5: RUN IT (safe now — reads only the two editor `metrics.json`):**
`python -m famail_temporal.analysis.cleanup_delta --dirty-run famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup --clean-run famail_temporal/results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned --out-dir famail_temporal/results/analysis/cleanup_delta`
Confirm the F_spatial baseline shift ≈ +0.021 (sink removal) and the F_causal edit-delta shift ≈ 0 (robustness).

- [ ] **Step 6: commit** `feat(analysis): editor-level dirty-vs-clean fairness delta (E22 editor)`

---

### Task 4: E17 — pareto CSV mirror + E15 — dual-Pareto + removed-id list (`pareto.py`/`figure.py`/`run_data_pareto.py`)

**Files:** Modify `famail_temporal/baselines/pareto.py`, `famail_temporal/baselines/figure.py`, `famail_temporal/baselines/run_data_pareto.py`; test `famail_temporal/baselines/tests/test_pareto_csv.py` (create).

**Interfaces:**
- `pareto.points_to_csv_rows(points) -> list[dict]` (pure) — flat `asdict` rows (E17).
- `figure.plot_pareto` is already metric-generic; the driver writes BOTH `pareto_causal.png` (metric `f_causal`) and `pareto_spatial.png` (metric `f_spatial`) (E15).
- `pareto.filtered_points` gains an optional capture of `removed_indices` per K (E15) → persisted to `pareto_removed_ids.json`.

- [ ] **Step 1: failing test** (`tests/test_pareto_csv.py`)
```python
from dataclasses import asdict
from famail_temporal.baselines.pareto import ParetoPoint, points_to_csv_rows


def test_points_to_csv_rows_flat():
    pts = [ParetoPoint("raw", 1.0, 0.10, 0.81, 0.7, 0.8, 0),
           ParetoPoint("filter@100", 0.999, 0.11, 0.82, 0.69, 0.8, 100)]
    rows = points_to_csv_rows(pts)
    assert rows[0]["label"] == "raw" and rows[1]["n_removed"] == 100
    assert set(rows[0]) == set(asdict(pts[0]))
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**
  - `points_to_csv_rows(points)`: `[asdict(p) for p in points]`.
  - In `run_data_pareto.main`, after `pareto_points.json`: write `pareto_points.csv` via `csv.DictWriter` over `points_to_csv_rows(points)`; and write `pareto_causal.png` + `pareto_spatial.png` (two `plot_pareto` calls with `metric="f_causal"` / `"f_spatial"`).
  - In `pareto.filtered_points`, additionally return/collect the `ranked[:k_eff]` trajectory ids per K; expose them so the driver writes `pareto_removed_ids.json` (`{f"filter@{k}": [ids...]}`). Keep `ParetoPoint` unchanged (carry the ids in a parallel dict the driver persists) to avoid a frozen-dataclass schema change.

- [ ] **Step 4: run → PASS** (the pure-row test; plus `python -c "import famail_temporal.baselines.run_data_pareto"`).

- [ ] **Step 5: RUN IT (safe now — `--edit-from-dir` is CPU, reads cleaned bundle read-only):**
`python -m famail_temporal.baselines.run_data_pareto --edit-from-dir famail_temporal/results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned --k-levels 100 500 1000 2293 5000 10000 --out-dir famail_temporal/results/analysis/pareto_cleaned`
(`2293` = the edited-N, for the equal-N point.) Confirm `pareto_points.{json,csv}`, `pareto_causal.png`, `pareto_spatial.png`, `pareto_removed_ids.json`.

- [ ] **Step 6: commit** `feat(pareto): CSV mirror + dual F_causal/F_spatial pareto + removed-id list (E15/E17)`

---

### Task 5: E23 — per-sink F_spatial decomposition (`sink_decomposition.py`)  [CODE NOW; EXECUTE AFTER SEQUENCE]

**Files:** Create `famail_temporal/analysis/sink_decomposition.py` + `tests/test_sink_decomposition.py`.

**Interfaces:**
- `sink_spatial_contributions(dense_spatial: np.ndarray, active_mask: np.ndarray, sink_cells_1idx: list[tuple]) -> dict` (pure) → per sink cell (converted to 0-idx via −1), the summed channel-0 (spatial) αᵢ over active t-blocks = that cell's F_spatial contribution; plus the total over all sinks.
- `decompose(dirty_dense_pkl, clean_dense_pkl, sink_cells, out_dir) -> Path` → reads both `fairness_attribution_dense.pkl` (`["spatial"]`,`["active_mask"]`), differences per sink → `sink_f_spatial_decomposition.json` (per sink: dirty contrib, clean contrib, delta; and the share each sink is of the total F_spatial shift).

- [ ] **Step 1: failing test** (pure, synthetic dense arrays)
```python
import numpy as np
from famail_temporal.analysis.sink_decomposition import sink_spatial_contributions


def test_sink_contribution_sums_active_spatial_channel():
    dense = np.zeros((4, 4, 2), dtype=np.float32)   # (gx,gy,T) spatial channel
    mask = np.zeros((4, 4, 2), dtype=bool)
    dense[2, 3, 0] = 0.05; mask[2, 3, 0] = True
    dense[2, 3, 1] = 0.02; mask[2, 3, 1] = True
    dense[2, 3, 1] = 0.02
    # sink at 1-indexed (3,4) -> 0-indexed (2,3); sum over active t = 0.07
    out = sink_spatial_contributions(dense, mask, [(3, 4)])
    assert round(out["per_sink"]["(3, 4)"], 4) == 0.07
    assert round(out["total"], 4) == 0.07
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement** `sink_spatial_contributions` (for each 1-idx `(x,y)`: `x0,y0=x-1,y-1`; `contrib = float(np.nansum(dense[x0,y0,:][mask[x0,y0,:]]))`) and `decompose` (load both dense pkls, call the pure fn on each, diff, compute shares; CLI `--dirty-export`, `--clean-export`, `--out-dir`; sink list = the 10 calibrated cells, headline (29,53)).

- [ ] **Step 4: run → PASS.**

- [ ] **Step 5: DEFERRED EXECUTION (AFTER `baaigffdf` completes — needs two `export_fairness_attributions` runs, one against the DIRTY bundle).** Documented procedure (do NOT run while the sequence is live):
  1. `python -m famail_temporal.evaluation.export_fairness_attributions --name clean` (cleaned bundle currently on disk).
  2. Safely swap to dirty: back up `source_data`+`cache`, restore `source_data_dirty` → regenerate cache → `export_fairness_attributions --name dirty` → restore cleaned `source_data`+`cache`. (Or, if a dirty cache snapshot exists, point at it.)
  3. `python -m famail_temporal.analysis.sink_decomposition --dirty-export <dirty_export_dir> --clean-export <clean_export_dir> --out-dir famail_temporal/results/analysis/sink_decomposition`.

- [ ] **Step 6: commit** `feat(analysis): per-sink F_spatial decomposition dirty-vs-clean (E23)`

---

### Task 6: E16 — before/after gradient-heatmap pair (`heatmap_pair.py`)  [CODE NOW; EXECUTE AFTER SEQUENCE]

**Files:** Create `famail_temporal/analysis/heatmap_pair.py` + `tests/test_heatmap_pair.py`. Reuse `visualization/gradient_heatmap/{render.py,loader.py,geometry.py}`.

**Interfaces:**
- `write_heatmap_png(bundle_npz_path, *, quantity, term, hour, out_png) -> Path` — loads a `gradient_viz_bundle.npz` via the viz `loader`, picks the field via `render.select_field`, computes `render.color_range`, calls `render.export_png` (headless bytes), writes the PNG to disk.
- A driver that, given a `_with_sinks` bundle and a `_cleaned` bundle, writes `heatmap_{with_sinks,cleaned}_{quantity}_{term}.png` + a difference panel.

- [ ] **Step 1: failing test** — `write_heatmap_png` on a synthetic minimal bundle npz (build the documented `gradient_viz_bundle.npz` schema with tiny arrays, assert a non-empty PNG file is written). If constructing the full schema is heavy, test the thin slice-selection helper instead and note it.

- [ ] **Step 2–4:** implement + green (reusing `render.export_png` / `select_field` / `color_range`; the orientation guard already lives in `precompute`).

- [ ] **Step 5: DEFERRED EXECUTION (AFTER sequence — the dirty bundle needs dirty data):** run `precompute` twice with distinct `--out` (`gradient_viz_bundle_with_sinks.npz` from the dirty bundle, `..._cleaned.npz` from the cleaned bundle — same safe-swap procedure as Task 5), then `heatmap_pair` to render the pair + diff.

- [ ] **Step 6: commit** `feat(analysis): before/after gradient-heatmap pair driver (E16)`

---

## Wave 2 — experiment-dependent (detail + implement AFTER `baaigffdf` finishes)
Pinned to the Plan-4 output schemas (confirmed by the L1-v2 smoke); placeholders until the real result dirs exist:
- **E22 experiment-level `cleanup_delta`** — dirty-vs-clean for L1-v2 (`sources.{src}.f_causal/f_spatial`), L2 (`per_source`/`paired`), weighted-BC (`per_arm`/`paired_vs_raw`/`dose_response`), variance (`per_seed_values`). Dirty baselines located (recon §3): `level1_table_v2/2026-06-18_full_run`, `level2_table/2026-06-18T17-27-34`, `weighted_bc_sweep/{sig_6seed_w10_w20_w30,placebo_6seed_w10_w30}`, `variance_suite/...2026-06-11...`.
- **Headline-table figures** — L1-v2 error-bar table (from `level1_v2_multiseed.json`), L2 negative-transfer CI plot (E14), weighted-BC **dose-response + edit-vs-most-fair-vs-placebo** figure (from `dose_response.json` + `paired_stats.json` + the new `most_fair_w*` arms), Fid-B component violins (E9/E36 per-seed arrays).
- **E40 intro teaser / E33 PoI overlay** — composite figures; design once the core figures exist.
- **Figures manifest + rerun-README** — wire `baselines/_bundle_index.register_figure(...)` (Plan 2) for every figure produced, and `write_rerun_readme(...)` over all cleaned result dirs.

## Self-Review
- **Spec coverage:** E31 (T2), E22-editor (T3), E17+E15 (T4), E23 (T5), E16 (T6); E22-experiment + headline figures + E40/E33 + bundle-index = Wave 2. All accounted for (none silently dropped).
- **Safety:** T5/T6 executions are explicitly deferred behind the source_data-swap constraint; T2/T3/T4 read editor outputs / metadata / cleaned-bundle read-only and are safe to run during the sequence.
- **Type consistency:** `_io` readers feed every driver; pure transforms (`dataset_summary`, `editor_delta`, `sink_spatial_contributions`, `points_to_csv_rows`) are tested with synthetic inputs matching the confirmed on-disk schemas.

## Execution Handoff
After the experiment sequence finishes: implement Wave 2, then run the final whole-branch review (opus) over the entire `data-cleanup-rerun` branch (triaging the Minor nits logged across Plans 1–4 + the most-fair double-compute), then superpowers:finishing-a-development-branch → merge to `main`.

Related: [[data-cleanup-rerun-pickup]], [[pickup-gps-sinks]], [[meeting40-plan]], spec §6/§6.2.
