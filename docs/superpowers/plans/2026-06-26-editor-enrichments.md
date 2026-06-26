# Editor Enrichments (Plan 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Persist three cheap, figure-ready enrichment artifacts from the trajectory editor on the cleaned data — the full per-trajectory attribution distribution (E6), per-edit origin/destination before/after fairness contributions in `trajectories.csv` (E7), and a compact per-iteration `convergence_curve.npz` of F_causal + F_spatial + the fidelity term (E8/E35) — so the optimization-dynamics and "we-edit-the-most-unfair-and-make-them-fairer" figures can be drawn without loading the 14.5 MB `histories.pkl`.

**Architecture:** All three captures are **pure reads of already-computed values** — `scored` from `rank_trajectories`, the `result.grid_before/grid_after` fairness grids, and `result.histories[*].iterations[*]` — serialized at the end of `evaluation/persistence.write()` (alongside the existing artifacts, before the `metrics.json` completion sentinel). Each capture is implemented as a **pure helper** (testable with plain numpy arrays / `SimpleNamespace` stubs) plus a thin write-site call. E6 additionally threads one new optional field, `all_trajectory_scores`, through the `ExperimentResult` dataclass so the full distribution reaches the writer.

**Tech Stack:** Python 3.12, numpy, csv, pytest. No GPU, no torch.

**Spec:** `docs/superpowers/specs/2026-06-25-data-cleanup-rerun-design.md` §6 (E6, E7, E8) + §6.2 (E35). Branch: `data-cleanup-rerun`.

## Global Constraints
- **NO ALGORITHM / NUMERIC CHANGE.** Every capture only READS values the editor already computed (`scored`, the fairness grids, the per-iteration `ModificationResult` fields) and writes them to disk. No edit to the objective, the ST-iFGSM loop, attribution, ranking, or selection. Because nothing in the trajectory-editing algorithm or its intermediate calculations changes, the [[feedback-algorithm-change-protocol]] gate does NOT apply — but if any task finds itself needing to alter a computed value, STOP and escalate.
- **TDD** every helper: pure functions first, tested in isolation (RED → GREEN → commit).
- **Grid channel convention is fixed** (do not reorder): channel `0 = spatial αᵢ`, `1 = causal αᵢ`, `2 = gini_dsr_contrib`, `3 = gini_asr_contrib`. Grid indexing is `grid[cell_x, cell_y, t_block, channel]`. (Confirmed in `_write_per_unit_attribution_csv`, `persistence.py:201-219`.)
- **Register every new artifact** in `write()`'s `artifact_paths` and `file_sizes` dicts so it appears in `metrics.json` (the bundle index), exactly like the existing artifacts.
- **Preserve `metrics.json`-written-LAST** invariant: all three new writes happen BEFORE the `metrics.json` write at the end of `write()`.
- **Backward-compatible result contract:** the new `ExperimentResult.all_trajectory_scores` field MUST have a default (`= None`); `write()` skips the E6 npz when it is `None` so existing constructions (e.g. `test_persistence._fake_result`) keep working unchanged.

---

## File Structure
- **Modify** `famail_temporal/evaluation/runner.py` — add one optional field `all_trajectory_scores: Optional[np.ndarray] = None` to `ExperimentResult`; populate it from `scored` in the `run_experiment` constructor (Task 1 only).
- **Modify** `famail_temporal/evaluation/persistence.py` — add three pure helpers (`_attribution_distribution_payload`, `_origin_dest_fairness`, `_convergence_curve`), extend `_write_trajectories_csv` with 8 columns, and add the npz/csv write+register calls inside `write()`.
- **Create** `famail_temporal/tests/test_editor_enrichments.py` — unit tests for the three helpers + integration assertions through `write()` (reusing `test_persistence._fake_result`).

---

### Task 1: E6 — full attribution distribution (`attribution_distribution.npz`)

**Files:**
- Modify: `famail_temporal/evaluation/runner.py` (`ExperimentResult` dataclass + the `run_experiment` return)
- Modify: `famail_temporal/evaluation/persistence.py` (`write()` + a pure helper)
- Test: `famail_temporal/tests/test_editor_enrichments.py` (create)

**Interfaces:**
- Consumes: `scored: List[Tuple[int, float]]` (already computed at `runner.py:278` via `rank_trajectories`, sorted ascending by score); `result.top_k_scores: List[float]` (the edited-set scores).
- Produces: `ExperimentResult.all_trajectory_scores: Optional[np.ndarray]` (the full per-trajectory score vector, float32, ascending). `persistence._attribution_distribution_payload(all_scores, edited_scores) -> dict` with keys `all_scores` (f32 array), `edited_scores` (f32 array), `n_total` (int64 0-d), `n_negative` (int64 0-d), `n_edited` (int64 0-d). A new file `attribution_distribution.npz` in the run dir, registered in `artifact_paths["attribution_distribution"]`.

- [ ] **Step 1: Write the failing test** (create `famail_temporal/tests/test_editor_enrichments.py`)

```python
"""Unit + integration tests for the editor enrichment captures (E6/E7/E8/E35)."""
import numpy as np
import pytest

from famail_temporal.evaluation import persistence as P


def test_attribution_distribution_payload_counts_and_arrays():
    all_scores = np.array([-3.0, -1.0, 0.0, 2.0], dtype=np.float32)
    edited = np.array([-3.0, -1.0], dtype=np.float32)
    p = P._attribution_distribution_payload(all_scores, edited)
    assert int(p["n_total"]) == 4
    assert int(p["n_negative"]) == 2      # strictly-negative αᵢ marks the editable pool
    assert int(p["n_edited"]) == 2
    np.testing.assert_array_equal(p["all_scores"], all_scores)
    np.testing.assert_array_equal(p["edited_scores"], edited)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_editor_enrichments.py -q`
Expected: FAIL — `AttributeError: module ... persistence has no attribute '_attribution_distribution_payload'`.

- [ ] **Step 3: Write minimal implementation**

In `persistence.py`, add the pure helper (near the other `_write_*` helpers):

```python
def _attribution_distribution_payload(all_scores, edited_scores) -> dict:
    """E6: full per-trajectory attribution distribution + the editable-pool counts.

    `all_scores` is every trajectory's selection αᵢ (sorted ascending);
    `edited_scores` is the subset actually edited (result.top_k_scores).
    n_negative counts strictly-negative αᵢ (the editable pool under the
    F-decomposition convention)."""
    all_scores = np.asarray(all_scores, dtype=np.float32)
    edited_scores = np.asarray(edited_scores, dtype=np.float32)
    return {
        "all_scores": all_scores,
        "edited_scores": edited_scores,
        "n_total": np.int64(all_scores.size),
        "n_negative": np.int64(int((all_scores < 0).sum())),
        "n_edited": np.int64(edited_scores.size),
    }
```

In `runner.py`, add the field to `ExperimentResult` AFTER `rounds` (it must keep a default since `rounds` has one):

```python
    rounds: List[RoundRecord] = field(default_factory=list)
    # E6: every trajectory's selection αᵢ (ascending), for the attribution
    # distribution figure. Optional so synthetic constructions stay valid.
    all_trajectory_scores: Optional[np.ndarray] = None
```

In `runner.py`'s `run_experiment` return (the `ExperimentResult(...)` constructor at ~line 421), add the keyword argument:

```python
            rounds=rounds,
            all_trajectory_scores=np.asarray([s for _, s in scored], dtype=np.float32),
        )
```

In `persistence.py`'s `write()`, after the `per_unit_attribution.csv` block (~line 320) and BEFORE the `metrics = {...}` dict, add:

```python
    if result.all_trajectory_scores is not None:
        path = out_dir / "attribution_distribution.npz"
        np.savez(path, **_attribution_distribution_payload(
            result.all_trajectory_scores, result.top_k_scores))
        artifact_paths["attribution_distribution"] = path.name
        file_sizes["attribution_distribution"] = path.stat().st_size
```

- [ ] **Step 4: Add the integration test, then run the file**

Append to `test_editor_enrichments.py`:

```python
def test_write_emits_attribution_distribution_npz(tmp_path):
    from famail_temporal.tests.test_persistence import _fake_result
    from dataclasses import replace
    result = replace(
        _fake_result(),
        all_trajectory_scores=np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float32),
        top_k_scores=[-2.0, -1.0],
    )
    out_dir = P.write(result, output_root=tmp_path)
    npz = np.load(out_dir / "attribution_distribution.npz")
    assert int(npz["n_total"]) == 4 and int(npz["n_negative"]) == 2
    assert int(npz["n_edited"]) == 2
    import json
    meta = json.loads((out_dir / "metrics.json").read_text())
    assert "attribution_distribution" in meta["artifact_paths"]


def test_write_skips_attribution_distribution_when_scores_absent(tmp_path):
    from famail_temporal.tests.test_persistence import _fake_result
    out_dir = P.write(_fake_result(), output_root=tmp_path)  # all_trajectory_scores defaults None
    assert not (out_dir / "attribution_distribution.npz").exists()
```

Run: `python -m pytest famail_temporal/tests/test_editor_enrichments.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Confirm no regression in the existing editor tests**

Run: `python -m pytest famail_temporal/tests/test_persistence.py famail_temporal/tests/test_runner.py -q`
Expected: PASS (the new optional field defaults None; existing constructions unaffected).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/evaluation/runner.py famail_temporal/evaluation/persistence.py famail_temporal/tests/test_editor_enrichments.py
git commit -m "feat(editor): persist full attribution distribution npz (E6)"
```

---

### Task 2: E7 — per-edit origin/destination before/after fairness columns in `trajectories.csv`

**Files:**
- Modify: `famail_temporal/evaluation/persistence.py` (`_write_trajectories_csv` + a pure helper)
- Test: `famail_temporal/tests/test_editor_enrichments.py` (extend)

**Interfaces:**
- Consumes: `result.grid_before`, `result.grid_after` (shape `(X, Y, T, 4)`), the per-edit `orig = h.original.pickup_cell`, `modc = h.modified.pickup_cell`, and the `tb` (t_block) already computed in `_write_trajectories_csv` (`persistence.py:150-152`).
- Produces: `persistence._origin_dest_fairness(grid_before, grid_after, orig, dest, tb) -> list[float]` returning 8 floats in header order; 8 new trailing columns in `trajectories.csv`.

- [ ] **Step 1: Write the failing test** (append to `test_editor_enrichments.py`)

```python
def test_origin_dest_fairness_reads_correct_channels():
    gb = np.zeros((4, 4, 2, 4), dtype=np.float32)
    ga = np.zeros((4, 4, 2, 4), dtype=np.float32)
    gb[1, 2, 0, 0] = 0.11   # origin spatial BEFORE
    gb[1, 2, 0, 1] = 0.13   # origin causal  BEFORE
    ga[1, 2, 0, 1] = 0.05   # origin causal  AFTER
    ga[3, 0, 0, 0] = 0.21   # dest spatial   AFTER
    ga[3, 0, 0, 1] = 0.23   # dest causal    AFTER
    vals = P._origin_dest_fairness(gb, ga, (1, 2), (3, 0), 0)
    # order: o_spatial_b, o_spatial_a, o_causal_b, o_causal_a,
    #        d_spatial_b, d_spatial_a, d_causal_b, d_causal_a
    assert vals[0] == pytest.approx(0.11)
    assert vals[2] == pytest.approx(0.13)
    assert vals[3] == pytest.approx(0.05)
    assert vals[5] == pytest.approx(0.21)
    assert vals[7] == pytest.approx(0.23)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_editor_enrichments.py -k origin_dest -q`
Expected: FAIL — `AttributeError: ... '_origin_dest_fairness'`.

- [ ] **Step 3: Write minimal implementation**

In `persistence.py`, add the pure helper:

```python
def _origin_dest_fairness(grid_before, grid_after, orig, dest, tb) -> list:
    """E7: spatial(ch0)+causal(ch1) αᵢ at the origin and destination cells,
    before and after the edit. Values may be NaN if a cell is inactive at tb."""
    ox, oy = int(orig[0]), int(orig[1])
    dx, dy = int(dest[0]), int(dest[1])
    return [
        float(grid_before[ox, oy, tb, 0]), float(grid_after[ox, oy, tb, 0]),
        float(grid_before[ox, oy, tb, 1]), float(grid_after[ox, oy, tb, 1]),
        float(grid_before[dx, dy, tb, 0]), float(grid_after[dx, dy, tb, 0]),
        float(grid_before[dx, dy, tb, 1]), float(grid_after[dx, dy, tb, 1]),
    ]
```

In `_write_trajectories_csv`, extend the `headers` list (append after `"sign_flip_rate"`):

```python
        "sign_flip_rate",
        "origin_spatial_attr_before", "origin_spatial_attr_after",
        "origin_causal_attr_before",  "origin_causal_attr_after",
        "dest_spatial_attr_before",   "dest_spatial_attr_after",
        "dest_causal_attr_before",    "dest_causal_attr_after",
    ]
```

In the row-write (the `writer.writerow([...])` call), the last current element is `sign_flip_rate,`. Replace the closing `])` so the 8 new values are appended:

```python
                sign_flip_rate,
                *_origin_dest_fairness(result.grid_before, result.grid_after, orig, modc, tb),
            ])
```

(`orig`, `modc`, and `tb` are already in scope — defined at `persistence.py:150-152`.)

- [ ] **Step 4: Add a header-presence integration test, then run the file**

Append to `test_editor_enrichments.py`:

```python
def test_trajectories_csv_has_origin_dest_columns(tmp_path):
    import csv
    from famail_temporal.tests.test_persistence import _fake_result
    out_dir = P.write(_fake_result(), output_root=tmp_path)  # histories=[] -> header only
    with open(out_dir / "trajectories.csv") as f:
        header = next(csv.reader(f))
    for col in ("origin_causal_attr_before", "dest_causal_attr_after",
                "origin_spatial_attr_before", "dest_spatial_attr_after"):
        assert col in header
```

Run: `python -m pytest famail_temporal/tests/test_editor_enrichments.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm no regression**

Run: `python -m pytest famail_temporal/tests/test_persistence.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/evaluation/persistence.py famail_temporal/tests/test_editor_enrichments.py
git commit -m "feat(editor): per-edit origin/dest before-after fairness columns in trajectories.csv (E7)"
```

---

### Task 3: E8/E35 — per-iteration `convergence_curve.npz` (F_causal + F_spatial + fidelity)

**Files:**
- Modify: `famail_temporal/evaluation/persistence.py` (`write()` + a pure helper)
- Test: `famail_temporal/tests/test_editor_enrichments.py` (extend)

**Interfaces:**
- Consumes: `result.histories: List[ModificationHistory]`; each `h.iterations: List[ModificationResult]` whose entries carry `f_causal`, `f_spatial`, `f_fidelity`, `objective_value` floats (recorded unconditionally per iteration — confirmed `modifier.py:511-516`).
- Produces: `persistence._convergence_curve(histories) -> dict` with keys `iteration` (int64), `mean_f_causal`, `mean_f_spatial`, `mean_f_fidelity`, `mean_objective` (float64), `n_contributing` (int64), all length = max iteration count across histories. A new file `convergence_curve.npz`, registered in `artifact_paths["convergence_curve"]`.

- [ ] **Step 1: Write the failing test** (append to `test_editor_enrichments.py`)

```python
from types import SimpleNamespace


def _mr(c, s, fi, o):
    return SimpleNamespace(f_causal=c, f_spatial=s, f_fidelity=fi, objective_value=o)


def test_convergence_curve_handles_ragged_iterations():
    h1 = SimpleNamespace(iterations=[_mr(0.5, 0.3, 0.1, 1.0), _mr(0.6, 0.35, 0.1, 0.9)])
    h2 = SimpleNamespace(iterations=[_mr(0.7, 0.4, 0.2, 1.2)])  # patience fired early
    c = P._convergence_curve([h1, h2])
    assert list(c["iteration"]) == [0, 1]
    assert list(c["n_contributing"]) == [2, 1]
    assert c["mean_f_causal"][0] == pytest.approx(0.6)    # (0.5 + 0.7) / 2
    assert c["mean_f_causal"][1] == pytest.approx(0.6)    # only h1 reached iter 1
    assert c["mean_f_spatial"][0] == pytest.approx(0.35)  # (0.3 + 0.4) / 2
    assert c["mean_f_fidelity"][1] == pytest.approx(0.1)


def test_convergence_curve_empty_histories():
    c = P._convergence_curve([])
    assert c["iteration"].size == 0 and c["mean_f_causal"].size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_editor_enrichments.py -k convergence -q`
Expected: FAIL — `AttributeError: ... '_convergence_curve'`.

- [ ] **Step 3: Write minimal implementation**

In `persistence.py`, add the pure helper:

```python
def _convergence_curve(histories) -> dict:
    """E8/E35: mean per-iteration F_causal / F_spatial / fidelity / objective
    across all edited trajectories. Iteration counts are ragged (patience fires
    at different points); each iteration averages only the trajectories that
    reached it, and n_contributing records how many that was."""
    fields = ("f_causal", "f_spatial", "f_fidelity", "objective_value")
    out_keys = ("mean_f_causal", "mean_f_spatial", "mean_f_fidelity", "mean_objective")
    if not histories:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return {"iteration": empty_i, "n_contributing": empty_i.copy(),
                **{k: empty_f.copy() for k in out_keys}}
    max_iters = max(len(h.iterations) for h in histories)
    means = {f: [] for f in fields}
    n_contrib = []
    for it in range(max_iters):
        vals = {f: [] for f in fields}
        for h in histories:
            if it < len(h.iterations):
                r = h.iterations[it]
                for f in fields:
                    vals[f].append(getattr(r, f))
        n_contrib.append(len(vals["f_causal"]))
        for f in fields:
            means[f].append(float(np.mean(vals[f])) if vals[f] else float("nan"))
    return {
        "iteration": np.arange(max_iters, dtype=np.int64),
        "mean_f_causal":   np.asarray(means["f_causal"],       dtype=np.float64),
        "mean_f_spatial":  np.asarray(means["f_spatial"],      dtype=np.float64),
        "mean_f_fidelity": np.asarray(means["f_fidelity"],     dtype=np.float64),
        "mean_objective":  np.asarray(means["objective_value"], dtype=np.float64),
        "n_contributing":  np.asarray(n_contrib,               dtype=np.int64),
    }
```

In `write()`, after the E6 attribution-distribution block and BEFORE the `metrics = {...}` dict, add:

```python
    path = out_dir / "convergence_curve.npz"
    np.savez(path, **_convergence_curve(result.histories))
    artifact_paths["convergence_curve"] = path.name
    file_sizes["convergence_curve"] = path.stat().st_size
```

- [ ] **Step 4: Add an integration test, then run the file**

Append to `test_editor_enrichments.py`:

```python
def test_write_emits_convergence_curve_npz(tmp_path):
    from famail_temporal.tests.test_persistence import _fake_result
    out_dir = P.write(_fake_result(), output_root=tmp_path)  # histories=[] -> empty curve
    assert (out_dir / "convergence_curve.npz").exists()
    npz = np.load(out_dir / "convergence_curve.npz")
    assert npz["iteration"].size == 0          # empty histories -> empty curve
    import json
    meta = json.loads((out_dir / "metrics.json").read_text())
    assert "convergence_curve" in meta["artifact_paths"]
```

Run: `python -m pytest famail_temporal/tests/test_editor_enrichments.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Confirm no regression across the editor suite**

Run: `python -m pytest famail_temporal/tests/test_persistence.py famail_temporal/tests/test_runner.py famail_temporal/tests/test_modifier.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/evaluation/persistence.py famail_temporal/tests/test_editor_enrichments.py
git commit -m "feat(editor): compact per-iteration convergence_curve.npz (f_causal+f_spatial+fidelity) (E8/E35)"
```

---

## Self-Review
- **Spec coverage:** E6 (Task 1 — `attribution_distribution.npz` with the α<0 editable-pool counts), E7 (Task 2 — origin/dest before/after spatial+causal columns), E8 (Task 3 — per-iteration mean F_causal), E35 (Task 3 — broadened to F_spatial + fidelity + objective per iteration). All four covered.
- **No-algorithm-change check:** every helper reads existing `result` fields; the only structural change is adding an OPTIONAL, default-`None` field to `ExperimentResult` and populating it from `scored` (which is already computed). Zero edits to the objective / ST-iFGSM / attribution / selection math.
- **Type consistency:** `_attribution_distribution_payload`, `_origin_dest_fairness`, `_convergence_curve` signatures match their call sites in `write()` and `_write_trajectories_csv`. Grid channel indices (0 spatial, 1 causal) match the fixed convention. `all_trajectory_scores` default `None` is honored by the `write()` skip-guard and by `_fake_result`.
- **Placeholder scan:** none — every step has complete code.

## Notes — what is NOT in this plan (deliberately, per the pickup decisions)
- The editor `--diagnostics` multi-seed edit-delta envelope is SKIPPED (locked decision: only L1-v2 multi-seed error bars are a costly extra, in Plan 4).
- `convergence_curve.npz` records the MEAN curve only (no per-trajectory traces) — that is the "no 14.5 MB pkl load" win; per-trajectory data remains in `histories.pkl` for anyone who needs it.

## ⛔ AFTER THIS PLAN'S CODE LANDS → PAUSE FOR THE EDITOR GPU RE-RUN (do NOT run without explicit user go-ahead)
Once Tasks 1-3 are merged-clean on the branch, the editor must be re-run on the **cleaned** bundle to produce the new `*_cleaned` edit-dir that every downstream experiment consumes. This is a ~2 h GPU run on the RTX 3070 — **PAUSE and get explicit user approval first** (execution-mode decision).

- **Command (to be confirmed at pause time, matching the original causal-emphasis config):**
  `python -m famail_temporal.evaluation.runner --name causal_emphasis_no-dedup_cleaned -k 10000 --alpha-spatial 0.2 --alpha-causal 0.7 --alpha-fidelity 0.1 [original flags]`
  → verify the exact `-k`, alphas, dedup, and discriminator flags against the OLD run's `metrics.json` (`config_overrides`/`command_line`) in `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup/metrics.json` BEFORE launching, so the only intended difference is the cleaned input data.
- **Output:** a NEW dir under `famail_temporal/results/<stamp>_..._causal_emphasis_no-dedup_cleaned/` carrying the three new artifacts (`attribution_distribution.npz`, enriched `trajectories.csv`, `convergence_curve.npz`).
- **EDIT-DIR HANDOFF (Plan 4):** the downstream runners (`run_level1_table.py:231`, `run_level2_table.py:296`, and the v2/L2/weighted-BC/variance/pareto runners) default `--edit-dir` to the OLD `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`. After the re-run, every experiment MUST be pointed at the new `_cleaned` edit-dir (via the `--edit-dir`/`--edit-from-dir` flag at run time, or by updating the defaults in Plan 4). Record the new dir name in the SDD ledger the moment the editor run completes.

Related: [[data-cleanup-rerun-pickup]], [[feedback-algorithm-change-protocol]], spec §6/§6.2.
