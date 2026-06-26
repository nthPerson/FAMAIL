# Experiment-Runner Enrichments (Plan 4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make every cleaned-data experiment run persist paper-grade data — surfacing values the runners already compute then discard (Fid-B per-component, Fid-A separation), adding cheap diagnostics (terminal-cell entropy + trip-length for the degeneracy check, t-based 95% CIs, dose-response table, effective-edited-fraction, reproducible placebo id sets, gate per-pair score arrays, terminal-cell histogram vectors), and adding L1-v2 multi-seed error bars — so the headline tables, the dose-response/trade-off curves, the L2 negative-transfer CI plot, and the edit-specificity panel all draw from persisted artifacts.

**Architecture:** All computation lives in ONE new pure, fully-unit-tested module `baselines/_enrich.py` (no torch import). The four experiment runners (`run_level2_table.py`, `run_weighted_bc_smoke.py`, `run_level1_table_v2.py`, `run_variance_suite.py`) change only **additively** — they collect already-computed values that were being dropped, call the pure helpers, and write new artifacts next to their existing outputs. The single structural change is L1-v2's new multi-seed wrapper (E18), which RUNS the existing scoring N times and aggregates — it changes no calculation.

**Tech Stack:** Python 3.12, numpy, scipy.stats (t/wilcoxon — already a dependency), pytest. No GPU, no new torch usage.

**Spec:** `docs/superpowers/specs/2026-06-25-data-cleanup-rerun-design.md` §6 (E9–E14, E18), §6.1 (E24–E28), §6.2 (E36). Branch: `data-cleanup-rerun`.

## Global Constraints
- **ADDITIVE ONLY — no change to any existing computed value.** Every task either (a) collects a value the runner already computes and currently drops, (b) computes a NEW diagnostic from data already in scope, or (c) writes a NEW artifact. Do NOT alter F_causal/F_spatial/Fidelity-A/Fidelity-B/gate math, the training, the rollout, or any existing output's values. Because no scoring/algorithm calculation changes, the [[feedback-algorithm-change-protocol]] gate does NOT apply — but if a task finds itself needing to change a computed value, STOP and escalate.
- **All logic in `_enrich.py` is pure + TDD'd.** Runner wiring is additive call-sites only; the runners are GPU-bound and not unit-tested here — their end-to-end validation happens at the experiment runs (the next pause). Where a runner change is a pure reshape, route it through an `_enrich.py` helper so it IS tested.
- **JSON-safe scalars in `*.json`; arrays go to `*.npz`.** Never dump a 4320-length histogram into JSON. Per-seed scalar arrays (already small) stay in the JSON `"values"` lists; large vectors (terminal-cell histograms, gate per-pair scores) go to `.npz`.
- **Register/point at the cleaned edit-dir at RUN time** (next pause), not in code: the new edit-dir is `famail_temporal/results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned`. Runs pass `--edit-dir`/`--edit-from-dir` explicitly. (Do not hardcode it in the runners.)
- **`N_CELLS = 4320`** (`= GX*GY = 48*90`, `baselines/gan/config.py`); flat index `x*GY + y`. **Fid-B components = `fe._STAT_KEYS_V2`** = `("length","mean_displacement","coverage","radius_of_gyration","net_displacement")` + `"terminal_cell"`.

---

## File Structure
- **Create** `famail_temporal/baselines/_enrich.py` — pure helpers: `t_ci`, `shannon_entropy_bits`, `degeneracy_scalars`, `effective_edited_fraction`, `dose_response_table`, `chosen_placebo_ids`.
- **Create** `famail_temporal/baselines/tests/test_enrich.py`.
- **Modify** `famail_temporal/baselines/run_level2_table.py` — surface E9 components + E11 degeneracy in `_evaluate_policy`; collect them in the seed loop; add t_ci; write `paired_stats.json` (E26). (E14/E25 per-seed arrays already in `level2_metrics.json`.)
- **Modify** `famail_temporal/baselines/run_weighted_bc_smoke.py` — collect E9 components; write `dose_response.json` (E10), `paired_stats.json` (E26), `chosen_ids.json` (E27); add `effective_edited_fraction` per weight + degeneracy scalars to `sweep.json` (E28). (E24 placebo + E25 F_spatial ride along.)
- **Modify** `famail_temporal/baselines/run_level1_table_v2.py` — multi-seed wrapper (E18) + per-component Fid-B per seed (E36) + terminal-cell histogram npz (E13) + gate per-pair score npz (E12).
- **Modify** `famail_temporal/baselines/run_variance_suite.py` — add raw per-seed value arrays to `aggregate.json` (convenience).

---

### Task 1: Pure enrichment helpers (`_enrich.py`)

**Files:**
- Create: `famail_temporal/baselines/_enrich.py`
- Test: `famail_temporal/baselines/tests/test_enrich.py`

**Interfaces (Produces):**
- `t_ci(values, confidence=0.95) -> tuple[float,float]` — t-based CI of the mean; `(nan,nan)` if <2 values.
- `shannon_entropy_bits(hist) -> float` — Shannon entropy (base-2) of a non-negative vector (normalized internally; 0.0 if empty/all-zero).
- `degeneracy_scalars(terminal_pickups, gen_cells, *, n_cells) -> dict` with keys `terminal_cell_entropy_bits`, `mean_trip_length`, `std_trip_length` (E11).
- `effective_edited_fraction(n_edited, n_total, w) -> float` — `(n_edited*w)/(n_edited*w + (n_total-n_edited))` (E28).
- `dose_response_table(per_arm, paired_vs_raw, weights) -> list[dict]` — flat rows `{w, delta_f_causal, wilcoxon_p, fidelity_b, fidelity_a}` (E10).
- `chosen_placebo_ids(raw_traj_ids, edited_id_set, placebo_seed, k=None) -> list[int]` — deterministic re-derivation of the placebo subset's trajectory_ids (E27).

- [ ] **Step 1: Write the failing test** (create `tests/test_enrich.py`)

```python
"""Unit tests for the pure runner-enrichment helpers (Plan 4)."""
import math
import numpy as np
import pytest

from famail_temporal.baselines import _enrich as E


def test_t_ci_basic_and_degenerate():
    lo, hi = E.t_ci([1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.95)
    assert lo < 3.0 < hi                      # CI brackets the mean (3.0)
    assert math.isnan(E.t_ci([1.0])[0])       # <2 values -> nan
    assert math.isnan(E.t_ci([])[0])


def test_shannon_entropy_bits():
    assert E.shannon_entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)  # uniform 4 -> 2 bits
    assert E.shannon_entropy_bits([1, 0, 0, 0]) == pytest.approx(0.0)  # one cell -> 0 bits
    assert E.shannon_entropy_bits([0, 0, 0]) == 0.0                    # empty -> 0
    assert E.shannon_entropy_bits(np.array([5.0, 5.0])) == pytest.approx(1.0)


def test_degeneracy_scalars():
    # 3 generated trajectories of lengths 2, 3, 4; terminals at 3 distinct cells
    gen_cells = [[(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2), (3, 3)]]
    terminal_pickups = [(1, 1, 0), (2, 2, 0), (3, 3, 0)]
    d = E.degeneracy_scalars(terminal_pickups, gen_cells, n_cells=4320)
    assert d["mean_trip_length"] == pytest.approx(3.0)
    assert d["std_trip_length"] == pytest.approx(1.0)            # ddof=1 of [2,3,4]
    assert d["terminal_cell_entropy_bits"] == pytest.approx(math.log2(3), abs=1e-6)  # 3 equally-used cells


def test_effective_edited_fraction():
    assert E.effective_edited_fraction(2000, 95297, 1) == pytest.approx(2000 / 95297)
    # w=30: (2000*30)/(2000*30 + 93297)
    assert E.effective_edited_fraction(2000, 95297, 30) == pytest.approx(60000 / (60000 + 93297))


def test_dose_response_table():
    per_arm = {
        "edited_w10": {"fidelity_b": {"mean": 0.30}, "fidelity_a": {"mean": 0.84}},
        "edited_w30": {"fidelity_b": {"mean": 0.31}, "fidelity_a": {"mean": 0.83}},
    }
    paired = {"f_causal": {
        "edited_w10": {"mean": 0.018, "wilcoxon_p": 0.06},
        "edited_w30": {"mean": 0.027, "wilcoxon_p": 0.03},
    }}
    rows = E.dose_response_table(per_arm, paired, [10, 30])
    assert rows[0] == {"w": 10, "delta_f_causal": 0.018, "wilcoxon_p": 0.06,
                       "fidelity_b": 0.30, "fidelity_a": 0.84}
    assert rows[1]["w"] == 30 and rows[1]["delta_f_causal"] == 0.027


def test_chosen_placebo_ids_deterministic():
    raw_ids = list(range(10))
    edited = {0, 1, 2}
    a = E.chosen_placebo_ids(raw_ids, edited, placebo_seed=12345, k=3)
    b = E.chosen_placebo_ids(raw_ids, edited, placebo_seed=12345, k=3)
    assert a == b                                  # deterministic
    assert len(a) == 3
    assert all(i not in edited for i in a)         # never picks edited ids
    # default k = len(edited)
    assert len(E.chosen_placebo_ids(raw_ids, edited, placebo_seed=1)) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -q`
Expected: FAIL — `ModuleNotFoundError: ... _enrich`.

- [ ] **Step 3: Write minimal implementation** (`_enrich.py`)

```python
"""Pure, dependency-light helpers for runner enrichment (Plan 4).
No torch import. Every function is unit-tested; runners call these at write-sites."""
from __future__ import annotations
import math
import random
import numpy as np
from famail_temporal.baselines.transmission import terminal_cell_histogram


def t_ci(values, confidence: float = 0.95):
    """t-based confidence interval of the MEAN. (nan, nan) if fewer than 2 values."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    from scipy.stats import t
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1)) / math.sqrt(n)
    h = sem * float(t.ppf(0.5 + confidence / 2.0, n - 1))
    return (mean - h, mean + h)


def shannon_entropy_bits(hist) -> float:
    """Shannon entropy (base-2) of a non-negative vector; normalized internally."""
    p = np.asarray(hist, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(-np.sum(p * np.log2(p)))


def degeneracy_scalars(terminal_pickups, gen_cells, *, n_cells) -> dict:
    """E11 collapse check: terminal-cell entropy (bits) + trip-length mean/std.
    Low entropy or near-1 trip length => degenerate generator."""
    hist = terminal_cell_histogram(terminal_pickups, n_cells=n_cells)
    lengths = [len(seq) for seq in gen_cells]
    return {
        "terminal_cell_entropy_bits": shannon_entropy_bits(hist),
        "mean_trip_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_trip_length": float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
    }


def effective_edited_fraction(n_edited, n_total, w) -> float:
    """E28: weight-adjusted edited mass = (n_edited*w) / (n_edited*w + n_unedited)."""
    n_edited = float(n_edited); n_unedited = float(n_total) - n_edited
    num = n_edited * float(w)
    denom = num + n_unedited
    return float(num / denom) if denom > 0 else 0.0


def dose_response_table(per_arm, paired_vs_raw, weights) -> list:
    """E10: flat rows w -> {delta_f_causal, wilcoxon_p, fidelity_b, fidelity_a}."""
    rows = []
    for w in weights:
        arm = f"edited_w{int(w)}"
        pc = paired_vs_raw.get("f_causal", {}).get(arm, {})
        a = per_arm.get(arm, {})
        rows.append({
            "w": int(w),
            "delta_f_causal": float(pc.get("mean", float("nan"))),
            "wilcoxon_p": pc.get("wilcoxon_p"),
            "fidelity_b": float(a.get("fidelity_b", {}).get("mean", float("nan"))),
            "fidelity_a": float(a.get("fidelity_a", {}).get("mean", float("nan"))),
        })
    return rows


def chosen_placebo_ids(raw_traj_ids, edited_id_set, placebo_seed, k=None) -> list:
    """E27: deterministic re-derivation of the placebo subset's trajectory_ids.
    Mirrors run_weighted_bc_smoke.random_subset_weight_vector's selection:
    sample k indices from the NON-edited positions with random.Random(placebo_seed),
    then map positions back to trajectory_ids."""
    edited = set(edited_id_set)
    non_edited_pos = [i for i, tid in enumerate(raw_traj_ids) if int(tid) not in edited]
    n = len(edited) if k is None else k
    chosen_pos = random.Random(placebo_seed).sample(non_edited_pos, n)
    return [int(raw_traj_ids[i]) for i in chosen_pos]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/_enrich.py famail_temporal/baselines/tests/test_enrich.py
git commit -m "feat(baselines): pure runner-enrichment helpers (t_ci/entropy/degeneracy/dose-response/placebo-ids) (E10/E11/E26/E27/E28)"
```

---

### Task 2: Surface dropped values + degeneracy + paired CI in Level-2 (`run_level2_table.py`)

**Files:**
- Modify: `famail_temporal/baselines/run_level2_table.py`
- Test: `famail_temporal/baselines/tests/test_enrich.py` (extend — pure-augmentation test only)

**Interfaces:**
- Consumes: `_enrich.degeneracy_scalars`, `_enrich.t_ci`.
- Produces: `_evaluate_policy` return dict gains `terminal_cell_entropy_bits`, `mean_trip_length`, `std_trip_length`. `level2_metrics.json` gains per-source per-seed arrays for `fidelity_b_per_component` (each of the 6 keys) + `fidelity_a_separation` + the 3 degeneracy scalars. New `paired_stats.json` = `result["paired"]` with a `t_ci` added to every leaf that has `diffs`.

**Wiring (read the file; the recon line refs may have shifted):**

1. In `_evaluate_policy` (≈line 185–280), where `gen_cells` and the terminal pickups are computed for the Fid-B terminal JS (the `_terminal_pickups_from_cells(gen_cells)` call ≈line 267), add to the return dict:
```python
        **_enrich.degeneracy_scalars(_terminal_pickups_from_cells(gen_cells), gen_cells, n_cells=gc.N_CELLS),
```
(import `from famail_temporal.baselines import _enrich` at module top; `gc` is already imported. Reuse the existing terminal-pickups variable if one is already bound rather than recomputing.)

2. In the seed loop (≈line 502–530), alongside the existing `per_seed_metric[metric][src].append(...)`, collect the dropped/new values into NEW accumulators:
```python
# init before the loop, mirroring per_seed_metric:
per_seed_components = {src: {c: [] for c in (*fe._STAT_KEYS_V2, "terminal_cell")} for src in _SOURCE_ORDER}
per_seed_sep = {src: [] for src in _SOURCE_ORDER}
per_seed_degen = {src: {k: [] for k in ("terminal_cell_entropy_bits", "mean_trip_length", "std_trip_length")} for src in _SOURCE_ORDER}
# inside the loop, after `m = _evaluate_policy(...)`:
for c in (*fe._STAT_KEYS_V2, "terminal_cell"):
    per_seed_components[src][c].append(float(m["fidelity_b_per_component"][c]))
per_seed_sep[src].append(float(m["fidelity_a_separation"]))
for k in ("terminal_cell_entropy_bits", "mean_trip_length", "std_trip_length"):
    per_seed_degen[src][k].append(float(m[k]))
```

3. In the `result` dict assembly (≈line 553–564), add:
```python
    "per_source_fidelity_b_components": per_seed_components,
    "per_source_fidelity_a_separation": per_seed_sep,
    "per_source_degeneracy": per_seed_degen,
```

4. After `level2_metrics.json` is written (≈line 574), add the paired_stats.json (E26) with t_ci:
```python
import copy
paired_ci = copy.deepcopy(result["paired"])
for metric, by_other in paired_ci.items():
    for other, leaf in by_other.items():
        if isinstance(leaf, dict) and "diffs" in leaf:
            leaf["t_ci"] = list(_enrich.t_ci(leaf["diffs"]))
(out_dir / "paired_stats.json").write_text(json.dumps(paired_ci, indent=2, default=float))
```

- [ ] **Step 1: Write the failing test** (append to `test_enrich.py` — test the t_ci augmentation as a pure op; the `_evaluate_policy`/seed-loop wiring is validated at the GPU run)

```python
def test_paired_stats_t_ci_augmentation_shape():
    # mirrors the in-runner augmentation: every leaf with 'diffs' gains a 't_ci' pair
    paired = {"f_causal": {"raw": {"diffs": [0.01, 0.02, 0.03, 0.015, 0.025], "mean": 0.02}}}
    leaf = paired["f_causal"]["raw"]
    leaf["t_ci"] = list(E.t_ci(leaf["diffs"]))
    assert len(leaf["t_ci"]) == 2 and leaf["t_ci"][0] < 0.02 < leaf["t_ci"][1]
```

- [ ] **Step 2: Run it (RED only if you add a not-yet-existing reference; here it should pass once `_enrich.t_ci` exists)**

Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -k t_ci -q`
Expected: PASS (t_ci already exists from Task 1).

- [ ] **Step 3: Apply the wiring above to `run_level2_table.py`.**

- [ ] **Step 4: Verify the runner still imports + the file parses**

Run: `python -c "import famail_temporal.baselines.run_level2_table as m; print('import OK')"`
Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -q`
Expected: both succeed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_level2_table.py famail_temporal/baselines/tests/test_enrich.py
git commit -m "feat(level2): surface Fid-B components + Fid-A separation + degeneracy + paired t_ci (E9/E11/E14/E26)"
```

---

### Task 3: weighted-BC enrichments (`run_weighted_bc_smoke.py`)

**Files:**
- Modify: `famail_temporal/baselines/run_weighted_bc_smoke.py`
- Test: (covered by `_enrich` tests; this task is additive wiring validated at the GPU run)

**Interfaces:**
- Consumes: `_enrich.dose_response_table`, `_enrich.t_ci`, `_enrich.effective_edited_fraction`, `_enrich.chosen_placebo_ids`.
- Produces: in `sweep.json` — gains `effective_edited_fraction` (a `{weight: fraction}` map) + per-arm `fidelity_b_per_component`/`fidelity_a_separation`/degeneracy per-seed arrays. New files `dose_response.json` (E10), `paired_stats.json` (E26), `chosen_ids.json` (E27). Placebo arm (E24) + per-seed F_spatial (E25) already present.

**Wiring (read the file; recon refs ≈ given):**

1. `_evaluate_policy` is imported from `run_level2_table` and now returns the degeneracy scalars (Task 2) — they flow automatically. In the sweep loop (≈line 275–305), where the 4 `_METRICS` are collected per arm, ALSO collect (mirror Task 2's accumulators, keyed by arm not source): `fidelity_b_per_component` (6 keys), `fidelity_a_separation`, and the 3 degeneracy scalars. Add them to `per_arm[arm]` in the `_ms`-assembly block (≈line 308–318) as `{"values": [...]}` arrays. (This makes the placebo arm carry the same schema — E24.)

2. After `sweep.json` is written (≈line 338), add:
```python
from famail_temporal.baselines import _enrich
# E10 dose-response
(out_dir / "dose_response.json").write_text(json.dumps(
    _enrich.dose_response_table(per_arm, paired, up_weights), indent=2, default=float))
# E26 paired_stats with t_ci
import copy
paired_ci = copy.deepcopy(paired)
for metric, by_arm in paired_ci.items():
    for arm, leaf in by_arm.items():
        if isinstance(leaf, dict) and "diffs" in leaf:
            leaf["t_ci"] = list(_enrich.t_ci(leaf["diffs"]))
(out_dir / "paired_stats.json").write_text(json.dumps(paired_ci, indent=2, default=float))
# E27 chosen id sets (edited + placebo per weight)
raw_ids = [int(t.trajectory_id) for t in raw_trajs]
chosen = {"edited_ids": sorted(int(i) for i in eids)}
for w in placebo_weights:
    chosen[f"random_w{int(w)}"] = _enrich.chosen_placebo_ids(raw_ids, eids, args.placebo_seed)
(out_dir / "chosen_ids.json").write_text(json.dumps(chosen, indent=2))
```

3. E28 — add the effective-edited-fraction map into the `result` dict BEFORE it is written (≈line 324–330):
```python
    "effective_edited_fraction": {
        str(int(w)): _enrich.effective_edited_fraction(len(eids), len(raw_trajs), w)
        for w in ([1.0] + up_weights)
    },
```

- [ ] **Step 1: Confirm the helpers exist (no new test needed — `_enrich` is tested in Task 1).**

Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -q` — PASS.

- [ ] **Step 2: Apply the wiring above to `run_weighted_bc_smoke.py`.**

- [ ] **Step 3: Verify import + parse**

Run: `python -c "import famail_temporal.baselines.run_weighted_bc_smoke as m; print('import OK')"`
Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/baselines/run_weighted_bc_smoke.py
git commit -m "feat(weighted-bc): dose-response + paired t_ci + chosen-ids + effective-edited-fraction + component/degeneracy arrays (E9/E10/E24/E26/E27/E28)"
```

---

### Task 4: Level-1 v2 multi-seed wrapper + per-component/gate/histogram artifacts (`run_level1_table_v2.py`)

**Files:**
- Modify: `famail_temporal/baselines/run_level1_table_v2.py`
- Test: `famail_temporal/baselines/tests/test_enrich.py` (extend — test ONLY the pure aggregation helper added here)

**Interfaces:**
- Consumes: `_enrich.t_ci`, `fe._score_identity_pairs`, `transmission.terminal_cell_histogram`.
- Produces: `--seeds` arg (default `"0"`, comma-list). When >1 seed, `level1_v2_metrics.json` carries per-source per-metric `{mean, std, values, t_ci}` across seeds + per-component Fid-B per-seed arrays (E18/E36). New `terminal_cell_histograms.npz` (raw/edited/bc/gan N_CELLS vectors, E13) and `gate_pair_scores.npz` (matched/mismatched per-pair HuMID probs, E12).

**This is the one structural change. Approach: extract the per-seed scoring body into a helper, loop it, aggregate. Do NOT change any scoring math.**

1. Add a `_mean_std` helper (copy L2's, ≈run_level2_table.py:533) and extend it with `t_ci`:
```python
def _mean_std_ci(vals):
    import numpy as np
    from famail_temporal.baselines._enrich import t_ci
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
            "t_ci": list(t_ci(vals))}
```

2. Add `--seeds` to the arg parser (keep the existing `--seed` working: if `--seeds` absent, use `[args.seed]`):
```python
ap.add_argument("--seeds", type=str, default=None,
                help="Comma-separated seeds for multi-seed error bars (E18). Defaults to [--seed].")
# in main(): seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
```

3. **Extract** the stochastic per-seed scoring (the BC/GAN `_train_and_generate_cond` calls + the Fidelity-A pass + Fidelity-B pass + bc/gan fairness, ≈lines 388–607) into a function `_score_one_seed(seed, *, <all the fixtures it reads: bundle, raw_trajs, driver_to_idx, groups, profiles, disc, eval_drivers, device, args, ...>) -> dict` returning the per-source dict (the `result["sources"]` payload for that seed) PLUS the per-component dicts PLUS the gate. The raw/edited FAIRNESS reads (`data_level_fairness`, `_edited_fairness_from_metrics`) are deterministic — compute them ONCE outside the loop and merge in. Move `set_all_seeds`/`random.Random(seed)` usage to use the loop `seed` (the training is already seeded inside `_train_and_generate_cond(seed=...)`).

4. Loop seeds, accumulate per-source per-metric value lists + per-component value lists, then aggregate each with `_mean_std_ci`. For the gate, compute per seed; report the seed-0 gate as the headline `result["gate"]` plus `result["gate_all_passed"] = all(...)`. Keep the single-seed path byte-compatible when `len(seeds)==1` (means==values[0]).

5. E13 — after the loop (or at seed 0), compute terminal-cell histograms for each source from the pickup tuple-lists already built in the Fid-B pass and write:
```python
from famail_temporal.baselines.transmission import terminal_cell_histogram
np.savez(out_dir / "terminal_cell_histograms.npz",
         raw=terminal_cell_histogram(raw_pickups, n_cells=gc.N_CELLS),
         edited=terminal_cell_histogram(edited_pickups, n_cells=gc.N_CELLS),
         bc=terminal_cell_histogram(bc_pickups_term, n_cells=gc.N_CELLS),
         gan=terminal_cell_histogram(gan_pickups_term, n_cells=gc.N_CELLS))
```

6. E12 — at the gate site (where `matched["raw"]`/`mismatched["raw"]` are in scope), capture per-pair scores and write:
```python
m_scores = fe._score_identity_pairs(disc, matched["raw"], batch_size=64, device=device)
mm_scores = fe._score_identity_pairs(disc, mismatched["raw"], batch_size=64, device=device)
np.savez(out_dir / "gate_pair_scores.npz", matched=np.asarray(m_scores), mismatched=np.asarray(mm_scores))
```

- [ ] **Step 1: Write the failing test** (append to `test_enrich.py` — test the `_mean_std_ci` aggregation logic in isolation by importing it from the runner)

```python
def test_l1v2_mean_std_ci_single_and_multi():
    from famail_temporal.baselines.run_level1_table_v2 import _mean_std_ci
    one = _mean_std_ci([0.81])
    assert one["mean"] == pytest.approx(0.81) and one["std"] == 0.0
    assert one["values"] == [0.81]
    multi = _mean_std_ci([0.80, 0.82, 0.81, 0.83, 0.79])
    assert multi["mean"] == pytest.approx(0.81)
    assert len(multi["t_ci"]) == 2 and multi["t_ci"][0] < 0.81 < multi["t_ci"][1]
```

- [ ] **Step 2: Run it to confirm RED** (the helper doesn't exist yet)

Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -k l1v2 -q`
Expected: FAIL — `ImportError: cannot import name '_mean_std_ci'`.

- [ ] **Step 3: Apply the wiring above. Keep all scoring math byte-identical; only restructure into the loop + add artifacts.**

- [ ] **Step 4: Verify import, parse, and the single-seed default still works**

Run: `python -c "import famail_temporal.baselines.run_level1_table_v2 as m; print('import OK')"`
Run: `python -m pytest famail_temporal/baselines/tests/test_enrich.py -q`
Expected: both succeed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_level1_table_v2.py famail_temporal/baselines/tests/test_enrich.py
git commit -m "feat(level1-v2): multi-seed wrapper + per-component Fid-B per seed + terminal-cell + gate-pair npz (E12/E13/E18/E36)"
```

---

### Task 5: variance-suite per-seed value arrays (`run_variance_suite.py`)

**Files:**
- Modify: `famail_temporal/baselines/run_variance_suite.py`
- Test: (additive; validated at the run — the per-seed `seed_{s}.json` already hold the data)

**Interfaces:**
- Produces: `aggregate.json` gains a `per_seed_values` block: for each of `b0`/`famail` and each `METRIC_KEYS` metric, the raw per-seed value array (so paired re-tests don't have to re-read every `seed_{s}.json`).

**Wiring:** In `main()`, the per-seed metric dicts are already accumulated (the `entry["b0"]`/`entry["famail"]` written to each `seed_{s}.json`). Keep an in-memory list of those per seed and, when assembling `aggregate.json` (≈line 300–322), add:
```python
"per_seed_values": {
    arm: {k: [float(e[arm][k]) for e in seed_entries] for k in METRIC_KEYS}
    for arm in ("b0", "famail")
},
```
where `seed_entries` is the list of per-seed `entry` dicts (collect it during the loop if not already retained).

- [ ] **Step 1: Apply the wiring (read the loop to find the accumulator; add `seed_entries.append(entry)` if needed).**

- [ ] **Step 2: Verify import + parse**

Run: `python -c "import famail_temporal.baselines.run_variance_suite as m; print('import OK')"`
Expected: OK.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/baselines/run_variance_suite.py
git commit -m "feat(variance): surface raw per-seed value arrays in aggregate.json (E14/E25 convenience)"
```

---

## Self-Review
- **Spec coverage:** E9 (T2/T3 surface components+separation), E10 (T3 dose_response), E11 (T1 degeneracy_scalars → T2/T3), E12 (T4 gate_pair_scores.npz), E13 (T4 terminal_cell_histograms.npz), E14 (T2 per-seed arrays already in json + t_ci; per-source Fid-B violin via E9 component arrays — the separate L2 trajectory_stats.npz is subsumed by L1-v2's existing one, noted here not silently dropped), E18 (T4 multi-seed), E24 (T3 placebo rides the same arm schema), E25 (already in per_arm/per_source `values`; surfaced), E26 (T2/T3 paired_stats.json + t_ci), E27 (T3 chosen_ids.json), E28 (T3 effective_edited_fraction), E36 (T4 per-component Fid-B per seed).
- **Additive-only check:** no task changes a scoring/training/rollout calculation; T4's multi-seed wrapper RUNS existing scoring N times. All else collects-dropped-values / computes-new-diagnostics / writes-new-files.
- **Type consistency:** `_enrich` signatures match their call-sites in T2–T4. `t_ci` returns a 2-tuple (serialized via `list(...)`). Degeneracy keys identical across T2/T3. Fid-B component keys = `_STAT_KEYS_V2 + ("terminal_cell",)` everywhere.
- **Placeholder scan:** T1 is fully code-complete. T2–T5 give exact additive snippets + the recon line refs; the implementer reads each runner to place them (the runners are not unit-testable without GPU — flagged, not a silent gap).

## ⛔ AFTER THIS PLAN'S CODE LANDS → PAUSE FOR THE EXPERIMENT RUNS (GPU, ~15h; needs explicit user go-ahead)
Runs serialize on the one GPU, each pointed at the NEW cleaned edit-dir `--edit-dir famail_temporal/results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned`. Order (spec §5/§8): **L1-v2 multi-seed** (`--seeds 0,1,2,3,4`; validates the cleaned edit-dir + frozen-discriminator gate first) → **weighted-BC + placebo** (`--seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30`) → **L2** (`--seeds 0,1,2,3,4`) → **variance** (`--seeds 0,1,2,3,4`). CPU analysis (Pareto `--edit-from-dir`, gradient-heatmap precompute, cell-histogram, attribution export) runs concurrently (no GPU contention). Confirm each command + the new edit-dir at pause time. Retire L1-v1 + metric_hardening (not re-run).

Related: [[data-cleanup-rerun-pickup]], [[weighted-bc-transfer]], [[feedback-algorithm-change-protocol]], spec §6/§6.1/§6.2.
