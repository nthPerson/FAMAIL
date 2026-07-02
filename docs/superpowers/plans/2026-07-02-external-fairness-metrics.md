# External Fairness Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute four established fairness metrics (supply/demand ratio, demographic parity, disparate impact, Theil index) before→after trajectory edit, over the active `(cell,t)` units, with bootstrap CIs, tables, and figures — for Shenzhen (3 feature sets) + SF sf12.

**Architecture:** Three new modules under `famail_temporal/baselines/`: a **pure** metric/grouping/bootstrap layer (numpy in, scalar out), an **IO** layer (bundle → service ratio `Y`, per-unit equity-axis demographics, edited demand grid), and a **run/report** layer (orchestration + JSON + markdown tables + figures + CLI). Grouping/regions are derived from per-cell demographic *values* (city-agnostic), not the Shenzhen-only district files.

**Tech Stack:** Python 3, numpy, matplotlib, pytest. Reuses `famail_temporal.data.loader.DataBundle`, `data.demographics.enrich_demographics`, `baselines.datasets.{pickup_unit_of,pickup_mass}`.

## Global Constraints

- Outcome variable: `Y = active_taxis / max(pickups, DEMAND_FLOOR)`, `DEMAND_FLOOR = 0.5` (verbatim `config.DEMAND_FLOOR`).
- Group label convention (per unit): `0 = advantaged (A)`, `1 = disadvantaged (D)`, `-1 = excluded`.
- Equity axes (always these 3, independent of the edit's objective feature set): `EQUITY_AXES = ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]`.
- Disadvantaged pole: `{"AvgHousingPricePerSqM": False, "CompPerCapita": False, "MigrantRatio": True}` (low housing, low comp, high migrant).
- City is selected by the `FAMAIL_CITY` env var at process launch (`DataBundle.load()` takes no city arg). SF runs as `FAMAIL_CITY=sf12 python -m ...`.
- Test dir: `famail_temporal/baselines/tests/`. Style: module-level `test_*` functions, `pytest.approx`, synthetic bundle via `from famail_temporal.tests.test_objective import _make_synthetic_bundle`.
- Run tests with: `python -m pytest <path> -v`.

## File Structure

- Create `famail_temporal/baselines/external_fairness.py` — pure metrics, grouping, regions, bootstrap.
- Create `famail_temporal/baselines/external_fairness_io.py` — bundle → `Y`, per-unit demographics, edited demand grid.
- Create `famail_temporal/baselines/run_external_fairness.py` — `assemble_results`, JSON/table/figure rendering, `main()` CLI.
- Create `famail_temporal/baselines/tests/test_external_fairness.py`
- Create `famail_temporal/baselines/tests/test_external_fairness_io.py`
- Create `famail_temporal/baselines/tests/test_run_external_fairness.py`

---

### Task 1: Core group-comparison metrics (SDR, DP, DI)

**Files:**
- Create: `famail_temporal/baselines/external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_external_fairness.py`

**Interfaces:**
- Produces: `supply_demand_ratio(Y, groups) -> dict{mean_disadvantaged,mean_advantaged,gap}`, `demographic_parity(Y, groups) -> float`, `disparate_impact(Y, groups) -> float`, and float wrappers `sdr_gap`, `sdr_mean_disadvantaged`, `sdr_mean_advantaged`. `Y`,`groups` are `(N,)` numpy arrays; `groups ∈ {0,1,-1}`.

- [ ] **Step 1: Write the failing test**

```python
# famail_temporal/baselines/tests/test_external_fairness.py
import numpy as np
import pytest

from famail_temporal.baselines import external_fairness as ef


def test_perfect_parity_dp_zero_di_one():
    Y = np.array([2.0, 2.0, 2.0, 2.0])
    groups = np.array([0, 0, 1, 1])          # A, A, D, D
    assert ef.demographic_parity(Y, groups) == pytest.approx(0.0)
    assert ef.disparate_impact(Y, groups) == pytest.approx(1.0)
    sdr = ef.supply_demand_ratio(Y, groups)
    assert sdr["gap"] == pytest.approx(0.0)


def test_skewed_case_hand_computed():
    # D units under-served: mean(D)=1, mean(A)=4
    Y = np.array([4.0, 4.0, 1.0, 1.0])
    groups = np.array([0, 0, 1, 1])
    assert ef.demographic_parity(Y, groups) == pytest.approx(3.0)   # A - D
    assert ef.disparate_impact(Y, groups) == pytest.approx(0.25)    # D / A
    sdr = ef.supply_demand_ratio(Y, groups)
    assert sdr["mean_disadvantaged"] == pytest.approx(1.0)
    assert sdr["mean_advantaged"] == pytest.approx(4.0)
    assert sdr["gap"] == pytest.approx(3.0)


def test_excluded_units_ignored():
    Y = np.array([4.0, 1.0, 99.0])
    groups = np.array([0, 1, -1])            # last excluded
    assert ef.demographic_parity(Y, groups) == pytest.approx(3.0)


def test_empty_group_returns_nan():
    Y = np.array([1.0, 2.0])
    groups = np.array([0, 0])                # no disadvantaged
    assert np.isnan(ef.demographic_parity(Y, groups))
    assert np.isnan(ef.disparate_impact(Y, groups))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'demographic_parity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# famail_temporal/baselines/external_fairness.py
"""Pure external fairness metrics over per-active-unit service ratios Y.

N-vector in, scalar/dict out; grid- and bundle-unaware (mirrors fairness/).
Group labels: 0 = advantaged (A), 1 = disadvantaged (D), -1 = excluded.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def _group_means(Y: np.ndarray, groups: np.ndarray) -> Tuple[float, float]:
    d = Y[groups == 1]
    a = Y[groups == 0]
    if d.size == 0 or a.size == 0:
        return float("nan"), float("nan")
    return float(d.mean()), float(a.mean())


def supply_demand_ratio(Y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    mean_d, mean_a = _group_means(Y, groups)
    return {
        "mean_disadvantaged": mean_d,
        "mean_advantaged": mean_a,
        "gap": mean_a - mean_d,
    }


def demographic_parity(Y: np.ndarray, groups: np.ndarray) -> float:
    mean_d, mean_a = _group_means(Y, groups)
    return mean_a - mean_d               # signed gap; 0 = parity


def disparate_impact(Y: np.ndarray, groups: np.ndarray) -> float:
    mean_d, mean_a = _group_means(Y, groups)
    if not np.isfinite(mean_a) or mean_a == 0.0:
        return float("nan")
    return mean_d / mean_a               # 1 = parity; < 0.8 = adverse


def sdr_gap(Y: np.ndarray, groups: np.ndarray) -> float:
    return supply_demand_ratio(Y, groups)["gap"]


def sdr_mean_disadvantaged(Y: np.ndarray, groups: np.ndarray) -> float:
    return supply_demand_ratio(Y, groups)["mean_disadvantaged"]


def sdr_mean_advantaged(Y: np.ndarray, groups: np.ndarray) -> float:
    return supply_demand_ratio(Y, groups)["mean_advantaged"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/external_fairness.py famail_temporal/baselines/tests/test_external_fairness.py
git commit -m "feat(external-fairness): SDR, demographic parity, disparate impact core metrics"
```

---

### Task 2: Theil index (between-region, on Y)

**Files:**
- Modify: `famail_temporal/baselines/external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_external_fairness.py`

**Interfaces:**
- Produces: `theil_index(Y, regions) -> float`. `regions` is `(N,)` int, `-1` excluded.

- [ ] **Step 1: Write the failing test**

```python
# append to test_external_fairness.py
def test_theil_zero_when_all_regions_equal():
    Y = np.array([3.0, 3.0, 3.0, 3.0])
    regions = np.array([0, 0, 1, 1])
    assert ef.theil_index(Y, regions) == pytest.approx(0.0, abs=1e-12)


def test_theil_scale_invariant():
    Y = np.array([1.0, 1.0, 5.0, 5.0])
    regions = np.array([0, 0, 1, 1])
    t1 = ef.theil_index(Y, regions)
    t2 = ef.theil_index(10.0 * Y, regions)
    assert t1 == pytest.approx(t2)
    assert t1 > 0.0


def test_theil_hand_computed_two_regions():
    # region means 1 and 3, equal sizes -> ybar=2
    # T = 0.5*(1/2)ln(1/2) + 0.5*(3/2)ln(3/2)
    Y = np.array([1.0, 1.0, 3.0, 3.0])
    regions = np.array([0, 0, 1, 1])
    expected = 0.5 * (0.5) * np.log(0.5) + 0.5 * (1.5) * np.log(1.5)
    assert ef.theil_index(Y, regions) == pytest.approx(expected)


def test_theil_excludes_negative_region_and_survives_zero_service():
    Y = np.array([0.0, 4.0, 4.0, 99.0])
    regions = np.array([0, 0, 1, -1])        # last excluded; zero-service in region 0
    # region0 mean=2, region1 mean=4, ybar over valid = (0+4+4)/3
    val = ef.theil_index(Y, regions)
    assert np.isfinite(val)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -k theil -v`
Expected: FAIL with `AttributeError: ... has no attribute 'theil_index'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to external_fairness.py
def theil_index(Y: np.ndarray, regions: np.ndarray) -> float:
    """Between-region Theil-T index of Y. regions: (N,) int, -1 excluded.

    T_between = sum_g (N_g/N) * (ybar_g/ybar) * ln(ybar_g/ybar).
    Zero-service units contribute 0 (limit y*ln y -> 0). Scale-invariant.
    """
    valid = regions >= 0
    y = Y[valid].astype(np.float64)
    r = regions[valid]
    n = y.size
    if n == 0:
        return float("nan")
    ybar = y.mean()
    if not np.isfinite(ybar) or ybar <= 0.0:
        return float("nan")
    total = 0.0
    for g in np.unique(r):
        yg = y[r == g]
        ybar_g = yg.mean()
        if ybar_g > 0.0:
            total += (yg.size / n) * (ybar_g / ybar) * np.log(ybar_g / ybar)
    return float(total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -k theil -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/external_fairness.py famail_temporal/baselines/tests/test_external_fairness.py
git commit -m "feat(external-fairness): between-region Theil index on Y"
```

---

### Task 3: Grouping strategies + regions

**Files:**
- Modify: `famail_temporal/baselines/external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_external_fairness.py`

**Interfaces:**
- Produces: `median_split(values, disadvantaged_high) -> groups(N,)`, `region_extremes(values, disadvantaged_high, frac=1/3) -> groups(N,)`, `regions_from_values(value_columns) -> regions(N,)`. `values` is `(N,)` float (NaN allowed → excluded); `value_columns` is a list of `(N,)` arrays.

- [ ] **Step 1: Write the failing test**

```python
# append to test_external_fairness.py
def test_median_split_disadvantaged_low():
    values = np.array([1.0, 2.0, 3.0, 4.0])       # median 2.5
    g = ef.median_split(values, disadvantaged_high=False)
    # low (<=2.5) is disadvantaged
    assert list(g) == [1, 1, 0, 0]


def test_median_split_disadvantaged_high_and_nan_excluded():
    values = np.array([1.0, 4.0, np.nan])
    g = ef.median_split(values, disadvantaged_high=True)
    assert g[2] == -1
    assert g[0] == 0 and g[1] == 1                # high=disadvantaged


def test_region_extremes_top_bottom_third():
    # 6 distinct region values -> frac 1/3 -> k=2 each end
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    g = ef.region_extremes(values, disadvantaged_high=False)  # low = disadv
    assert g[0] == 1 and g[1] == 1                # 10,20 bottom -> D
    assert g[4] == 0 and g[5] == 0               # 50,60 top -> A
    assert g[2] == -1 and g[3] == -1             # middle excluded


def test_region_extremes_groups_by_distinct_value():
    # region-constant values: two regions of size 3
    values = np.array([5.0, 5.0, 5.0, 9.0, 9.0, 9.0])
    g = ef.region_extremes(values, disadvantaged_high=True)   # high = disadv
    assert list(g[:3]) == [0, 0, 0]              # value 5 = advantaged
    assert list(g[3:]) == [1, 1, 1]              # value 9 = disadvantaged


def test_regions_from_values_maps_profiles():
    housing = np.array([1.0, 1.0, 2.0, np.nan])
    comp = np.array([9.0, 9.0, 8.0, 8.0])
    r = ef.regions_from_values([housing, comp])
    assert r[0] == r[1]                          # same profile
    assert r[0] != r[2]                          # different profile
    assert r[3] == -1                            # NaN -> excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -k "split or region" -v`
Expected: FAIL (`no attribute 'median_split'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to external_fairness.py
def median_split(values: np.ndarray, disadvantaged_high: bool) -> np.ndarray:
    groups = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    if not finite.any():
        return groups
    med = np.median(values[finite])
    high = finite & (values > med)
    low = finite & (values <= med)
    if disadvantaged_high:
        groups[high] = 1
        groups[low] = 0
    else:
        groups[low] = 1
        groups[high] = 0
    return groups


def region_extremes(
    values: np.ndarray, disadvantaged_high: bool, frac: float = 1.0 / 3.0,
) -> np.ndarray:
    """Rank distinct region values; bottom/top `frac` of regions -> D/A."""
    groups = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    uniq = np.unique(values[finite])
    n_reg = uniq.size
    if n_reg < 2:
        return groups
    k = max(1, int(round(frac * n_reg)))
    if 2 * k > n_reg:
        k = n_reg // 2
    low_vals = uniq[:k]
    high_vals = uniq[-k:]
    is_low = finite & np.isin(values, low_vals)
    is_high = finite & np.isin(values, high_vals)
    if disadvantaged_high:
        groups[is_high] = 1
        groups[is_low] = 0
    else:
        groups[is_low] = 1
        groups[is_high] = 0
    return groups


def regions_from_values(value_columns: Sequence[np.ndarray]) -> np.ndarray:
    """Map each unit's demographic profile (row across axes) to an int region
    id. Any NaN across the columns -> -1 (excluded)."""
    stacked = np.stack([np.asarray(c, dtype=np.float64) for c in value_columns],
                       axis=1)
    regions = np.full(stacked.shape[0], -1, dtype=np.int64)
    finite = np.all(np.isfinite(stacked), axis=1)
    if not finite.any():
        return regions
    _, inv = np.unique(stacked[finite], axis=0, return_inverse=True)
    regions[finite] = inv
    return regions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -k "split or region" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/external_fairness.py famail_temporal/baselines/tests/test_external_fairness.py
git commit -m "feat(external-fairness): median-split, region-extremes grouping + region derivation"
```

---

### Task 4: Paired unit-level bootstrap

**Files:**
- Modify: `famail_temporal/baselines/external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_external_fairness.py`

**Interfaces:**
- Produces: `paired_bootstrap(Y_before, Y_after, specs, B=1000, seed=0, ci=0.95) -> dict`. `specs` is a list of `(name: str, fn: Callable[[np.ndarray, np.ndarray], float], labels: np.ndarray)`. Returns `{name: {"before":(lo,hi), "after":(lo,hi), "delta":(lo,hi), "n_dropped": int}}`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_external_fairness.py
def test_bootstrap_deterministic_and_brackets_point():
    rng = np.random.default_rng(0)
    Yb = rng.uniform(0.5, 2.0, size=200)
    Ya = Yb + 0.3                                    # uniform improvement
    groups = np.where(np.arange(200) % 2 == 0, 0, 1)
    specs = [("dp", ef.demographic_parity, groups)]
    out1 = ef.paired_bootstrap(Yb, Ya, specs, B=200, seed=7)
    out2 = ef.paired_bootstrap(Yb, Ya, specs, B=200, seed=7)
    assert out1 == out2                              # determinism
    lo, hi = out1["dp"]["delta"]
    assert lo <= 0.0 <= hi or (lo <= (ef.demographic_parity(Ya, groups)
                                      - ef.demographic_parity(Yb, groups)) <= hi)


def test_bootstrap_counts_empty_group_drops():
    # a group that can vanish under resampling of a tiny sample
    Yb = np.array([1.0, 1.0, 2.0])
    Ya = np.array([1.5, 1.5, 2.0])
    groups = np.array([0, 0, 1])                     # single disadvantaged unit
    specs = [("di", ef.disparate_impact, groups)]
    out = ef.paired_bootstrap(Yb, Ya, specs, B=300, seed=1)
    assert out["di"]["n_dropped"] >= 1               # some resamples drop unit 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -k bootstrap -v`
Expected: FAIL (`no attribute 'paired_bootstrap'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to external_fairness.py
def paired_bootstrap(
    Y_before: np.ndarray,
    Y_after: np.ndarray,
    specs: List[Tuple[str, Callable[[np.ndarray, np.ndarray], float], np.ndarray]],
    B: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, Dict[str, object]]:
    """Resample unit indices with replacement (shared per replicate); recompute
    each spec's metric on before/after; percentile CIs on before/after/delta.
    Non-finite replicates are dropped and counted."""
    rng = np.random.default_rng(seed)
    N = Y_before.shape[0]
    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 * (1.0 + ci) / 2.0
    acc = {name: {"before": [], "after": [], "delta": [], "dropped": 0}
           for name, _, _ in specs}
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        yb = Y_before[idx]
        ya = Y_after[idx]
        for name, fn, labels in specs:
            lab = labels[idx]
            b = fn(yb, lab)
            a = fn(ya, lab)
            if not (np.isfinite(b) and np.isfinite(a)):
                acc[name]["dropped"] += 1
                continue
            acc[name]["before"].append(b)
            acc[name]["after"].append(a)
            acc[name]["delta"].append(a - b)

    def _ci(vals: List[float]) -> Tuple[float, float]:
        if len(vals) == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, lo_q)),
                float(np.percentile(vals, hi_q)))

    out: Dict[str, Dict[str, object]] = {}
    for name, d in acc.items():
        out[name] = {
            "before": _ci(d["before"]),
            "after": _ci(d["after"]),
            "delta": _ci(d["delta"]),
            "n_dropped": int(d["dropped"]),
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py -v`
Expected: PASS (all tests in file).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/external_fairness.py famail_temporal/baselines/tests/test_external_fairness.py
git commit -m "feat(external-fairness): paired unit-level bootstrap for before/after/delta CIs"
```

---

### Task 5: IO — per-unit demographics + service ratio Y

**Files:**
- Create: `famail_temporal/baselines/external_fairness_io.py`
- Test: `famail_temporal/baselines/tests/test_external_fairness_io.py`

**Interfaces:**
- Consumes: `DataBundle` (fields `mask_3d`, `active_taxis_3d`, `pickup_3d`); `config.DEMAND_FLOOR`, `config.SOURCE_DATA_DIR`; `data.demographics.enrich_demographics`.
- Produces: `EQUITY_AXES`, `DISADVANTAGED_HIGH`, `per_unit_demographics(bundle, selected_grid=None) -> Dict[str, np.ndarray]`, `service_ratio_Y(pickup_3d, bundle) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# famail_temporal/baselines/tests/test_external_fairness_io.py
import numpy as np
import pytest

from famail_temporal.baselines import external_fairness_io as io
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_service_ratio_matches_manual():
    bundle = _make_synthetic_bundle()
    Y = io.service_ratio_Y(bundle.pickup_3d, bundle)
    mask = bundle.mask_3d
    demand = bundle.pickup_3d[mask]
    supply = bundle.active_taxis_3d[mask]
    expected = supply / np.maximum(demand, 0.5)
    assert Y.shape == (int(mask.sum()),)
    np.testing.assert_allclose(Y, expected, rtol=1e-9)


def test_per_unit_demographics_injected_grid_shapes_and_values():
    bundle = _make_synthetic_bundle()
    gx, gy, _ = bundle.mask_3d.shape
    # synthetic (gx, gy, 3) grid: axis j = constant j+1 everywhere
    sel = np.zeros((gx, gy, 3), dtype=np.float64)
    for j in range(3):
        sel[..., j] = j + 1
    demo = io.per_unit_demographics(bundle, selected_grid=sel)
    n = int(bundle.mask_3d.sum())
    for j, axis in enumerate(io.EQUITY_AXES):
        assert demo[axis].shape == (n,)
        np.testing.assert_allclose(demo[axis], j + 1)


def test_equity_axes_and_pole_constants():
    assert io.EQUITY_AXES == ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]
    assert io.DISADVANTAGED_HIGH["MigrantRatio"] is True
    assert io.DISADVANTAGED_HIGH["AvgHousingPricePerSqM"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness_io.py -v`
Expected: FAIL (`ModuleNotFoundError: external_fairness_io`).

- [ ] **Step 3: Write minimal implementation**

```python
# famail_temporal/baselines/external_fairness_io.py
"""IO layer: bundle -> service ratio Y, per-unit equity-axis demographics, and
the edited (after) demand grid. City-agnostic (reads cell_demographics.pkl)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from famail_temporal import config
from famail_temporal.data.demographics import enrich_demographics
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.datasets import pickup_unit_of, pickup_mass

EQUITY_AXES: List[str] = ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]
DISADVANTAGED_HIGH: Dict[str, bool] = {
    "AvgHousingPricePerSqM": False,
    "CompPerCapita": False,
    "MigrantRatio": True,
}


def _enriched_selected_grid() -> np.ndarray:
    """(GX, GY, 3) raw values for EQUITY_AXES from cell_demographics.pkl."""
    with open(config.SOURCE_DATA_DIR / "cell_demographics.pkl", "rb") as f:
        raw = pickle.load(f)
    enriched, names = enrich_demographics(
        raw["demographics_grid"], list(raw["feature_names"]),
    )
    idx = [names.index(a) for a in EQUITY_AXES]
    return enriched[..., idx].astype(np.float64)


def per_unit_demographics(
    bundle: DataBundle, selected_grid: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """{axis: (N,) raw value per active unit}. NaN where cell unmapped."""
    sel = _enriched_selected_grid() if selected_grid is None else selected_grid
    mask = bundle.mask_3d
    out: Dict[str, np.ndarray] = {}
    for j, axis in enumerate(EQUITY_AXES):
        per_cell_t = np.broadcast_to(sel[:, :, j][:, :, None], mask.shape)
        out[axis] = per_cell_t[mask].astype(np.float64)
    return out


def service_ratio_Y(pickup_3d: np.ndarray, bundle: DataBundle) -> np.ndarray:
    """Y = supply/demand over active units (F_causal convention)."""
    mask = bundle.mask_3d
    demand_N = pickup_3d[mask].astype(np.float64)
    supply_N = bundle.active_taxis_3d[mask].astype(np.float64)
    return supply_N / np.maximum(demand_N, config.DEMAND_FLOOR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness_io.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/external_fairness_io.py famail_temporal/baselines/tests/test_external_fairness_io.py
git commit -m "feat(external-fairness): IO for service ratio Y and per-unit equity demographics"
```

---

### Task 6: IO — edited (after) demand grid reconstruction

**Files:**
- Modify: `famail_temporal/baselines/external_fairness_io.py`
- Test: `famail_temporal/baselines/tests/test_external_fairness_io.py`

**Interfaces:**
- Consumes: `histories.pkl` entries with `.original` / `.modified` `Trajectory`; `pickup_unit_of`, `pickup_mass`, `config.DEMAND_FLOOR`.
- Produces: `build_edited_pickup_3d(bundle, edit_dir) -> np.ndarray (GX,GY,T)`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_external_fairness_io.py
import pickle as _pickle
from types import SimpleNamespace

from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj_at(x, y, time_bucket=0):
    return Trajectory(states=[TrajectoryState(
        x_grid=int(x), y_grid=int(y), time_bucket=int(time_bucket), day_index=0)])


def test_build_edited_pickup_relocates_mass(tmp_path):
    bundle = _make_synthetic_bundle()
    mask = bundle.mask_3d
    xs, ys, ts = np.where(mask)
    # pick an active origin unit with a high pickup, and a distinct active dest
    demand_vals = bundle.pickup_3d[mask]
    o = int(np.argmax(demand_vals))
    ox, oy, ot = int(xs[o]), int(ys[o]), int(ts[o])
    d = next(i for i in range(len(xs)) if (xs[i], ys[i], ts[i]) != (ox, oy, ot))
    dx, dy, dt = int(xs[d]), int(ys[d]), int(ts[d])
    from famail_temporal.data.aggregation import time_bucket_to_hour  # noqa
    # build trajectories whose terminal state maps to (ox,oy,ot)/(dx,dy,dt)
    # t_block == time_bucket's hour block; use time_bucket = hour so block==hour
    orig = _traj_at(ox, oy, time_bucket=ot)
    modif = _traj_at(dx, dy, time_bucket=dt)
    histories = [SimpleNamespace(original=orig, modified=modif)]
    with open(tmp_path / "histories.pkl", "wb") as f:
        _pickle.dump(histories, f)

    before = bundle.pickup_3d.copy()
    after = io.build_edited_pickup_3d(bundle, tmp_path)
    mass_o = 1.0 / (int(bundle.n_hours_per_block[ot]) * bundle.n_days)
    mass_d = 1.0 / (int(bundle.n_hours_per_block[dt]) * bundle.n_days)
    assert before[ox, oy, ot] - after[ox, oy, ot] == pytest.approx(mass_o)
    assert after[dx, dy, dt] - before[dx, dy, dt] == pytest.approx(mass_d)
```

Note: `_traj_at` uses `time_bucket = t_block` so that `hour_to_block_index(time_bucket_to_hour(tb))` lands on the intended block. Since `TIME_BLOCKS` are hourly (`hour_h` for h in 0..23) and `t_block` ∈ 0..23, choosing `time_bucket` equal to the block's hour is valid. If `time_bucket_to_hour` expects 0..287, use the block's start hour directly (0..23 is a valid bucket-as-hour). Verify against `data/aggregation.py` during implementation and adjust the `time_bucket` argument so `pickup_unit_of` returns the chosen `(x,y,t_block)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness_io.py -k build_edited -v`
Expected: FAIL (`no attribute 'build_edited_pickup_3d'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to external_fairness_io.py
def build_edited_pickup_3d(bundle: DataBundle, edit_dir) -> np.ndarray:
    """After-edit demand grid: relocate each edited pickup's per-event mass
    from its original to modified cell (modifier convention). Subtraction is
    floored at DEMAND_FLOOR; addition is unflored."""
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    pickup_3d = bundle.pickup_3d.copy()
    floor = config.DEMAND_FLOOR
    for h in histories:
        ox, oy, ot = pickup_unit_of(h.original)
        mx, my, mt = pickup_unit_of(h.modified)
        reduced = float(pickup_3d[ox, oy, ot]) - pickup_mass(bundle, ot)
        pickup_3d[ox, oy, ot] = max(reduced, floor)
        pickup_3d[mx, my, mt] = float(pickup_3d[mx, my, mt]) + pickup_mass(bundle, mt)
    return pickup_3d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness_io.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/external_fairness_io.py famail_temporal/baselines/tests/test_external_fairness_io.py
git commit -m "feat(external-fairness): reconstruct after-edit demand grid from histories"
```

---

### Task 7: Orchestration — assemble_results

**Files:**
- Create: `famail_temporal/baselines/run_external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_run_external_fairness.py`

**Interfaces:**
- Consumes: all of `external_fairness` (metrics/grouping/bootstrap) + `external_fairness_io.{EQUITY_AXES,DISADVANTAGED_HIGH}`.
- Produces: `GROUPINGS = ("district_extremes", "median_split")`; `assemble_results(Y_before, Y_after, demo, seed=0, B=1000) -> dict`. `demo` is `{axis: (N,) values}`. Result dict schema below.

Result schema:
```
{
  "theil": {"before": f, "after": f, "delta": f, "delta_ci": (lo,hi), "n_dropped": int},
  "metrics": { axis: { grouping: {
      "group_sizes": {"n_disadvantaged": int, "n_advantaged": int, "n_excluded": int},
      "supply_demand_ratio": {"before": {...}, "after": {...}, "delta_gap": f, "gap_ci": (lo,hi)},
      "demographic_parity": {"before": f, "after": f, "delta": f, "delta_ci": (lo,hi)},
      "disparate_impact":  {"before": f, "after": f, "delta": f, "delta_ci": (lo,hi)},
  }}},
}
```

- [ ] **Step 1: Write the failing test**

```python
# famail_temporal/baselines/tests/test_run_external_fairness.py
import numpy as np
import pytest

from famail_temporal.baselines import run_external_fairness as rx
from famail_temporal.baselines import external_fairness_io as io


def _synthetic_arrays(n_per_region=20):
    # 4 regions with distinct housing/comp/migrant profiles
    housing, comp, migrant, Yb = [], [], [], []
    profiles = [(1.0, 1.0, 0.8), (2.0, 2.0, 0.6),
                (3.0, 3.0, 0.4), (4.0, 4.0, 0.2)]
    base_Y = [1.0, 2.0, 3.0, 4.0]                 # poor regions under-served
    for (h, c, m), y in zip(profiles, base_Y):
        housing += [h] * n_per_region
        comp += [c] * n_per_region
        migrant += [m] * n_per_region
        Yb += [y] * n_per_region
    demo = {"AvgHousingPricePerSqM": np.array(housing),
            "CompPerCapita": np.array(comp),
            "MigrantRatio": np.array(migrant)}
    Yb = np.array(Yb)
    Ya = Yb + np.where(Yb < 2.5, 1.0, 0.0)        # lift under-served regions
    return Yb, Ya, demo


def test_assemble_results_schema_and_improvement():
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=100)
    assert set(res["metrics"].keys()) == set(io.EQUITY_AXES)
    for axis in io.EQUITY_AXES:
        for g in rx.GROUPINGS:
            entry = res["metrics"][axis][g]
            assert "demographic_parity" in entry
            # lifting under-served regions reduces the parity gap magnitude
            dp = entry["demographic_parity"]
            assert abs(dp["after"]) <= abs(dp["before"]) + 1e-9
            # disparate impact moves toward 1
            di = entry["disparate_impact"]
            assert di["after"] >= di["before"] - 1e-9
    assert "delta" in res["theil"]
    assert res["theil"]["after"] <= res["theil"]["before"] + 1e-9


def test_assemble_results_ci_present():
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=50)
    entry = res["metrics"]["MigrantRatio"]["district_extremes"]
    lo, hi = entry["demographic_parity"]["delta_ci"]
    assert lo <= entry["demographic_parity"]["delta"] <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_external_fairness.py -v`
Expected: FAIL (`ModuleNotFoundError: run_external_fairness`).

- [ ] **Step 3: Write minimal implementation**

```python
# famail_temporal/baselines/run_external_fairness.py
"""Compute + report external fairness metrics before/after edit. See
docs/superpowers/plans/2026-07-02-external-fairness-metrics.md."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io

GROUPINGS: Tuple[str, str] = ("district_extremes", "median_split")


def _groups_for(values: np.ndarray, axis: str, grouping: str) -> np.ndarray:
    high = io.DISADVANTAGED_HIGH[axis]
    if grouping == "district_extremes":
        return ef.region_extremes(values, disadvantaged_high=high)
    if grouping == "median_split":
        return ef.median_split(values, disadvantaged_high=high)
    raise ValueError(f"unknown grouping {grouping!r}")


def assemble_results(
    Y_before: np.ndarray, Y_after: np.ndarray,
    demo: Dict[str, np.ndarray], seed: int = 0, B: int = 1000,
) -> dict:
    regions = ef.regions_from_values([demo[a] for a in io.EQUITY_AXES])
    specs: List[Tuple[str, object, np.ndarray]] = [("theil", ef.theil_index, regions)]
    metrics: Dict[str, dict] = {}
    for axis in io.EQUITY_AXES:
        metrics[axis] = {}
        for g in GROUPINGS:
            groups = _groups_for(demo[axis], axis, g)
            metrics[axis][g] = {
                "group_sizes": {
                    "n_disadvantaged": int((groups == 1).sum()),
                    "n_advantaged": int((groups == 0).sum()),
                    "n_excluded": int((groups == -1).sum()),
                },
                "supply_demand_ratio": {
                    "before": ef.supply_demand_ratio(Y_before, groups),
                    "after": ef.supply_demand_ratio(Y_after, groups),
                    "delta_gap": ef.sdr_gap(Y_after, groups) - ef.sdr_gap(Y_before, groups),
                },
                "demographic_parity": {
                    "before": ef.demographic_parity(Y_before, groups),
                    "after": ef.demographic_parity(Y_after, groups),
                    "delta": (ef.demographic_parity(Y_after, groups)
                              - ef.demographic_parity(Y_before, groups)),
                },
                "disparate_impact": {
                    "before": ef.disparate_impact(Y_before, groups),
                    "after": ef.disparate_impact(Y_after, groups),
                    "delta": (ef.disparate_impact(Y_after, groups)
                              - ef.disparate_impact(Y_before, groups)),
                },
            }
            specs.append((f"dp::{axis}::{g}", ef.demographic_parity, groups))
            specs.append((f"di::{axis}::{g}", ef.disparate_impact, groups))
            specs.append((f"sdrgap::{axis}::{g}", ef.sdr_gap, groups))

    boot = ef.paired_bootstrap(Y_before, Y_after, specs, B=B, seed=seed)

    # attach CIs
    theil_before = ef.theil_index(Y_before, regions)
    theil_after = ef.theil_index(Y_after, regions)
    result = {
        "theil": {
            "before": theil_before, "after": theil_after,
            "delta": theil_after - theil_before,
            "delta_ci": boot["theil"]["delta"],
            "n_dropped": boot["theil"]["n_dropped"],
        },
        "metrics": metrics,
    }
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = metrics[axis][g]
            e["demographic_parity"]["delta_ci"] = boot[f"dp::{axis}::{g}"]["delta"]
            e["disparate_impact"]["delta_ci"] = boot[f"di::{axis}::{g}"]["delta"]
            e["supply_demand_ratio"]["gap_ci"] = boot[f"sdrgap::{axis}::{g}"]["delta"]
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_external_fairness.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_external_fairness.py famail_temporal/baselines/tests/test_run_external_fairness.py
git commit -m "feat(external-fairness): assemble before/after results across metrics x axes x groupings"
```

---

### Task 8: Rendering — JSON + markdown tables

**Files:**
- Modify: `famail_temporal/baselines/run_external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_run_external_fairness.py`

**Interfaces:**
- Produces: `write_json(result, out_dir, meta) -> Path`, `render_markdown(result, meta) -> str`, `render_combined_table(named_results) -> str`. `named_results` is a list of `(label, result)`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_run_external_fairness.py
import json


def test_write_json_and_markdown(tmp_path):
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=30)
    meta = {"dataset": "shenzhen-primary", "edit_dir": "x", "seed": 0, "B": 30}
    path = rx.write_json(res, tmp_path, meta)
    loaded = json.loads(path.read_text())
    assert loaded["meta"]["dataset"] == "shenzhen-primary"
    assert "theil" in loaded

    md = rx.render_markdown(res, meta)
    assert "Demographic parity" in md
    assert "Disparate impact" in md
    assert "Theil" in md
    assert "| Before | After | Delta |" in md or "Before" in md


def test_combined_table():
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=20)
    md = rx.render_combined_table([("shenzhen", res), ("sf", res)])
    assert "shenzhen" in md and "sf" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_external_fairness.py -k "json or combined" -v`
Expected: FAIL (`no attribute 'write_json'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to run_external_fairness.py
import json
from pathlib import Path


def _fmt(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _fmt_ci(ci) -> str:
    lo, hi = ci
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def write_json(result: dict, out_dir, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, **result}
    path = out_dir / "external_fairness.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def render_markdown(result: dict, meta: dict) -> str:
    lines: List[str] = [f"# External fairness — {meta.get('dataset','')}", ""]
    lines.append(f"**Edit dir:** `{meta.get('edit_dir','')}`  ·  "
                 f"**B:** {meta.get('B','')}  ·  **seed:** {meta.get('seed','')}")
    lines.append("")
    t = result["theil"]
    lines.append("## Theil index (between-region, on Y)")
    lines.append("| Before | After | Delta | Δ 95% CI |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(f"| {_fmt(t['before'])} | {_fmt(t['after'])} | "
                 f"{t['delta']:+.4f} | {_fmt_ci(t['delta_ci'])} |")
    lines.append("")
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = result["metrics"][axis][g]
            gs = e["group_sizes"]
            lines.append(f"## {axis} — {g}  "
                         f"(D={gs['n_disadvantaged']}, A={gs['n_advantaged']}, "
                         f"excl={gs['n_excluded']})")
            lines.append("| Metric | Before | After | Delta | Δ 95% CI |")
            lines.append("|---|---:|---:|---:|---:|")
            dp = e["demographic_parity"]
            di = e["disparate_impact"]
            sd = e["supply_demand_ratio"]
            lines.append(f"| Supply/demand gap | {_fmt(sd['before']['gap'])} | "
                         f"{_fmt(sd['after']['gap'])} | {sd['delta_gap']:+.4f} | "
                         f"{_fmt_ci(sd['gap_ci'])} |")
            lines.append(f"| Demographic parity | {_fmt(dp['before'])} | "
                         f"{_fmt(dp['after'])} | {dp['delta']:+.4f} | "
                         f"{_fmt_ci(dp['delta_ci'])} |")
            lines.append(f"| Disparate impact | {_fmt(di['before'])} | "
                         f"{_fmt(di['after'])} | {di['delta']:+.4f} | "
                         f"{_fmt_ci(di['delta_ci'])} |")
            lines.append("")
    return "\n".join(lines)


def render_combined_table(named_results: List[Tuple[str, dict]]) -> str:
    lines = ["# External fairness — cross-dataset comparison", "",
             "| Dataset | Theil Δ | DP Δ (migrant/extremes) | DI Δ (migrant/extremes) |",
             "|---|---:|---:|---:|"]
    for label, res in named_results:
        e = res["metrics"]["MigrantRatio"]["district_extremes"]
        lines.append(f"| {label} | {res['theil']['delta']:+.4f} | "
                     f"{e['demographic_parity']['delta']:+.4f} | "
                     f"{e['disparate_impact']['delta']:+.4f} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_external_fairness.py -v`
Expected: PASS (all tests in file).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_external_fairness.py famail_temporal/baselines/tests/test_run_external_fairness.py
git commit -m "feat(external-fairness): JSON + markdown table + combined-table rendering"
```

---

### Task 9: Figures (before→after with CI error bars)

**Files:**
- Modify: `famail_temporal/baselines/run_external_fairness.py`
- Test: `famail_temporal/baselines/tests/test_run_external_fairness.py`

**Interfaces:**
- Produces: `write_figure(result, out_dir, meta) -> Path` (a forest plot of Δ ± 95% CI per metric×axis×grouping).

- [ ] **Step 1: Write the failing test**

```python
# append to test_run_external_fairness.py
def test_write_figure_creates_png(tmp_path):
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=30)
    meta = {"dataset": "shenzhen-primary"}
    path = rx.write_figure(res, tmp_path, meta)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_external_fairness.py -k figure -v`
Expected: FAIL (`no attribute 'write_figure'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to run_external_fairness.py — put matplotlib import at top of function
def write_figure(result: dict, out_dir, meta: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, float, float, float]] = []
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = result["metrics"][axis][g]
            for mname, key in (("DP", "demographic_parity"),
                               ("DI", "disparate_impact")):
                d = e[key]
                lo, hi = d["delta_ci"]
                rows.append((f"{mname} {axis[:6]}/{g[:4]}", d["delta"], lo, hi))
    rows.append(("Theil", result["theil"]["delta"],
                 *result["theil"]["delta_ci"]))

    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1))
    ax.errorbar(deltas, y, xerr=[los, his], fmt="o", capsize=3)
    ax.axvline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ (after − before)")
    ax.set_title(f"External fairness Δ — {meta.get('dataset','')}")
    fig.tight_layout()
    path = out_dir / "external_fairness_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_external_fairness.py -v`
Expected: PASS (all tests in file).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_external_fairness.py famail_temporal/baselines/tests/test_run_external_fairness.py
git commit -m "feat(external-fairness): forest-plot figure of before/after deltas with CIs"
```

---

### Task 10: CLI wiring + real-data run

**Files:**
- Modify: `famail_temporal/baselines/run_external_fairness.py`

**Interfaces:**
- Consumes: `DataBundle.load()`, `io.{service_ratio_Y,per_unit_demographics,build_edited_pickup_3d}`, `assemble_results`, `write_json`, `render_markdown`, `write_figure`.
- Produces: `main(argv=None) -> int` with flags `--edit-dir`, `--dataset`, `--out-dir`, `--seed`, `--bootstrap`, `--combine`.

- [ ] **Step 1: Add `main()` (no new failing unit test — exercised by the smoke run in Step 3)**

```python
# append to run_external_fairness.py
import argparse
import sys
from famail_temporal import config
from famail_temporal.data.loader import DataBundle


def _run_one(edit_dir: Path, dataset: str, out_dir: Path,
             seed: int, B: int) -> dict:
    bundle = DataBundle.load()
    Y_before = io.service_ratio_Y(bundle.pickup_3d, bundle)
    after_pickup = io.build_edited_pickup_3d(bundle, edit_dir)
    Y_after = io.service_ratio_Y(after_pickup, bundle)
    demo = io.per_unit_demographics(bundle)
    result = assemble_results(Y_before, Y_after, demo, seed=seed, B=B)
    meta = {"dataset": dataset, "city": config.CITY, "edit_dir": str(edit_dir),
            "seed": seed, "B": B, "n_active": int(bundle.mask_3d.sum())}
    write_json(result, out_dir, meta)
    (out_dir / "report.md").write_text(render_markdown(result, meta))
    write_figure(result, out_dir, meta)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.run_external_fairness")
    ap.add_argument("--edit-dir", type=Path, required=False,
                    help="Results dir with histories.pkl for the edit")
    ap.add_argument("--dataset", default=None,
                    help="Label for outputs (e.g. shenzhen-primary, sf12)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--combine", nargs="+", type=Path, default=None,
                    help="external_fairness.json paths to combine into one table")
    args = ap.parse_args(argv)

    if args.combine:
        named = []
        for p in args.combine:
            payload = json.loads(Path(p).read_text())
            named.append((payload["meta"].get("dataset", str(p)), payload))
        out = args.out_dir or Path(config.PACKAGE_ROOT) / "baselines" / \
            "external_fairness" / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "combined.md").write_text(render_combined_table(named))
        print(f"wrote {out / 'combined.md'}")
        return 0

    if not args.edit_dir:
        ap.error("--edit-dir is required (unless --combine)")
    dataset = args.dataset or f"{config.CITY}"
    out_dir = args.out_dir or (Path(config.PACKAGE_ROOT) / "baselines" /
                               "external_fairness" / "results" / dataset)
    _run_one(args.edit_dir, dataset, out_dir, args.seed, args.bootstrap)
    print(f"wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest famail_temporal/baselines/tests/test_external_fairness.py famail_temporal/baselines/tests/test_external_fairness_io.py famail_temporal/baselines/tests/test_run_external_fairness.py -v`
Expected: PASS (all).

- [ ] **Step 3: Smoke-run on the real Shenzhen PRIMARY edit dir**

Identify the current Shenzhen PRIMARY causal-emphasis cleaned results dir under `famail_temporal/results/` (the headline run; e.g. the one referenced by `baselines/run_metric_hardening.py::DEFAULT_EDIT_DIR`, or the latest `*_causal_emphasis*` dir). Then:

Run:
```bash
python -m famail_temporal.baselines.run_external_fairness \
  --edit-dir famail_temporal/results/<SHENZHEN_PRIMARY_DIR> \
  --dataset shenzhen-primary --bootstrap 1000 --seed 0
```
Expected: prints `wrote outputs to .../shenzhen-primary`; that dir contains `external_fairness.json`, `report.md`, `external_fairness_delta.png`. Sanity-check `report.md`: DP/DI/SDR deltas finite; DI moves toward 1 or DP gap shrinks on at least the migrant axis.

- [ ] **Step 4: Cross-check DI against the existing `compute_di` (migrant axis, Shenzhen)**

Run this one-off check (interactive or a scratch script) and confirm our `disparate_impact` on the migrant/district-extremes grouping is directionally consistent with `district_metrics.di_from_bundle_and_pickup_grid` on the before grid:
```python
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines import district_metrics as dm, external_fairness_io as io, external_fairness as ef
b = DataBundle.load()
Y = io.service_ratio_Y(b.pickup_3d, b)
demo = io.per_unit_demographics(b)
g = ef.region_extremes(demo["MigrantRatio"], disadvantaged_high=True)
print("ours DI:", ef.disparate_impact(Y, g))
print("compute_di primary:", dm.di_from_bundle_and_pickup_grid(b, b.pickup_3d)["di_primary"])
```
Expected: both < 1 (disadvantaged under-served) and of comparable magnitude (they will not be identical — `compute_di` groups strictly top-3/bottom-3 by hukou and uses its own Y; ours groups by migrant terciles — but the *direction* must agree). Note any large discrepancy for discussion; do not "fix" silently.

- [ ] **Step 5: Run the remaining three datasets + combined table, then commit**

```bash
# Shenzhen sensitivity feature-set edit dirs (default env):
python -m famail_temporal.baselines.run_external_fairness --edit-dir famail_temporal/results/<SHZ_GDP_COMP_DIR> --dataset shenzhen-gdp-comp
python -m famail_temporal.baselines.run_external_fairness --edit-dir famail_temporal/results/<SHZ_LOGPOP_DIR> --dataset shenzhen-logpop
# SF sf12 (env var selects the city + its cell_demographics.pkl):
FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_external_fairness --edit-dir famail_temporal/results/2026-06-30T23-13-33_sf12-fair-ce --dataset sf12
# combined cross-dataset table:
python -m famail_temporal.baselines.run_external_fairness --combine \
  famail_temporal/baselines/external_fairness/results/*/external_fairness.json \
  --out-dir famail_temporal/baselines/external_fairness/results
git add famail_temporal/baselines/run_external_fairness.py
git commit -m "feat(external-fairness): CLI main + combine mode; run on Shenzhen (3 sets) + SF sf12"
```
Note: results dirs under `baselines/external_fairness/results/` are data artifacts — confirm `.gitignore` covers them (mirror how `baselines/metric_hardening/results/` is handled) before committing; do not commit large artifacts.

---

## Self-Review

- **Spec coverage:** §3 outcome Y (T5), §4 all four metrics (T1,T2), §5 both groupings × 3 axes + city-agnostic regions (T3,T7), §6 data flow incl. after-grid reconstruction (T5,T6,T10), §7 paired unit bootstrap (T4), §8 architecture 3 modules (T1-T10), §9 JSON+tables+figures (T8,T9), §10 tests (every task), §11 SF (env var + demographic regions, T10). Covered.
- **Placeholder scan:** every code step has runnable code; the two real-data specifics (exact Shenzhen dir name; verifying `time_bucket_to_hour` semantics in the T6 test) are called out explicitly with how to resolve, not left as silent TODOs.
- **Type consistency:** group label convention `{0,1,-1}` consistent across T1/T3/T4/T7; `groups` vs `regions` kept distinct (DP/DI/SDR take `groups`, Theil takes `regions`); bootstrap `specs` are `(name, float-fn, labels)` and only float-returning fns are passed (SDR uses `sdr_gap`, not the dict fn); `assemble_results` schema matches what `render_markdown`/`write_figure` read (`delta_ci`, `gap_ci`, `group_sizes`).

## Execution Handoff

Two execution options — see below.
