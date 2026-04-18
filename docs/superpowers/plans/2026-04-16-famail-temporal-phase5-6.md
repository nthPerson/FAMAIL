# FAMAIL-Temporal Implementation Plan — Phases 5–6

> **MODEL REQUIREMENT — OPUS ONLY:** Same as the main plan file.
>
> **Prerequisite:** Phases 1–4 complete (all tests passing).

**Scope:** Phase 5 (Preprocess + DataBundle loader) and Phase 6 (Fidelity port).

---

## Phase 5: Preprocess + DataBundle (Tasks 17–19)

### Task 17: MultiStreamData dataclass stub in fidelity/context.py

DataBundle needs a type-safe container for the five multi-stream context dicts. The full MultiStreamContextBuilder is added in Task 24.

**Files:**
- Create: famail_temporal/fidelity/context.py (stub)
- Create: famail_temporal/tests/test_ms_data.py

- [ ] **Step 1: Write failing test**

In famail_temporal/tests/test_ms_data.py:

    import dataclasses
    import numpy as np
    import pytest

    from famail_temporal.fidelity.context import MultiStreamData


    def test_multistream_data_is_frozen():
        ms = MultiStreamData(
            driving_trajs={0: []},
            seeking_trajs={0: []},
            profile_features={0: np.zeros(11)},
            seeking_days={0: []},
            driving_days={0: []},
        )
        assert dataclasses.is_dataclass(ms)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ms.driving_trajs = {}

- [ ] **Step 2: Run test (expect failure)**

    pytest famail_temporal/tests/test_ms_data.py -v

- [ ] **Step 3: Write famail_temporal/fidelity/context.py (stub)**

    """
    Multi-stream context builder. Full MultiStreamContextBuilder is added in Task 24.
    """

    from __future__ import annotations
    from dataclasses import dataclass
    from typing import Dict, List

    import numpy as np


    @dataclass(frozen=True)
    class MultiStreamData:
        """Bundle of the five multi-stream inputs.

        All dicts keyed by driver_idx (int, 0..49). Coordinates in driving_trajs
        and seeking_trajs are 1-indexed [1-48, 1-90].
        """
        driving_trajs: Dict[int, List]
        seeking_trajs: Dict[int, List]
        profile_features: Dict[int, np.ndarray]
        seeking_days: Dict[int, List[int]]
        driving_days: Dict[int, List[int]]

- [ ] **Step 4: Run test (expect pass)**

    pytest famail_temporal/tests/test_ms_data.py -v

- [ ] **Step 5: Commit**

    git add famail_temporal/fidelity/context.py famail_temporal/tests/test_ms_data.py
    git commit -m "feat(fidelity): add MultiStreamData dataclass stub"

---

### Task 18: DataBundle dataclass in data/loader.py

**Files:**
- Create: famail_temporal/data/loader.py
- Create: famail_temporal/tests/test_data_loader.py

- [ ] **Step 1: Write failing test**

In famail_temporal/tests/test_data_loader.py:

    import dataclasses
    import numpy as np
    import pytest
    import torch.nn as nn

    from famail_temporal.data.active_mask import UnitIndexMap
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.fairness.g0_power_basis import G0Function
    from famail_temporal.fidelity.context import MultiStreamData


    def test_databundle_is_frozen_dataclass():
        assert dataclasses.is_dataclass(DataBundle)
        mask = np.zeros((2, 2, 4), dtype=bool)
        umap = UnitIndexMap.from_mask(mask, grid_shape=(2, 2))
        bundle = DataBundle(
            pickup_3d=np.zeros((2, 2, 4), dtype=np.float32),
            dropoff_3d=np.zeros((2, 2, 4), dtype=np.float32),
            active_taxis_3d=np.zeros((2, 2, 4), dtype=np.float32),
            mask_3d=mask,
            unit_map=umap,
            n_hours_per_block=np.array([3, 6, 4, 11], dtype=np.int32),
            n_days=65,
            g0_func=G0Function(coefficients=np.zeros(4), d_min=0.01, d_max=10.0),
            hat_matrices={'I_minus_H_demo': np.eye(1), 'M': np.eye(1)},
            trajectories=[],
            multi_stream=MultiStreamData(
                driving_trajs={}, seeking_trajs={},
                profile_features={}, seeking_days={}, driving_days={},
            ),
            discriminator=nn.Identity(),
        )
        assert bundle.n_days == 65
        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.n_days = 100

- [ ] **Step 2: Run test (expect failure)**

    pytest famail_temporal/tests/test_data_loader.py -v

- [ ] **Step 3: Write famail_temporal/data/loader.py (dataclass only)**

    """DataBundle dataclass — immutable container."""

    from __future__ import annotations
    from dataclasses import dataclass
    from typing import Dict, List

    import numpy as np
    import torch.nn as nn

    from famail_temporal.data.active_mask import UnitIndexMap
    from famail_temporal.fairness.g0_power_basis import G0Function
    from famail_temporal.fidelity.context import MultiStreamData
    from famail_temporal.utils.trajectory import Trajectory


    @dataclass(frozen=True)
    class DataBundle:
        pickup_3d: np.ndarray
        dropoff_3d: np.ndarray
        active_taxis_3d: np.ndarray
        mask_3d: np.ndarray
        unit_map: UnitIndexMap
        n_hours_per_block: np.ndarray
        n_days: int
        g0_func: G0Function
        hat_matrices: Dict[str, np.ndarray]
        trajectories: List[Trajectory]
        multi_stream: MultiStreamData
        discriminator: nn.Module

- [ ] **Step 4: Run test (expect pass)**

    pytest famail_temporal/tests/test_data_loader.py -v

- [ ] **Step 5: Commit**

    git add famail_temporal/data/loader.py famail_temporal/tests/test_data_loader.py
    git commit -m "feat(data): add DataBundle frozen dataclass"

---

### Task 19: Preprocess script + DataBundle.load() + cache I/O

This task integrates multiple modules. Split into 10 sub-steps; commit once at the end.

**Files:**
- Create: famail_temporal/data/cache_io.py
- Modify: famail_temporal/data/loader.py
- Create: famail_temporal/preprocess.py
- Modify: famail_temporal/data/__init__.py
- Create: famail_temporal/raw_data/README.md
- Create: famail_temporal/cache/README.md
- Modify: famail_temporal/tests/test_data_loader.py
- Create: famail_temporal/tests/conftest.py

**Prerequisite:** Ten raw files must be copied into famail_temporal/raw_data/ — see the README written in Step 5.

- [ ] **Step 1: Write famail_temporal/data/cache_io.py**

    """Cache I/O helpers."""

    from __future__ import annotations
    import pickle as _pkl
    from pathlib import Path
    from typing import Any

    from famail_temporal import config


    def cache_path(artifact_name: str, include_features: bool = False) -> Path:
        suffix = config.cache_suffix(include_features=include_features)
        return config.CACHE_DIR / f"{artifact_name}_{suffix}.pkl"


    def save_artifact(artifact_name: str, data: Any,
                      include_features: bool = False) -> Path:
        path = cache_path(artifact_name, include_features=include_features)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            _pkl.dump(data, f)
        return path


    def load_artifact(artifact_name: str, include_features: bool = False) -> Any:
        path = cache_path(artifact_name, include_features=include_features)
        if not path.exists():
            raise FileNotFoundError(
                f"Cache artifact missing: {path}. "
                "Run: python -m famail_temporal.preprocess"
            )
        with open(path, "rb") as f:
            return _pkl.load(f)


    def load_raw(filename: str) -> Any:
        """Load a raw .pkl from the raw_data directory."""
        path = config.RAW_DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data missing: {path}. See raw_data/README.md."
            )
        with open(path, "rb") as f:
            return _pkl.load(f)

- [ ] **Step 2: Write famail_temporal/preprocess.py**

    """
    Preprocess raw_data/ → cache/.

    Run:    python -m famail_temporal.preprocess
    Force:  python -m famail_temporal.preprocess --force
    """

    from __future__ import annotations
    import argparse
    import sys

    import numpy as np

    from famail_temporal import config
    from famail_temporal.data.active_mask import UnitIndexMap, compute_active_mask
    from famail_temporal.data.aggregation import (
        aggregate_pickup_dropoff,
        aggregate_active_taxis,
        block_n_hours,
        dataset_n_days,
    )
    from famail_temporal.data.cache_io import cache_path, load_raw, save_artifact
    from famail_temporal.fairness.g0_power_basis import fit as fit_g0
    from famail_temporal.fairness.hat_matrices import precompute_hat_matrices
    from famail_temporal.utils.seeding import set_all_seeds


    def run(force: bool = False) -> None:
        set_all_seeds(config.DEFAULT_SEED)

        def _should_write(name, include_features=False) -> bool:
            if force:
                return True
            return not cache_path(name, include_features).exists()

        print("[preprocess] Loading raw data ...", flush=True)
        pickup_dropoff_raw = load_raw("pickup_dropoff_counts.pkl")
        active_taxis_raw = load_raw("active_taxis_5x5_hourly.pkl")
        demographics_raw = load_raw("cell_demographics.pkl")
        district_raw = load_raw("grid_to_district_mapping.pkl")

        demographics_grid = demographics_raw['demographics_grid']
        demo_feature_names = list(demographics_raw['feature_names'])
        valid_mask = district_raw['valid_mask']

        for feat in config.DEMOGRAPHIC_FEATURES:
            if feat not in demo_feature_names:
                raise ValueError(
                    f"Demographic feature '{feat}' not found in raw: "
                    f"{demo_feature_names}"
                )

        if isinstance(active_taxis_raw, dict) and 'data' in active_taxis_raw:
            active_taxis_raw = active_taxis_raw['data']

        n_days = max(dataset_n_days(pickup_dropoff_raw),
                     dataset_n_days(active_taxis_raw))
        print(f"[preprocess] n_days = {n_days}", flush=True)

        print("[preprocess] Aggregating to (48, 90, T) ...", flush=True)
        pickup_3d, dropoff_3d = aggregate_pickup_dropoff(
            pickup_dropoff_raw, n_days=n_days
        )
        active_taxis_3d = aggregate_active_taxis(active_taxis_raw, n_days=n_days)

        if _should_write("pickup_counts"):
            save_artifact("pickup_counts", pickup_3d)
        if _should_write("dropoff_counts"):
            save_artifact("dropoff_counts", dropoff_3d)
        if _should_write("active_taxis"):
            save_artifact("active_taxis", active_taxis_3d)

        print("[preprocess] Computing active mask ...", flush=True)
        feat_indices = [demo_feature_names.index(f)
                        for f in config.DEMOGRAPHIC_FEATURES]
        demographics_selected = demographics_grid[..., feat_indices].astype(np.float32)
        mask_3d = compute_active_mask(active_taxis_3d, valid_mask, demographics_selected)

        n_total = int(mask_3d.sum())
        assert n_total >= config.MIN_TOTAL_ACTIVE_UNITS, (
            f"Only {n_total} active units — below MIN_TOTAL_ACTIVE_UNITS="
            f"{config.MIN_TOTAL_ACTIVE_UNITS}"
        )

        unit_map = UnitIndexMap.from_mask(mask_3d, grid_shape=config.GRID_DIMS)
        for t in range(config.T):
            assert unit_map.units_per_block[t] >= config.MIN_ACTIVE_UNITS_PER_BLOCK, (
                f"Block {config.TIME_BLOCKS[t][0]} has only "
                f"{unit_map.units_per_block[t]} active units"
            )
        print(
            f"[preprocess] n_active_units = {unit_map.n_units} "
            f"(per block: {unit_map.units_per_block.tolist()})",
            flush=True,
        )

        if _should_write("active_mask"):
            save_artifact("active_mask", mask_3d)
        if _should_write("unit_index_map"):
            save_artifact("unit_index_map", unit_map)

        print("[preprocess] Extracting active-unit vectors ...", flush=True)
        D_vec = pickup_3d[mask_3d]
        S_vec = active_taxis_3d[mask_3d]
        demo_flat = demographics_selected.reshape(-1, demographics_selected.shape[-1])

        unit_demographics = np.empty(
            (unit_map.n_units, len(config.DEMOGRAPHIC_FEATURES)),
            dtype=np.float32,
        )
        for i in range(unit_map.n_units):
            flat_cell = unit_map.to_flat_cell(i)
            unit_demographics[i] = demo_flat[flat_cell]

        assert np.isfinite(unit_demographics).all(), (
            "Demographics for active units contain NaN"
        )

        print("[preprocess] Fitting g0(D) ...", flush=True)
        D_clamped = np.maximum(D_vec, config.DEMAND_FLOOR)
        Y_vec = S_vec / D_clamped
        g0_func, g0_diag = fit_g0(D_clamped, Y_vec)
        print(f"[preprocess] g0 diagnostics: {g0_diag}", flush=True)
        if _should_write("g0_power_basis"):
            save_artifact("g0_power_basis", g0_func)

        print("[preprocess] Precomputing hat matrices ...", flush=True)
        hat = precompute_hat_matrices(
            demands=D_clamped,
            demographic_features=unit_demographics,
            feature_names=config.DEMOGRAPHIC_FEATURES,
        )
        if _should_write("hat_matrices", include_features=True):
            save_artifact("hat_matrices", hat, include_features=True)

        print("[preprocess] Done.", flush=True)


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Preprocess FAMAIL-Temporal raw data."
        )
        parser.add_argument("--force", action="store_true",
                            help="Overwrite existing cache.")
        args = parser.parse_args()
        try:
            run(force=args.force)
        except FileNotFoundError as e:
            print(f"[preprocess] ERROR: {e}", file=sys.stderr)
            sys.exit(1)

- [ ] **Step 3: Append .load() to data/loader.py**

Append the following to famail_temporal/data/loader.py (below the DataBundle dataclass):

    import pickle as _pkl
    import random as _random

    from famail_temporal.data.cache_io import load_artifact
    from famail_temporal.data.aggregation import block_n_hours, dataset_n_days
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


    def _parse_trajectory(traj_data, trajectory_id, driver_id):
        if not isinstance(traj_data, list) or len(traj_data) < 2:
            return None
        states = []
        for state_data in traj_data:
            if len(state_data) >= 4:
                states.append(TrajectoryState(
                    x_grid=int(state_data[0]) - 1,
                    y_grid=int(state_data[1]) - 1,
                    time_bucket=int(state_data[2]),
                    day_index=int(state_data[3]),
                ))
        if len(states) < 2:
            return None
        return Trajectory(
            trajectory_id=trajectory_id, driver_id=driver_id, states=states,
        )


    def _load_trajectories(max_trajectories=None, max_drivers=None):
        from famail_temporal import config
        path = config.RAW_DATA_DIR / "passenger_seeking_trajs_45-800.pkl"
        with open(path, "rb") as f:
            data = _pkl.load(f)
        driver_keys = list(data.keys())
        if max_drivers:
            driver_keys = driver_keys[:max_drivers]
        all_trajs = []
        for did in driver_keys:
            for td in data[did]:
                all_trajs.append((did, td))
        if max_trajectories and len(all_trajs) > max_trajectories:
            _random.seed(config.DEFAULT_SEED)
            all_trajs = _random.sample(all_trajs, max_trajectories)
        out = []
        for i, (did, td) in enumerate(all_trajs):
            t = _parse_trajectory(td, trajectory_id=i, driver_id=did)
            if t is not None:
                out.append(t)
        return out


    def _load_multi_stream():
        from famail_temporal import config
        def _load(filename):
            path = config.RAW_DATA_DIR / filename
            with open(path, "rb") as f:
                return _pkl.load(f)
        driving = {int(k): v for k, v in _load("ms_driving_trajs.pkl").items()}
        seeking = {int(k): v for k, v in _load("ms_seeking_trajs.pkl").items()}
        profile_raw = _load("ms_profile_features.pkl")
        raw_features = profile_raw.get("features_normalized", profile_raw)
        profile = {int(k): v for k, v in raw_features.items()}
        seeking_days = {int(k): v for k, v in _load("ms_seeking_calendar_days.pkl").items()}
        driving_days = {int(k): v for k, v in _load("ms_driving_calendar_days.pkl").items()}
        return MultiStreamData(
            driving_trajs=driving, seeking_trajs=seeking,
            profile_features=profile,
            seeking_days=seeking_days, driving_days=driving_days,
        )


    def _load_discriminator_stub():
        import torch.nn as _nn
        return _nn.Identity()


    def _bundle_load(max_trajectories=None, max_drivers=None):
        from famail_temporal import config
        pickup_3d = load_artifact("pickup_counts")
        dropoff_3d = load_artifact("dropoff_counts")
        active_taxis_3d = load_artifact("active_taxis")
        mask_3d = load_artifact("active_mask")
        unit_map = load_artifact("unit_index_map")
        g0_func = load_artifact("g0_power_basis")
        hat_matrices = load_artifact("hat_matrices", include_features=True)

        assert unit_map.n_units == hat_matrices['I_minus_H_demo'].shape[0], (
            f"unit_map.n_units ({unit_map.n_units}) != hat matrix "
            f"shape[0] ({hat_matrices['I_minus_H_demo'].shape[0]})"
        )
        assert pickup_3d.shape == dropoff_3d.shape == active_taxis_3d.shape == (
            config.GRID_DIMS[0], config.GRID_DIMS[1], config.T,
        )

        n_hours_per_block = np.array(
            [block_n_hours(t) for t in range(config.T)], dtype=np.int32,
        )

        raw_pickup_path = config.RAW_DATA_DIR / "pickup_dropoff_counts.pkl"
        with open(raw_pickup_path, "rb") as f:
            raw_pickup = _pkl.load(f)
        n_days = dataset_n_days(raw_pickup)

        trajectories = _load_trajectories(
            max_trajectories=max_trajectories, max_drivers=max_drivers,
        )
        multi_stream = _load_multi_stream()

        try:
            from famail_temporal.fidelity.checkpoint import load_discriminator
            ckpt_path = (
                config.DISCRIMINATOR_CHECKPOINT_DIR
                / config.DISCRIMINATOR_CHECKPOINT_FILENAME
            )
            if ckpt_path.exists():
                discriminator = load_discriminator(ckpt_path)
            else:
                discriminator = _load_discriminator_stub()
        except (ImportError, ModuleNotFoundError):
            discriminator = _load_discriminator_stub()

        return DataBundle(
            pickup_3d=pickup_3d.copy(),
            dropoff_3d=dropoff_3d.copy(),
            active_taxis_3d=active_taxis_3d.copy(),
            mask_3d=mask_3d.copy(),
            unit_map=unit_map,
            n_hours_per_block=n_hours_per_block,
            n_days=n_days,
            g0_func=g0_func,
            hat_matrices=hat_matrices,
            trajectories=trajectories,
            multi_stream=multi_stream,
            discriminator=discriminator,
        )


    DataBundle.load = classmethod(
        lambda cls, max_trajectories=None, max_drivers=None:
            _bundle_load(max_trajectories, max_drivers)
    )

- [ ] **Step 4: Update famail_temporal/data/__init__.py**

    """Data loading and aggregation."""

    from famail_temporal.data.loader import DataBundle
    from famail_temporal.data.active_mask import UnitIndexMap, compute_active_mask
    from famail_temporal.data.aggregation import (
        hour_to_block_index,
        time_bucket_to_hour,
        block_n_hours,
        dataset_n_days,
        aggregate_pickup_dropoff,
        aggregate_active_taxis,
    )

    __all__ = [
        "DataBundle", "UnitIndexMap", "compute_active_mask",
        "hour_to_block_index", "time_bucket_to_hour", "block_n_hours",
        "dataset_n_days",
        "aggregate_pickup_dropoff", "aggregate_active_taxis",
    ]

- [ ] **Step 5: Write famail_temporal/raw_data/README.md**

Create the README with a table of the ten required files and copying instructions. See the design spec section 13.8 for the full content. Key table rows:

| Filename | Source |
|---|---|
| passenger_seeking_trajs_45-800.pkl | source_data/passenger_seeking_trajs_45-800.pkl |
| pickup_dropoff_counts.pkl | source_data/pickup_dropoff_counts.pkl |
| active_taxis_5x5_hourly.pkl | source_data/active_taxis_5x5_hourly.pkl |
| cell_demographics.pkl | source_data/cell_demographics.pkl |
| grid_to_district_mapping.pkl | source_data/grid_to_district_mapping.pkl |
| ms_driving_trajs.pkl | discriminator/multi_stream/extracted_data/driving_trajs.pkl |
| ms_seeking_trajs.pkl | discriminator/multi_stream/extracted_data/seeking_trajs.pkl |
| ms_profile_features.pkl | discriminator/multi_stream/extracted_data/profile_features.pkl |
| ms_seeking_calendar_days.pkl | discriminator/multi_stream/extracted_data/seeking_calendar_days.pkl |
| ms_driving_calendar_days.pkl | discriminator/multi_stream/extracted_data/driving_calendar_days.pkl |

Plus a "Copying (from repo root)" section with the ten cp commands for each file.

- [ ] **Step 6: Write famail_temporal/cache/README.md**

Describe the filename scheme `{artifact}_T{T}_thr{threshold}[_feat-...].pkl` and list the artifact types. See design spec section 13.9 for the full content.

- [ ] **Step 7: Create famail_temporal/tests/conftest.py**

    """Pytest fixtures and markers for famail_temporal tests."""

    import pytest

    from famail_temporal.utils.seeding import set_all_seeds


    def pytest_addoption(parser):
        parser.addoption("--run-slow", action="store_true", default=False,
                         help="Run tests marked @pytest.mark.slow")


    def pytest_configure(config):
        config.addinivalue_line(
            "markers", "slow: mark test as slow (deselected by default)"
        )


    def pytest_collection_modifyitems(config, items):
        if config.getoption("--run-slow"):
            return
        skip_slow = pytest.mark.skip(reason="need --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


    @pytest.fixture(autouse=True)
    def seeded():
        set_all_seeds(42)

- [ ] **Step 8: Append slow integration test to tests/test_data_loader.py**

    import pytest


    @pytest.mark.slow
    def test_databundle_load_real_data():
        """End-to-end DataBundle.load() — skip if raw data missing."""
        from famail_temporal import config
        from famail_temporal.data.loader import DataBundle

        required = [
            config.RAW_DATA_DIR / "pickup_dropoff_counts.pkl",
            config.RAW_DATA_DIR / "active_taxis_5x5_hourly.pkl",
            config.RAW_DATA_DIR / "cell_demographics.pkl",
            config.RAW_DATA_DIR / "grid_to_district_mapping.pkl",
        ]
        for path in required:
            if not path.exists():
                pytest.skip(f"Raw data missing: {path}")

        cache_files = list(config.CACHE_DIR.glob("*.pkl"))
        if not cache_files:
            pytest.skip("Cache empty — run preprocess first")

        bundle = DataBundle.load(max_trajectories=10, max_drivers=2)
        assert bundle.pickup_3d.shape == (*config.GRID_DIMS, config.T)
        assert bundle.unit_map.n_units >= config.MIN_TOTAL_ACTIVE_UNITS
        assert bundle.unit_map.n_units == bundle.hat_matrices['I_minus_H_demo'].shape[0]

- [ ] **Step 9: Run fast tests (expect pass; slow tests skipped)**

    pytest famail_temporal/tests/ -v

- [ ] **Step 10: Commit**

    git add famail_temporal/data/cache_io.py \
            famail_temporal/data/loader.py \
            famail_temporal/data/__init__.py \
            famail_temporal/preprocess.py \
            famail_temporal/raw_data/README.md \
            famail_temporal/cache/README.md \
            famail_temporal/tests/conftest.py \
            famail_temporal/tests/test_data_loader.py
    git commit -m "feat(data): preprocess + DataBundle.load() + cache I/O"

---

## Phase 6: Fidelity port (Tasks 20–25)

### Task 20: Port FeatureNormalizer to fidelity/model.py

**Files:**
- Source reference (do not modify): discriminator/model/model.py lines 30-113
- Create: famail_temporal/fidelity/model.py
- Create: famail_temporal/tests/test_fidelity_model.py

- [ ] **Step 1: Write failing test**

    """Tests for fidelity.model components."""
    import torch

    from famail_temporal.fidelity.model import FeatureNormalizer


    def test_feature_normalizer_output_shape():
        norm = FeatureNormalizer()
        x = torch.randn(2, 20, 4)
        out = norm(x)
        assert out.shape == (2, 20, 4)


    def test_feature_normalizer_has_buffers():
        norm = FeatureNormalizer()
        buffers = {name for name, _ in norm.named_buffers()}
        assert len(buffers) > 0

- [ ] **Step 2: Run test (expect failure)**

    pytest famail_temporal/tests/test_fidelity_model.py -v

- [ ] **Step 3: Port FeatureNormalizer**

Copy the FeatureNormalizer class (lines 30-113) verbatim from discriminator/model/model.py into famail_temporal/fidelity/model.py. Prepend this module docstring:

    """
    Multi-stream Siamese discriminator — inference-only port.

    Only four classes are ported from discriminator/model/model.py:
      - FeatureNormalizer
      - SiameseLSTMEncoder
      - ProfileEncoder
      - MultiStreamSiameseDiscriminator

    Training loops, dataset classes, and five deprecated alternate
    architectures (SiameseLSTMDiscriminator, TransformerEncoder,
    SiameseTransformerDiscriminator, SiameseLSTMDiscriminatorV2) are
    intentionally excluded.

    Classes are lifted verbatim from the original model.py.
    """

    from __future__ import annotations

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

Then paste the FeatureNormalizer class.

- [ ] **Step 4: Run test (expect pass)**

    pytest famail_temporal/tests/test_fidelity_model.py -v

- [ ] **Step 5: Commit**

    git add famail_temporal/fidelity/model.py famail_temporal/tests/test_fidelity_model.py
    git commit -m "feat(fidelity): port FeatureNormalizer class"

---

### Task 21: Port SiameseLSTMEncoder and ProfileEncoder

**Files:**
- Source reference: discriminator/model/model.py lines 116-287
- Modify: famail_temporal/fidelity/model.py
- Modify: famail_temporal/tests/test_fidelity_model.py

- [ ] **Step 1: Append failing tests**

    from famail_temporal.fidelity.model import SiameseLSTMEncoder, ProfileEncoder


    def test_siamese_lstm_encoder_shape():
        enc = SiameseLSTMEncoder(input_dim=4, hidden_dim=64)
        x = torch.randn(3, 15, 4)
        out = enc(x)
        assert out.shape[0] == 3
        assert out.shape[-1] in (64, 128)  # uni- or bidirectional


    def test_profile_encoder_shape():
        enc = ProfileEncoder(input_dim=11, hidden_dim=32)
        x = torch.randn(3, 11)
        out = enc(x)
        assert out.shape == (3, 32)

- [ ] **Step 2: Run tests (expect failure)**

    pytest famail_temporal/tests/test_fidelity_model.py -v

- [ ] **Step 3: Port both classes**

Copy SiameseLSTMEncoder (lines 116-232) and ProfileEncoder (lines 233-287) verbatim from discriminator/model/model.py into famail_temporal/fidelity/model.py.

DO NOT copy these deprecated classes:
  - SiameseLSTMDiscriminator
  - TransformerEncoder
  - SiameseTransformerDiscriminator
  - SiameseLSTMDiscriminatorV2

If the constructor signatures differ from the test examples (e.g., parameter named `lstm_hidden` instead of `hidden_dim`), update the test to match the actual class.

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_fidelity_model.py -v

- [ ] **Step 5: Commit**

    git add famail_temporal/fidelity/model.py famail_temporal/tests/test_fidelity_model.py
    git commit -m "feat(fidelity): port SiameseLSTMEncoder and ProfileEncoder"

---

### Task 22: Port MultiStreamSiameseDiscriminator

**Files:**
- Source reference: discriminator/model/model.py line 838 to end of class
- Modify: famail_temporal/fidelity/model.py
- Modify: famail_temporal/tests/test_fidelity_model.py

- [ ] **Step 1: Append failing test**

    from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator


    def test_multistream_discriminator_shape():
        model = MultiStreamSiameseDiscriminator()
        model.train(False)
        batch_size, n_trajs, seq_len = 2, 5, 20
        x1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
        x2 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
        driving_1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
        driving_2 = driving_1.clone()
        profile_1 = torch.randn(batch_size, 11)
        profile_2 = profile_1.clone()

        with torch.no_grad():
            out = model(x1, x2, driving_1=driving_1, driving_2=driving_2,
                        profile_1=profile_1, profile_2=profile_2)
        assert out.shape[0] == batch_size

- [ ] **Step 2: Run test (expect failure)**

    pytest famail_temporal/tests/test_fidelity_model.py -v

- [ ] **Step 3: Port the class**

Copy MultiStreamSiameseDiscriminator (line 838 through end of class body) verbatim into famail_temporal/fidelity/model.py. Preserve all forward logic.

If the test fails with kwarg naming mismatches (e.g., `driving_traj_1` instead of `driving_1`), update the test to match the actual signature.

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_fidelity_model.py -v

- [ ] **Step 5: Commit**

    git add famail_temporal/fidelity/model.py famail_temporal/tests/test_fidelity_model.py
    git commit -m "feat(fidelity): port MultiStreamSiameseDiscriminator"

---

### Task 23: fidelity/checkpoint.py — load_discriminator

**Files:**
- Create: famail_temporal/fidelity/checkpoint.py
- Create: famail_temporal/tests/test_fidelity_checkpoint.py
- Create: famail_temporal/discriminator_checkpoints/README.md

**Prerequisite:** Copy a canonical checkpoint to famail_temporal/discriminator_checkpoints/default/best.pt.

- [ ] **Step 1: Write failing test**

    """Tests for fidelity.checkpoint."""
    import pytest

    from famail_temporal import config
    from famail_temporal.fidelity.checkpoint import load_discriminator


    @pytest.mark.slow
    def test_load_discriminator_inference_mode():
        ckpt_path = (
            config.DISCRIMINATOR_CHECKPOINT_DIR
            / config.DISCRIMINATOR_CHECKPOINT_FILENAME
        )
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not present at {ckpt_path}")
        model = load_discriminator(ckpt_path)
        assert not model.training
        for p in model.parameters():
            assert not p.requires_grad

- [ ] **Step 2: Run test (expect skip or fail)**

    pytest famail_temporal/tests/test_fidelity_checkpoint.py --run-slow -v

- [ ] **Step 3: Write famail_temporal/fidelity/checkpoint.py**

    """Load a pre-trained MultiStreamSiameseDiscriminator checkpoint."""

    from __future__ import annotations
    from pathlib import Path

    import torch

    from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator


    class MissingArchitectureConfig(RuntimeError):
        pass


    def load_discriminator(checkpoint_path: Path) -> MultiStreamSiameseDiscriminator:
        """Load weights, switch to inference mode, freeze parameters."""
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False,
        )

        arch_config = checkpoint.get("architecture_config", None)
        if arch_config is None:
            model = MultiStreamSiameseDiscriminator()
        else:
            model = MultiStreamSiameseDiscriminator(**arch_config)

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            if arch_config is None:
                raise MissingArchitectureConfig(
                    "Checkpoint state dict does not match default architecture. "
                    "The checkpoint is missing 'architecture_config'. Add it "
                    "via a one-time preprocessing step."
                ) from e
            raise

        model.train(False)
        for p in model.parameters():
            p.requires_grad = False
        return model

- [ ] **Step 4: Write famail_temporal/discriminator_checkpoints/README.md**

Document the canonical path (default/best.pt), provenance (copy from discriminator/model/checkpoints/20260316_223817/best.pt), expected checkpoint format (model_state_dict + architecture_config), and how to substitute alternate checkpoints by editing config.DISCRIMINATOR_CHECKPOINT_FILENAME.

- [ ] **Step 5: Run slow test if checkpoint present**

    pytest famail_temporal/tests/test_fidelity_checkpoint.py --run-slow -v

- [ ] **Step 6: Commit**

    git add famail_temporal/fidelity/checkpoint.py \
            famail_temporal/tests/test_fidelity_checkpoint.py \
            famail_temporal/discriminator_checkpoints/README.md
    git commit -m "feat(fidelity): add load_discriminator + README"

---

### Task 24: MultiStreamContextBuilder

**Files:**
- Source reference: trajectory_modification/multi_stream_context.py
- Modify: famail_temporal/fidelity/context.py
- Create: famail_temporal/tests/test_ms_context.py

- [ ] **Step 1: Write failing test**

    """Tests for MultiStreamContextBuilder."""
    import numpy as np
    import torch

    from famail_temporal.fidelity.context import (
        MultiStreamContextBuilder, MultiStreamData,
    )
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


    def _synthetic_ms_data(n_drivers=3, n_trajs=10, seq_len=20):
        driving, seeking, profile = {}, {}, {}
        seeking_days, driving_days = {}, {}
        for d in range(n_drivers):
            driving[d] = [
                [[float(i + 1), float(i + 2), i, 1] for i in range(seq_len)]
                for _ in range(n_trajs)
            ]
            seeking[d] = [
                [[float(i + 1), float(i + 2), i, 1] for i in range(seq_len)]
                for _ in range(n_trajs)
            ]
            profile[d] = np.zeros(11, dtype=np.float32)
            seeking_days[d] = [1] * n_trajs
            driving_days[d] = [1] * n_trajs
        return MultiStreamData(
            driving_trajs=driving, seeking_trajs=seeking,
            profile_features=profile,
            seeking_days=seeking_days, driving_days=driving_days,
        )


    def _make_trajectory(driver_id, seq_len=20):
        states = [
            TrajectoryState(
                x_grid=float(i), y_grid=float(i + 1),
                time_bucket=50 + i, day_index=1,
            )
            for i in range(seq_len)
        ]
        return Trajectory(trajectory_id=0, driver_id=driver_id, states=states)


    def test_builder_returns_x1_x2():
        ms = _synthetic_ms_data()
        builder = MultiStreamContextBuilder(ms, device="cpu", seed=42)
        t_orig = _make_trajectory(driver_id=0)
        t_mod = _make_trajectory(driver_id=0)
        kw = builder.build_fidelity_kwargs(t_orig, t_mod)
        assert "x1" in kw
        assert "x2" in kw
        assert kw["x1"].shape == kw["x2"].shape


    def test_same_driver_branch_invariant():
        ms = _synthetic_ms_data()
        builder = MultiStreamContextBuilder(ms, device="cpu", seed=42)
        t_orig = _make_trajectory(driver_id=0)
        t_mod = _make_trajectory(driver_id=0)
        kw = builder.build_fidelity_kwargs(t_orig, t_mod)
        if "driving_1" in kw and "driving_2" in kw:
            assert torch.equal(kw["driving_1"], kw["driving_2"])
        if "profile_1" in kw and "profile_2" in kw:
            assert torch.equal(kw["profile_1"], kw["profile_2"])

- [ ] **Step 2: Run test (expect failure)**

    pytest famail_temporal/tests/test_ms_context.py -v

- [ ] **Step 3: Append MultiStreamContextBuilder to fidelity/context.py**

Open trajectory_modification/multi_stream_context.py and copy:
- The "Decision 1-4" comment block at the top of the file (preserve verbatim as a docstring above the class)
- The entire MultiStreamContextBuilder class

Append both to famail_temporal/fidelity/context.py. Adapt the constructor to accept a single MultiStreamData instance:

    class MultiStreamContextBuilder:
        def __init__(
            self,
            multi_stream_data: "MultiStreamData",
            n_trajs: int = 5,
            fill_strategy: str = "sample",
            device: str = "cpu",
            seed: int = 42,
        ):
            self.driving_trajs = multi_stream_data.driving_trajs
            self.seeking_trajs = multi_stream_data.seeking_trajs
            self.profile_features = multi_stream_data.profile_features
            self.seeking_days = multi_stream_data.seeking_days
            self.driving_days = multi_stream_data.driving_days
            self.n_trajs = n_trajs
            self.fill_strategy = fill_strategy
            self.device = device
            self.seed = seed
            # Copy remaining initialization verbatim from original __init__

Preserve all four design decisions (same-driver branches; sample fill; 1-indexed coordinate conversion; slot-0 gradient flow).

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_ms_context.py -v

- [ ] **Step 5: Commit**

    git add famail_temporal/fidelity/context.py famail_temporal/tests/test_ms_context.py
    git commit -m "feat(fidelity): port MultiStreamContextBuilder"

---

### Task 25: fidelity/compute.py — compute_ffidelity

**Files:**
- Create: famail_temporal/fidelity/compute.py
- Create: famail_temporal/tests/test_fidelity_compute.py
- Modify: famail_temporal/fidelity/__init__.py

- [ ] **Step 1: Write failing test**

    """Tests for fidelity.compute."""
    import torch

    from famail_temporal.fidelity.compute import compute_ffidelity
    from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator


    def test_compute_ffidelity_in_unit_interval():
        torch.manual_seed(0)
        model = MultiStreamSiameseDiscriminator()
        model.train(False)
        for p in model.parameters():
            p.requires_grad = False

        batch_size, n_trajs, seq_len = 1, 5, 15
        x1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
        x2 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
        driving_1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
        driving_2 = driving_1.clone()
        profile_1 = torch.randn(batch_size, 11)
        profile_2 = profile_1.clone()

        ms_kwargs = {
            "x1": x1, "x2": x2,
            "driving_1": driving_1, "driving_2": driving_2,
            "profile_1": profile_1, "profile_2": profile_2,
        }
        tau = torch.rand(1, seq_len, 4)
        tau_prime = tau.clone()

        f, _ = compute_ffidelity(model, tau, tau_prime, ms_kwargs)
        assert 0.0 <= float(f) <= 1.0

- [ ] **Step 2: Run test (expect failure)**

    pytest famail_temporal/tests/test_fidelity_compute.py -v

- [ ] **Step 3: Write famail_temporal/fidelity/compute.py**

    """
    Compute F_fidelity = Discriminator(tau, tau_prime).

    cuDNN workaround: cuDNN's RNN backward requires training mode, but we
    need inference-mode behavior while allowing gradient flow through the
    LSTM for ST-iFGSM. Disabling cuDNN for the forward pass uses the
    pure-PyTorch LSTM implementation, which supports backward in inference
    mode.
    """

    from __future__ import annotations
    from typing import Dict, Tuple

    import torch


    def compute_ffidelity(
        discriminator: torch.nn.Module,
        tau_features: torch.Tensor,
        tau_prime_features: torch.Tensor,
        multi_stream_kwargs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """Forward the discriminator; return F_fidelity in [0, 1] + debug dict."""
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            if "x1" in multi_stream_kwargs and "x2" in multi_stream_kwargs:
                x1 = multi_stream_kwargs["x1"]
                x2 = multi_stream_kwargs["x2"]
                extra = {
                    k: v for k, v in multi_stream_kwargs.items()
                    if k not in {"x1", "x2"}
                }
                similarity = discriminator(x1, x2, **extra)
            else:
                similarity = discriminator(tau_features, tau_prime_features)

        f_fidelity = similarity.mean() if similarity.dim() > 0 else similarity
        f_fidelity = torch.clamp(f_fidelity, 0.0, 1.0)
        return f_fidelity, {"similarity_raw": float(similarity.mean())}

- [ ] **Step 4: Update famail_temporal/fidelity/__init__.py**

    """Fidelity term: discriminator-based realism check."""

    from famail_temporal.fidelity.context import (
        MultiStreamData, MultiStreamContextBuilder,
    )
    from famail_temporal.fidelity.model import (
        FeatureNormalizer, SiameseLSTMEncoder, ProfileEncoder,
        MultiStreamSiameseDiscriminator,
    )
    from famail_temporal.fidelity.checkpoint import (
        load_discriminator, MissingArchitectureConfig,
    )
    from famail_temporal.fidelity.compute import compute_ffidelity

    __all__ = [
        "MultiStreamData", "MultiStreamContextBuilder",
        "FeatureNormalizer", "SiameseLSTMEncoder", "ProfileEncoder",
        "MultiStreamSiameseDiscriminator",
        "load_discriminator", "MissingArchitectureConfig",
        "compute_ffidelity",
    ]

- [ ] **Step 5: Run tests (expect pass)**

    pytest famail_temporal/tests/test_fidelity_compute.py -v

- [ ] **Step 6: Commit**

    git add famail_temporal/fidelity/compute.py \
            famail_temporal/fidelity/__init__.py \
            famail_temporal/tests/test_fidelity_compute.py
    git commit -m "feat(fidelity): compute_ffidelity with cuDNN workaround"

---

**End of Phase 5–6 file.** Continue with 2026-04-16-famail-temporal-phase7-8.md.
