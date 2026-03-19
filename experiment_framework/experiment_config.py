"""
Experiment configuration for the FAMAIL Experiment & Analysis Framework.

Provides ExperimentConfig (single run) and SweepConfig (parameter sweep)
dataclasses. All parameters for attribution, modification, objective weights,
discriminator, and instrumentation are consolidated here.
"""

from __future__ import annotations

import json
import itertools
from copy import deepcopy
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentConfig:
    """Complete configuration for a FAMAIL experiment run.

    Consolidates all parameters for the two-phase trajectory modification
    pipeline (Phase 1: attribution/selection, Phase 2: ST-iFGSM editing)
    plus instrumentation and output settings.

    The framework uses soft_cell gradient mode exclusively for differentiable
    gradient computation through the objective function terms.
    """

    # ---- Experiment Metadata ----
    experiment_name: str = "default"
    description: str = ""
    random_seed: int = 42

    # ---- Phase 1: Attribution ----
    top_k: int = 10
    selection_method: str = "top_k"        # "top_k" or "diverse"
    lis_weight: float = 0.5
    dcd_weight: float = 0.5

    # ---- Phase 2: ST-iFGSM ----
    alpha: float = 0.1                     # step size
    epsilon: float = 2.0                   # max perturbation (grid cells)
    max_iterations: int = 50
    convergence_threshold: float = 1e-6

    # ---- Objective Weights ----
    alpha_spatial: float = 0.33
    alpha_causal: float = 0.33
    alpha_fidelity: float = 0.34

    # ---- F_causal Formulation ----
    causal_formulation: str = "option_b"   # "baseline", "option_b", "option_c"

    # ---- Soft Cell Assignment ----
    neighborhood_size: int = 5
    temperature: float = 1.0
    temperature_annealing: bool = True
    tau_max: float = 1.0
    tau_min: float = 0.1

    # ---- Discriminator ----
    use_discriminator: bool = True
    discriminator_checkpoint: str = "checkpoints/20260316_223817/best.pt"
    seeking_fill_strategy: str = "sample"
    device: str = "auto"                   # "auto", "cuda", "cpu"

    # ---- Data ----
    max_trajectories: int = 100

    # ---- Instrumentation ----
    record_gradient_decomposition: bool = False  # 3x backward cost per iter
    snapshot_every_n: int = 1                    # global snapshot interval
    record_debug_dicts: bool = True

    # ---- Output ----
    output_dir: str = "experiment_results"

    # ---- Grid ----
    grid_dims: Tuple[int, int] = (48, 90)

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.alpha > 0, f"Step size alpha must be positive, got {self.alpha}"
        assert self.epsilon > 0, f"Epsilon must be positive, got {self.epsilon}"
        assert self.max_iterations > 0, f"max_iterations must be positive, got {self.max_iterations}"
        assert self.top_k > 0, f"top_k must be positive, got {self.top_k}"

        total = self.alpha_spatial + self.alpha_causal + self.alpha_fidelity
        assert abs(total - 1.0) < 0.05, (
            f"Objective weights should sum to ~1.0, got {total:.4f} "
            f"(spatial={self.alpha_spatial}, causal={self.alpha_causal}, "
            f"fidelity={self.alpha_fidelity})"
        )

        assert self.causal_formulation in ("baseline", "option_b", "option_c"), (
            f"Unknown causal_formulation: {self.causal_formulation}"
        )
        assert self.selection_method in ("top_k", "diverse"), (
            f"Unknown selection_method: {self.selection_method}"
        )
        assert self.seeking_fill_strategy in ("sample", "replicate", "single"), (
            f"Unknown seeking_fill_strategy: {self.seeking_fill_strategy}"
        )
        assert self.device in ("auto", "cuda", "cpu"), (
            f"Unknown device: {self.device}"
        )
        assert self.snapshot_every_n >= 1, (
            f"snapshot_every_n must be >= 1, got {self.snapshot_every_n}"
        )

    def normalize_weights(self) -> None:
        """Normalize objective weights to sum to 1.0."""
        total = self.alpha_spatial + self.alpha_causal + self.alpha_fidelity
        if total > 0:
            self.alpha_spatial /= total
            self.alpha_causal /= total
            self.alpha_fidelity /= total

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        # Convert tuple to list for JSON compatibility
        d['grid_dims'] = list(d['grid_dims'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """Create from dict, handling type conversions."""
        d = d.copy()
        if 'grid_dims' in d and isinstance(d['grid_dims'], list):
            d['grid_dims'] = tuple(d['grid_dims'])
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str) -> None:
        """Save to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def with_overrides(self, **kwargs) -> 'ExperimentConfig':
        """Create a new config with specific fields overridden."""
        return replace(self, **kwargs)


@dataclass
class SweepConfig:
    """Configuration for parameter sweeps.

    Generates the Cartesian product of all sweep parameter values,
    creating one ExperimentConfig per combination.
    """

    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)

    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all combinations (Cartesian product) of sweep params."""
        if not self.sweep_params:
            return [deepcopy(self.base_config)]

        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        configs = []

        for combo in itertools.product(*param_values):
            overrides = dict(zip(param_names, combo))
            name_parts = [f"{k}={v}" for k, v in overrides.items()]
            overrides['experiment_name'] = f"sweep_{'_'.join(name_parts)}"
            cfg = replace(self.base_config, **overrides)
            configs.append(cfg)

        return configs

    def __len__(self) -> int:
        if not self.sweep_params:
            return 1
        total = 1
        for values in self.sweep_params.values():
            total *= len(values)
        return total
