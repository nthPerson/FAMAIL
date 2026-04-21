"""Per-trajectory removal record + summary."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


RemovalCategory = Literal[
    "out_of_bounds",
    "degenerate_length",
    "no_matching_count",
    "temporal_order",
]


@dataclass
class RemovalRecord:
    driver_id: str
    driver_idx: int | None
    trajectory_index_within_driver: int
    kind: Literal["seeking", "driving"]
    which_invariant: int
    failing_values: dict[str, Any]
    n_states_before_removal: int
    removal_reason_category: RemovalCategory

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_id": self.driver_id,
            "driver_idx": self.driver_idx,
            "trajectory_index_within_driver": self.trajectory_index_within_driver,
            "kind": self.kind,
            "which_invariant": self.which_invariant,
            "failing_values": self.failing_values,
            "n_states_before_removal": self.n_states_before_removal,
            "removal_reason_category": self.removal_reason_category,
        }


@dataclass
class RemovalSummary:
    total_seeking_extracted: int = 0
    total_driving_extracted: int = 0
    removals: list[RemovalRecord] = field(default_factory=list)

    def counts_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.removals:
            out[r.removal_reason_category] = out.get(r.removal_reason_category, 0) + 1
        return out

    def total_extracted(self) -> int:
        return self.total_seeking_extracted + self.total_driving_extracted

    def removal_rate(self) -> float:
        total = self.total_extracted()
        return len(self.removals) / total if total > 0 else 0.0
