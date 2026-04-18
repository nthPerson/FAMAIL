"""Tests for the MultiStreamData stub."""
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
