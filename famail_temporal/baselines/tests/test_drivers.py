import pytest

from famail_temporal.baselines.gan.drivers import (
    build_driver_index, invert_driver_index, group_by_driver, driver_idxs_for,
)


class _Stub:
    def __init__(self, driver_id):
        self.driver_id = driver_id


def test_build_driver_index_is_sorted_and_contiguous():
    trajs = [_Stub(7), _Stub(2), _Stub(7), _Stub(5)]
    m = build_driver_index(trajs)
    assert m == {2: 0, 5: 1, 7: 2}          # sorted driver_id -> contiguous idx


def test_invert_driver_index():
    m = {2: 0, 5: 1, 7: 2}
    assert invert_driver_index(m) == {0: 2, 1: 5, 2: 7}


def test_group_by_driver_counts():
    trajs = [_Stub(7), _Stub(2), _Stub(7), _Stub(5)]
    g = group_by_driver(trajs)
    assert set(g) == {2, 5, 7}
    assert len(g[7]) == 2 and len(g[2]) == 1 and len(g[5]) == 1


def test_driver_idxs_for_aligned():
    # Map built from the full 3-driver corpus, then applied to a 2-driver
    # subset (verifies an externally-built map is applied index-aligned).
    full = [_Stub(7), _Stub(2), _Stub(7), _Stub(5)]
    m = build_driver_index(full)            # {2: 0, 5: 1, 7: 2}
    trajs = [_Stub(7), _Stub(2), _Stub(7)]
    assert driver_idxs_for(trajs, m) == [2, 0, 2]


def test_driver_idxs_for_unknown_raises():
    m = {2: 0}
    with pytest.raises(KeyError):
        driver_idxs_for([_Stub(99)], m)
