"""Tests for data.aggregation."""
import pytest

from famail_temporal.data.aggregation import hour_to_block_index


@pytest.mark.parametrize("hour,expected", [
    (7, 0), (9, 0), (10, 1), (15, 1),
    (16, 2), (19, 2), (20, 3), (23, 3),
    (0, 3), (6, 3),
])
def test_hour_to_block_index(hour, expected):
    assert hour_to_block_index(hour) == expected


def test_invalid_hour_raises():
    with pytest.raises(ValueError):
        hour_to_block_index(24)
