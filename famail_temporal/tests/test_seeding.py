"""Tests for utils.seeding."""
import numpy as np
import torch

from famail_temporal.utils.seeding import set_all_seeds


def test_numpy_reproducible():
    set_all_seeds(123)
    a = np.random.rand(5)
    set_all_seeds(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_torch_reproducible():
    set_all_seeds(123)
    a = torch.rand(5)
    set_all_seeds(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_python_random_reproducible():
    import random
    set_all_seeds(123)
    a = [random.random() for _ in range(5)]
    set_all_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b
