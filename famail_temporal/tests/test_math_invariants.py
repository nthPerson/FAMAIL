"""Mathematical invariants across the fairness math stack.

These guard properties that a reviewer could verify by hand from the
equations in the Methods section. Failure of any of these tests means
the paper's mathematical claims no longer hold.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from famail_temporal.fairness import (
    compute_fcausal_torch,
    per_unit_attribution,
    precompute_hat_matrices,
    pairwise_gini,
    compute_fspatial,
    hat_matrices_to_torch,
)


def test_I_minus_H_idempotent():
    """(I - H_demo)^2 == (I - H_demo) — residual-maker is a projector.

    This is the mathematical property that justifies treating (I - H_demo)R as
    the component of R that is orthogonal to the demographic subspace.
    """
    rng = np.random.RandomState(100)
    N = 60
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = hat['I_minus_H_demo']
    np.testing.assert_allclose(IH @ IH, IH, atol=1e-9)


def test_M_idempotent_and_centers_ones():
    """M^2 == M and M @ 1 == 0 — centering matrix is a projector onto the
    mean-zero subspace."""
    rng = np.random.RandomState(101)
    N = 60
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    M = hat['M']
    np.testing.assert_allclose(M @ M, M, atol=1e-9)
    np.testing.assert_allclose(M @ np.ones(N), np.zeros(N), atol=1e-9)


def test_attribution_sum_property_multiple_seeds():
    """Sum_i per_unit_attribution_i == 1 - F_causal across many random R.

    This is the load-bearing decomposition identity: the per-unit attribution
    vector sums EXACTLY to the pooled r^2_demo = 1 - F_causal. It holds because
    both (I - H) and M are idempotent (see tests above). The paper's attribution
    heatmaps and trajectory-ranking pipeline depend on this property.
    """
    rng = np.random.RandomState(102)
    N = 100
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    IH = tensors['I_minus_H_demo']
    M = tensors['M']
    for seed in range(5):
        R = torch.from_numpy(np.random.RandomState(seed + 200).randn(N) * 3.0).float()
        f = compute_fcausal_torch(R, IH, M)
        attr = per_unit_attribution(R, IH, M)
        diff = abs(float(attr.sum()) - (1.0 - float(f)))
        assert diff < 1e-5, (
            f"Attribution sum invariant broken at seed={seed}: "
            f"diff={diff:.2e}"
        )


def test_fcausal_zero_when_R_in_demographic_span():
    """R in span([1, standardized(demographics)]) => F_causal == 0.

    If R is perfectly explained by demographics, demographics explain 100% of
    R's variance, so the "unexplained fraction" F_causal is zero. This is the
    lower extreme of the fairness spectrum: maximum demographic-driven disparity.
    """
    rng = np.random.RandomState(103)
    N = 70
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    X_scaled = StandardScaler().fit_transform(demo)
    # R = intercept + weighted combination of scaled demographics -> lies in span([1, X_scaled])
    R = torch.from_numpy(1.0 + 0.5 * X_scaled[:, 0] + 0.2 * X_scaled[:, 1]).float()
    f = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])
    assert float(f) < 1e-4


def test_fcausal_one_when_R_orthogonal_to_demographic_span():
    """R orthogonal to span([1, X_demo]) => F_causal == 1.

    If R is entirely in the complement of the demographic subspace, demographics
    explain 0% of its variance. This is the upper extreme: the residuals
    carry no demographically-correlated signal.
    """
    rng = np.random.RandomState(104)
    N = 70
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    # Project a random vector through (I - H) — result is orthogonal to span([1, X_demo])
    v = rng.randn(N)
    R_np = hat['I_minus_H_demo'] @ v
    R = torch.from_numpy(R_np).float()
    f = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])
    assert float(f) > 1.0 - 1e-4


def test_fcausal_in_unit_interval_over_random_R():
    """For any finite R, F_causal is in [0, 1] — it's a normalized ratio.

    This is a basic sanity property: F_causal is always interpretable as a
    "fraction of residual variance NOT explained by demographics."
    """
    rng = np.random.RandomState(105)
    N = 80
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    for seed in range(10):
        R = torch.from_numpy(np.random.RandomState(seed + 300).randn(N) * 5.0).float()
        f = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])
        assert 0.0 <= float(f) <= 1.0, (
            f"F_causal out of [0,1] at seed={seed}: f={float(f)}"
        )


def test_fcausal_scale_invariant():
    """F_causal(c*R) == F_causal(R) for c != 0.

    F_causal is a ratio of quadratic forms, so it is invariant under
    scaling of R. This means the measure is dimensionless and insensitive
    to the units in which Y and g_0(D) are expressed.
    """
    rng = np.random.RandomState(106)
    N = 60
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    R = torch.from_numpy(rng.randn(N) * 2.0).float()
    f1 = float(compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M']))
    f2 = float(compute_fcausal_torch(R * 7.3, tensors['I_minus_H_demo'], tensors['M']))
    f3 = float(compute_fcausal_torch(-R, tensors['I_minus_H_demo'], tensors['M']))
    assert abs(f1 - f2) < 1e-4
    assert abs(f1 - f3) < 1e-4


def test_gini_scale_invariance():
    """Gini(c*x) == Gini(x) for c > 0.

    Gini is a dimensionless measure of inequality — unchanged by multiplicative
    rescaling. This justifies interpreting F_spatial as fairness "on its own
    scale" regardless of the units of DSR and ASR.
    """
    rng = np.random.RandomState(107)
    x = torch.from_numpy(rng.rand(50) * 10.0).float()
    g1 = float(pairwise_gini(x))
    for c in [0.1, 1.0, 7.3, 100.0]:
        g2 = float(pairwise_gini(x * c))
        assert abs(g1 - g2) < 1e-4, f"Gini scale invariance broken at c={c}: {g1} vs {g2}"


def test_gini_zero_when_all_equal():
    """Gini(constant vector) == 0 — perfectly equal distributions have zero
    inequality.
    """
    for n in [10, 50, 100]:
        values = torch.full((n,), 3.0)
        g = float(pairwise_gini(values))
        assert g < 1e-6, f"Gini non-zero at n={n}: {g}"


def test_gini_bounded_above_by_n_minus_1_over_n():
    """Pairwise Gini on an n-vector is bounded above by (n-1)/n.

    Achieved when all mass concentrates at one unit. For n=10 this is 0.9.
    """
    for n in [10, 20, 100]:
        x = torch.zeros(n)
        x[0] = 100.0
        g = float(pairwise_gini(x))
        expected = (n - 1) / n
        assert abs(g - expected) < 1e-4, (
            f"At n={n}: expected (n-1)/n={expected}, got {g}"
        )


def test_fspatial_one_when_service_equal():
    """F_spatial == 1 when DSR and ASR are constant across units.

    If every unit has the same DSR and ASR (perfect spatial equality), F_spatial
    achieves its maximum value of 1. This is the "perfect fairness" baseline.
    """
    for N in [10, 50, 200]:
        pickup = torch.full((N,), 3.0)
        dropoff = torch.full((N,), 3.0)
        active = torch.full((N,), 5.0)
        f, _ = compute_fspatial(pickup, dropoff, active)
        assert float(f) > 0.999, f"F_spatial not ~1 at N={N}: {float(f)}"


def test_fspatial_bounded_in_unit_interval():
    """F_spatial is in [0, 1] for any non-negative inputs."""
    rng = np.random.RandomState(108)
    for seed in range(5):
        r = np.random.RandomState(seed + 400)
        N = 80
        pickup = torch.from_numpy(r.rand(N) * 5.0).float()
        dropoff = torch.from_numpy(r.rand(N) * 5.0).float()
        active = torch.from_numpy(r.rand(N) * 3.0 + 1.0).float()
        f, _ = compute_fspatial(pickup, dropoff, active)
        assert 0.0 <= float(f) <= 1.0, f"F_spatial out of [0,1] at seed={seed}: {float(f)}"


def test_fcausal_and_fspatial_both_increase_toward_fairness():
    """Both metrics are defined so that HIGHER == MORE FAIR.

    This is a semantic invariant: both F_causal and F_spatial are in [0, 1]
    with 1 = perfectly fair, 0 = maximally unfair. This test confirms the
    sign conventions match — a shift toward equal service increases F_spatial,
    and a shift of R out of the demographic span increases F_causal.
    """
    rng = np.random.RandomState(109)
    N = 60
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)

    # F_spatial: unequal -> equal should INCREASE F_spatial
    pickup_unequal = torch.ones(N)
    pickup_unequal[0] = 10.0  # one unit dominates
    dropoff = torch.ones(N)
    active = torch.ones(N) * 2.0
    f_unequal, _ = compute_fspatial(pickup_unequal, dropoff, active)
    pickup_equal = torch.ones(N)
    f_equal, _ = compute_fspatial(pickup_equal, dropoff, active)
    assert float(f_equal) > float(f_unequal), (
        f"F_spatial did not increase with equality: unequal={float(f_unequal)}, equal={float(f_equal)}"
    )

    # F_causal: R-in-span -> R-orthogonal should INCREASE F_causal
    demo = rng.randn(N, 3)
    X_scaled = StandardScaler().fit_transform(demo)
    R_in_span = torch.from_numpy(1.0 + 0.5 * X_scaled[:, 0]).float()
    R_orth_np = hat['I_minus_H_demo'] @ rng.randn(N)
    R_orth = torch.from_numpy(R_orth_np).float()
    f_in_span = float(compute_fcausal_torch(R_in_span, tensors['I_minus_H_demo'], tensors['M']))
    f_orth = float(compute_fcausal_torch(R_orth, tensors['I_minus_H_demo'], tensors['M']))
    assert f_orth > f_in_span, (
        f"F_causal did not increase with orthogonality: in_span={f_in_span}, orth={f_orth}"
    )


def test_attribution_sum_holds_at_production_scale():
    """Attribution invariant must hold at realistic production scale (N=1000).

    Float32 matmul accumulation at large N introduces precision loss; this
    test verifies that the invariant still holds within an acceptable tolerance
    at the scale the pipeline will actually use.
    """
    rng = np.random.RandomState(110)
    N = 1000
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    R = torch.from_numpy(rng.randn(N) * 2.0).float()
    f = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])
    attr = per_unit_attribution(R, tensors['I_minus_H_demo'], tensors['M'])
    # Loosen tolerance to account for float32 fp accumulation at this N
    diff = abs(float(attr.sum()) - (1.0 - float(f)))
    assert diff < 1e-3, f"Attribution invariant broken at N=1000: diff={diff:.2e}"
