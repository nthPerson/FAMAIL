"""Mathematical invariants across the fairness math stack.

These guard properties that a reviewer could verify by hand from the
equations in the Methods section. Failure of any of these tests means
the paper's mathematical claims no longer hold.

Sign convention (1/N-shifted decomposition, see
``docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``):

    Σᵢ per_cell_fairness_attribution_*ᵢ == F  (not 1 - F)
    αᵢ > 0 → cell contributes more than 1/N baseline to fairness
    αᵢ < 0 → cell drags fairness below baseline (priority for modification)
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from famail_temporal.fairness import (
    compute_fcausal_torch,
    per_cell_fairness_attribution_causal,
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
    """Σᵢ per_cell_fairness_attribution_causalᵢ == F_causal across many random R.

    This is the load-bearing decomposition identity under the 1/N-shifted
    formulation: αᵢ = 1/N − ((MR)ᵢ² − ((I−H)R)ᵢ²) / R'MR sums to F_causal
    because (a) the 1/N terms sum to 1 and (b) the explained-variance ratio
    sums to 1 − F_causal.
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
        attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
        diff = abs(float(attr.sum()) - float(f))
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
    pickup_unequal[0] = 10.0
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
    """Sum invariant must hold at realistic production scale (N=1000).

    Float32 matmul accumulation at large N introduces precision loss; this
    test verifies the 1/N-shifted decomposition still sums to F_causal
    within an acceptable tolerance at the scale the pipeline will actually use.
    """
    rng = np.random.RandomState(110)
    N = 1000
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    R = torch.from_numpy(rng.randn(N) * 2.0).float()
    f = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])
    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
    diff = abs(float(attr.sum()) - float(f))
    assert diff < 1e-3, f"Attribution invariant broken at N=1000: diff={diff:.2e}"


def test_per_cell_attribution_can_be_negative():
    """Individual αᵢ_causal CAN be negative — the sign is informative.

    Under the 1/N-shifted decomposition, αᵢ < 0 means demographics explain
    MORE than the 1/N baseline of the cell's residual variance — i.e. the
    cell drags F_causal below the uniform-fairness baseline. This is a
    priority signal for the trajectory-modification algorithm.

    The full sum stays bounded in [0, 1] (= F_causal) because the
    1/N terms sum to 1 and the explained-variance terms sum to (1 - F_causal).
    """
    rng = np.random.RandomState(115)
    N = 50
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)

    for trial_seed in range(5):
        trial_rng = np.random.RandomState(trial_seed + 500)
        R = torch.from_numpy(trial_rng.randn(N)).float()
        attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
        f_causal = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])

        attr_sum = float(attr.sum())
        assert 0.0 <= attr_sum <= 1.0 + 1e-5, (
            f"At trial {trial_seed}: attribution sum {attr_sum} out of [0,1]"
        )
        assert abs(attr_sum - float(f_causal)) < 1e-5, (
            f"At trial {trial_seed}: sum {attr_sum} != F_causal {float(f_causal)}"
        )

    # At least one trial should produce an αᵢ < 0 (drag cell). With random R,
    # some cells inevitably have higher explained-variance contribution than
    # the 1/N baseline.
    seen_negative = False
    for trial_seed in range(20):
        trial_rng = np.random.RandomState(trial_seed + 600)
        R = torch.from_numpy(trial_rng.randn(N)).float()
        attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
        if (attr < 0).any():
            seen_negative = True
            break
    assert seen_negative, (
        "No trial produced a negative per-cell attribution — the property "
        "that αᵢ can flag drag cells must be demonstrable in practice."
    )


def test_fcausal_handles_zero_residual_degenerate_case():
    """F_causal for R = 0 must be finite and = 1.0 (degenerate branch).

    When R = 0, there is no residual variance to explain. The `ss_tot < eps`
    guard in compute_fcausal_torch returns 1.0 (perfectly fair by convention).
    Without this guard, 0/0 would produce NaN and silently poison downstream
    computations. This test locks the convention.
    """
    rng = np.random.RandomState(116)
    N = 50
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    R_zero = torch.zeros(N)
    f = compute_fcausal_torch(R_zero, tensors['I_minus_H_demo'], tensors['M'])
    assert torch.isfinite(f), f"F_causal(0) not finite: {float(f)}"
    assert float(f) == 1.0, f"F_causal(0) != 1.0: {float(f)}"


def test_per_cell_attribution_handles_zero_residual():
    """per_cell_fairness_attribution_causal for R = 0 must be finite (≈ 1/N each).

    Complementary to the F_causal degenerate test: when R = 0, every cell
    sits at the 1/N baseline because the explained-variance term is 0.
    Sum ≈ 1.0 = F_causal in the degenerate case.
    """
    rng = np.random.RandomState(117)
    N = 50
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    tensors = hat_matrices_to_torch(hat)
    R_zero = torch.zeros(N)
    attr = per_cell_fairness_attribution_causal(R_zero, tensors['X_demo'], tensors['XtX_inv'])
    assert torch.isfinite(attr).all(), "attribution contains non-finite values"
    # Each cell sits at 1/N baseline; sum ≈ 1.0 (degenerate F_causal = 1.0).
    assert abs(float(attr.sum()) - 1.0) < 1e-4
    assert torch.allclose(attr, torch.full_like(attr, 1.0 / N), atol=1e-5)


def test_precompute_hat_matrices_rejects_small_N():
    """precompute_hat_matrices must reject N < max(10, p+1).

    The pooled F_causal framework requires at least enough active units to
    uniquely determine the linear projection onto demographics. Concretely:
    - N must exceed the number of columns in the design matrix (p + 1 for
      intercept + demographics).
    - The enforced minimum is max(10, p+1) to ensure meaningful statistics.

    This invariant is load-bearing: if it were weakened, rank-deficient hat
    matrices could be constructed silently, leading to division-by-zero in
    downstream F_causal.
    """
    import pytest
    rng = np.random.RandomState(118)
    with pytest.raises(ValueError):
        precompute_hat_matrices(np.ones(5), rng.randn(5, 3), ["a", "b", "c"])


def test_spatial_gini_decomposition_sums_to_gini():
    """sum(per_unit_gini_decomposition(x)) == pairwise_gini(x) for random x."""
    from famail_temporal.fairness.spatial import (
        per_unit_gini_decomposition, pairwise_gini,
    )
    torch.manual_seed(17)
    for _ in range(5):
        n = int(torch.randint(2, 100, (1,)).item())
        values = torch.rand(n) * 10.0 + 0.01
        assert torch.isclose(
            per_unit_gini_decomposition(values).sum(),
            pairwise_gini(values),
            atol=1e-6,
        )
