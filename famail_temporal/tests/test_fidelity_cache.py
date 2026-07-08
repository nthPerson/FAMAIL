"""Equivalence + benchmark tests for the per-trajectory constant-encoding cache.

The cache (MultiStreamSiameseDiscriminator.cache_constant_streams, wired into
TrajectoryModifier.modify_single) reuses the iteration-invariant stream
encodings (trip_s1, driving pair, profile pair) across one trajectory's
ST-iFGSM loop, recomputing only the modified seeking branch (trip_s2).

Correctness bar (mission): BITWISE-identical objective values, per-term values,
and leaf gradients vs the uncached path — proven here directly at the
discriminator level (F_fidelity + grad) and end-to-end through modify_single
(objective + every term + gradient-norm + the whole cumulative-delta trajectory)
for varied trajectory lengths across BOTH trim and lift modes (the G1 test).
"""
import contextlib

import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.modifier import TrajectoryModifier
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.fidelity.compute import compute_ffidelity
from famail_temporal.fidelity.context import (
    MultiStreamContextBuilder, MultiStreamData,
)
from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.tests.test_modifier import _interior_active_cell

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_discriminator():
    """Real MultiStreamSiameseDiscriminator, inference mode, frozen params.

    Random-init weights are fine: bitwise transparency of the cache is a
    property of the forward/backward graph, independent of the weight values.
    """
    torch.manual_seed(0)
    d = MultiStreamSiameseDiscriminator()
    d.train(False)
    for p in d.parameters():
        p.requires_grad = False
    return d.to(DEVICE)


def _ms_builder(n_drivers=3, n_trajs=8, seq_len=20):
    driving, seeking, profile, sdays, ddays = {}, {}, {}, {}, {}
    rng = np.random.RandomState(0)
    for d in range(n_drivers):
        driving[d] = [
            [[float((i % 40) + 1), float((i % 40) + 2), i, 1] for i in range(seq_len)]
            for _ in range(n_trajs)
        ]
        seeking[d] = [
            [[float((i % 40) + 1), float((i % 40) + 3), i, 1] for i in range(seq_len)]
            for _ in range(n_trajs)
        ]
        profile[d] = rng.randn(11).astype(np.float32)
        sdays[d] = [1] * n_trajs
        ddays[d] = [1] * n_trajs
    ms = MultiStreamData(
        driving_trajs=driving, seeking_trajs=seeking, profile_features=profile,
        seeking_days=sdays, driving_days=ddays,
    )
    return MultiStreamContextBuilder(ms, device=str(DEVICE), seed=42)


def _stay_traj(x, y, tb, n, driver_id=0):
    return Trajectory(
        trajectory_id=0, driver_id=driver_id,
        states=[TrajectoryState(x_grid=float(x), y_grid=float(y),
                                time_bucket=tb, day_index=1) for _ in range(n)],
    )


# ───────────────────────── direct discriminator-level ────────────────────────

def test_discriminator_cache_bitwise_forward_and_grad():
    """Within one cache context, reusing the constant streams across 5 forwards
    (each with a different x2 slot 0) yields F_fidelity AND grad w.r.t. x2 that
    are torch.equal to the uncached recompute."""
    disc = _build_discriminator()
    builder = _ms_builder()
    traj = _stay_traj(3, 4, 90, n=12, driver_id=0)
    kw = builder.build_fidelity_kwargs(traj, traj)
    kw = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in kw.items()}
    base_x2 = kw["x2"].detach()

    def perturbed(it):
        x2 = base_x2.clone()
        # move slot 0's last real row (mimics the modifier's per-iter splice)
        x2[0, 0, -1, 0] = x2[0, 0, -1, 0] + 0.1 * (it + 1)
        x2[0, 0, -1, 1] = x2[0, 0, -1, 1] - 0.05 * (it + 1)
        x2.requires_grad_(True)
        return x2

    def forward(x2):
        kwargs = {**kw, "x2": x2}
        f, _ = compute_ffidelity(disc, None, None, kwargs)
        (g,) = torch.autograd.grad(f, x2)
        return f.detach(), g.detach()

    # Cached: one context, five reuse-forwards.
    cached = []
    with disc.cache_constant_streams():
        for it in range(5):
            cached.append(forward(perturbed(it)))
    # Uncached reference.
    uncached = [forward(perturbed(it)) for it in range(5)]

    for it, ((fc, gc), (fu, gu)) in enumerate(zip(cached, uncached)):
        assert torch.equal(fc, fu), f"F_fidelity mismatch @ iter {it}: {fc} vs {fu}"
        assert torch.equal(gc, gu), f"grad mismatch @ iter {it}"


def test_cache_context_restores_state():
    """The context manager leaves no cache active after exit (and nests safely)."""
    disc = _build_discriminator()
    assert disc._fidelity_cache is None
    with disc.cache_constant_streams():
        assert disc._fidelity_cache == {} or isinstance(disc._fidelity_cache, dict)
        with disc.cache_constant_streams():
            assert isinstance(disc._fidelity_cache, dict)
        assert isinstance(disc._fidelity_cache, dict)
    assert disc._fidelity_cache is None
    # Exception-safe.
    with pytest.raises(RuntimeError):
        with disc.cache_constant_streams():
            raise RuntimeError("boom")
    assert disc._fidelity_cache is None


# ───────────────────────── end-to-end (G1) equivalence ───────────────────────

def _run_modify(mode, n_states, disc, cache_on):
    """Fresh bundle+objective+builder+modifier, run modify_single, return the
    per-iteration ModificationResult list. Rebuilding everything (and the
    seeded builder) makes the cache-on and cache-off runs bit-comparable."""
    import dataclasses
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=7)
    bundle = dataclasses.replace(bundle, discriminator=disc)
    obj = FAMAILObjective(
        bundle, alpha_spatial=0.2, alpha_causal=0.7, alpha_fidelity=0.1,
    ).to(DEVICE)
    builder = _ms_builder()
    m = TrajectoryModifier(
        objective=obj, bundle=bundle, multi_stream_builder=builder,
        max_iterations=5, alpha=1.0, diagnostics_enabled=False, patience=None,
        device=DEVICE,
    )
    if not cache_on:
        # Force the uncached path: nullcontext instead of the real cache CM.
        m.objective.discriminator.cache_constant_streams = (
            lambda: contextlib.nullcontext()
        )
    x, y, tb = _interior_active_cell(bundle, lo=3, hi=4)
    traj = _stay_traj(x, y, tb, n=n_states, driver_id=0)
    return m.modify_single(traj, mode=mode).iterations


@pytest.mark.parametrize("mode", ["trim", "lift"])
@pytest.mark.parametrize("n_states", [6, 15, 50])
def test_modify_single_cache_bitwise_end_to_end(mode, n_states):
    """G1: cached vs uncached modify_single are bitwise-identical, per iteration,
    for objective, every term, gradient norm, and the cumulative-delta path."""
    disc = _build_discriminator()
    on = _run_modify(mode, n_states, disc, cache_on=True)
    off = _run_modify(mode, n_states, disc, cache_on=False)

    assert len(on) == len(off) and len(on) > 0
    for it, (a, b) in enumerate(zip(on, off)):
        assert a.objective_value == b.objective_value, f"total @ {it}"
        assert a.f_spatial == b.f_spatial, f"f_spatial @ {it}"
        assert a.f_causal == b.f_causal, f"f_causal @ {it}"
        assert a.f_fidelity == b.f_fidelity, f"f_fidelity @ {it}"
        assert a.gradient_norm == b.gradient_norm, f"grad_norm @ {it}"
        assert np.array_equal(a.cumulative_delta, b.cumulative_delta), \
            f"cumulative_delta @ {it}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="benchmark needs CUDA")
@pytest.mark.parametrize("seq_len", [20, 50])
def test_cache_speedup_benchmark(capsys, seq_len):
    """Micro-benchmark of the per-iteration fidelity forward+backward — the
    ~70-75% per-iter component the cache targets, exercised exactly as
    modify_single does (only x2 slot-0's last row depends on the (2,) leaf;
    context rows detached; cudnn disabled). The cache eliminates the CONSTANT
    stream forward encodes (~15 of ~20 LSTM rows) while the modified branch and
    its backward stay live. Tolerant of the ~35% background GPU load.

    NB: this isolates the optimized component; end-to-end per-iter speedup is
    this scaled by the fidelity share of the iteration."""
    disc = _build_discriminator()
    builder = _ms_builder()
    traj = _stay_traj(3, 4, 90, n=seq_len, driver_id=0)
    kw = builder.build_fidelity_kwargs(traj, traj)
    kw = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in kw.items()}
    x2_ctx = kw["x2"].detach().clone()

    def fidelity_iter(delta):
        x2 = x2_ctx.clone()
        x2[0, 0, -1, 0] = x2_ctx[0, 0, -1, 0] + delta[0]
        x2[0, 0, -1, 1] = x2_ctx[0, 0, -1, 1] + delta[1]
        f, _ = compute_ffidelity(disc, None, None, {**kw, "x2": x2})
        (g,) = torch.autograd.grad(f, delta)
        return g

    def bench(cache_on, reps=30, rounds=5):
        # CUDA-event timing (device time; robust to WSL2 host-clock jitter and
        # the background run's host-side scheduling stalls). Report the median
        # round to further reject contention spikes.
        cm = disc.cache_constant_streams() if cache_on else contextlib.nullcontext()
        times = []
        with cm:
            fidelity_iter(torch.zeros(2, device=DEVICE, requires_grad=True))  # warmup
            for r in range(rounds):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for it in range(reps):
                    d = torch.zeros(2, device=DEVICE, requires_grad=True)
                    d.data[0] = 0.01 * it
                    fidelity_iter(d)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / reps)
        return sorted(times)[len(times) // 2]

    t_off = bench(cache_on=False)
    t_on = bench(cache_on=True)
    speedup = t_off / t_on
    with capsys.disabled():
        print(f"\n[cache benchmark seq_len={seq_len}] fidelity fwd+bwd: "
              f"uncached={t_off:.1f}ms  cached={t_on:.1f}ms  speedup={speedup:.2f}x")
    assert speedup > 1.3, f"expected a speedup, got {speedup:.2f}x"
