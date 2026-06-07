# Straight-Through (Hard-Metric) Editing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in straight-through estimator so the trajectory editor optimizes/selects/gates on the **realizable hard-grid** F-metrics (forward = hard, gradient = soft), then run one experiment to determine whether the +0.0128 editing ceiling is an optimization artifact or intrinsic.

**Architecture:** Entirely inside `modify_single` ([modifier.py](famail_temporal/algorithm/modifier.py)): when `use_ste` is on, build the hard grid (full pickup mass at `int(current_pickup)` — the exact cell the persist step writes) and stitch `hard + (soft − soft.detach())` so the objective's *value* is the hard metric while gradients still flow through the soft assignment. The existing best-iterate tracking and acceptance gate then become hard-based automatically (they read the objective's value). The multi-loop engine and CLI already exist; we just thread one boolean. Default off ⇒ bit-identical to today.

**Tech Stack:** Python, NumPy, PyTorch, pytest. Spec: `docs/superpowers/specs/2026-06-06-famail-ste-hard-metric-editing-design.md`. Branch: `algorithm-improvements`.

**Key facts for the engineer:**
- This is a **gated** algorithm change, explicitly directed by the user. Default-off must be bit-identical (the regression suite is the guard).
- `int(current_pickup)` (NOT `argmax(probs)`) is the snap cell — the soft assignment measures distance to cell *centers* (`int+0.5`) while the pickup sits at the integer corner, so `argmax` can tie-break wrong at integer coords (e.g. iter-0). `int(current_pickup)` matches the persist's `int(modified.pickup_state.x_grid)`.
- The STE stitch: `hard_probs` is a one-hot (no grad) → `hard_3d` is constant; `soft_3d - soft_3d.detach()` is numerically zero but carries the soft gradient. So `objective_grid` forwards as `hard_3d` and back-props through `soft_3d`.
- Test bundle builder: `from famail_temporal.tests.test_objective import _make_synthetic_bundle`. Synthetic bundles carry an `nn.Identity` discriminator → build the objective with `FAMAILObjective(bundle, alpha_fidelity=0.0)`.
- `nn.Module` dispatches `obj(...)` → `self.forward`, so test stubs override `obj.forward` (and use `diagnostics_enabled=False` to take the single-backward path).
- Security hook blocks the literal `eval` + `(` token — not needed here.
- Commits: stage named files only (never `git add -A`); end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

---

## Task 1: Config flag `STE_ENABLED`

**Files:**
- Modify: `famail_temporal/config.py` (after the `ITERATIVE_TOPK_MAX_EDITS` line)
- Test: `famail_temporal/tests/test_config_multiloop.py` (append)

- [ ] **Step 1: Write the failing test** — append to `famail_temporal/tests/test_config_multiloop.py`:

```python
def test_ste_default_is_off():
    from famail_temporal import config
    assert config.STE_ENABLED is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_config_multiloop.py::test_ste_default_is_off -v`
Expected: FAIL — `AttributeError: module 'famail_temporal.config' has no attribute 'STE_ENABLED'`

- [ ] **Step 3: Add the constant** — in `famail_temporal/config.py`, immediately after the `ITERATIVE_TOPK_MAX_EDITS: int = 1` line, add:

```python
# Straight-through (hard-metric) editing (spec 2026-06-06). When True, modify_single
# evaluates/selects/gates on the realizable HARD grid (forward = hard, gradient =
# soft) instead of the soft relaxation, closing the soft-vs-hard gap of §8.7.
# Default False = historical soft behavior (bit-identical).
STE_ENABLED: bool = False
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest famail_temporal/tests/test_config_multiloop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/config.py famail_temporal/tests/test_config_multiloop.py
git commit -m "feat(config): add STE_ENABLED knob (straight-through hard-metric editing)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Straight-through estimator in `modify_single`

**Files:**
- Modify: `famail_temporal/algorithm/modifier.py` (`__init__` signature + body; the ST-iFGSM loop ~line 403-432)
- Test: `famail_temporal/tests/test_modifier.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `famail_temporal/tests/test_modifier.py`:

```python
def test_use_ste_default_false():
    bundle = _make_synthetic_bundle()
    m = TrajectoryModifier(
        objective=FAMAILObjective(bundle, alpha_fidelity=0.0), bundle=bundle)
    assert m.use_ste is False


def test_ste_runs_and_gradient_flows():
    """With STE on, modify_single runs end-to-end and the soft gradient still
    flows (some iteration has a nonzero gradient norm)."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=5,
                           use_ste=True)
    x, y, tb = _active_cell_and_bucket(bundle)
    h = m.modify_single(_make_test_trajectory(pickup_xy=(x, y), time_bucket=tb))
    assert isinstance(h, ModificationHistory)
    assert any(it.gradient_norm > 0 for it in h.iterations)


def test_ste_feeds_concentrated_hard_grid():
    """STE hands the objective a grid with the pickup mass concentrated in ONE
    cell (hard); the soft path spreads it over the neighborhood — so the two
    grids differ in more than a single cell of the trajectory's t_block slice."""
    import torch as _t
    bundle = _make_synthetic_bundle()
    x, y, tb = _active_cell_and_bucket(bundle)
    t_block = bundle.unit_map.to_time_block(0)

    def captured_grid(use_ste):
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        grids = []

        def rec(soft_pickup_3d=None, **kw):
            grids.append(soft_pickup_3d.detach().clone())
            return (_t.tensor(1.0, requires_grad=True),
                    {"f_spatial": _t.tensor(0.0), "f_causal": _t.tensor(0.0),
                     "f_fidelity": _t.tensor(0.0)})

        obj.forward = rec  # type: ignore[method-assign]
        m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=1,
                               patience=None, diagnostics_enabled=False,
                               use_ste=use_ste)
        m.modify_single(_make_test_trajectory(pickup_xy=(x, y), time_bucket=tb))
        return grids[0]

    soft_grid = captured_grid(False)
    ste_grid = captured_grid(True)
    n_diff = int((_t.abs(soft_grid[:, :, t_block] - ste_grid[:, :, t_block])
                  > 1e-9).sum())
    assert n_diff > 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest famail_temporal/tests/test_modifier.py::test_use_ste_default_false famail_temporal/tests/test_modifier.py::test_ste_feeds_concentrated_hard_grid -v`
Expected: FAIL — `TrajectoryModifier.__init__() got an unexpected keyword argument 'use_ste'` (and `AttributeError: ... 'use_ste'`).

- [ ] **Step 3a: Add `use_ste` to `__init__`** — add the parameter to the `TrajectoryModifier.__init__` signature, immediately after `epsilon_cap: float | None = None,`:

```python
        use_ste: bool | None = None,
```

And in the `__init__` body, immediately after the `self.epsilon_cap = (...)` assignment block, add:

```python
        # Straight-through (hard-metric) editing toggle (see config.STE_ENABLED).
        self.use_ste = config.STE_ENABLED if use_ste is None else use_ste
```

- [ ] **Step 3b: Insert the STE stitch in the loop** — in `modify_single`, the loop currently computes `soft_3d` and later calls the objective on it. Right AFTER this block (the `soft_3d = inject_soft_counts_into_3d(...)` call, ~line 403-406):

```python
            soft_3d = inject_soft_counts_into_3d(
                base_3d, probs, (orig_cx, orig_cy), t_block,
                k=self.soft_assign.k, pickup_mass=pickup_mass,
            )
```

insert:

```python
            # (c2) Straight-through hard-metric grid (opt-in). Forward value =
            # the HARD (realizable) grid: full pickup mass at int(current_pickup),
            # the exact cell the persist step writes. Gradient flows via the soft
            # assignment (soft_3d - soft_3d.detach()). This makes best-iterate +
            # the acceptance gate select on the metric actually deployed (§8.8).
            # int(current_pickup), NOT argmax(probs): soft uses cell centers, so
            # argmax can tie-break wrong at integer coords.
            if self.use_ste:
                k_half = self.soft_assign.k
                snap_x, snap_y = int(current_pickup[0]), int(current_pickup[1])
                ox, oy = snap_x - orig_cx + k_half, snap_y - orig_cy + k_half
                hard_probs = torch.zeros_like(probs)
                if 0 <= ox < probs.shape[0] and 0 <= oy < probs.shape[1]:
                    hard_probs[ox, oy] = 1.0
                hard_3d = inject_soft_counts_into_3d(
                    base_3d, hard_probs, (orig_cx, orig_cy), t_block,
                    k=k_half, pickup_mass=pickup_mass,
                )
                objective_grid = hard_3d + (soft_3d - soft_3d.detach())
            else:
                objective_grid = soft_3d
```

- [ ] **Step 3c: Feed the stitched grid to the objective** — change the `(d) Forward through FAMAILObjective` call so it uses `objective_grid` instead of `soft_3d`:

```python
            # (d) Forward through FAMAILObjective
            total, terms = self.objective(
                soft_pickup_3d=objective_grid,
                tau_features=tau_features,
                tau_prime_features=tau_prime_features,
                multi_stream_kwargs=ms_kwargs,
            )
```

(Everything downstream — best-iterate, the non-regression gate, the persist — is unchanged; with STE on, `total`/`terms` now carry hard-grid values automatically.)

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_modifier.py -v`
Expected: PASS — the 3 new tests pass AND every pre-existing modifier test still passes (`use_ste` defaults to False ⇒ `objective_grid = soft_3d`, the exact prior path).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/algorithm/modifier.py famail_temporal/tests/test_modifier.py
git commit -m "feat(modifier): straight-through hard-metric editing (opt-in via use_ste)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Runner wiring (`--ste` flag → run_experiment → modifier)

**Files:**
- Modify: `famail_temporal/evaluation/runner.py` (`run_experiment` signature + modifier construction; `_build_arg_parser`; `main`)
- Test: `famail_temporal/tests/test_runner.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `famail_temporal/tests/test_runner.py`:

```python
def test_cli_parses_ste_flag():
    from famail_temporal.evaluation.runner import _build_arg_parser
    assert _build_arg_parser().parse_args(["-k", "10", "--ste"]).ste is True
    assert _build_arg_parser().parse_args(["-k", "10"]).ste is False


def test_run_experiment_ste_runs(tiny_bundle):
    result = run_experiment(k=4, use_ste=True)
    assert len(result.rounds) == 1  # default single round
    assert len(result.modified_trajectory_ids) >= 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest famail_temporal/tests/test_runner.py::test_cli_parses_ste_flag famail_temporal/tests/test_runner.py::test_run_experiment_ste_runs -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'ste'` / `run_experiment() got an unexpected keyword argument 'use_ste'`.

- [ ] **Step 3a: Add the `run_experiment` parameter** — add to the `run_experiment(...)` signature, after `iterative_topk_max_edits: Optional[int] = None,`:

```python
    use_ste: Optional[bool] = None,
```

- [ ] **Step 3b: Pass it to the modifier** — in the `modifier = TrajectoryModifier(...)` construction, add the `use_ste` kwarg (alongside `accept_rule=accept_rule, epsilon_cap=epsilon_cap`):

```python
            accept_rule=accept_rule,
            epsilon_cap=epsilon_cap,
            use_ste=use_ste,
        )
```

- [ ] **Step 3c: Add the CLI flag** — in `_build_arg_parser`, after the `--iterative-topk-max-edits` argument, add:

```python
    p.add_argument("--ste", action="store_true",
                   help="Straight-through hard-metric editing: optimize/select/gate "
                        "on the realizable hard grid (forward=hard, grad=soft). "
                        "Off by default (config.STE_ENABLED).")
```

- [ ] **Step 3d: Thread it in `main`** — in the `run_experiment(...)` call inside `main`, add (use `None` when the flag is absent so a `--override STE_ENABLED=...` is still respected):

```python
        use_ste=(True if args.ste else None),
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_runner.py -v`
Expected: PASS — new tests pass AND all pre-existing runner tests still pass.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/evaluation/runner.py famail_temporal/tests/test_runner.py
git commit -m "feat(runner): --ste flag wiring for straight-through editing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Engine smoke test + full regression

**Files:**
- Test: `famail_temporal/tests/test_editing_loop.py` (append)

- [ ] **Step 1: Write the test** — append to `famail_temporal/tests/test_editing_loop.py`:

```python
def test_ste_multiloop_runs_and_is_valid():
    """STE multi-loop runs end-to-end through the engine and returns a valid
    result. (The quantitative STE-vs-soft comparison is the real-data E2
    experiment, not a synthetic unit test — synthetic data is too small to show
    the round-2+ degradation that STE fixes.)"""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, use_ste=True, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=8, mode="batch", max_rounds=5,
        round_convergence_tol=1e-5, round_patience=2)
    assert isinstance(result, EditingLoopResult)
    assert result.stop_reason in ("converged", "pool_exhausted", "max_rounds")
    assert len(result.rounds) >= 1
    assert all(r.f_causal == r.f_causal for r in result.rounds)  # no NaN
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest famail_temporal/tests/test_editing_loop.py -v`
Expected: PASS (`_make_modifier` forwards `use_ste` via `**kw` to `TrajectoryModifier`).

- [ ] **Step 3: Full regression suite**

Run: `python -m pytest famail_temporal/tests/ -q --ignore=famail_temporal/tests/test_runner_real_data.py`
Expected: ALL PASS (the `--ste` path is additive; defaults preserve behavior). Report exact counts.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/tests/test_editing_loop.py
git commit -m "test(editing-loop): STE multi-loop smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Experiment E2 — STE multi-loop (GPU; checkpoint with the user)

> NOT TDD. Surface the round curve to the user; do not patch the algorithm to "fix" an unexpected result. Reference points: baseline +0.0128 (`results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`); soft multi-loop A3 (`results/2026-06-06T18-10-53_A3_multiloop_C2_objective_afi0`, round 1 +0.01271 → final +0.01213, degraded).

- [ ] **Step 1: Run E2 (STE multi-loop) — identical to A3 except `--ste`**

```bash
python -m famail_temporal.evaluation.runner \
  --name E2_ste_multiloop_C2_objective_afi0 -k 10000 \
  --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap 2 --accept-rule objective --ste \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```

- [ ] **Step 2: Read the round curve + final deltas** from the new `results/*E2_ste_multiloop*/metrics.json` (the `rounds` block + `deltas`). Build the comparison:
  - **Round 1 (STE single-pass)** vs baseline +0.0128 and soft single-pass A3-r1 +0.01271 — does selecting on the hard metric change the single pass?
  - **Rounds 2+** vs A3's degrading curve — does STE accumulate, plateau (non-degrading), or (it shouldn't) degrade?

- [ ] **Step 3: Present to the user and pause.** The interpretation is the deliverable:
  - STE final **> +0.0128** ⇒ the ceiling was an optimization artifact; STE is a new best (→ STATUS/memory update + decide whether it ships).
  - STE final **≈ +0.0128, non-degrading** ⇒ the ceiling is **intrinsic** (confirms ε=5 / editable-slice evidence); the editing method's limit is fundamental.

---

## Task 6: Document §8.8 (after E2)

**Files:**
- Modify: `famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md` (append `### 8.8`)

- [ ] **Step 1: Write §8.8** — append a calibration entry covering: the STE mechanism (forward=hard, grad=soft; best-iterate/gate become hard-based for free); the E2 round curve; and the verdict — whether the +0.0128 ceiling is an **optimization artifact** (STE accumulates past it) or **intrinsic** (STE plateaus, non-degrading). Cross-reference §8.7. If STE beats +0.0128, also update `STATUS.md` + memory (new shipped config); if it plateaus, record that the ceiling is confirmed fundamental and the soft-vs-hard gap, while real, is not what bounds ΔF_causal.

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md
git commit -m "docs(methodology): §8.8 straight-through editing — ceiling disambiguation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (author)

- **Spec coverage:** §2 STE stitch → T2; §3 auto-hard selection/gate → T2 (no extra code, verified by the concentration + regression tests); §4 CLI/config → T1, T3; §5 backward-compat → default-off in T1/T2/T3 + full suite in T4; §6 experiment → T5; §7 testing → T2 (default-off, grad-flows, feeds-hard), T3 (wiring), T4 (smoke + regression); §8 docs → T6.
- **Testing honesty:** synthetic data can't robustly show STE's round-2+ advantage (it's a real-data-scale effect), so the unit tests verify the *mechanism* (STE feeds the hard grid, gradient flows, default-off bit-identical) and the *outcome* is the E2 experiment. This is called out in T4's docstring.
- **Naming consistency:** config `STE_ENABLED`; modifier param/attr `use_ste`; runner param `use_ste`; CLI `--ste`; local `objective_grid`, `hard_probs`, `hard_3d`, `k_half`, `snap_x/snap_y`, `ox/oy`. The snap cell is `int(current_pickup)` everywhere (never `argmax(probs)`).
