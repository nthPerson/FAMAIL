# FAMAIL-Temporal Implementation Plan — Phase 9

> **MODEL REQUIREMENT — OPUS ONLY:** Same as the main plan file.
>
> **Prerequisite:** Phases 1–8 complete. All tests passing.

**Scope:** Phase 9 — Documentation. Sub-directory READMEs + top-level README. The three READMEs in `raw_data/`, `cache/`, and `discriminator_checkpoints/` were already written in earlier tasks.

Target: each sub-README is ~60–120 lines, focused and scannable. The top-level README links to all six sub-READMEs plus the three data-directory READMEs.

---

## Phase 9: Documentation (Tasks 33–34)

### Task 33: Sub-directory READMEs

Six READMEs in a single task (they share a common template). Each follows the spec section 13 template: Purpose → Files → Key design choices → API surface → Dependencies → Paper-section hook.

**Files:**
- Create: famail_temporal/data/README.md
- Create: famail_temporal/fairness/README.md
- Create: famail_temporal/fidelity/README.md
- Create: famail_temporal/algorithm/README.md
- Create: famail_temporal/utils/README.md
- Create: famail_temporal/tests/README.md

- [ ] **Step 1: Write famail_temporal/data/README.md**

    # data/

    Ingest raw .pkl files and produce the canonical active-unit representation
    used by the entire algorithm.

    ## Files

    - `loader.py` — `DataBundle` dataclass + `DataBundle.load()` entry point
    - `aggregation.py` — raw data → (48, 90, T) tensors + hour-to-block mapping
    - `active_mask.py` — `UnitIndexMap` + `compute_active_mask()`
    - `cache_io.py` — read/write helpers for the `cache/` directory

    ## Key design choices

    **Canonical active-unit ordering.** The `UnitIndexMap` defines a single
    cell-major, block-within-cell ordering of active units. This ordering is
    built once at preprocess time, serialized, and asserted at every load. All
    downstream math operates on N-vectors indexed by this ordering.

    **Unified mean-hourly aggregation.** All three base tensors (`pickup_3d`,
    `dropoff_3d`, `active_taxis_3d`) use the same rule: mean hourly rate within
    block, mean across days. This lets `Y = S/D` have a clean
    "taxis-per-pickup-per-hour" interpretation and eliminates the dual-tensor
    runtime-rescale present in the legacy code.

    **Active at (cell, t) granularity.** A unit is active iff
    `active_taxis_3d[c, t] > 0.5` AND the cell is within Shenzhen AND all
    selected demographic features for that cell are finite. Unlike the legacy
    2D active filter, this captures cells that are busy at some times but
    dormant at others.

    **`DataBundle` is frozen.** Mutation raises `FrozenInstanceError`.

    ## API surface

        from famail_temporal.data import (
            DataBundle, UnitIndexMap, compute_active_mask,
            hour_to_block_index, time_bucket_to_hour, block_n_hours,
            aggregate_pickup_dropoff, aggregate_active_taxis,
        )

        bundle = DataBundle.load(max_trajectories=100)

    ## Dependencies

    `config`, `utils/`, `fairness.g0_power_basis`, `fairness.hat_matrices`
    (for preprocessing; runtime loads only pre-computed artifacts).

    ## Paper-section hook

    "Data Preparation" in Methods. The canonical active-unit ordering and the
    `(cell, t)` active filter are the two ideas that distinguish this work's
    data pipeline from the legacy 2D approach.

- [ ] **Step 2: Write famail_temporal/fairness/README.md**

    # fairness/

    Pooled fairness metrics over the N active `(cell, t)` units. Every function
    here takes N-vectors as input and returns either scalars or N-vectors.
    Nothing in this directory knows about grid dimensions or time blocks.

    ## Files

    - `spatial.py` — pooled pairwise Gini + `compute_fspatial`
    - `causal.py` — `compute_fcausal` + `per_unit_attribution(_signed)`
    - `hat_matrices.py` — `precompute_hat_matrices` + `compute_fcausal_torch`
    - `g0_power_basis.py` — `G0Function` + `fit()`

    ## Key design choices

    **Pooled, not stratified.** One Gini over all active units (not one per
    time block averaged together), and one F_causal from a single
    (n_active × n_active) hat-matrix regression. This gives a unified
    spatiotemporal equity number whose decomposition into per-unit attribution
    is mathematically exact.

    **Option B is the sole causal formulation.** No dispatch, no "baseline" or
    "option_c" alternatives. `F_causal = R'(I-H_demo)R / R'MR` where
    `R = Y - g_0(D)` and `H_demo` projects onto `[1, standardized(demographics)]`.

    **Per-unit attribution sums to `1 - F_causal`.** Because `(I - H)` and `M`
    are idempotent, the decomposition
    `attribution_i = ((MR)_i^2 - ((I-H)R)_i^2) / R'MR` has the exact property
    that the per-unit scores sum to `1 - F_causal`. This is both a
    useful diagnostic and a clean publishable result.

    **Power basis for g_0, isotonic for diagnostics.** `g_0(D)` is fit with a
    4-parameter power basis `[1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)]` because the
    hat-matrix math requires a linear model. Isotonic regression is also fit
    during preprocess for diagnostic comparison — if the two disagree
    materially, something is wrong.

    **N-vector inputs only.** `compute_fspatial(pickup_N, dropoff_N,
    active_taxis_N)` does not know the grid exists. This enforces a clean
    separation: grid arithmetic happens in `algorithm/objective.py`; fairness
    math happens here.

    ## API surface

        from famail_temporal.fairness import (
            compute_fspatial, compute_fcausal,
            per_unit_attribution, per_unit_attribution_signed,
            precompute_hat_matrices, compute_fcausal_torch,
            G0Function, fit_g0,
        )

    ## Dependencies

    `config` only. No imports from `data/`, `algorithm/`, or `fidelity/`.

    ## Paper-section hook

    "Fairness Metrics" in Methods. The per-unit attribution decomposition
    surfaces again in the Results section's attribution heatmaps and in the
    trajectory-ranking experiment.

- [ ] **Step 3: Write famail_temporal/fidelity/README.md**

    # fidelity/

    Discriminator-based realism check. The discriminator is treated as an
    opaque pre-trained artifact — only inference-time machinery is ported.

    ## Files

    - `model.py` — four inference-only classes from the V3 discriminator
      (`FeatureNormalizer`, `SiameseLSTMEncoder`, `ProfileEncoder`,
      `MultiStreamSiameseDiscriminator`)
    - `checkpoint.py` — `load_discriminator(path) -> model`
    - `context.py` — `MultiStreamData` + `MultiStreamContextBuilder`
    - `compute.py` — `compute_ffidelity(model, tau, tau_prime, ms_kwargs)`

    ## What was and wasn't ported

    **Ported (4 classes from discriminator/model/model.py, 1297 lines total):**
    `FeatureNormalizer`, `SiameseLSTMEncoder`, `ProfileEncoder`,
    `MultiStreamSiameseDiscriminator`. Combined, these comprise the full V3
    inference graph.

    **Excluded:** Training loops (`train.py`, `trainer.py`), dataset classes
    (`dataset.py`), training dashboard, and five deprecated architectures
    (`SiameseLSTMDiscriminator`, `TransformerEncoder`,
    `SiameseTransformerDiscriminator`, `SiameseLSTMDiscriminatorV2`, and any
    other legacy variants). All checkpoint-generation code remains in the
    legacy `discriminator/` directory — we consume checkpoints, we don't
    create them.

    ## Key design choices

    **cuDNN backward-in-inference workaround.** cuDNN's RNN backward requires
    training mode, but we need inference-mode behavior (no dropout) while
    allowing gradient flow through the LSTM for ST-iFGSM. Disabling cuDNN for
    the forward pass via `torch.backends.cudnn.flags(enabled=False)` uses the
    pure-PyTorch LSTM implementation, which supports backward in inference
    mode. Without this, `pickup_tensor.grad` silently comes back as zero.

    **Four multi-stream builder decisions preserved verbatim:**
      1. Both Siamese branches represent the same driver
      2. Seeking fill strategy is "sample" (N=5 trajectories, slot 0 is target)
      3. Coordinates converted 0-indexed → 1-indexed for the V3 model (+1 to x, y)
      4. Gradient flows through slot 0 of `x2` only

    **`ALPHA_FIDELITY = 0` cleanly skips fidelity.** Useful for ablations
    ("pure fairness with no realism constraint") and for fast tests that don't
    need the discriminator checkpoint.

    ## API surface

        from famail_temporal.fidelity import (
            MultiStreamData, MultiStreamContextBuilder,
            FeatureNormalizer, SiameseLSTMEncoder, ProfileEncoder,
            MultiStreamSiameseDiscriminator,
            load_discriminator, MissingArchitectureConfig,
            compute_ffidelity,
        )

    ## Dependencies

    `config`, `utils/trajectory`.

    ## Paper-section hook

    "Fidelity Term" in Methods; checkpoint provenance and hyperparameters in
    Supplementary.

- [ ] **Step 4: Write famail_temporal/algorithm/README.md**

    # algorithm/

    Orchestration of the objective and the ST-iFGSM trajectory modification
    loop. This is the only directory that imports from both `data/` and
    `fairness/` and `fidelity/`.

    ## Files

    - `soft_cell_assignment.py` — `SoftCellAssignment` module +
      `inject_soft_counts_into_3d` helper
    - `attribution.py` — per-unit attribution → per-trajectory ranking
    - `objective.py` — `FAMAILObjective` (the orchestrator)
    - `modifier.py` — `TrajectoryModifier` (ST-iFGSM loop)

    ## Key design choices

    **Single grid↔unit conversion point.** The `(48, 90, T) → (N,)` gather
    happens in exactly one place: `FAMAILObjective.forward()`. Every fairness
    module downstream receives N-vectors. This invariant prevents an entire
    class of shape-mismatch bugs from the legacy codebase.

    **`pickup_N` carries gradient for both spatial and causal terms.** One
    tensor serves as both "pickup count for Gini" and "demand for F_causal".
    This simplifies the gradient graph and halves the backward-pass cost
    compared to a design with separate demand and pickup tensors.

    **Delta-tensor pattern for autograd-safe 3D injection.** The soft
    assignment module produces a 2D probability distribution; we inject it
    into `soft_pickup_3d[:, :, t_block]` using `out = base + delta` where
    `delta` is scatter-filled. In-place assignment on a `.clone()` was tried
    and turned out subtle; the delta pattern is unambiguously correct.

    **Per-trajectory attribution inherits from its pickup's (cell, t) unit.**
    Not path-aware (for now) — a trajectory's score is the per-unit
    attribution of its pickup, not an aggregate across all states the
    trajectory visits. This matches the "we only perturb the pickup"
    modification strategy.

    **Cross-trajectory ordering via shared `_base_pickup_3d`.** When
    trajectory B is modified after trajectory A, B's optimization sees the
    updated base (A's modification is already applied). This matches the
    legacy semantics and stabilizes convergence.

    **Pickup-in-inactive-unit safeguard.** Before starting ST-iFGSM, we check
    that the neighborhood around the pickup contains at least one active unit
    in the relevant time block. Trajectories whose pickups are isolated in
    inactive regions are skipped with a warning.

    ## Gradient flow (ASCII)

        pickup_tensor (x, y, requires_grad)
              |
              v
        SoftCellAssignment  -->  probs_2d (sum to 1)
              |
              v
        inject_soft_counts_into_3d (delta-tensor pattern)
              |
              v
        soft_pickup_3d[:, :, t*]  (only t* slice carries grad)
              |
              v (gather via mask_3d)
        pickup_N (N-vector)
              |
              |---> compute_fspatial (Gini over DSR + ASR)
              |
              '---> compute_fcausal (R'(I-H)R / R'MR)
                       |
                       v
                     total
                       |
                       v
                   .backward()
                       |
                       v
              pickup_tensor.grad  (2D gradient for ST-iFGSM step)

    ## API surface

        from famail_temporal.algorithm import (
            FAMAILObjective, TrajectoryModifier,
            ModificationResult, ModificationHistory,
            SoftCellAssignment, inject_soft_counts_into_3d,
            compute_per_unit_attribution, rank_trajectories, select_top_k,
        )

    ## Dependencies

    `data/`, `fairness/`, `fidelity/`, `utils/`, `config`.

    ## Paper-section hook

    "Algorithm" in Methods, including the gradient flow diagram, the
    attribution ranking, and the ST-iFGSM update rule. Per-unit attribution
    heatmaps land in Results.

- [ ] **Step 5: Write famail_temporal/utils/README.md**

    # utils/

    Shared utilities with no domain-specific knowledge. Everything here is
    trivially testable and has no fairness or trajectory-modification logic.

    ## Files

    - `seeding.py` — `set_all_seeds(seed)` unifies `random`, `numpy`, `torch`,
      `torch.cuda` seeds in one call
    - `trajectory.py` — `Trajectory` and `TrajectoryState` dataclasses
      (ported verbatim from the legacy trajectory module)

    ## Key design choices

    **One `set_all_seeds` call for reproducibility.** Every top-level script
    and the pytest autouse fixture calls `set_all_seeds(config.DEFAULT_SEED)`
    at the start. `MultiStreamContextBuilder`'s context-sampling seed also
    resolves via this.

    **Trajectory representation is unchanged from legacy.** The `Trajectory`
    and `TrajectoryState` dataclasses are the same shape as the legacy version
    — they are already clean and the 4-element state vector (x, y, time, day)
    is the paper's canonical representation.

    ## API surface

        from famail_temporal.utils import (
            set_all_seeds,
            Trajectory, TrajectoryState,
        )

    ## Dependencies

    Nothing except `config`.

    ## Paper-section hook

    Briefly mentioned in the Reproducibility appendix.

- [ ] **Step 6: Write famail_temporal/tests/README.md**

    # tests/

    Every task in the implementation plan is TDD: failing test first, then
    implementation. This directory is populated incrementally as the build
    progresses.

    ## Running tests

    Fast tests (< 10 s total) — synthetic fixtures only:

        pytest famail_temporal/tests/

    Full suite (adds integration tests that load real data and the real
    discriminator checkpoint):

        pytest famail_temporal/tests/ --run-slow

    ## Test categories

    **Mathematical invariants** — `test_math_invariants.py`,
    `test_spatial_fairness.py`, `test_causal_fairness.py`,
    `test_g0_power_basis.py`, `test_hat_matrices.py`. These guard properties
    that a reviewer might verify by hand from the equations:

      - `sum_i per_unit_attribution_i == 1 - F_causal`
      - `F_causal = 0` when `R` lies in the demographic span
      - `(I - H_demo)^2 = (I - H_demo)` (idempotent projection)
      - `rank(H_demo) == n_features + 1` (no demographic collinearity)
      - Gini is scale-invariant

    **Bug-class regression guards** — `test_gradient_flow.py`,
    `test_soft_cell_assignment.py`, `test_fidelity_compute.py`. These watch
    for specific bug classes the legacy codebase hit:

      - Silently-zero gradients due to cuDNN-in-inference
      - Gradients leaking across time blocks
      - Shape mismatches between unit count and hat matrix dimension

    **Integration tests** — `test_modifier_integration.py`,
    `test_data_loader.py` (slow). Exercise the modifier loop end-to-end on
    synthetic data and verify properties like:

      - Convergence or monotone improvement of the total objective
      - `pickup_3d` total mass is preserved within `EPS` after one modification
      - Final pickup distance from original is within the epsilon-ball

    ## Fixtures

    `conftest.py` provides:
      - `seeded` (autouse) — sets all seeds before every test
      - `--run-slow` option — enables tests marked `@pytest.mark.slow`

    Individual test files may define local helpers (e.g., `_make_synthetic_bundle`
    in `test_objective.py`, imported by the algorithm and integration tests).

    ## Paper-section hook

    Reproducibility appendix references the test categorization scheme.
    Specific invariant tests (e.g., the attribution-sum property) may be
    cited in Methods as guarantees of the decomposition's mathematical
    correctness.

- [ ] **Step 7: Verify all READMEs render as markdown**

    ls -la famail_temporal/*/README.md famail_temporal/cache/README.md \
           famail_temporal/raw_data/README.md \
           famail_temporal/discriminator_checkpoints/README.md

Expected: 9 README.md files (data, fairness, fidelity, algorithm, utils, tests, cache, raw_data, discriminator_checkpoints).

- [ ] **Step 8: Commit**

    git add famail_temporal/data/README.md \
            famail_temporal/fairness/README.md \
            famail_temporal/fidelity/README.md \
            famail_temporal/algorithm/README.md \
            famail_temporal/utils/README.md \
            famail_temporal/tests/README.md
    git commit -m "docs: sub-directory READMEs for all six core modules"

---

### Task 34: Top-level README.md

**Files:**
- Create: famail_temporal/README.md

- [ ] **Step 1: Write famail_temporal/README.md**

    # FAMAIL-Temporal

    Standalone implementation of the temporally-aware fairness reformulation
    of the FAMAIL trajectory modification algorithm. Produces a clean,
    self-contained codebase suitable for publication alongside the research
    paper.

    ## What this codebase does

    Takes a corpus of expert taxi-driver trajectories (Shenzhen, 2016) and
    edits their pickup locations to improve spatial and causal fairness at
    a pooled `(cell, t)` granularity — specifically, four time blocks
    (morning peak, midday, evening peak, night) instead of the legacy 2D-only
    formulation. A pre-trained discriminator constrains edits to remain
    realistic.

    ## Objective

        L = alpha_spatial * F_spatial + alpha_causal * F_causal + alpha_fidelity * F_fidelity

    - `F_spatial` — pooled Gini of Demand-Service Ratio and Arrival-Service
      Ratio over all active `(cell, t)` units
    - `F_causal` — Option B demographic residual independence, measured via
      hat-matrix projection over the same active units
    - `F_fidelity` — similarity between original and modified trajectories as
      judged by a frozen multi-stream Siamese discriminator

    Modification is by ST-iFGSM over pickup coordinates:
    `delta = clip(alpha * sign(grad L), -epsilon, epsilon)`, bounded in an
    epsilon-ball around the original pickup cell.

    ## Quickstart

        # 1. Install dependencies (torch, numpy, scikit-learn, pytest)
        cd famail_temporal
        pip install -r requirements.txt

        # 2. Copy raw data (see raw_data/README.md for the full list)
        # From repo root: cp source_data/*.pkl famail_temporal/raw_data/ ...
        # Also copy the discriminator checkpoint:
        # cp discriminator/model/checkpoints/20260316_223817/best.pt \
        #    famail_temporal/discriminator_checkpoints/default/best.pt

        # 3. Preprocess the raw data
        python -m famail_temporal.preprocess

        # 4. Run the tests (fast tests ~10 s; --run-slow adds full data tests)
        pytest famail_temporal/tests/
        pytest famail_temporal/tests/ --run-slow

    ## Using the API

        from famail_temporal.data import DataBundle
        from famail_temporal.algorithm import (
            FAMAILObjective, TrajectoryModifier,
            compute_per_unit_attribution, rank_trajectories, select_top_k,
        )
        from famail_temporal.fidelity import MultiStreamContextBuilder

        # Load everything
        bundle = DataBundle.load(max_trajectories=1000)

        # Compute attribution, select top-k trajectories to modify
        attribution, _ = compute_per_unit_attribution(bundle)
        scored = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
        top_k_indices = select_top_k(scored, k=100)

        # Modify
        objective = FAMAILObjective(bundle)
        ms_builder = MultiStreamContextBuilder(bundle.multi_stream)
        modifier = TrajectoryModifier(
            objective=objective,
            bundle=bundle,
            multi_stream_builder=ms_builder,
        )
        histories = modifier.modify_batch(
            [bundle.trajectories[i] for i in top_k_indices],
        )

    ## Directory layout

    - `config.py` — single source of truth for every knob in the system
    - `preprocess.py` — one-time script that produces everything in `cache/`
    - `data/` — ingest + canonical active-unit ordering
      ([README](data/README.md))
    - `fairness/` — pooled F_spatial and F_causal + per-unit attribution
      ([README](fairness/README.md))
    - `fidelity/` — discriminator port + multi-stream context builder
      ([README](fidelity/README.md))
    - `algorithm/` — objective, soft cell assignment, ST-iFGSM modifier
      ([README](algorithm/README.md))
    - `utils/` — shared helpers (seeding, trajectory dataclasses)
      ([README](utils/README.md))
    - `tests/` — TDD test suite
      ([README](tests/README.md))
    - `raw_data/` — input files (gitignored; see
      [README](raw_data/README.md) for copy instructions)
    - `cache/` — derived artifacts from preprocess (gitignored;
      [README](cache/README.md))
    - `discriminator_checkpoints/` — pre-trained discriminator weights
      (gitignored; [README](discriminator_checkpoints/README.md))

    ## Design spec

    See
    [../docs/superpowers/specs/2026-04-16-famail-temporal-design.md](../docs/superpowers/specs/2026-04-16-famail-temporal-design.md)
    for the full design document that motivated this rewrite, including the
    13 identified snags and their resolutions.

    ## Key design commitments

    These are the non-negotiable architectural invariants that every
    contributor should understand before editing:

    1. **One active-unit ordering.** `UnitIndexMap` defines a canonical
       cell-major, block-within-cell ordering used by every fairness and
       attribution computation. Shape-mismatch assertions guard against drift.
    2. **Single grid↔unit conversion.** The only `(48, 90, T) → (N,)` gather
       happens in `FAMAILObjective.forward()`. Fairness modules see
       N-vectors only.
    3. **Gradient flow only through `pickup_counts`.** The only tensor that
       varies during ST-iFGSM is `soft_pickup_3d`, and only in one time-block
       slice. All other inputs are frozen.
    4. **No external dependencies.** This directory has zero imports from
       outside `famail_temporal/`. See `requirements.txt`.

    ## Reproducibility

    Every top-level script calls `utils.seeding.set_all_seeds(config.DEFAULT_SEED)`
    before doing anything stochastic. The `seeded` pytest autouse fixture
    does the same before every test. The multi-stream context builder uses
    the same seed. Running preprocess twice produces bit-identical cache
    artifacts.

    ## Citing this code

    Cite the paper once it is published. Until then, cite this repository.

- [ ] **Step 2: Verify README renders**

    cat famail_temporal/README.md | head -80

- [ ] **Step 3: Run the full test suite one final time**

    pytest famail_temporal/tests/

Expected: all fast tests pass (slow tests remain skipped without `--run-slow`).

- [ ] **Step 4: Commit**

    git add famail_temporal/README.md
    git commit -m "docs: top-level README linking all sub-READMEs + quickstart"

---

## Implementation complete

At this point:

- All 34 tasks executed
- All fast tests pass
- All ten subdirectory READMEs written
- The algorithm is end-to-end functional

**Final verification checklist:**

- [ ] `pytest famail_temporal/tests/` — all fast tests green
- [ ] `pytest famail_temporal/tests/ --run-slow` — all tests green (if real data + checkpoint copied)
- [ ] `python -m famail_temporal.preprocess` — completes without errors (if raw data copied)
- [ ] `find famail_temporal -name README.md | wc -l` — returns `10` (top-level + 9 subdirs)
- [ ] `git log --oneline famail_temporal/` — 34+ commits, conventional-commit format

**Deliverables produced:**

1. `famail_temporal/` — the standalone rewrite directory
2. `docs/superpowers/specs/2026-04-16-famail-temporal-design.md` — the design spec
3. `docs/superpowers/plans/2026-04-16-famail-temporal*.md` — these four plan files (main + phase5-6 + phase7-8 + phase9)
4. Ten README.md files documenting every part of the new directory
5. Test suite covering mathematical invariants, bug-class regressions, and end-to-end integration

The algorithm is now ready for experimentation. Future extensions (T=24 hourly granularity, path-aware attribution, dashboards) have clean hooks in the design — see the "Out of scope" section of the design spec.
