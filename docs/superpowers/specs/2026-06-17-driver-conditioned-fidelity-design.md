# Driver-Conditioned Generation + Identity-Aware Fidelity (Level-1 v2) — Design Spec

**Date:** 2026-06-17
**Status:** Draft for review
**Branch:** `two-level-paper`
**Context:** Level-1 v1 ([`2026-06-17-level1-data-quality-table-design.md`](2026-06-17-level1-data-quality-table-design.md)) shipped a 4-source data-quality table, but its HuMID **Fidelity-A** failed the validation gate and was reported untrusted. Root cause (confirmed): HuMID is an **identity** model (same-driver?) used on **agent-agnostic, single-stream** generated data — deeply out of distribution (the frozen checkpoint was trained on 5-trajectory-per-stream, 3-stream, driver-keyed inputs; v1 fed one seeking-only trajectory). This spec makes multi-agency first-class so HuMID becomes well-posed, and enriches the discriminator-free **Fidelity-B**. The GAN-architecture stabilization work (runaway-G / strong-critic) is explicitly **on hold** and out of scope here.

---

## 1. Goal

Make the BC/GAN baselines **driver-conditioned** so generated trajectories carry a driver identity, then (a) score **Fidelity-A** with HuMID used near its trained regime — a *matched-vs-mismatched driver* gate that is well-posed — and (b) extend **Fidelity-B** with standard human-mobility realism statistics. Deliver a Level-1 table **v2** where Fidelity-A is trustworthy (gate can actually pass) and Fidelity-B is a richer realism cross-check.

**Success criteria:** (1) driver-conditioned BC + GAN generators that condition generation on a target driver; (2) a Fidelity-A whose validation gate is a matched-vs-mismatched separation test (not the v1 OOD reduced-mode call); (3) an enriched Fidelity-B; (4) a regenerated Level-1 v2 table + results doc. HuMID remains **frozen, read-only**.

---

## 2. Scope

### In scope
- Optional **driver conditioning** added to the generator + training/rollout plumbing (backward compatible: absent → today's unconditioned behavior, unchanged numerics).
- A **per-driver index map** and per-driver trajectory grouping.
- **Identity-aware Fidelity-A**: build HuMID inputs near the trained regime (per-driver trajectory *sets*, **seeking + profile** streams, driving omitted symmetrically), and a **matched-vs-mismatched** validation gate.
- **Enriched Fidelity-B**: add per-trajectory mobility stats (radius of gyration, net displacement) + a corpus-level terminal-cell-distribution JS, alongside the existing length/displacement/coverage.
- A driver-conditioned **Level-1 v2 orchestrator** + results doc + training curves (reusing the existing curve-capture).

### Out of scope
- **GAN architecture/training stabilization** (spectral norm, hinge/relativistic loss, n_critic/TTUR retuning) — separate, on hold.
- The **real-vs-generated realism classifier** — explicitly dropped (user decision).
- Any change to the editing algorithm, F_causal/F_spatial formulas, ε, or the HuMID model/checkpoint (reused frozen, read-only). No edits to `algorithm/`, `fairness/`, `fidelity/`.
- Multi-seed fidelity (v1's single-representative-seed choice stands).
- Having the generator emit the **driving** stream (a noted future upgrade; here driving is omitted symmetrically from both HuMID branches).

---

## 3. Locked design decisions

1. **Driver conditioning is additive + optional.** Extend `TrajectoryLSTM` with an optional driver embedding folded into the existing additive context (`cell + tblock + driver`); a `driver_idx=None` path preserves current behavior bit-for-bit. Training/rollout take an optional parallel `driver_idxs` list. No new generator class.
2. **Driver index map.** Build `driver_to_idx` from the corpus (`Trajectory.driver_id`; ~50 drivers); `n_drivers = len(map)`. Persist the map with the run for reproducible conditioned generation.
3. **Fidelity-A uses HuMID near its trained regime** (per the checkpoint's `model_config`: `n_trajs_per_stream=5`, `combination_mode='concatenation'`, `streams=(seeking,driving,profile)`): represent each driver by a **set of N=5 trajectories** per branch. Streams used = **seeking + profile**, with **driving omitted from BOTH branches** (symmetric graceful degradation) since generated trajectories have no driving stream. Profiles come from `fidelity/context.py`'s per-driver `profile_features` (read-only). *(Caveat resolution (i)+(ii): attach the driver's real profile AND rely on the matched-vs-mismatched gate.)*
4. **Validation gate = matched vs mismatched.** `high_matched` = HuMID(real-d set, gen-for-d set); `low_mismatched` = HuMID(real-d set, gen-for-d′ set). Gate passes iff `high_matched − low_mismatched ≥ MARGIN` and `high_matched > low_mismatched`. Keep `MARGIN = 0.2` (module constant) unless evidence says otherwise. All means persisted regardless.
5. **Fidelity-A per source** = mean matched same-agent probability over driver sets (how well gen-for-d reads as driver d). Gated/flagged by the matched-vs-mismatched gate (replacing v1's real-vs-collapsed/shuffled gate).
6. **Enriched Fidelity-B statistics.** Per-trajectory scalars: length, mean per-step displacement, spatial coverage (existing) + **radius of gyration** + **net displacement** (origin→terminal Euclidean). Corpus-level: **terminal-cell (pickup) distribution JS** vs raw (reuse `terminal_cell_histogram` + `jensen_shannon_divergence`). Aggregate = mean of all component JS values; all in bits, lower = better.
7. **Forward-only, frozen HuMID** under `torch.no_grad()`; `train(False)`; +1 coords; the mask convention from v1 (True = valid step) carries over. Generated HuMID inputs are constructed to the discriminator's trained input format (mirroring `fidelity/context.py`, read-only).
8. **Single representative seed** for BC/GAN generation (v1 §3.3 stands).

---

## 4. Component A — Driver-conditioned generation

### A.1 Generator (`baselines/gan/generator.py`)
- `TrajectoryLSTM.__init__(..., n_drivers: int | None = None)` → if set, `self.driver_embed = nn.Embedding(n_drivers, embed_dim)`.
- `forward(tokens, ctx_cell, ctx_tblock, driver_idx=None)`, `step(..., driver_idx=None)`, `step_embed(..., driver_idx=None)`: when `driver_idx` is provided, `ctx = cell_embed(ctx_cell) + tblock_embed(ctx_tblock) + driver_embed(driver_idx)`; else unchanged. `driver_idx=None` with `n_drivers=None` is exactly today's path (regression-tested).

### A.2 Training + rollout plumbing
Thread an **optional** `driver_idxs: List[int] | None = None` (index-aligned with `sequences`/`contexts`) through:
- `train_mle(model, sequences, contexts, *, driver_idxs=None, ...)` — per minibatch, build the `driver_idx` batch and pass to `model(...)`.
- `adversarial_finetune(..., driver_idxs=None, ...)` — same; pass `driver_idx` into `gumbel_rollout`.
- `gumbel_rollout(model, cc, tb, *, driver_idx=None, ...)` and the rollout decoders (`generate_trajectories`, `generate_pickups`, `sample_terminal_cells_batched`) — pass `driver_idx` through to `model.step`.
None given → identical to current behavior (backward-compatible; existing tests unaffected).

### A.3 Driver map + grouping (`baselines/gan/` helper, new)
- `build_driver_index(trajectories) -> {driver_id: idx}` and its inverse; deterministic ordering.
- `group_by_driver(trajectories) -> {driver_idx: [Trajectory, ...]}` for the N=5 set construction in Fidelity-A.

### A.4 Conditioned generation for the table
- BC = MLE-only driver-conditioned; GAN = MLE + adversarial driver-conditioned (current critic — stabilization is out of scope).
- For fidelity: generate **for each real seed's driver** (gen_i conditioned on `filtered_train[i].driver_id`) so gen_i is a driver-d_i trajectory; also generate **for a wrong driver** (d′ ≠ d) for the mismatched gate set.

---

## 5. Component B — Identity-aware Fidelity-A (HuMID)

### B.1 Input construction (`baselines/fidelity_eval.py`, new helpers; read-only mirror of `fidelity/context.py`)
- `driver_set_branch(trajs_or_cells, driver_idx, *, n=5) -> HuMID-branch input` for the **seeking** stream (N trajectories per branch, padded; +1 coords; the seeking feature dimension and packing pinned at plan time by reading `fidelity/context.py` / `SiameseLSTMEncoder(input_dim=…)`), plus the **profile** stream from `profile_features[driver_idx]`. **Driving = None** for both branches.
- Real branch: from a driver's real trajectories. Generated branch: from driver-conditioned generated trajectories (`generated_to_disc_tensor` per trajectory, assembled into the N-set), with the **target driver's** profile attached.

### B.2 Scoring + gate (`baselines/fidelity_eval.py`)
- `humid_identity_fidelity(disc, driver_pairs, *, batch_size, device) -> {mean, std, n}` — mean same-agent probability over (real-d set, gen-for-d set) branch pairs, forward-only.
- `identity_validation_gate(disc, *, matched_pairs, mismatched_pairs, batch_size, device, margin=GATE_MARGIN) -> {high_matched, low_mismatched, margin, passed}` — passes iff `high_matched − low_mismatched ≥ margin` and `high_matched > low_mismatched`.
- Per source Fidelity-A = `humid_identity_fidelity(matched_pairs).mean`; `fidelity_a_trusted = gate.passed`.

### B.3 Fallback (decision (ii))
If the N=5 multi-stream construction proves impractical at plan time, fall back to **single-trajectory seeking+profile** matched-vs-mismatched pairs (still well-posed because the identity signal is now real). The matched-vs-mismatched gate is the load-bearing trust mechanism either way.

---

## 6. Component C — Enriched Fidelity-B (discriminator-free)

### C.1 Per-trajectory statistics (`baselines/fidelity_eval.py::trajectory_statistics`, extended)
Add to the existing `{length, mean_displacement, coverage}`:
- `radius_of_gyration` = `sqrt(mean_i ||r_i − r_cm||²)`, `r_cm` = centroid of visited `(x,y)` (0.0 if length < 2).
- `net_displacement` = Euclidean distance origin→terminal `(x,y)` (0.0 if length < 2).
(All keys remain backward-compatible additions; existing callers/tests that read the three original keys are unaffected.)

### C.2 Corpus-level distribution (`baselines/fidelity_eval.py`, new)
- `terminal_cell_distribution_js(source_cells, raw_cells) -> float` — JS (bits) between the source's and raw's terminal-cell (pickup) histograms (reuse `terminal_cell_histogram` + `jensen_shannon_divergence`).

### C.3 Aggregate
`distributional_fidelity` extended to the 5 per-trajectory stats (shared-grid JS each) + the terminal-cell JS; `aggregate` = mean of all component JS values; per-component values reported. Lower = more faithful; raw vs raw = 0.

---

## 7. Integration — Level-1 table v2

Driver-conditioned orchestration (extend the v1 pipeline or a sibling `run_level1_table_v2.py` — packaging decided at plan time; **reuse** `fidelity_eval`, the fairness path, the edited-fairness-from-`metrics_after` fix, persistence, and curve capture). Four sources: raw, FAM-AIL edited, **BC (driver-conditioned)**, **GAN (driver-conditioned)** × {F_causal, F_spatial, **Fidelity-A** (identity, matched-vs-mismatched gated), **Fidelity-B** (enriched)}. Persist `level1_v2_metrics.json` (incl. the gate, per-component Fidelity-B, the driver map), `level1_v2_table.md`, `training_curves.json`, `trajectory_stats.npz`; emit a `LEVEL1_V2_RESULTS.md`.

---

## 8. Data flow

```
trajectories ─► build_driver_index / group_by_driver ─► driver_idx per trajectory
re-train BC (MLE, driver-cond)  ─► generate_trajectories(..., driver_idxs) ─► gen-for-d
re-train GAN (MLE+adv, driver-cond) ─► generate_trajectories(..., driver_idxs) ─► gen-for-d (+ gen-for-d′)

each source ─► terminal cells ─► pickups_to_pickup_3d ─► data_level_fairness ─► F_causal, F_spatial
              (edited fairness from edit metrics_after, per the v1 fix)

HuMID(frozen) ─┬─ identity_validation_gate(matched: real-d vs gen-for-d ; mismatched: real-d vs gen-for-d′) ─► trusted?
               └─ humid_identity_fidelity(real-d vs gen-for-d per source) ─► Fidelity-A (gated)

each source stats vs raw ─► distributional_fidelity (length/disp/coverage/RoG/net-disp + terminal-cell JS) ─► Fidelity-B

assemble ─► level1_v2_metrics.json + level1_v2_table.md + LEVEL1_V2_RESULTS.md (+ curves)
```

---

## 9. Components (files)

- **Modify:** `baselines/gan/generator.py` (optional driver embedding); `baselines/gan/train_mle.py`, `train_adversarial.py`, `rollout.py` (optional `driver_idxs` plumbing); `baselines/fidelity_eval.py` (identity Fidelity-A + enriched stats + terminal-cell JS).
- **Create:** a driver-map/grouping helper (`baselines/gan/drivers.py` or in `sequences.py`); the v2 orchestrator (module or flag); `baselines/LEVEL1_V2_RESULTS.md`.
- **Read-only reuse:** `fidelity/context.py` (per-driver `profile_features`, multi-stream input format), `fidelity/model.py` (forward signature), `fidelity/checkpoint.py` (`load_discriminator`).
- **Tests:** generator driver-conditioning (incl. `driver_idx=None` regression); driver map/grouping; identity Fidelity-A + matched-vs-mismatched gate (stub discriminator); enriched stats (RoG, net-disp, terminal-cell JS); v2 orchestrator pure helpers + alignment.

---

## 10. Error handling & edge cases

- **Unknown driver_id** at generation (not in the map): raise a clear error (or map to a reserved "unknown" index — decided at plan time; default: raise).
- **Driver with < N trajectories** for the N=5 set: sample with replacement (note it) or reduce N for that driver; never crash.
- **Profile missing** for a driver index: fall back to zero/default profile with a logged warning (graceful degradation; HuMID already supports a missing stream).
- **`driver_idx=None`** path must be numerically identical to today (regression test).
- **Empty generated rollout**: excluded from sets + stats, counted in `n_empty` (as v1).
- **length < 2**: RoG and net_displacement = 0.0.
- **Gate may still fail**: non-fatal; Fidelity-A flagged untrusted; Fidelity-B leads (same fallback discipline as v1). The matched-vs-mismatched design is expected to separate *if* the generator captured driver style — a genuine, interpretable test of that.

---

## 11. Testing

Unit (no GPU, stub discriminator where needed):
- generator: `driver_idx` changes logits vs `None`; `driver_idx=None` identical to pre-change (regression); shape checks.
- driver map: deterministic index assignment; inverse; grouping counts.
- identity Fidelity-A: stub discriminator scoring same-driver sets high / different-driver low → gate passes; constant stub → gate fails; mean/std/n correct; batching.
- enriched stats: hand-built trajectory → known RoG, net-displacement; length<2 → 0.0; terminal-cell JS identical→0, disjoint→high.
- v2 orchestrator: result-dict/JSON round-trip; render; driver-conditioned `_train_and_generate` alignment (contexts/driver_idxs/filtered_train index-aligned).

Real-data smoke (GPU, manual): driver-conditioned BC+GAN, matched-vs-mismatched gate verdict, populated v2 table + curves.

---

## 12. Open decisions — resolved / deferred

- HuMID stream set: **seeking + profile, driving omitted symmetrically** (resolved, §3.3).
- Gate: **matched vs mismatched**, margin 0.2 (resolved, §3.4).
- Fidelity-B additions: **RoG + net-displacement + terminal-cell JS** (resolved, §6).
- Real-vs-generated classifier: **dropped** (resolved).
- GAN stabilization: **deferred** (separate effort).
- **Plan-time details to pin (by reading code read-only):** exact seeking feature dimension + N-set packing format from `fidelity/context.py`; profile accessor API; v2 packaging (flag vs new module); unknown-driver policy.
