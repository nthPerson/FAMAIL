# Level-1 v2 Methodology: Driver-Conditioned Generation + Identity-Aware Fidelity

**Status:** implementation complete; full run done 2026-06-18 (see [`LEVEL1_V2_RESULTS.md`](LEVEL1_V2_RESULTS.md) for numbers — gate PASSED, Fidelity-A trusted).
**Branch:** `two-level-paper`.
**Design spec:** [`docs/superpowers/specs/2026-06-17-driver-conditioned-fidelity-design.md`](../../docs/superpowers/specs/2026-06-17-driver-conditioned-fidelity-design.md).
**Implementation plan:** [`docs/superpowers/plans/2026-06-17-driver-conditioned-fidelity.md`](../../docs/superpowers/plans/2026-06-17-driver-conditioned-fidelity.md).

This document is written to make paper-writing straightforward: it states the motivation, the architectures, the evaluation construction, and the design decisions (and their justifications) in full. It is the companion to the numeric results doc.

---

## 1. Motivation

The Level-1 "data-quality" argument compares four data **sources** — `raw` (real Shenzhen taxi trajectories), `edited` (FAM-AIL fairness-edited), `bc` (behavior-cloning / MLE-generated), and `gan` (adversarially fine-tuned) — on **fairness** (causal F_causal, spatial F_spatial; 1 = fairest) and **fidelity** (realism). The claim is that fairness-editing produces data that is both *fairer* and *as faithful* as the alternatives, so it is a better source for downstream imitation learning.

Level-1 **v1** shipped this table but its discriminator-based fidelity (**Fidelity-A**, using the frozen HuMID Siamese discriminator) **failed its validation gate** and had to be reported "untrusted," leaving the discriminator-free distributional metric (**Fidelity-B**) as the only fidelity signal.

**Root cause (confirmed empirically).** HuMID (`MultiStreamSiameseDiscriminator`, an ST-SiameseNet after Ren et al., KDD 2020) is an **identity** model: it answers "do these two trajectory *sets* belong to the same driver?" It was trained on **5 trajectories per branch, three streams** (seeking + driving + profile), keyed by driver. v1 fed it a **single seeking-only trajectory** per branch (N=1, auto zero-padded to the trained N=5; driving and profile defaulted to the model's zero embeddings). That input is far out of distribution, so the v1 gate measured `real-vs-real ≈ real-vs-shuffled ≈ 0.668` — no separation, hence "untrusted."

The fix is **not** to change HuMID (it is frozen and reused read-only). It is to (a) make the generated data carry a **driver identity** so HuMID is applicable, and (b) construct HuMID's inputs **near its trained regime**. That is what v2 does, in three components.

---

## 2. Component A — Driver-conditioned generation

### 2.1 The three models (and how they relate)

A frequent question (Dr. Zhang, Meeting 38): *is the GAN's critic the same model as the discriminator used in trajectory editing and in fidelity scoring?* **No — there are three distinct models:**

| Model | File | Role | Architecture |
|---|---|---|---|
| **Generator** `TrajectoryLSTM` | `baselines/gan/generator.py` | produces BC/GAN trajectories | conditional autoregressive LSTM over the grid-cell vocabulary; context injected **additively** |
| **GAN critic** `SequenceCritic` | `baselines/gan/critic.py` | adversarial realism signal during GAN fine-tuning | single-stream, **unconditioned** `Embedding→LSTM→Linear(1)` realism scorer |
| **HuMID** `MultiStreamSiameseDiscriminator` | `fidelity/model.py` | **same** model used in (i) the trajectory-editing fidelity term and (ii) Level-1 Fidelity-A | 3-stream **Siamese** identity discriminator (seeking + driving + profile), N=5 trajectories/branch |

The GAN critic is simpler than, and independent of, HuMID. HuMID is the shared, frozen identity model — it is never trained here; it is consumed forward-only under `torch.no_grad()`.

### 2.2 Additive, optional driver embedding

`TrajectoryLSTM` already conditions generation on a start cell and start time-block by **adding** their embeddings to every input-token embedding:

```
ctx = cell_embed(start_cell) + tblock_embed(start_tblock)        # (B, E)
```

Driver conditioning extends this additively with an optional driver embedding:

```
ctx = cell_embed(start_cell) + tblock_embed(start_tblock) + driver_embed(driver_idx)
```

- `TrajectoryLSTM(..., n_drivers=N)` creates `driver_embed = nn.Embedding(N, embed_dim)`; built **without** `n_drivers` there is no driver embedding at all.
- `forward` / `step` / `step_embed` take an optional `driver_idx` (a `(B,)` long tensor). When it is `None`, the model is **bit-for-bit identical** to the original unconditioned generator (regression-tested), so all prior baselines are unaffected.
- The optional `driver_idxs` list is threaded (index-aligned with the training sequences/contexts) through `train_mle`, `adversarial_finetune` (and its Gumbel-softmax rollout), and the rollout decoders (`generate_trajectories`, `generate_pickups`, `sample_terminal_cells_batched`). In the training loops the per-batch driver tensor is built from the **same permuted index list** used to gather the batch, so each trajectory trains under its correct driver.

This design (additive, optional, no new generator class) keeps driver-conditioning a clean superset of the existing pipeline.

### 2.3 Driver index map

`Trajectory.driver_id` is an integer in `[0, 49]` (50 Shenzhen drivers). `build_driver_index` maps the distinct driver ids to contiguous embedding indices `[0, n_drivers)` in sorted order (deterministic, persisted to `driver_index.json`). `group_by_driver` groups trajectories by driver id for the Fidelity-A set construction. The per-driver **profile** vectors (11-dim, z-score-normalized) come from `bundle.multi_stream.profile_features` (the same artifact HuMID was trained on), keyed by the original driver id.

---

## 3. Component B — Identity-aware Fidelity-A

### 3.1 Branch construction (mirrors the editing algorithm)

Fidelity-A is scored exactly the way HuMID is used **inside the trajectory-editing algorithm** (`fidelity/context.py`). Each HuMID *branch* is:

- **Seeking stream:** N = 5 trajectories. **Slot 0 = the trajectory under test**; **slots 1–4 = the same driver's real context trajectories** (sampled, with replacement if a driver has < 4). All coordinates are 1-indexed (`+1`), matching HuMID's training; padding masks mark valid steps.
- **Profile stream:** that driver's real 11-dim profile vector.
- **Driving stream:** **omitted from both branches** (passed as `None`), so HuMID symmetrically uses its fixed zero `driving_default_embedding`. Generated trajectories have no driving stream, so omitting it from *both* branches keeps the comparison fair (graceful degradation, supported by the model).

The discriminator then asks: *does the slot-0 trajectory look like it came from the same driver as the real context + profile?* This is precisely the fidelity question, evaluated in HuMID's trained regime (N=5, seeking + profile), which is the key departure from v1.

### 3.2 Per-source Fidelity-A

For each source `S` and each evaluation driver `d`, a **matched** pair compares
`(real-d slot-0 + real-d context, profile d)` against `(S-of-d slot-0 + real-d context, profile d)` — same driver. `S-of-d` is:
- `raw` → a different (disjoint where possible) real trajectory of driver `d`;
- `edited` → an edited trajectory of driver `d`;
- `bc` / `gan` → a trajectory **generated conditioned on driver `d`** (start contexts taken from `d`'s real trajectories).

**Fidelity-A(S) = mean matched same-agent probability** over all evaluation drivers (higher = more faithful to driver identity). Because the slot-0 trajectory is the only thing that varies between the real and source branches (context and profile are held to driver `d`), Fidelity-A isolates how well the source trajectory itself preserves driver-`d` style.

### 3.3 The real-anchored validation gate

The gate tests whether HuMID is **well-posed in our construction** — i.e., whether it actually separates same-driver from different-driver here — independent of any generator's quality:

- `high_matched` = mean over drivers of `HuMID(real-d slot-0 + real-d context/profile, another real-d slot-0 + real-d context/profile)` — same driver.
- `low_mismatched` = mean over `(d, d′≠d)` of `HuMID(real-d branch, real-d′ slot-0 + real-d′ context/profile)` — different driver (d′ = the next evaluation driver, wrap-around).
- **Gate passes iff `high_matched − low_mismatched ≥ 0.2` and `high_matched > low_mismatched`.**

When the gate passes, Fidelity-A is reported as **trusted** for all sources. (GPU validation on the real checkpoint before the full run gave `high_matched ≈ 0.82`, `low_mismatched ≈ 0.005` — a decisive pass, in contrast to v1's degenerate `0.668 ≈ 0.668`.)

**Per-source separation diagnostic.** For each source we also report `separation(S) = Fidelity-A(S) − mismatched(S)`, where `mismatched(S)` pairs the real-`d` branch against `S-of-d′` (the source's trajectory for the *other* driver `d′`, with `d′` context + profile). For `bc`/`gan` this is the spec's "did the generator capture driver-specific style?" test: a large separation means generated-for-`d` reads as driver `d` and generated-for-`d′` does not.

### 3.4 Design decision: real-anchored gate (deviation from spec §3.4)

The design spec (§3.4) anchored the gate on **generated** data (`gen-for-d` vs `gen-for-d′`). We instead anchor the **trust verdict** on **real** data (`real-d` vs `real-d′`), because:

1. **Decoupling.** The real-anchored gate measures *"is the metric well-posed in our regime?"* — a property of HuMID + our input construction, **not** of any generator. The spec's gen-anchored gate conflates metric validity with generator quality: a *collapsed* GAN would fail it and thereby (incorrectly) invalidate the metric for *all* sources, including raw and edited.
2. **It is the direct fix for v1.** v1 failed precisely because real-vs-real did not separate from real-vs-garbage in the OOD construction. Showing real-vs-real-same-driver ≫ real-vs-different-driver in the proper construction is the cleanest demonstration that the metric is now trustworthy.
3. **No information is lost.** The gen-anchored separations the spec asked for are still computed and persisted (the per-source `separation` diagnostics, §3.3), so the gate can be reinterpreted from the saved means without re-running.

This was confirmed with the user before implementation. It is a **strict superset** of the spec: real-anchored trust verdict **plus** all per-source (incl. gen-based) separations.

---

## 4. Component C — Enriched Fidelity-B (discriminator-free)

Fidelity-B compares each source's trajectory-statistic distributions to `raw` via Jensen-Shannon divergence (bits; lower = more faithful; `raw` vs `raw` = 0). v2 enriches it with two standard human-mobility statistics and a corpus-level spatial check:

**Per-trajectory statistics** (each histogrammed on a shared grid across all four sources, then JS vs raw):
- `length` — number of steps (existing).
- `mean_displacement` — mean Euclidean step length (existing).
- `coverage` — number of distinct cells visited (existing).
- **`radius_of_gyration`** — `sqrt(mean_i ||r_i − r_cm||²)`, with `r_cm` the centroid of the visited `(x, y)` cells. The canonical spatial-spread measure in human-mobility analysis. `0.0` for trajectories of length < 2.
- **`net_displacement`** — Euclidean distance from origin to terminal cell. Distinguishes commute-like (large) from local/loop trajectories (small). `0.0` for length < 2.

**Corpus-level statistic:**
- **`terminal_cell_distribution_js`** — JS divergence between a source's and raw's **terminal-cell (pickup) distribution** over the grid (reuses `terminal_cell_histogram` + `jensen_shannon_divergence`). Catches spatial drop-off mismatch that per-trajectory scalars miss.

**Aggregate Fidelity-B(S)** = mean of the five per-trajectory JS values **and** the terminal-cell JS. All component values are persisted in `fidelity_b_per_component` for the paper's breakdown table.

The original three-key behavior is preserved for v1: `distributional_fidelity` / `stat_ranges` are key-parameterized and default to the original three statistics, so v1 callers and tests are unchanged; v2 explicitly passes the five-key set.

---

## 5. Evaluation protocol & data flow

```
trajectories ─► build_driver_index / group_by_driver ─► driver_idx per trajectory, profiles per driver
re-train BC  (MLE, driver-conditioned)        ─► generate_trajectories(..., driver_idxs) ─► gen-for-d
re-train GAN (MLE + WGAN-GP, driver-conditioned) ─► generate_trajectories(..., driver_idxs) ─► gen-for-d

Fidelity-A (HuMID, frozen):
  per driver d, partner d' = next driver:
    matched(S,d)    = (real-d + ctx d, prof d)  vs  (S-of-d  + ctx d,  prof d)     # same driver
    mismatched(S,d) = (real-d + ctx d, prof d)  vs  (S-of-d' + ctx d', prof d')    # different driver
  gate          = identity_validation_gate(matched=raw_matched, mismatched=raw_mismatched)   # real-anchored
  Fidelity-A(S) = mean matched same-agent prob ;  separation(S) = Fidelity-A(S) − mean mismatched(S)

Fidelity-B (discriminator-free):
  per-trajectory {length, mean_disp, coverage, radius_of_gyration, net_displacement} ─► shared-grid JS vs raw
  + terminal-cell distribution JS vs raw ─► aggregate (mean), per-component reported

Fairness:
  raw    = data_level_fairness(bundle)
  edited = edit run's metrics_after  (relocates pickups within the full corpus; recomputing from the
           modified subset would be a sparse, non-comparable grid — same basis as raw)
  bc/gan = driver-conditioned generate_pickups ─► pickups_to_pickup_3d ─► data_level_fairness

assemble ─► level1_v2_metrics.json + level1_v2_table.md + training_curves.json
            + trajectory_stats.npz + driver_index.json
```

**Settings (full run).** 50 evaluation drivers (all have ≥ 6 real trajectories), 20 matched pairs/driver, MLE 20 epochs, adversarial 3 epochs (WGAN-GP, `n_critic = 5`), Fidelity-B sample 5000, single representative seed 0. HuMID frozen/read-only throughout.

---

## 6. Reproducibility

```bash
python -m famail_temporal.baselines.run_level1_table_v2 \
  --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
  --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto
```

Outputs (per run, under the chosen `--out-dir`): `level1_v2_metrics.json` (full result incl. gate, per-source Fidelity-A/separation, per-component Fidelity-B), `level1_v2_table.md`, `training_curves.json` (per-batch + per-epoch MLE/adversarial curves), `trajectory_stats.npz` (the five v2 statistics per source), `driver_index.json` (the persisted driver→index map).

**Out of scope (deliberately).** GAN architecture/training stabilization (spectral norm, hinge/relativistic loss, TTUR, `n_critic` retuning) — a separate, on-hold effort; here the existing `SequenceCritic` is used as-is. The real-vs-generated realism classifier was dropped (user decision). Multi-seed fidelity (the single-representative-seed choice stands; the variance suite remains the multi-seed authority for fairness).

---

## 7. Code map

| Concern | Location |
|---|---|
| Driver-conditioned generator | `baselines/gan/generator.py` (`TrajectoryLSTM`, optional `n_drivers`/`driver_idx`) |
| Driver map + grouping | `baselines/gan/drivers.py` |
| Training/rollout plumbing | `baselines/gan/{train_mle,gumbel,train_adversarial,rollout}.py` (optional `driver_idxs`) |
| Identity Fidelity-A + gate | `baselines/fidelity_eval.py` (`build_identity_branch`, `humid_identity_fidelity`, `identity_validation_gate`) |
| Enriched Fidelity-B | `baselines/fidelity_eval.py` (`trajectory_statistics`, key-parameterized `distributional_fidelity`/`stat_ranges`, `terminal_cell_distribution_js`) |
| Orchestrator (v2 table) | `baselines/run_level1_table_v2.py` |
| HuMID (frozen, read-only) | `fidelity/model.py`, `fidelity/context.py`, `fidelity/checkpoint.py` |
| Tests | `baselines/tests/test_{generator_driver_cond,drivers,train_rollout_driver_cond,fidelity_eval_identity,fidelity_eval_enriched,run_level1_table_v2}.py` |
