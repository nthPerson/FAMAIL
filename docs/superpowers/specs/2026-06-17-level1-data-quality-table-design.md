# Level-1 Data-Quality Table + Fidelity Metric — Design Spec

**Date:** 2026-06-17
**Status:** Approved (brainstorming complete)
**Context:** Meeting 38 (2026-06-11) restructured the paper around a Two-Level Argument (see memory `project_paper_argument.md`). This spec covers **Level 1 — Data Quality**: the claim that FAM-AIL's edited data is higher quality than data produced by generative baselines, measured on causal fairness, spatial fairness, and **fidelity** (realism). Level 2 (usability) is out of scope here.

---

## 1. Goal

Produce a reproducible **Level-1 data-quality table** — 4 data sources × 4 metrics — that answers Dr. Zhang's explicit "double check" (does FAM-AIL outperform the baselines on data quality?), with the new fidelity axis implemented two ways: a **HuMID-discriminator metric guarded by a validation gate**, and an independent **discriminator-free distributional metric** as cross-check.

The table:

| Source | F_causal | F_spatial | Fidelity-A (HuMID) | Fidelity-B (distributional divergence, lower=better) |
|---|---|---|---|---|
| Raw | reference | reference | high anchor | 0 (self) |
| FAM-AIL edited | (have ~0.818) | (have) | reuse edit-time pairing | compute |
| BC-generated | compute | compute | compute | compute |
| GAN-generated | compute | compute | compute | compute |

**Success criteria:** a committed paper-ready `LEVEL1_RESULTS.md` with the filled table, a persisted per-run artifact dir, and a validation-gate verdict that determines whether Fidelity-A is trustworthy. The headline question — *is FAM-AIL edited data higher quality than raw / BC-generated / GAN-generated?* — is answered with numbers.

---

## 2. Scope

### In scope
- Two fidelity metrics (A: HuMID paired, gated; B: discriminator-free distributional) on generated/edited/raw trajectory data.
- A validation gate that tests whether the HuMID discriminator behaves sanely on the (out-of-distribution) reduced inputs before its numbers are trusted.
- Full-trajectory generation capture (the existing generation path keeps only terminal cells).
- A Level-1 table orchestrator CLI + persisted artifacts + canonical results doc.

### Out of scope
- Level-2 usability experiments (BC-on-edited vs BC-on-raw downstream performance) — a separate spec.
- Multi-seed fidelity (v1 uses a single representative seed per generated source; fairness columns retain their existing 5-seed numbers from the variance suite). Multi-seed fidelity is a documented future upgrade.
- Any change to the trajectory-editing algorithm, F_causal/F_spatial formulas, or ε. The HuMID discriminator is reused **frozen, inference-only, read-only**.
- Training a new discriminator.

---

## 3. Locked design decisions

1. **Fidelity-A is gated.** The HuMID discriminator's numbers on generated data are trusted only if the validation gate (§6) passes; otherwise Fidelity-B becomes primary and Fidelity-A is reported but flagged untrusted.
2. **Fidelity-B is discriminator-free** (independent cross-check): distributional divergence over interpretable trajectory statistics, sharing zero machinery with the HuMID model.
3. **Single representative seed** for BC/GAN generation in v1 (~40 min total GPU). Rationale: fidelity is a coarse realism check, gated and cross-checked; multi-seed is a later upgrade.
4. **Time/day synthesis for generated trajectories:** per-step `time_bucket` = the generation context's time block (held constant); `day_index` = the paired real trajectory's `day_index`. (The generator does not model time/day; this is a documented fabrication, and it is precisely what the validation gate stress-tests.)
5. **Fidelity-B statistics:** trajectory **length**, mean **per-step displacement** (Euclidean distance between consecutive cells), **spatial coverage** (count of unique cells visited).
6. **Forward-only discriminator use.** Level-1 scoring needs no gradient, so calls run under `torch.no_grad()`; the editing-time cuDNN-backward workaround is unnecessary here.
7. **Coordinate convention:** the discriminator was trained on 1-indexed coords `[1-48, 1-90]`; pipeline trajectories are 0-indexed `[0-47, 0-89]`. Add +1 to x,y when building discriminator tensors (mirrors `fidelity/context.py` Decision 3).
8. **Reduced-mode discriminator call:** single seeking trajectory per branch (legacy 3D `[B, L, 4]`), `driving`/`profile` left `None` so the model uses its learned default embeddings. This is the OOD condition the gate validates.

---

## 4. Data sources

| Source | Trajectory provenance | Full trajectories available? | Cost |
|---|---|---|---|
| **Raw** | `bundle.trajectories` | Yes (native `Trajectory` objects with driver_id + [x,y,time,day] states) | free |
| **FAM-AIL edited** | `<edit_dir>/histories.pkl` `.modified` | Yes (full `Trajectory` objects; pickup relocated) | free |
| **BC-generated** | re-train one B0 (MLE-only, 20 epochs), capture full rollouts | via new `generate_trajectories` | ~8 min |
| **GAN-generated** | one WGAN run (`--gan-loss wgan-gp`), capture full rollouts | via new `generate_trajectories` | ~32 min |

**Pairing for Fidelity-A:** generation iterates over contexts derived from the training trajectory list (after the `max_tokens` filter). The i-th generated trajectory pairs with the i-th surviving training trajectory (the real trajectory whose `(start_cell, time_block)` seeded it). `generate_trajectories` must preserve this index alignment so the orchestrator can recover `(real_i, gen_i)` pairs.

**Edited-data Fidelity-A** uses the native `(original_i, modified_i)` pairing from `histories.pkl` — the same pairing F_fidelity used during editing.

**Fairness columns (F_causal, F_spatial)** are recomputed per source for internal consistency: each source's demand grid → `data_level_fairness`. Raw = corpus; edited = the edit after-grid (or recomputed from edited trajectories' terminal cells); BC/GAN = `pickups_to_pickup_3d` over the generated trajectories' terminal cells. (The variance-suite 5-seed mean±std numbers remain the authoritative fairness figures cited in the paper; this table's single-seed fairness cells are for table-internal coherence and must be reported as single-seed.)

---

## 5. Fidelity-A — HuMID paired (primary, gated)

**Definition.** For a set of `(real, gen)` trajectory pairs, Fidelity-A = mean over pairs of `discriminator(real, gen)` (same-agent probability ∈ [0,1]). Higher = the generated trajectory reads as more realistic / same-source as its real counterpart.

**Inputs to the discriminator.** Each trajectory → `[L, 4]` tensor of `(x_grid+1, y_grid+1, time_bucket, day_index)`:
- Real trajectory: from `Trajectory` states.
- Generated trajectory: from the captured cell sequence (cells → x,y), with synthesized time/day per §3.4.

**Call.** Reduced mode (§3.8): `discriminator(x1=real_batch, x2=gen_batch)` with driving/profile `None`, under `torch.no_grad()`. Batched over pairs (batch size configurable). Returns per-pair probability; Fidelity-A = mean (+ std, n for traceability).

**Per source:**
- Raw: high anchor — `discriminator(real_i, real_i)` self-pairs (or same-driver pairs); expected high.
- Edited: `discriminator(original_i, modified_i)` over histories pairs.
- BC-generated / GAN-generated: `discriminator(real_i, gen_i)` over context-aligned pairs.

**Gating.** Fidelity-A values are emitted always, but the orchestrator marks them `trusted: true|false` based on the validation gate (§6). If untrusted, the results doc leads with Fidelity-B.

---

## 6. Validation gate (runs first; gates all Fidelity-A numbers)

**Purpose.** The discriminator was trained on real, multi-stream, driver-keyed data; we feed it single, partially-fabricated, identity-less trajectories. The gate tests whether it still ranks real above garbage in this reduced regime.

**Categories scored** (mean same-agent probability over a sample of pairs):
- `high_real_real` — real_i vs real_i (or same-driver real); **expect high**.
- `low_collapsed` — real_i vs a collapsed GAN trajectory (the len-~52 WGAN output); **expect low**.
- `low_shuffled` — real_i vs a length-matched random/shuffled trajectory; **expect low**.

**Pass criterion.** Gate passes iff `high_real_real − max(low_collapsed, low_shuffled) ≥ MARGIN` (default `MARGIN = 0.2`, a module constant) AND the ordering `high_real_real > both lows` holds. The numeric margin and all three category means are persisted regardless of pass/fail.

**On fail.** Do not abort. Mark Fidelity-A `trusted: false`; the orchestrator and results doc lead with Fidelity-B and explicitly note the discriminator was not validated for this OOD use.

---

## 7. Fidelity-B — discriminator-free distributional (independent cross-check)

**Definition.** For each statistic in {length, mean per-step displacement, spatial coverage}, build a normalized histogram of that statistic over a source's trajectories and over the raw trajectories, then compute the Jensen-Shannon divergence (bits, reusing `baselines/transmission.jensen_shannon_divergence`) between the source histogram and the raw histogram. Fidelity-B per source = {per-statistic JS, aggregate = mean of the three}. **Lower = more faithful** (raw vs raw = 0).

**Statistics (per trajectory):**
- `length` = number of cells (steps).
- `mean_displacement` = mean Euclidean distance between consecutive `(x,y)` cells (0 for length<2).
- `coverage` = count of unique `(x,y)` cells visited.

**Binning.** Each statistic histogrammed over a fixed shared bin grid (computed from the pooled raw+all-sources range, fixed bin count, e.g. 50 bins) so histograms are comparable. Bin spec is a module constant.

**Why this catches the failure the metric exists for:** a collapsed GAN (length ~52 vs real ~18) yields a length histogram far from raw → high length-JS → low fidelity, independent of any discriminator.

---

## 8. Components (all under `famail_temporal/baselines/`; no editing-algorithm changes)

### 8.1 `baselines/fidelity_eval.py` (new)
Pure-ish evaluation functions (take trajectory data + a discriminator; no training, no global state):
- `real_to_disc_tensor(traj) -> Tensor[L,4]` — `Trajectory` → discriminator input (+1 coords).
- `generated_to_disc_tensor(cells, time_bucket, day_index) -> Tensor[L,4]` — generated cell list → discriminator input (+1 coords, synthesized time/day).
- `humid_paired_fidelity(discriminator, pairs, *, batch_size, device) -> dict{mean, std, n}` — batched forward under `no_grad`, mean same-agent probability. `pairs` = list of `(left_tensor, right_tensor)`.
- `trajectory_statistics(traj_or_cells) -> dict{length, mean_displacement, coverage}` — accepts a `Trajectory` or a cell list.
- `distributional_fidelity(source_stats, raw_stats, *, bins) -> dict{per_stat: {length, mean_displacement, coverage}, aggregate}` — histogram + JS per statistic; lower=better.
- `validation_gate(discriminator, *, real_pairs, collapsed_pairs, shuffled_pairs, batch_size, device, margin) -> dict{high_real_real, low_collapsed, low_shuffled, margin, passed}`.

### 8.2 `baselines/gan/rollout.py` (modify — additive)
- `generate_trajectories(model, contexts, *, max_len, device, gen_batch_size, temperature=1.0, progress=False) -> List[List[int]]` — one full cell sequence per context, index-aligned with `contexts`. Reuses the existing autoregressive decode (`sample_trajectory_cells` logic or a batched equivalent). Existing `generate_pickups` / `sample_terminal_cells_batched` unchanged.

### 8.3 `baselines/run_level1_table.py` (new)
Orchestrator CLI. Flags: `--edit-dir` (default canonical no-dedup k-10000), `--mle-epochs` (default 20), `--adv-epochs` (default 3, applies to the GAN source ONLY; BC is always MLE-only / `adv_epochs=0`), `--gan-loss` (default `wgan-gp`), `--n-critic` (default 5, the critic-heavy WGAN schedule), `--fidelity-sample-size` (default 5000), `--seed` (default 0, shared by both generated sources in single-seed v1), `--max-tokens`, `--gen-batch-size`, `--device`, `--out-dir`, `--quiet`. **BC source is strictly MLE-only (B0); it never receives adversarial fine-tuning.**
Flow:
1. `DataBundle.load()`; load edited trajectories (`load_edited_trajectories`) + histories pairs.
2. Build the four source trajectory lists (raw, edited; BC + GAN via re-train + `generate_trajectories`).
3. Per source: terminal-cell demand grid → `data_level_fairness` → F_causal, F_spatial (single-seed; labelled as such).
4. `load_discriminator()`; build the validation-gate pair samples (real-real, real-collapsed [from the GAN source], real-shuffled); run `validation_gate`.
5. Per source: `humid_paired_fidelity` (Fidelity-A) on the source's pairs; `trajectory_statistics` + `distributional_fidelity` (Fidelity-B) vs raw.
6. Assemble the table; mark Fidelity-A trusted/untrusted from the gate; persist.

### 8.4 Persistence
- Per-run dir `famail_temporal/results/level1_table/<YYYY-MM-DDTHH-MM-SS>/` (gitignored): `level1_metrics.json` (all numbers incl. gate), `level1_table.md` (rendered table + gate verdict), `trajectory_stats.npz` (per-source statistic arrays).
- Canonical `famail_temporal/baselines/LEVEL1_RESULTS.md` (tracked; paper-ready table + interpretation + reproduction command), filled from the real run.

---

## 9. Data flow

```
bundle.trajectories ─────────────────────────► raw source
histories.pkl(.original/.modified) ──────────► edited source (+ native pairs)
re-train B0 ─► generate_trajectories ─► BC source (+ context pairs)
re-train WGAN ─► generate_trajectories ─► GAN source (+ context pairs, collapse sample)

each source ─► terminal cells ─► pickups_to_pickup_3d ─► data_level_fairness ─► F_causal, F_spatial

load_discriminator ─┬─► validation_gate(real-real, real-collapsed, real-shuffled) ─► trusted?
                    └─► humid_paired_fidelity(per-source pairs) ─► Fidelity-A (gated)

each source stats vs raw stats ─► distributional_fidelity ─► Fidelity-B

table assembler ─► level1_metrics.json + level1_table.md + LEVEL1_RESULTS.md
```

---

## 10. Error handling & edge cases

- **Discriminator checkpoint missing:** `load_discriminator()` raises specifically; orchestrator surfaces a clear message naming the expected path.
- **Validation gate fails:** non-fatal; Fidelity-A flagged `trusted:false`; Fidelity-B leads. The results doc states the gate failed and why it matters.
- **Empty generated trajectory** (rollout produced 0 in-vocab cells): excluded from pairing and from statistics; counted in a `n_empty` diagnostic.
- **length < 2** for displacement: `mean_displacement = 0.0`.
- **max_tokens filter alignment:** `generate_trajectories` and the real-pair recovery must use the identical filtered ordering used in training/generation, or pairing is silently wrong — this must be explicit in the implementation and covered by a test.
- **0-indexed vs 1-indexed:** every discriminator tensor builder applies +1; a test asserts this.
- **GAN collapse sample for the gate:** drawn from the actual GAN-generated source (the long trajectories), so the gate tests the real degraded case, not a synthetic stand-in.

---

## 11. Testing

Unit (no GPU, stub discriminator where needed):
- `distributional_fidelity`: identical stat distributions → JS≈0; disjoint → high; per-stat keys present; aggregate = mean.
- `trajectory_statistics`: a hand-built trajectory → known length, displacement, coverage; length<2 → displacement 0.
- `real_to_disc_tensor` / `generated_to_disc_tensor`: +1 coord conversion; generated time/day synthesis matches §3.4; shape `[L,4]`.
- `humid_paired_fidelity`: stub discriminator returning fixed probs → mean/std/n correct; batching across >1 batch correct.
- `validation_gate`: stub discriminator that scores same-length pairs high and length-mismatched low → gate passes with margin; a constant-output stub → gate fails. Verifies the pass/fail logic and the persisted fields.
- `generate_trajectories`: tiny synthetic model/contexts → one sequence per context, index-aligned, specials stripped, len ≤ max_len.
- `run_level1_table` pure helpers: table/JSON serialization round-trip (numpy-safe).

Real-data smoke (manual, GPU, ~40 min): full `run_level1_table` run on the canonical edit dir; inspect `level1_table.md` + gate verdict; confirm the four-source table is populated and FAM-AIL's causal-fairness standing is reported.

---

## 12. Open decisions — resolved

- Single representative seed (v1) — **yes** (§3.3).
- Time/day synthesis convention — **context block + paired real day** (§3.4).
- Fidelity-B statistics — **length, displacement, coverage** (§3.5).
- Gate margin — **0.2 default, module constant** (§6).
- All resolved; no TBDs remain.
