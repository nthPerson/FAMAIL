# Experiments Section + Trim+Lift Re-run Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the trim+lift re-run campaign (Q0–Q8) with maximal reproducibility collection, and draft `paper/sections/04_experiments.tex` (claims ladder 5.1–5.7) whose every number comes from trim+lift corpora.

**Architecture:** Two interleaved tracks. Track A: a committed, resumable campaign driver runs the GPU queue serially (per-item completion checkpoints), each run wrapped by a ledger helper that captures environment, checksums, and provenance; results are curated into `PAPER/` as they land. Track B: the section is drafted up-front with `% TODO(run:Q<n> → <artifact>)` markers; each completed stage triggers curation + slot-in behind the compile/lint gates. Q0 is a hard human checkpoint that gates the whole campaign.

**Tech Stack:** existing famail_temporal runners (verified entry points below), bash driver (α-sweep pattern), one new tested Python utility (`run_ledger.py`), LaTeX/acmart + `paper/lint.sh`.

**Spec:** `docs/superpowers/specs/2026-07-11-experiments-section-design.md` (approved 2026-07-11).

## Global Constraints

- **Ledger discipline: nothing runs without a ledger row.** Every campaign run is wrapped by
  `run_ledger.py start`/`finish` (Task 1) which records: queue id, exact command, git SHA,
  frozen-editor gate (`git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py`
  must be empty), config note, seeds, wall time, artifact dir, status — into
  `famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md` — and captures `environment.json` +
  SHA-256 checksums into the artifact dir.
- **Over-collect:** never delete per-seed JSONs, manifests, `timings.jsonl`, or logs; curation COPIES
  into `PAPER/`, never moves.
- **α overrides are mandatory on every editor run:** config defaults are `(0.33, 0.33, 0.34)` — all
  edit runs pass `--override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1`.
  `TAIL_LEN=4` / `LIFT_BUDGET=None` (trim+lift mode) are already config defaults.
- **City switch:** `FAMAIL_CITY=sf12` env prefix for all SF runs (re-homes cache/source/discriminator).
  Shenzhen = no env var.
- **Feature-set switch = a committed one-line edit** of `config.DEMOGRAPHIC_FEATURES`
  (`famail_temporal/config.py:70-74`) — never `--override` (it cannot build lists). The three sets:
  PRIMARY `["AvgHousingPricePerSqM","CompPerCapita","MigrantRatio"]`;
  HGC `["AvgHousingPricePerSqM","GDPperCapita","CompPerCapita"]`;
  4FEAT `["AvgHousingPricePerSqM","CompPerCapita","MigrantRatio","LogPopDensity"]`.
  Caches are feature-suffixed and already exist for all three sets (verified 2026-07-11). Every flip is
  its own commit (`paper-campaign: config -> <set>`); config must be back on PRIMARY at every task end
  unless the task says otherwise.
- **GPU serialization:** one GPU job at a time on the RTX 3070 eGPU. If CUDA disappears, suspect
  enclosure power (known gotcha), then resume the driver — every stage is skip-if-done.
- **Daemonization:** long runs use `nohup setsid ... >> <log> 2>&1 &` with `PYTHONUNBUFFERED=1`.
- **Paper gates after every Track-B task:**
  `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
  (both exit 0). Trim-only numbers (0.0144/0.0128/0.0139 family) appear ONLY in subsection 5.3 with
  `% lint-allow: ablation`. All `paper/README.md` conventions apply (provenance `% src:` comments,
  supply-tier labels, p=0.031 pairing, SF reproduces-not-beats, no "54%").
- **Slot markers:** a pending number is written as `X.XXXX% TODO(run:Q<n> -> <exact artifact path>)`
  — the lint script does not flag TODO; the final audit (Task 18) requires zero remaining `TODO(run:`.
- **Commit after every task**; campaign stages also commit their curated `PAPER/` artifacts.

## Verified entry points (evidence-checked 2026-07-11; do not improvise flags)

| family | command skeleton |
|---|---|
| Weighted-BC sweep | `python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir <filtered> --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir <out>` |
| L1v2 four-source | `python -m famail_temporal.baselines.run_level1_table_v2 --edit-dir <filtered> --seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto --out-dir <out>` |
| Variance suite | `python -m famail_temporal.baselines.run_variance_suite --edit-dir <filtered> --seeds 0,1,2,3,4 --out-dir <out>` |
| Editor (alt sets) | `python -m famail_temporal.evaluation.runner -k 10000 --name <name> --device auto --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1` |
| Infeasible-trim filter | `python -m famail_temporal.analysis.filter_infeasible_trims --edit-dir <run dir>` → `<run dir>_filtered` |
| External metrics | `python -m famail_temporal.baselines.run_external_fairness --edit-dir <dir> --dataset <label> --bootstrap 1000 --seed 0` (add `--delta-supply <npz>` for supply-lift dirs per its CLI) |
| Perturbation arms | `python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir <headline filtered> --mode {ifgsm,fgsm,random} --seed 0 --device auto --score-fidelity` (+ `--no-random-start` ablation for ifgsm/fgsm) |
| Tier-2 recount | `python -m famail_temporal.analysis.supply_recount --edit-dir <arm> --city shenzhen --persist-grids` |
| Comparison table | `python -m famail_temporal.baselines.assemble_baseline_table --arm-dirs <arm dirs...> --famail-json <stub> --raw-json <stub> --out famail_temporal/baselines/baseline_table` |
| Filter@K Pareto | `python -m famail_temporal.baselines.run_data_pareto --edit-from-dir <filtered> --out-dir <out>` |
| α summary | `python -m famail_temporal.analysis.alpha_sweep_summary` |

Headline filtered dirs: SZ `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered`,
SF `famail_temporal/results/2026-07-08T22-43-06_supply_lift_v1_sf12_filtered`.

## File Structure

```
famail_temporal/analysis/run_ledger.py            NEW — ledger/env/checksum helper (tested)
famail_temporal/analysis/tests/test_run_ledger.py NEW
famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md NEW — the committed ledger (force-add; results/ is gitignored)
famail_temporal/results/experiments_campaign/driver.sh  NEW — committed campaign driver (force-add)
famail_temporal/baselines/famail_headline_stub.json     NEW — hand-authored 6-row-table stub
famail_temporal/baselines/raw_stub.json                 NEW — hand-authored 6-row-table stub
paper/sections/04_experiments.tex                 THE Track-B deliverable
PAPER/REPRODUCIBILITY.md                          NEW — capstone claims→artifacts→ledger map
PAPER/supply-lift/{data,by_feature_set,tables}/…  curated additions per stage
PAPER/baselines/comparison/…                      6-row table + provenance
PAPER/objective-motivation/weight-sensitivity/…   α bundle + DECISION.md
```

---

### Task 1: Reproducibility infrastructure (`run_ledger.py` + ledger + backfill)

**Files:**
- Create: `famail_temporal/analysis/run_ledger.py`, `famail_temporal/analysis/tests/test_run_ledger.py`,
  `famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md`

**Interfaces:**
- Produces: CLI `python -m famail_temporal.analysis.run_ledger start|finish|env` used by every later
  stage. `start --queue-id Q<n> --cmd '<cmd>' --artifact-dir <dir> [--config-note '<txt>'] [--ledger <path>]`
  appends a `LAUNCHED` row (timestamp, git SHA, frozen-gate PASS/FAIL, cmd, config note, artifact dir) and
  writes `<dir>/environment.json`. `finish --queue-id Q<n> --artifact-dir <dir> [--ledger <path>]` flips the
  row to `DONE` with end-time + wall-time and appends SHA-256 checksums of `<dir>/*.json` + `<dir>/*.npz`
  to `<dir>/PROVENANCE.md` (created if absent).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the campaign run-ledger helper."""
import json, re, subprocess, sys
from pathlib import Path
from famail_temporal.analysis import run_ledger as rl


def test_start_appends_launched_row_and_environment(tmp_path):
    ledger = tmp_path / "LEDGER.md"; art = tmp_path / "art"; art.mkdir()
    rc = rl.main(["start", "--queue-id", "Q9", "--cmd", "echo hi",
                  "--artifact-dir", str(art), "--config-note", "PRIMARY",
                  "--ledger", str(ledger)])
    assert rc == 0
    text = ledger.read_text()
    assert "Q9" in text and "LAUNCHED" in text and "echo hi" in text and "PRIMARY" in text
    env = json.loads((art / "environment.json").read_text())
    assert "python" in env and "torch" in env and "pip_freeze_sha256" in env


def test_finish_flips_row_and_writes_checksums(tmp_path):
    ledger = tmp_path / "LEDGER.md"; art = tmp_path / "art"; art.mkdir()
    (art / "metrics.json").write_text('{"a": 1}')
    rl.main(["start", "--queue-id", "Q9", "--cmd", "x", "--artifact-dir",
             str(art), "--ledger", str(ledger)])
    rc = rl.main(["finish", "--queue-id", "Q9", "--artifact-dir", str(art),
                  "--ledger", str(ledger)])
    assert rc == 0
    assert "DONE" in ledger.read_text()
    prov = (art / "PROVENANCE.md").read_text()
    assert "metrics.json" in prov and re.search(r"[0-9a-f]{64}", prov)


def test_frozen_gate_recorded(tmp_path):
    ledger = tmp_path / "LEDGER.md"; art = tmp_path / "art"; art.mkdir()
    rl.main(["start", "--queue-id", "Q9", "--cmd", "x", "--artifact-dir",
             str(art), "--ledger", str(ledger)])
    assert re.search(r"frozen-gate:(PASS|FAIL)", ledger.read_text())
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest famail_temporal/analysis/tests/test_run_ledger.py -q`
Expected: FAIL / ImportError (`run_ledger` does not exist).

- [ ] **Step 3: Implement `run_ledger.py` (minimal)**

Module with: `_git_sha()`, `_frozen_gate()` (runs the git diff above, returns "PASS" iff empty output),
`_environment()` (python version, `torch.__version__`, `torch.version.cuda`,
`torch.cuda.get_device_name(0)` guarded by availability, and the sha256 of `pip freeze` output stored
as `pip_freeze_sha256` plus the full freeze text as `pip_freeze`), `start`/`finish` subcommands
appending/patching markdown rows keyed by queue-id (row format:
`| Q<id> | <status> | <start> | <end> | <wall> | <sha> | frozen-gate:<r> | <config note> | <artifact dir> | <cmd> |`),
`finish` computing wall time from the row's start timestamp and appending a `## Checksums (<utc>)`
section of `sha256sum` lines to `<dir>/PROVENANCE.md`. Default `--ledger` =
`famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest famail_temporal/analysis/tests/test_run_ledger.py -q` → 3 passed.
Also: `python -m pytest famail_temporal/analysis/ -q` → no regressions.

- [ ] **Step 5: Create the committed ledger skeleton + BACKFILL rows**

Create `famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md` with a header explaining purpose/columns, the
row table, and a **Backfill** section: one `DONE (backfilled)` row each for the already-run artifacts the
paper will cite — the SZ + SF supply-lift headline runs (+ their `_filtered` derivations), the SZ
weighted-BC sweep, the 9 demographic-oversampling arms, and the 5 α-sweep points + anchor — each with its
recorded command (from the manifests/PROVENANCE cited in `PAPER/supply-lift/data_provenance.md` and
`famail_temporal/baselines/STATUS.md`) and artifact dir. Mark backfilled rows explicitly (environment
captured post hoc where recoverable from manifests; noted as such).

- [ ] **Step 6: Commit (force-add the ledger — results/ is gitignored)**

```bash
cd /home/robert/FAMAIL
git add famail_temporal/analysis/run_ledger.py famail_temporal/analysis/tests/test_run_ledger.py
git add -f famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md
git commit -m "campaign: run-ledger helper (env capture, checksums, frozen-gate) + backfilled ledger"
```

---

### Task 2: Campaign driver script

**Files:**
- Create: `famail_temporal/results/experiments_campaign/driver.sh` (committed via force-add)

**Interfaces:**
- Consumes: `run_ledger.py` (Task 1); the verified entry-point commands table above.
- Produces: `bash driver.sh <stage>` (stages: `q1 q2 q3 q4 q5 q6a q6b q7 q8a q8b`), `bash driver.sh --status`.
  Later tasks launch stages through this driver ONLY.

- [ ] **Step 1: Write the driver**

Bash script modeled on `famail_temporal/results/alpha_sweep/driver.sh`: `set -euo pipefail`;
`PYTHONUNBUFFERED=1`; a `DONE_MARKER` convention (each stage's final artifact path checked for
skip-if-done); each stage = `run_ledger start` → the exact command(s) from the entry-point table with
the stage's concrete dirs (Tasks 11–16 give each stage's full command block — the driver contains them
verbatim) → `run_ledger finish`. `--status` prints per-stage DONE/PENDING by checking the marker
artifacts. Stages q6a/q6b REFUSE to run unless `grep -q 'GDPperCapita' famail_temporal/config.py` /
`grep -q 'LogPopDensity' ...` respectively (config-flip guard), and q8a/q8b likewise — the flips
themselves are Task-15/16 steps, not driver actions.

- [ ] **Step 2: Verify statically**

Run: `bash -n famail_temporal/results/experiments_campaign/driver.sh && bash famail_temporal/results/experiments_campaign/driver.sh --status`
Expected: syntax OK; all stages PENDING.

- [ ] **Step 3: Commit**

```bash
git add -f famail_temporal/results/experiments_campaign/driver.sh
git commit -m "campaign: committed resumable driver (stages q1-q8b, skip-if-done, --status)"
```

---

### Task 3: Q0 — the α-sweep HARD CHECKPOINT (gates the campaign)

**Files:**
- Create: `PAPER/objective-motivation/weight-sensitivity/{README.md,DECISION.md,alpha_sweep_summary.md,alpha_pareto.png,alpha_sweep_summary.json}`
- Modify: `PAPER/objective-motivation/MOTIVATION.md` ("Why these weights" subsection),
  `paper/sections/03_methodology.tex` (the `% TODO(alpha-sweep)` block)

- [ ] **Step 1: Confirm s80 completion** — `bash famail_temporal/results/alpha_sweep/driver.sh --status`
  shows all 5 DONE (if PENDING, stop and wait; do not proceed).
- [ ] **Step 2: Run the summary** — `python -m famail_temporal.analysis.alpha_sweep_summary`
  → writes md/png/json to `famail_temporal/results/alpha_sweep/summary/`; expect 6 rows, 0 pending.
  Ledger: `run_ledger start/finish --queue-id Q0` around it (artifact dir = the summary dir).
- [ ] **Step 3: Write the weight-decision memo** `DECISION.md`: the 6-point table; flatness assessment
  (span of ΔF_causal across points; partial finding was +0.0217..+0.0226); which point the criterion
  (max ΔF_causal s.t. ΔF_spatial ≥ 0) selects and the margin vs shipped; ΔF_spatial trend; a
  recommendation with reasoning. Copy summary md/png/json into the bundle dir.
- [ ] **Step 4: HARD STOP — present the memo to Robert for the keep-vs-re-anchor decision.**
  Do NOT launch any campaign stage before his answer.
  - **Keep:** proceed to Step 5.
  - **Re-anchor to α\*:** append the re-anchor addendum to the ledger; re-run the SZ + SF headline
    editors at α\* (runner command from the entry-point table with the new overrides, then
    `filter_infeasible_trims` on each), update every `<filtered>` path in Tasks 11–16 and the drafted
    sections' numbers, update methodology §3.2's stated weights, and only then continue.
- [ ] **Step 5 (keep-branch): fold in** — rewrite `MOTIVATION.md` "Why these weights" final paragraph to
  cite the empirical 6-point frontier + the trim+lift +0.0222 headline (replacing the trim-only +0.0128
  sentence); replace `03_methodology.tex`'s `% TODO(alpha-sweep)` block with the final frontier sentence
  (state flatness explicitly; keep the sensitivity-analysis framing; `% src:` the summary artifacts).
  Gates: paper compile + lint.
- [ ] **Step 6: Commit** — `git add PAPER/objective-motivation/ paper/ famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md && git commit -m "campaign(Q0): alpha-sweep checkpoint — memo, weight-sensitivity bundle, MOTIVATION + methodology fold-in"`

---

### Task 4: Draft 5.1 Experimental Setup

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (all; ~0.5 page): two cities + one-line stuck-GPS cleanup (`% src: PAPER/argument/02_datasets.md`);
editor config (α=(0.2,0.7,0.1), ε=2, k=10,000 SZ / 2,000 SF, budget splits 2,455+7,545 / 1,324+629
post-filter, `% src: PAPER/supply-lift/FINDINGS.md §2, LIFT_ALGORITHM_REFERENCE.md §4.3`); paired-seed
protocol + n=6 Wilcoxon floor + n=5 convention (`% src: PAPER/argument/04_evaluation.md`); unit-level
paired bootstrap B=1000 + first-order caveat (`% src: PAPER/external-metrics/FINDINGS.md §1,§5`);
the three-ring metric firewall paragraph (`% src: LIFT_ALGORITHM_REFERENCE.md §13`); external-metric
definitions — group levels, DP gap, DI ratio, between-region Theil — with the DP≡gap disclosure
(`% src: PAPER/external-metrics/FINDINGS.md §4.3`).

Steps: write → gates → self-check against list → commit `paper: experiments 5.1 setup`.

---

### Task 5: Draft 5.3 Trim-only vs trim+lift ablation

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (~0.4 page + 1 table; the ONLY home of trim-only numbers, every such line marked
`% lint-allow: ablation`): table with rows {ΔF_causal SZ: +0.0144 → +0.0222; ΔF_causal SF: +0.0139 →
+0.0328; ΔF_spatial SZ: −0.0009 → +0.0064; mean(Y|D) SZ: +0.000 (flat 7.0734) → +0.0468 CI-sig;
SF migrant DP: n.s. → −0.0758 CI-sig} (`% src: PAPER/supply-lift/FINDINGS.md §3,§5.1;
PAPER/external-metrics/FINDINGS.md §2-4; PAPER/argument/05_results_shenzhen.md §1;
PAPER/argument/06_results_sf.md §1`); prose: what the ablation certifies (every delta attributable to the
lift mechanism — the two-phase control from methodology §3.5); the leveling-down evidence line
(trim-only closes the gap with the under-served level flat) as the ablation's qualitative row.
No slot markers needed — all numbers exist.

Steps: write → gates → self-check → commit `paper: experiments 5.3 trim vs trim+lift ablation`.

---

### Task 6: Draft 5.2 Data-level fairness (Shenzhen)

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (~0.7 page + 2 tables): internal deltas (F_causal 0.7988→0.8210, F_spatial
0.1034→0.1098; `% src: PAPER/supply-lift/data/shz_primary_filtered_metrics.json`); external-metrics
table (migrant/extremes DI 0.3325→0.3480 +0.0155 CI [0.0128,0.0182]; DP 14.1989→13.3412 −0.8576;
Theil 0.1550→0.1468 −0.0082; group levels disadvantaged 7.0734→7.1203 / advantaged 21.2723→20.4615;
`% src: PAPER/supply-lift/tables/shenzhen-primary-filtered.md`); mean(Y|D) +0.0468 CI [+0.0022,+0.0932]
labeled **design-targeted** (ring 2); the channel-decomposition table (supply +0.0091 tier-1
[+0.0054,+0.0130] / +0.0242 tier-2 [+0.0208,+0.0279], both significant; demand +0.0378 n.s.; tier labels
mandatory; `% src: PAPER/supply-lift/data/shz_primary_filtered_channel_decomposition.json` +
`FINDINGS.md §4`); tier-2 recount validity one-liner (reproduces production grid exactly, MAE 0.0,
100% history matching; `% src: FINDINGS.md §4 / LIFT_ALGORITHM_REFERENCE.md §8 G2`); fidelity stability
(Fid-A 0.8457 vs raw 0.8489; lift-mode −0.0031 ≤ trim-mode −0.0059) + lift's Fidelity-B cost (0.2645 vs
trim 0.1601) disclosed as by-design (`% src: FINDINGS.md §6.1`); skip-on-infeasible (115/2,455 reverted,
rule-first, favorable-direction disclosure; `% src: FINDINGS.md §8`); oracle G0 (threshold +0.3;
supply-only ceiling +0.786 = 2.6×; full +0.882 — fulfills §3.4's promise; `% src:
PAPER/supply-lift/data/oracle.json / LIFT_ALGORITHM_REFERENCE.md §2`).

Steps: write → gates → self-check → commit `paper: experiments 5.2 data-level fairness (SZ)`.

---

### Task 7: Draft 5.5 Baselines

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (~0.6 page + the 6-row table skeleton): framing sentence (fidelity/editing-quality
baselines, not competing fairness methods — fairness expected NOT to improve under perturbation arms;
`% src: PAPER/baselines/README.md`); the naming discipline (arms are "iFGSM/FGSM with random restart";
vanilla-δ=0 is a provable no-op, shown as an ablation row; `% src: baselines/STATUS.md paper-facing
notes`); demographic oversampling (targeted +0.0153 dose-monotone +0.0059/+0.0097/+0.0153; placebo
−0.0172; DP explosion +2.8 with the mechanism sentence (fabricated supply lands in advantaged cells);
pool exhaustion 8,241 < 10,000 → 1,759 with-replacement; 10.5% inflation vs FAMAIL 0%; fidelity
not-scored-by-construction disclosure; `% src: PAPER/baselines/demographic-oversampling/FINDINGS.md`);
the 6-row cross-arm table with perturbation-arm cells as slot markers
`% TODO(run:Q1 -> famail_temporal/baselines/baseline_table/…)`.

Steps: write → gates → self-check → commit `paper: experiments 5.5 baselines (oversampling live, perturbation slotted)`.

---

### Task 8: Draft 5.4 Downstream propagation (Shenzhen)

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (~0.7 page + 2 tables): L1 four-source table SKELETON with every cell a
`% TODO(run:Q3 -> famail_temporal/results/level1_table_v2/<supply-lift out>/level1_v2_multiseed.json)`
marker (the re-emitted-generators run) + prose that does not depend on exact values (four sources, gate
construction, what disqualification means); vanilla-BC null (w=1: +0.0023, p=.156 n.s.;
`% src: PAPER/supply-lift/data/weighted_bc_paired_stats.json / FINDINGS.md §7`); weighted-BC
dose-response table (+0.0232/+0.0280/+0.0310, all 6/6 p=.031 with mean Δ + t-CI framing per convention)
+ **the new F_spatial propagation** (+0.0042/+0.0048/+0.0057, 6/6) + controls (random −0.0011/−0.0027
n.s.; most-fair +0.0034/+0.0014/+0.0007 n.s.) (`% src: FINDINGS.md §7`); model-level variance slot
`% TODO(run:Q5 -> famail_temporal/results/variance_suite/<supply-lift out>/aggregate.json)`; the
rollout-allocation boundary paragraph — drain −0.0029 @w30 vs trim-era −0.0048, attenuated ~40% NOT
reversed, 0/6 seeds, seeking-share n.s.; claim levels 1–2, disclose level 3; motivates training-side
future work (`% src: PAPER/supply-lift/data/rollout_supplylift_summary.json / FINDINGS.md §9`;
the trim-era −0.0048 line marked `% lint-allow: ablation` context).

Steps: write → gates → self-check → commit `paper: experiments 5.4 downstream propagation (SZ)`.

---

### Task 9: Draft 5.6 Robustness & sensitivity

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (~0.4 page + 1 table + 1 figure): the α-Pareto frontier figure
(`\includegraphics` of `figures/alpha_pareto.png` copied from the Q0 bundle) + the frontier/criterion
sentence consistent with Task 3's fold-in (`% src: PAPER/objective-motivation/weight-sensitivity/`);
the per-set matrix table SKELETON — rows: before-edit F_causal, editor Δ, external DI/DP/Theil Δ,
L1 verdict, vanilla null, WBC w30, variance — columns PRIMARY/HGC/4FEAT, with PRIMARY column live from
existing artifacts and both alternate columns slot markers `% TODO(run:Q6-Q8 -> …per-set artifact…)`;
filter@K Pareto slot `% TODO(run:Q7 -> famail_temporal/results/analysis/pareto_supplylift/…)` + one
sentence (editing beats filtering on the F_causal objective).

Steps: write (copy `alpha_pareto.png` into `paper/figures/`) → gates → self-check → commit
`paper: experiments 5.6 robustness + alpha frontier`.

---

### Task 10: Draft 5.7 External validity — San Francisco

**Files:** Modify: `paper/sections/04_experiments.tex`

Required content (~0.7 page + 1 table): dual claim (F_causal 0.8752→0.9079 +0.0328; F_spatial +0.0180;
Fid-A 0.9581 ≈ raw 0.9578; `% src: PAPER/supply-lift/data/sf12_filtered_metrics.json / FINDINGS.md §5-6`);
external metrics (Theil −0.0081 CI-sig; migrant DP −0.0758 CI-sig **under district-extremes** + DI
+0.0061 — with the median-split n.s. caveat stated; `% src: PAPER/supply-lift/tables/sf12-filtered.md /
FINDINGS.md §5.1`); the supply channel (+0.0195 CI-sig, explicitly **tier-1-labeled lower bound**; tier-2
recount not plumbed for SF — disclosure; `% src: sf12_filtered_channel_decomposition.json / FINDINGS §10`);
**the mean(Y|D) tension presented as BOTH readings** (total −0.0330 CI-sig negative because lift routes
demand INTO under-served cells vs every external metric improving — the demand-endogeneity connection to
§3.4) closed with `% TODO(PI-framing): Zhang decision — which SF fairness story leads; both readings kept
until then` (`% src: FINDINGS.md §5.2`); SF L1 table slot `% TODO(run:Q4 -> …)`; SF weighted-BC slot
`% TODO(run:Q2 -> …)`; SF variance slot `% TODO(run:Q5 -> …)`; the SF caveat block (ACS proxies not
hukou; 12-driver subsample; 14.9% raw adjacency violations — edited corpus MORE compliant than raw;
GAN-collapse divergence to be re-checked against Q4's fresh run; `% src: PAPER/argument/07_limitations.md
§5-7, FINDINGS.md §10`).

Steps: write → gates → self-check → commit `paper: experiments 5.7 SF external validity (tension surfaced)`.

---

### Task 11: Stage Q1 — perturbation arms + 6-row comparison table + slot 5.5

**Files:**
- Create: `famail_temporal/baselines/famail_headline_stub.json`, `famail_temporal/baselines/raw_stub.json`,
  `PAPER/baselines/comparison/{README.md,comparison_table.md}` (+ copied JSONs)
- Modify: `paper/sections/04_experiments.tex` (5.5 slots), driver stage q1 already contains commands

- [ ] **Step 1: Author the two stubs** per the schema documented at the top of
  `famail_temporal/baselines/assemble_baseline_table.py` (`arm.mode`, `arm.n_edited`,
  `fairness.{f_causal,f_spatial}_{before,after}`): `raw_stub.json` = before-values only (F_causal
  0.79880 both before/after? No — raw row: before=after=0.79880 / 0.10343, n_edited 0);
  `famail_headline_stub.json` = mode "famail-trim+lift", n_edited 9,885 (2,340+7,545), f_causal
  0.79880→0.82101, f_spatial 0.10343→0.10978. `% values src: PAPER/supply-lift/FINDINGS.md §3` (JSON
  comment field `"_src"`).
- [ ] **Step 2: Launch** `bash famail_temporal/results/experiments_campaign/driver.sh q1` (daemonized) —
  stage runs: 3 arms (`--mode ifgsm/fgsm/random --seed 0 --device auto --score-fidelity`), the 2
  `--no-random-start` ablation arms, per-arm `run_external_fairness` + `supply_recount`, then
  `assemble_baseline_table` with `--arm-dirs` = the 3 new arms + the 3 targeted oversampling d-dose arms'
  s0 dirs (`famail_temporal/results/2026-07-10T*_baseline_demo_oversample_targeted_d10000_s0_shenzhen`)
  + stubs. All wrapped in ledger start/finish.
- [ ] **Step 3: Verify + curate** — table renders 6+ rows; perturbation arms show no fairness
  improvement (expected direction; if an arm IMPROVES fairness materially, stop and surface — do not
  bury); vanilla no-op rows ≈ 0 change. Copy table + arm metrics to `PAPER/baselines/comparison/` with
  README + provenance.
- [ ] **Step 4: Slot 5.5** — replace `TODO(run:Q1` markers with values + `% src:` comments; gates; commit
  `paper+campaign(Q1): perturbation arms + 6-row comparison table`.

---

### Task 12: Stage Q2 — SF weighted-BC sweep + slot 5.7

- [ ] **Step 1: Launch** `bash driver.sh q2` (daemonized): `FAMAIL_CITY=sf12 python -m
  famail_temporal.baselines.run_weighted_bc_smoke --edit-dir famail_temporal/results/2026-07-08T22-43-06_supply_lift_v1_sf12_filtered
  --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir
  famail_temporal/results/weighted_bc_sweep/supply_lift_v1_sf12_filtered_6seed` (ledger-wrapped).
- [ ] **Step 2: Verify + curate** — `paired_stats.json` has all 10 arms × 6 seeds; copy
  manifest/paired-stats/dose-response to `PAPER/supply-lift/data/` as `sf12_weighted_bc_*.json`; check
  qualitative expectations (edited arms recover; controls null-or-negative) — surface anomalies, don't fix.
- [ ] **Step 3: Slot 5.7** WBC cells; gates; commit `paper+campaign(Q2): SF weighted-BC under trim+lift`.

---

### Task 13: Stages Q3+Q4 — L1v2 re-runs (re-emitted generators), both cities + slot 5.4/5.7

- [ ] **Step 1: Launch q3** (SZ): `python -m famail_temporal.baselines.run_level1_table_v2 --edit-dir
  famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered --seeds 0,1,2,3,4
  --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto --out-dir
  famail_temporal/results/level1_table_v2/supply_lift_shz_5seed` (ledger-wrapped; generators retrained
  from scratch by construction — this run trains BC + WGAN-GP fresh).
- [ ] **Step 2: Launch q4** (SF): same module with `FAMAIL_CITY=sf12`, `--edit-dir <sf12 filtered>`,
  `--out-dir famail_temporal/results/level1_table_v2/supply_lift_sf12_5seed`.
- [ ] **Step 3: Verify + curate** — identity gate PASSED both cities (else stop + surface); note whether
  SZ GAN collapses again (Fidelity-B magnitude) and whether SF GAN stays healthy — the 5.7 divergence
  note updates to match THIS run's outcome, whatever it is. Copy `level1_v2_multiseed.json` +
  `level1_v2_table.md` per city to `PAPER/supply-lift/data/`.
- [ ] **Step 4: Slot** the 5.4 and 5.7 L1 tables; gates; commit
  `paper+campaign(Q3-Q4): L1v2 four-source tables under trim+lift (fresh generators)`.

---

### Task 14: Stage Q5 — variance suites, both cities + slot 5.4/5.7

- [ ] **Step 1: Launch q5**: SZ `python -m famail_temporal.baselines.run_variance_suite --edit-dir
  <shz filtered> --seeds 0,1,2,3,4 --out-dir famail_temporal/results/variance_suite/supply_lift_shz_5seed`;
  then SF with `FAMAIL_CITY=sf12` + sf12 dirs (ledger-wrapped, serial).
- [ ] **Step 2: Verify + curate** — expect nulls (within noise band); copy `aggregate.json` per city to
  `PAPER/supply-lift/data/` as `variance_supplylift_{shz,sf12}.json`.
- [ ] **Step 3: Slot** 5.4 + 5.7 variance cells (n=5 convention wording: effect-vs-noise); gates; commit
  `paper+campaign(Q5): model-level variance under trim+lift`.

---

### Task 15: Stage Q6+Q7 — alternate-set trim+lift edit runs + external metrics + Pareto

- [ ] **Step 1: Config flip → HGC** — edit `famail_temporal/config.py` `DEMOGRAPHIC_FEATURES` to
  `["AvgHousingPricePerSqM","GDPperCapita","CompPerCapita"]`; run
  `python -m pytest famail_temporal/data/ famail_temporal/fairness/ -q` (sanity); commit
  `paper-campaign: config -> housing-gdp-comp`.
- [ ] **Step 2: Launch q6a** — runner `-k 10000 --name supply_lift_v1_shz_hgc --device auto` + the three
  α overrides; then `filter_infeasible_trims --edit-dir <new dir>`; ledger-wrapped; ~8h.
- [ ] **Step 3: Config flip → 4FEAT** (add `"LogPopDensity"` to the PRIMARY list); commit; launch q6b
  (`--name supply_lift_v1_shz_4feat`) + filter; ~8h.
- [ ] **Step 4: q7 externals** — under the matching config per set:
  `run_external_fairness --edit-dir <set filtered> --dataset shenzhen-{hgc,4feat}-supplylift
  --bootstrap 1000 --seed 0` (+ `--delta-supply` per its CLI); plus the PRIMARY filter@K Pareto
  `run_data_pareto --edit-from-dir <shz PRIMARY filtered> --out-dir
  famail_temporal/results/analysis/pareto_supplylift` (config must be PRIMARY for this one — order it
  before the flips or after the flip-back).
- [ ] **Step 5: Flip config back → PRIMARY**; commit `paper-campaign: config -> PRIMARY (restore)`.
- [ ] **Step 6: Curate** — new `PAPER/supply-lift/by_feature_set/{housing-gdp-comp,housing-comp-migrant-logpopdensity}/`
  (metrics.json copies, external tables, PROVENANCE) + Pareto CSV to `PAPER/supply-lift/tables/`.
- [ ] **Step 7: Slot** 5.6 editor-Δ + external columns + filter@K sentence; gates; commit
  `paper+campaign(Q6-Q7): alternate-set trim+lift editors + externals + pareto`.

---

### Task 16: Stage Q8 — per-set downstream matrix + complete 5.6

- [ ] **Step 1: Config flip → HGC**; commit. Launch q8a: L1v2 (`--edit-dir <hgc filtered> --seeds
  0,1,2,3,4 …`), weighted-BC sweep (full 10-arm argv against `<hgc filtered>`), variance suite — serial,
  each ledger-wrapped (~1 day).
- [ ] **Step 2: Config flip → 4FEAT**; commit. Launch q8b: same block against `<4feat filtered>` (~1 day).
- [ ] **Step 3: Flip back → PRIMARY**; commit.
- [ ] **Step 4: Curate** into the per-set `PAPER/supply-lift/by_feature_set/` dirs; regenerate the
  cross-set comparison table (pattern of `PAPER/feature_selection/tables/comparison_across_sets.md`,
  new file `PAPER/supply-lift/tables/comparison_across_sets_supplylift.md`).
- [ ] **Step 5: Slot** the full 5.6 matrix; verify the directional story reproduces per set (surface any
  set where it does not — plainly, in the table); gates; commit
  `paper+campaign(Q8): per-set downstream matrix under trim+lift`.

---

### Task 17: REPRODUCIBILITY.md capstone + ledger cross-check

**Files:** Create: `PAPER/REPRODUCIBILITY.md`; Modify: `PAPER/README.md` (one pointer line)

- [ ] **Step 1: Cross-check** — enumerate every table/figure in `04_experiments.tex`; for each, verify
  the chain cell → `% src:` artifact → ledger row → command exists. Any gap = fix the ledger/backfill
  first.
- [ ] **Step 2: Write the capstone** — one section per Experiments table/figure: claim → curated
  `PAPER/` artifact path → raw artifact dir → ledger row id → exact command → environment note. Plus a
  preamble describing the ledger, checksum, and environment-capture discipline.
- [ ] **Step 3:** `grep -c 'TODO(run:' paper/sections/04_experiments.tex` → **0** expected (all slots
  filled). The `TODO(PI-framing)` marker MAY remain (it is Zhang's decision, not a missing number).
- [ ] **Step 4:** gates; commit `paper: REPRODUCIBILITY capstone + ledger cross-check`.

---

### Task 18: Final two-agent audit + fix wave

- [ ] **Step 1:** Dispatch the read-only **number/convention auditor** (same brief as the methodology
  round, extended to `04_experiments.tex`: every number → `% src:` file verification; conventions incl.
  tier labels, p=0.031 pairing, lint-allow scoping; `\cite`/`\ref` integrity) and the **FAMAIL-fidelity
  reviewer** (overclaim scan against FINDINGS/REFERENCE boundary list: SF tension presented-not-resolved;
  rollout boundary disclosed; oversampling result not inverted; firewall ring labels; ablation-only
  trim-only numbers), in parallel.
- [ ] **Step 2:** Apply the fix wave (record disputed-finding dispositions in the commit message).
- [ ] **Step 3:** Final gates + `grep -ci 'undefined' paper/main.log` → 0; commit
  `paper: experiments audit fix wave`.

---

## Execution notes

- **Interleaving model:** Tasks 4–10 (writing) proceed while stages run; the controller launches a stage,
  writes, and returns on completion notification. Stage order is fixed (Q1→Q2→Q3→Q4→Q5→Q6a→Q6b→Q7→Q8a→Q8b);
  writing order is 4,5,6,7,8,9,10.
- **Q0 gates everything:** no stage launches before Task 3's Robert decision.
- **Anomaly protocol:** campaign results that contradict expectations (a perturbation arm improving
  fairness; a failed identity gate; a non-null control) are SURFACED to Robert with the artifact, never
  smoothed over in prose — house norm.
- **Re-anchor contingency:** if Q0 chooses re-anchor, Task 3 Step 4's addendum path replaces every
  `<filtered>` reference in Tasks 11–16 with the new-α dirs; Tasks 5/6 numbers re-slot; the ablation
  gains an α column note. Everything else is unchanged.
