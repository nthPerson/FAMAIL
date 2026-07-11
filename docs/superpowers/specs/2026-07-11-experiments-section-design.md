# Design: Experiments section + trim+lift re-run campaign

**Date:** 2026-07-11 · **Status:** approved (Robert, 2026-07-11, with three amendments folded in below)
**Task provenance:** continuation of the paper build (spec `2026-07-10-methodology-section-design.md`);
fills `paper/sections/04_experiments.tex`. Deadlines: abstract to Zhang ~Jul 17, KDD abstract ≈ Jul 19,
paper Jul 26.

## Goal

Draft the full Experiments section AND execute the compute campaign that makes every reported number
come from the **trim+lift (supply-lift) editor's corpora** — no trim-era numbers outside the
trim-vs-trim+lift ablation. The two tracks are one deliverable: the section is structurally complete
early, with pending cells wired to queue items that slot in as runs land.

## Decisions locked during brainstorm (Robert, 2026-07-11)

1. **Era policy = FULL RE-RUN BILL.** Everything the paper reports is recomputed on trim+lift corpora.
2. **Feature-set robustness = full downstream matrix per set** (editor + external metrics + L1 + vanilla
   null + weighted-BC + variance for both alternate Shenzhen sets), days of GPU accepted; runs sequenced
   to maximize parallel progress (writing + CPU work interleave with the serial GPU queue).
   *(Correction recorded during brainstorm: the trim-era DID run the full per-set downstream matrix —
   `PAPER/feature_selection/tables/comparison_across_sets.md` — so the harness infrastructure exists
   and is proven; this campaign re-runs it on trim+lift corpora.)*
3. **Generator rows are RE-EMITTED, not reused.** The L1 bc-/gan-generated rows are retrained/re-emitted
   from scratch on both cities even though they are edit-independent — "no room for mistakes that could
   open us up for trouble with reviewers." The same re-emit-over-reuse preference applies anywhere else
   a reused artifact would otherwise appear.
4. **Section structure = claims ladder with SF as its own subsection** (5.1–5.7 below).
5. **Reproducibility discipline = maximal** ("loaves of bread, not breadcrumbs") — the artifact set in
   §Track-A below; cheap-to-collect items are collected by default; nothing that materially inflates
   runtimes.
6. **Q0 is a HARD CHECKPOINT:** when the α-sweep's final point lands, perform a careful, documented
   review of the full frontier BEFORE the campaign proceeds — if a stronger α-configuration emerges,
   the campaign re-anchors (see §Campaign-gate).

## Track A — the re-run campaign

### GPU queue (serial RTX 3070; criticality-ordered so writing unblocks earliest)

| # | run | est. | unblocks |
|---|---|---|---|
| **Q0** | α-sweep s80 (already running) → `alpha_sweep_summary` + **HARD CHECKPOINT review** (below) → MOTIVATION.md fold-in + §3.2 `TODO(alpha-sweep)` resolution + `PAPER/objective-motivation/weight-sensitivity/` bundle | lands ~this AM | 5.6; **gates Q1–Q8** |
| Q1 | 3 perturbation arms (iFGSM / FGSM / random-jitter; run-book in `baselines/STATUS.md`) → 6-row cross-arm table (CPU) → `PAPER/baselines/comparison/` | minutes | 5.5 |
| Q2 | **SF weighted-BC sweep + controls** on the sf12 supply-lift filtered corpus (10 arms × 6 seeds) | hours | 5.7 Pillar-2 |
| Q3 | **SZ L1v2 full re-run**: BC + WGAN-GP generators retrained/re-emitted (multiseed) + all four sources scored (F_causal, Fidelity-A gate, Fidelity-B) against the filtered supply-lift corpus | hours | 5.4 L1 table |
| Q4 | **SF L1v2 full re-run** (same protocol, sf12) | ~1–2h | 5.7 |
| Q5 | SZ + SF model-level variance suites (b0 vs FAMAIL, 5 seeds × 2 arms each) on supply-lift corpora | hours | 5.4 / 5.7 |
| Q6 | 2 alternate-feature-set trim+lift edit runs: `{housing,gdp,comp}`, `{housing,comp,migrant,logpopdensity}` | ~8h each | gates Q7–Q8 |
| Q7 | Per-set external metrics + filter@K Pareto rescores (CPU) | minutes | 5.6 |
| Q8 | Per-set downstream matrix: L1 (with re-emitted generators), vanilla null, weighted-BC sweep + controls, variance — for both alternate sets | ~1–2 days | 5.6 full matrix |

Total ≈ 3–5 GPU-days. All runs daemonized (`nohup setsid`, `PYTHONUNBUFFERED=1`, logs), resumable with
per-item completion checkpoints and a `--status` view (the α-sweep driver pattern). CPU work (external
metrics, table assembly, curation, drafting) interleaves continuously.

### Campaign gate — the Q0 α-sweep review (HARD CHECKPOINT)

When s80 lands: run `python -m famail_temporal.analysis.alpha_sweep_summary`, then produce a short
**weight-decision memo** (committed, e.g. `PAPER/objective-motivation/weight-sensitivity/DECISION.md`)
covering: the full 6-point frontier; flatness assessment (current 4-point partial: ΔF_causal spans
+0.0217..+0.0226 — flat); which point the documented criterion (max ΔF_causal s.t. ΔF_spatial ≥ 0)
selects and by how much; ΔF_spatial behavior; and a recommendation. **Surface to Robert for the
keep-vs-re-anchor decision — do not decide unilaterally.**
- **Keep (0.2, 0.7, 0.1):** campaign proceeds as queued; the frontier is reported as weight-insensitivity.
- **Re-anchor to a stronger α:** the SZ + SF headline edit runs are re-executed at the new α FIRST
  (~8h + ~40min), all downstream queue items re-anchor to the new corpora, and every dependent number
  already drafted is re-slotted. The ablation (5.3) and methodology §3.2's stated weights update
  accordingly. This is the expensive branch — hence the careful review before it.

### Reproducibility artifact set (the "loaves of bread" discipline)

Collected for **every** campaign run; all cheap (seconds each), none affect experiment runtimes:

1. **`famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md`** — one row per run: queue id, exact command,
   git commit SHA, frozen-editor gate result (`git diff main -- famail_temporal/algorithm/
   famail_temporal/evaluation/runner.py` empty → recorded), config/feature set, seeds, start/end +
   wall-time, hardware note, artifact dir, status. **Nothing runs without a ledger row.**
2. **Per-run `environment.json`** — python/torch/CUDA versions, GPU name, `pip freeze` capture.
3. **Per-result-dir `PROVENANCE.md`** (supply-lift pattern) + SHA-256 checksums of key artifacts
   (metrics.json, *.npz, paired-stats).
4. **Over-collection**: per-seed raw JSONs, paired stats, manifests, and run logs retained — never
   only aggregates.
5. **Committed driver scripts** — the campaign queue itself is a committed, resumable script; a
   reviewer can re-execute the campaign end to end.
6. **`PAPER/REPRODUCIBILITY.md`** (capstone) — the claims→artifacts map: every Experiments table/figure
   → curated `PAPER/` artifact → ledger row → exact command. Extends the existing `data_provenance.md`
   pattern paper-wide; this is the single entry point a reviewer follows.
7. **Curation as runs land**: `PAPER/supply-lift/by_feature_set/`, additions to
   `PAPER/supply-lift/data/` (SF WBC, L1v2 refresh, variance), `PAPER/baselines/comparison/`,
   `PAPER/objective-motivation/weight-sensitivity/`.

## Track B — `04_experiments.tex` (claims ladder; SF as its own subsection)

Budget ~3–3.5 two-column pages; ~6 tables + 2–3 figures (dose-response, α-Pareto frontier,
channel-decomposition/forest candidate). Merging tables to fit is drafting freedom; dropping
disclosures is not.

- **5.1 Setup** — two cities + stuck-GPS cleanup one-liner; editor config (α, k, ε, budget split);
  paired-seed protocol; bootstrap B=1000 (first-order caveat); the n=6 Wilcoxon-floor and n=5
  conventions; the **three-ring metric firewall** paragraph (optimized / design-targeted / external);
  external-metric definitions (DP, DI, Theil, group levels) incl. the **DP≡gap disclosure**.
- **5.2 Data-level fairness (Shenzhen)** — internal (F_causal +0.0222, F_spatial +0.0064);
  external-metrics table (DI +0.0155, DP −0.858, Theil −0.0082, CIs); mean(Y|D) +0.0468 (labeled
  design-targeted) with the **channel decomposition** (supply +0.0091 tier-1 / +0.0242 tier-2, both
  significant, tiers always labeled; demand n.s.); fidelity stability + lift's Fidelity-B cost
  disclosed; skip-on-infeasible provenance (115/2,455 reverted; rule-first; favorable-direction
  disclosure); oracle G0 numbers (fulfills methodology §3.4's promise).
- **5.3 Trim-only vs trim+lift ablation** — the Zhang-mandated subsection; the ONLY home of trim-only
  numbers (`% lint-allow: ablation`): ΔF_causal +0.0144→+0.0222 (SZ), +0.0139→+0.0328 (SF); F_spatial
  −0.0009→+0.0064; disadvantaged group level flat (7.0734→7.0734)→+0.0468; SF migrant DP
  n.s.→significant; the leveling-down evidence row.
- **5.4 Downstream propagation (Shenzhen)** — L1 four-source table (Q3, re-emitted generators) →
  vanilla-BC null (w=1) → weighted-BC dose-response (+0.0232/+0.0280/+0.0310, 6/6) **+ the new
  F_spatial propagation** (+0.0042/+0.0048/+0.0057) + both controls null → model-level variance (Q5) →
  the **rollout-allocation boundary**: drain attenuated ~40%, NOT reversed (claim levels 1–2, disclose
  level 3; motivates training-side future work).
- **5.5 Baselines** — perturbation arms (Q1) framed per Meeting-41 as fidelity/editing-quality
  baselines, expected NOT to improve fairness; the arms are "iFGSM/FGSM with random restart" with the
  vanilla-no-op ablation row (`--no-random-start`); FGSM numbers from the corrected engine (`6da3d27`+);
  demographic oversampling (targeted +0.0153 dose-monotone / placebo −0.0172 / DP explosion +2.8 /
  8,241-pool exhaustion disclosure / 10.5% inflation vs FAMAIL's 0%); the 6-row cross-arm table.
- **5.6 Robustness & sensitivity** — the per-set matrix under trim+lift (Q6–Q8); the α-Pareto frontier
  figure + criterion note (Q0); filter@K Pareto (editing beats filtering).
- **5.7 External validity — San Francisco** — dual claim (+0.0328; Fidelity-A stable); external metrics
  (migrant DP/DI now significant under district-extremes; **median-split migrant n.s. caveat stated**);
  supply channel replicates (+0.0195, explicitly tier-1-labeled as a lower bound; tier-2 recount not
  plumbed for SF); **the mean(Y|D) tension presented as BOTH readings with a `% TODO(PI-framing)` flag
  for Dr. Zhang** (ratio reading negative vs external-metrics reading uniformly positive —
  demand-endogeneity connection); SF L1 (Q4) / WBC (Q2) / variance (Q5); SF caveat block (ACS proxies,
  12 drivers, 14.9% raw adjacency, GAN-collapse divergence note re-checked against Q4's fresh run).

### Slotting protocol

Pending cells carry `% TODO(run:Q<n> → <exact artifact path>)` markers. Each queue completion triggers:
curation into `PAPER/`, a slot-in edit, compile + lint gates, ledger update, commit. The section is
never blocked on the queue — prose and structure land first.

### Verification

1. Compile + lint gates per task (existing `paper/lint.sh`; trim-only numbers outside 5.3 fail lint).
2. End-state two-agent audit (number/convention auditor + FAMAIL-fidelity reviewer), as run for the
   methodology section.
3. **Ledger cross-check**: every table cell → curated artifact → ledger row → command. The
   `PAPER/REPRODUCIBILITY.md` capstone is built from this check.

## Out of scope

- Introduction / related work / conclusion content.
- Final figure production beyond the three named (placeholders + one regenerated dose-response and
  α-Pareto are in scope; polished figure design is its own task).
- The Overleaf port (Robert's workflow).
- Resolving the SF mean(Y|D) framing or the F_demo rename (PI decisions — flagged, not made).

## Dependencies & risks

- **The Q0 re-anchor branch** is the schedule risk: re-anchoring adds ~9h before anything else and
  invalidates drafted numbers. Mitigation: the checkpoint memo + Robert's explicit decision before
  Q1 launches; drafting starts with 5.1/5.3-independent prose that survives either branch.
- **Serial GPU**: Q6+Q8 (~2–3 days) is the long tail; it lands in 5.6 last — the section is
  reviewable (and abstract-supporting) before it completes.
- **eGPU flakiness**: if CUDA disappears, it is likely the enclosure power (known gotcha) — resume via
  the drivers' checkpoints.
- Weighted-BC harness works on any edit dir (proven on the supply-lift corpus); per-set caches are
  feature-suffixed and coexist (proven trim-era).

## Sources read for this design

`PAPER/supply-lift/{FINDINGS.md, LIFT_ALGORITHM_REFERENCE.md, tables/, data/}`,
`PAPER/external-metrics/FINDINGS.md`, `PAPER/argument/{04,05,06}`, 
`PAPER/baselines/{README.md, demographic-oversampling/FINDINGS.md}`, `PAPER/feature_selection/`
(3-set matrix), `famail_temporal/baselines/STATUS.md` run-books, α-sweep driver + summary tool state
(4/5 DONE at design time).
