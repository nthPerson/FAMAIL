# Design — FAMAIL paper-argument documentation (`PAPER/argument/`)

**Date:** 2026-07-01 · **Status:** approved design, ready for implementation planning.

## Purpose

Create a set of markdown documents under **`PAPER/argument/`** that encapsulate the FAMAIL project's complete
paper argument — motivation, datasets, fairness-metric theory, evaluation procedures, and results/findings — grounded
in the final, results-backed state of the `PAPER/` deliverable. The docs serve a **dual purpose**:

1. **Paper-argument foundation** — the rigorous, current articulation of the project's case (successor to the now-stale
   `famail_temporal/baselines/PAPER_ARGUMENT_PLAN.md` and `docs/two_level_argument.md`).
2. **Presentation context for a generic slide-building agent** — self-contained enough that an agent can build a
   research-team progress deck from `PAPER/argument/` alone.

**Product-agnostic:** no document may name any specific tool/product. Agent-facing guidance is phrased for a generic
"presentation agent" / "an agent building slides."

## Context (why now, and what's stale)

All experiments are complete and on `main`: the Shenzhen study across three demographic feature sets (PRIMARY
`{housing, comp, migrant}`) and the San Francisco (SF Cabspotting / sf12) external-validity replication, all curated
in `PAPER/` with three adversarial-review rounds. The two pre-existing argument docs predate this state:
- `famail_temporal/baselines/PAPER_ARGUMENT_PLAN.md` (2026-06-25) — right two-pillar *skeleton*, but stale numbers
  (F_causal 0.818, +0.0128, pre-cleanup 3-feature), figure placeholders, Pillar 2 written as "future".
- `docs/two_level_argument.md` (Meeting 38) — older "two-level" framing, Pillar 2 as an unrecovered negative.

Both are **left in place** but get a one-line "superseded by `PAPER/argument/`" pointer at the top. They are not deleted
(historical value; referenced elsewhere).

## Resolved decisions (from the brainstorm)

1. **Structure:** an indexed doc-set in a new `PAPER/argument/` directory (not one monolithic file).
2. **Self-containment:** self-contained argument — headline numbers and key tables embedded as markdown; existing
   figures referenced by path (not regenerated, not copied); a short "Provenance" footer per doc pointing to source
   JSONs / methodology docs.
3. **Theory depth:** core formulas + plain-language intuition + a curated Resources section (internal methodology docs
   + external method lineage). Full derivations stay in the methodology docs, linked.
4. **Dataset emphasis:** Shenzhen = primary study; SF = external-validity replication (reproduces the whole argument
   with no algorithm change, and Pillar 2 is sharper). Results are split into a Shenzhen doc and an SF doc.

## Conventions & defaults (apply to every doc)

- **Thesis (PI-approved, Meeting 40):** FAMAIL is a **fairness-oriented data-augmentation method** — edit a small,
  targeted unfair slice of real trajectories, then upweight that slice during policy training so the fairness
  propagates. Two pillars: (P1) the edited data is the *fairest faithful* source; (P2) vanilla BC averages the edit
  away, but importance-weighting recovers it edit-specifically. Validated on two cities.
- **Audience:** technical peers (ML / fairness / mobility-literate). Motivate the taxi-service-equity domain; assume ML
  fluency.
- **Metric naming:** use `F_causal` with the associational caveat; add a one-line note that a rename to `F_demo` is a
  pending PI decision. Do **not** pre-emptively rename.
- **Honesty:** a candid standalone `07_limitations.md`, plus inline caveats where load-bearing. Treat the three
  adversarial-review rounds as a credibility asset.
- **Numbers discipline:** every cited fairness number is a **seed MEAN** (never seed-0). Pull numbers from the
  authoritative source artifacts (below), not from prose. `p = 0.03125` is the n=6 Wilcoxon floor (sign-unanimity, not
  a magnitude) — always pair it with the mean Δ + t-CI + dose-response.
- **Figure references:** reference existing PNGs by their `PAPER/`-relative path so a presentation agent knows exactly
  which visual belongs on which slide. Do not regenerate or copy figures.
- **Provenance footer:** each content doc ends with a short "Sources / provenance" list (the JSONs, tables, and
  methodology docs it draws from).

## Directory layout

```
PAPER/argument/
  README.md                 index + generic-agent entry point + suggested slide outline (docs → sections)
  00_overview.md            the elevator argument + headline-numbers table + money-figure pointers
  01_motivation_goals.md    why service-equity in mobility data; edit-vs-generate; contributions/goals
  02_datasets.md            Shenzhen (primary) + SF (external validity); compatibility rationale; cleanup
  03_fairness_theory.md     F_causal / F_spatial / Fidelity-A / Fidelity-B + the editor + Resources
  04_evaluation.md          two-pillar experimental design + validation gate + statistical conventions
  05_results_shenzhen.md    Shenzhen primary results (3 feature sets, two-pillar, pareto, robustness)
  06_results_sf.md          SF external-validity results (dual claim, two-pillar reproduction, head-to-head)
  07_limitations.md         candid limitations + open questions + review-credibility note
```

Each file has **one clear purpose**, is independently readable, and communicates with the others only through explicit
cross-links — so a presentation agent (or a reader) can consume any one without the others, and each maps to a slide
section.

## Per-document spec

For each doc: **P** = purpose, **C** = contents/sections, **F** = figures to reference, **S** = authoritative sources.

### `README.md` — index + generic-agent entry point
- **P:** orient a reader or a presentation agent; give the one-paragraph thesis; provide a suggested slide outline.
- **C:** (1) one-paragraph thesis; (2) reading order + one-line description of each doc; (3) a **"Suggested slide
  outline"** mapping docs → deck sections (e.g. Title/Thesis ← 00; Motivation ← 01; Data ← 02; Metrics ← 03; How we
  evaluated ← 04; Shenzhen results ← 05; A second city ← 06; Limitations & next ← 07); (4) a short **"For a
  presentation agent"** note — generic directives (lead with 00's thesis; use the headline-numbers table; drop the
  referenced figures on the corresponding slides; keep the associational-`F_causal` caveat on any fairness-metric
  slide); (5) pointers to the deeper `PAPER/` artifacts (per-set READMEs, `feature_selection/`, `second-dataset/`,
  `reviews/`).
- **S:** this spec; `PAPER/README.md`.

### `00_overview.md` — the elevator argument
- **P:** the whole argument in ~1–2 pages.
- **C:** problem (demographic-driven service inequity in real mobility data) → approach (edit the unfair slice rather
  than generate synthetic data) → two-pillar result → external validity (SF) → framing (data augmentation).
  A **headline-numbers table** (Shenzhen PRIMARY + SF side by side: editor ΔF_causal, L1 edited-fairest, L2 null,
  weighted-BC w30 recovery, controls). "In one figure" pointers.
- **F:** `by_feature_set/housing-comp-migrant/figures/fig_dose_response.png` (the money figure);
  `.../fig_l1_data_quality.png`.
- **S:** `05_results_shenzhen.md` + `06_results_sf.md` (numbers must agree with those docs);
  `PAPER/by_feature_set/housing-comp-migrant/README.md`; `PAPER/second-dataset/FINDINGS.md`.

### `01_motivation_goals.md`
- **P:** motivate the problem and state goals/contributions.
- **C:** taxi/mobility service inequity and why it matters; why imitation-learned demand models inherit and can amplify
  it; why *editing real data* beats *generating* synthetic data (keep human fidelity, target the unfair slice);
  the data-augmentation positioning; explicit contributions (a fairness metric + attribution, an editor, the
  two-pillar training recipe, two-city validation).
- **S:** `docs/two_level_argument.md` (umbrella-claim framing), `PAPER_ARGUMENT_PLAN.md` (thesis), FINDINGS §8.

### `02_datasets.md`
- **P:** describe both datasets and why they were chosen.
- **C:** **Shenzhen (primary):** 50-driver sample, 48×90 grid @ 0.01°, T=24 hourly, demographics resolve to **10
  district profiles**, the three feature sets (before-edit F_causal 0.799 / 0.807 / 0.725) and why `{housing, comp,
  migrant}` is PRIMARY; the **stuck-GPS cleanup** (10 calibrated sink cells across 9 driver plates, 106,677 phantom
  pickups; headline sink grid (29,53)). **SF Cabspotting (external validity):** 536 taxis / ~11.2M pings → the
  fleet-density regime discovery → **sf12** density-matched subsample (12 drivers, baseline F_causal 0.8752), 32×30
  grid, ACS 2006–2010 demographics (housing/comp/migrant filled with ACS values). **Compatibility rationale:** the
  dual claim requires dense per-driver traces + persistent driver IDs (F_fidelity can't score OD pairs), which rules
  out OD-only US data (NYC TLC / Chicago / DC); SF is the only compatible US set. A small comparison table.
- **F:** `second-dataset/figures/sf_supply_demand.png` (the regime diagnostic);
  `shared_cleanup/figures/sink_spatial_attr_before_after.png`.
- **S:** `PAPER/shared_cleanup/` (cleanup counts), `PAPER/second-dataset/FINDINGS.md` §1–3,
  `project_second_dataset_compat` memory, the three editor metrics JSONs (before-edit F_causal).

### `03_fairness_theory.md`
- **P:** the theoretical foundation for the metrics + editor, at "core formulas + intuition" depth.
- **C:** for each of **F_causal**, **F_spatial**, **Fidelity-A**, **Fidelity-B**: the defining formula, the
  plain-language intuition, and the key caveat.
  - F_causal `= R'(I−H_demo)R / R'MR`, `R = Y − g₀(D)` (1 = fairest); FWL/residualization intuition (service residual
    after removing demand, then partialling out demographics); **associational, not causal**; **10 district-DOF /
    ecological-fallacy** caveat; the per-cell attribution αᵢ (one sentence, pointer for the full decomposition).
  - F_spatial (spatial-attribution / Gini channel-0), 1 = fairest.
  - Fidelity-A (frozen HuMID-style driver-conditioned 3-stream Siamese discriminator, same-driver probability).
  - Fidelity-B (discriminator-free Jensen–Shannon divergence of trajectory-statistic distributions vs raw).
  - **The editor:** per-(cell,time) attribution → ST-iFGSM signed-gradient edit of the pickup cell within an ε=2 L∞
    ball; weighted objective (causal-emphasis α = 0.2/0.7/0.1).
  - **Resources:** internal — `F_CAUSAL_METHODOLOGY_NOTES.md`, `FAIRNESS_DECOMPOSITION_FORMULATION.md`,
    `docs/mathematical_foundations.md`, `docs/site/methodology/*`, `TRAJECTORY_EDITING_METHODOLOGY.md`; external
    lineage — cGAIL (imitation-learning base), HuMID/Ren (the identity discriminator), FGSM/iFGSM (the editing step),
    Frisch–Waugh–Lovell (the residualization). *(Name the external lineage as grounded in the repo's methodology docs;
    exact bibliographic references to be finalized by the authors.)*
- **S:** the methodology docs above; `famail_temporal/fairness/`, `famail_temporal/config.py`.

### `04_evaluation.md`
- **P:** how the argument was measured.
- **C:** the two-pillar experimental design — **L1 data-quality** (four sources: raw / edited / BC-gen / GAN-gen, scored
  on F_causal, F_spatial, Fidelity-A, Fidelity-B); **L2 vanilla transfer** (driver-conditioned BC on each source,
  paired edited−raw); **weighted-BC recovery** (upweight the edited demos; dose-response w10/20/30; **random-placebo**
  and **most-fair-select** controls); **model-level variance** (b0 vs FAMAIL). The **real-anchored Fidelity-A
  validation gate** (matched vs mismatched). **Statistical conventions:** paired seeds, the n=6 Wilcoxon floor
  (0.03125 = sign-unanimity), lead with mean Δ + t-CI + monotone dose-response + null controls; n=5 nulls reported vs
  the cross-seed noise band; the deterministic (std=0) L1 data-level gap.
- **S:** `PAPER/by_feature_set/housing-comp-migrant/README.md`, `PAPER/second-dataset/FINDINGS.md` §6,
  `LEVEL1_V2_METHODOLOGY.md`.

### `05_results_shenzhen.md`
- **P:** the Shenzhen primary findings.
- **C:** editor dual-metric (PRIMARY F_causal 0.7988→0.8132 Δ+0.0144, F_spatial 0.1034→0.1025); **Pillar 1** (edited
  0.8132 fairest > raw 0.7988 ≈ bc 0.7980, gan 0.8089 disqualified by distributional collapse; Fidelity-A ~0.848);
  **L2** vanilla null (−0.0012); **Pillar 2** weighted-BC recovery **+0.0205 / +0.0278 / +0.0311** (w10/20/30, 6/6,
  t-CIs exclude 0); **edit ≫ select > random** (most_fair_w30 +0.0004 null, ~70× ratio; random null; filter@K *lowers*
  F_causal — Pareto); model-level variance null (−0.0011±0.0032); the **3-way feature-set robustness** (the argument
  reproduces under all three sets; only the F_causal scale shifts).
- **F:** `by_feature_set/housing-comp-migrant/figures/{fig_dose_response,fig_l1_data_quality,fig_l2_negative_transfer,
  fig_fidb_components,pareto_causal_hcm,pareto_spatial_hcm}.png`; `feature_selection/figures/fig_feature_robustness.png`.
- **S:** `PAPER/by_feature_set/*/data/*.json`, `PAPER/feature_selection/tables/comparison_across_sets.md`.

### `06_results_sf.md`
- **P:** the SF external-validity findings.
- **C:** the dual claim (F_causal 0.8752→0.8891 Δ+0.0139; F_fidelity 0.968; +0.0199 full-pool; fidelity inert as a
  gradient); discriminator val-AUC 0.998; **two-pillar reproduction** — Pillar 1 (edited 0.8891 fairest > raw 0.8752 ≈
  bc 0.8789 ≈ gan 0.8794; Fidelity-A 0.958), L2 null (+0.0004±0.0033), Pillar 2 recovery **+0.0296 / +0.0348 /
  +0.0387** (w10/20/30, 6/6) with **both controls negative** (random −0.0071/−0.0095; most-fair −0.0117/−0.0068/
  −0.0027) → sharper than Shenzhen; variance null (−0.0005±0.0043); the **head-to-head table** (Shenzhen vs SF); the
  honest **GAN-did-not-collapse** divergence (SF Fidelity-B 0.027) and why it's not load-bearing.
- **F:** `second-dataset/figures/sf_supply_demand.png` (+ optionally reference the SF eval tables).
- **S:** `PAPER/second-dataset/FINDINGS.md` §5–7, `PAPER/second-dataset/data/*.json`, `tables/*.csv`.

### `07_limitations.md`
- **P:** candid limitations + open questions.
- **C:** F_causal is associational (partial R² on 10 district profiles) + ecological-fallacy exposure; n=6/n=5 Wilcoxon
  floors + no multiple-comparison survival (evidence rests on CIs + dose-response + controls); the deterministic L1
  data-level gap (no sampling CI); **profile-dominated fidelity** (F_fidelity certifies identity preservation, not
  trajectory-shape realism — true on *both* cities); **GAN did not collapse on SF** (the Shenzhen "GAN disqualified"
  sub-claim doesn't transfer); small-n (SF 12 drivers, 5–6 seeds); SF demographics are ACS proxies (migrant =
  foreign-born share, not hukou), so cross-city absolute F_causal is not comparable; the pending `F_causal → F_demo`
  rename; **open questions** (what training procedure best realizes the data-level fairness; other model classes such
  as GAN/WGAN on edited data). Close with the credibility note: three adversarial-review rounds, 0 of 29 findings
  refuted + a third PRIMARY/branch review.
- **S:** `PAPER/reviews/`, `PAPER/second-dataset/FINDINGS.md` §9.

## Figure inventory (existing `PAPER/`-relative paths; reference, do not regenerate)

- `by_feature_set/housing-comp-migrant/figures/`: `fig_dose_response.png`, `fig_l1_data_quality.png`,
  `fig_l2_negative_transfer.png`, `fig_fidb_components.png`, `pareto_causal_hcm.png`, `pareto_spatial_hcm.png`
- `feature_selection/figures/fig_feature_robustness.png`
- `shared_cleanup/figures/sink_spatial_attr_before_after.png`
- `second-dataset/figures/sf_supply_demand.png`

## Authoritative number sources (pull means, never seed-0)

- Shenzhen PRIMARY: `PAPER/by_feature_set/housing-comp-migrant/data/{editor_hcm_metrics,L1v2_hcm_multiseed,
  L2_hcm_metrics,weighted_bc_hcm_{sweep,paired_stats},variance_hcm_aggregate}.json` + `tables/pareto_points_hcm.csv`.
- Cross-set: `PAPER/feature_selection/tables/comparison_across_sets.md`.
- Cleanup: `PAPER/shared_cleanup/` + `famail_temporal/source_data/processing_metadata.json`.
- SF: `PAPER/second-dataset/data/{sf12_dual_metrics,sf12_fairoff_k2000_metrics,eval_*}.json` + `tables/*.csv`;
  narrative `PAPER/second-dataset/FINDINGS.md`.

## Non-goals (YAGNI)

- No new experiments, no figure regeneration, no re-running analysis. Docs consume existing artifacts only.
- No editing of the numbers in the existing `PAPER/` artifacts (this is additive documentation).
- No full mathematical derivations inline (they live in the methodology docs, linked).
- No product-specific presentation instructions.
- No new figures created for these docs (reference existing ones).

## Success criteria

- `PAPER/argument/` contains the README + 8 docs, each self-bounded and internally consistent.
- Every embedded number is a seed mean traceable to a listed source; numbers agree across `00_overview`,
  `05_results_shenzhen`, and `06_results_sf`.
- Every figure reference resolves to an existing file.
- The two stale argument docs carry a "superseded by `PAPER/argument/`" pointer.
- No document names any specific product/tool; agent guidance is generic.
- A reader (or a presentation agent) can build a coherent research-team progress deck from `PAPER/argument/` alone.

## Implementation notes

- Build order that respects dependencies: `05_results_shenzhen` + `06_results_sf` first (they anchor the numbers),
  then `00_overview` (must agree with them), then the remaining docs, then `README.md` (indexes everything), then the
  two stale-doc pointers.
- Verify numbers against the source JSONs as each results doc is written (mean-not-seed-0 discipline; the recurring
  transcription pitfall).
