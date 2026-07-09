# Objective-function motivation — the literature-grounded *why*

This bundle is the **literature-grounded motivation** for the FAMAIL editing objective
`L = α_spatial·F_spatial + α_causal·F_causal + α_fidelity·F_fidelity` and its ST-iFGSM editor: why each term
exists, how its design follows from prior work, how FAMAIL is positioned against that work, and how the
predictable reviewer objections are answered. It is the paper-facing companion to the objective's *operational*
description.

**Relation to [`../argument/03_fairness_theory.md`](../argument/03_fairness_theory.md):** doc `03` gives the
*what* — the formulas, intuition, and caveats a reviewer needs to state the claims correctly. This bundle gives
the *why + how* — the supporting literature and the drafted motivation prose. `03` links here for the full
lineage; this bundle links back to `03` for the formulas.

## Reading order

| # | doc | contents |
|---|---|---|
| 1 | [`MOTIVATION.md`](MOTIVATION.md) | paper-ready per-component *why + how*: `F_causal`, `F_spatial`, `F_fidelity`, the ST-iFGSM editor + soft discretization, the downstream upweighting, and the weight/scalarization justification |
| 2 | [`REVIEWER_DEFENSE.md`](REVIEWER_DEFENSE.md) | anticipated KDD-reviewer objections → literature-grounded rebuttals |
| 3 | [`LEVELING_DOWN.md`](LEVELING_DOWN.md) | the egalitarian-ethics + fair-ML framing of over-service reduction, and the demand-endogeneity thread that unifies it |
| 4 | [`REFERENCES.md`](REFERENCES.md) | the consolidated, verified reference list — the single citation source-of-truth |

## Provenance & conventions

- **Citations verified 2026-07-08** against arXiv / DOI / ACM DL / IEEE Xplore / DBLP / Crossref. `REFERENCES.md`
  is the **single citation source-of-truth**; the other docs cite by *surname + year* against it. Known
  corrections are baked in (e.g. cGAIL = *IEEE Trans. Big Data* 2022; ST-iFGSM = Hu et al. KDD 2023; the Zheng
  et al. result is cited by its absolute service-gap reduction, not by any percentage figure).
- **No new experimental numbers** are introduced here. Empirical claims trace to the authoritative results docs
  ([`../argument/05_results_shenzhen.md`](../argument/05_results_shenzhen.md),
  [`../argument/06_results_sf.md`](../argument/06_results_sf.md), `../external-metrics/`, `../second-dataset/`)
  and, for the weight-selection facts, to `famail_temporal/baselines/STATUS.md` and the trajectory-editing
  methodology doc. The `(ΔF_spatial, ΔF_causal)` weight-Pareto sweep is described as a **planned sensitivity**.
- **`F_causal` is associational**, a partial R² on ~10 district profiles — never presented as a causal estimate;
  a rename to `F_demo` is pending.

## Related

- [`../argument/04_evaluation.md`](../argument/04_evaluation.md), [`../argument/05_results_shenzhen.md`](../argument/05_results_shenzhen.md) — the downstream (Pillar-2) experimental design and results.
- [`../external-metrics/LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md) — the structural proof and mechanism behind the leveling-down property that `LEVELING_DOWN.md` frames.
