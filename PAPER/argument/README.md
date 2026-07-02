# FAMAIL paper argument — documentation set

**Thesis.** FAMAIL is a fairness-oriented **data-augmentation** method for imitation-learned taxi
demand: *edit* a small, attribution-targeted slice of real trajectories to make the data fairer along
a demographic axis while keeping it realistic, then *upweight* that edited slice during training so
the fairness propagates into the model instead of being averaged away. The argument has two pillars —
(1) the edited data is the fairest *faithful* source (editing beats generating), and (2) vanilla
training averages the edit away but upweighting **recovers** it *edit-specifically* — and it
reproduces on a second city with no algorithm change.

This directory is the self-contained argument record. Every experimental number is a seed mean
traceable to a source JSON, and every figure is referenced by its `PAPER/`-relative path (figures are
never regenerated or copied here).

## Reading order

| # | doc | contents |
|---|---|---|
| — | [`00_overview.md`](00_overview.md) | the elevator argument + the headline-numbers table (Shenzhen ∥ SF) |
| 1 | [`01_motivation_goals.md`](01_motivation_goals.md) | why mobility inequity matters; why *edit* rather than *generate*; contributions |
| 2 | [`02_datasets.md`](02_datasets.md) | Shenzhen (primary) + SF (external validity); compatibility rationale; stuck-GPS cleanup |
| 3 | [`03_fairness_theory.md`](03_fairness_theory.md) | F_causal / F_spatial / Fidelity-A / Fidelity-B + the ST-iFGSM editor + Resources |
| 4 | [`04_evaluation.md`](04_evaluation.md) | the two-pillar experimental design + validation gate + statistical conventions |
| 5 | [`05_results_shenzhen.md`](05_results_shenzhen.md) | **primary results** (authoritative Shenzhen numbers) |
| 6 | [`06_results_sf.md`](06_results_sf.md) | **external-validity results** (authoritative SF numbers) + head-to-head |
| 7 | [`07_limitations.md`](07_limitations.md) | candid limitations, open questions, and the adversarial-review credibility note |

The two results docs (05, 06) are **authoritative** for all numbers; `00_overview.md` summarizes them
and must agree.

## Suggested slide outline

| slide section | source doc |
|---|---|
| Title / thesis | `00_overview.md` |
| Motivation (why this matters, edit vs generate) | `01_motivation_goals.md` |
| Data (two cities, cleanup) | `02_datasets.md` |
| Fairness metrics & the editor | `03_fairness_theory.md` |
| How we evaluated (two pillars, conventions) | `04_evaluation.md` |
| Results — Shenzhen (primary) | `05_results_shenzhen.md` |
| A second city — San Francisco | `06_results_sf.md` |
| Limitations & what's next | `07_limitations.md` |

## For a presentation agent

Directives for any agent building slides from this set:

- **Lead with the thesis in `00_overview.md`** and put its **headline-numbers table** on an early
  results-summary slide.
- **The numbers in `05_results_shenzhen.md` and `06_results_sf.md` are authoritative** — use them
  verbatim; do not recompute, round differently, or infer new values.
- **Place each doc's referenced figures on the corresponding slides**, using the `PAPER/`-relative
  paths exactly as given (e.g. the dose-response and L1 data-quality figures are the "money figures").
  Do not regenerate figures.
- **Keep the associational-`F_causal` caveat on any fairness-metric slide** (1 = fairest; a partial R²
  on 10 district profiles; a rename to `F_demo` is pending). Do not present it as a causal estimate.
- **Frame SF as reproducing / on par with Shenzhen, not beating it** — F_causal is city-specific and
  associational, so absolute cross-city magnitudes are not commensurable.
- **When stating the weighted-BC significance,** pair `p = 0.03125` with the mean Δ, the t-CIs, the
  monotone dose-response, and the null/negative control arms — it is an n = 6 sign-unanimity floor, not
  an effect size.
- **Do not name any specific authoring tool or product** anywhere in the deck.

## Deeper artifacts

For the full per-experiment tables, figures, and provenance behind this argument:

- `PAPER/by_feature_set/` — the three demographic feature sets (PRIMARY `housing-comp-migrant/` +
  two sensitivity sets), each with README, data JSONs, figures, and tables.
- `PAPER/feature_selection/` — the 3-way feature-set robustness comparison + selection rationale.
- `PAPER/second-dataset/` — the SF bundle (`FINDINGS.md` is the comprehensive synthesis).
- `PAPER/reviews/` — the three adversarial-review rounds.
- `PAPER/shared_cleanup/` — the stuck-GPS cleanup + F_spatial decomposition (demographic-independent).
