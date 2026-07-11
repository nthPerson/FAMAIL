# PAPER/baselines/ — data-augmentation baseline arms (Mission 3)

Curated, committed results bundle for **Mission 3** of the Meeting-41 plan: the additional
**data-augmentation baselines** that contextualize the FAMAIL editor's results. One subdirectory per
baseline approach; each is self-describing (README + FINDINGS + tables/figures with provenance), mirroring
the conventions of the sibling `PAPER/` bundles. The cross-arm comparison table gets its own
`comparison/` directory so no single arm's bundle has to be re-opened when the table lands.

**Framing (Meeting-41, canonical):** these arms are **fidelity / editing-quality baselines, NOT competing
fairness methods** — none of them optimizes a fairness objective, so fairness is expected NOT to improve
under the perturbation arms; the point of the comparison is that FAMAIL's gains come from its objective,
not from bounded editing (or corpus resampling) per se. All arms are scored against the **supply-lift
(trim+lift)** headline, canonical for all PAPER reporting (decision 2026-07-09).

## Layout & status

| dir | approach | kind | status |
|---|---|---|---|
| `demographic-oversampling/` | duplicate real trajectories from disadvantaged regions (phantom drivers, additive demand+supply) + random-oversampling placebo | resampling | ✅ **RUN (2026-07-10, SZ, 9 arms)** — see its `FINDINGS.md` |
| `ifgsm/` | iterative signed-gradient attack on the frozen HuMID discriminator (PGD-style random restart) | perturbation | ⏳ PENDING GPU (held by the α-sweep) |
| `fgsm/` | single full-budget signed-gradient step (random restart) | perturbation | ⏳ PENDING GPU |
| `random-jitter/` | seeded ε-jitter, direction placebo for the gradient arms | perturbation | ⏳ PENDING GPU |
| `comparison/` | the 6-row cross-arm table (raw / FAMAIL / ifgsm / fgsm / random / oversampling) | — | ⏳ lands with the GPU runs |

## Scope note — what does NOT live here

The Level-1/Level-2 comparisons against **BC-generated and GAN-generated data** are part of the two-pillar
argument itself and live inside `PAPER/by_feature_set/` (moving them would break those bundles'
self-containment); `PAPER/external-metrics/` and `PAPER/supply-lift/` are editor results, not baselines.
Nothing else in `PAPER/` is baseline work that is cleanly separable — this directory therefore holds the
Mission-3 data-augmentation baseline family only.

## Code & provenance

Code (all standalone; zero changes to the frozen editor): `famail_temporal/baselines/{stifgsm_baseline,
run_stifgsm_baseline,demographic_oversampling,run_demographic_oversampling,assemble_baseline_table}.py`
(+ 5 test files). Run-books and working notes: `famail_temporal/baselines/STATUS.md` (Mission-3 section).
Candidate selection lit-scan (citation-verified): `famail_temporal/baselines/DATA_AUG_BASELINE_CANDIDATES.md`.
Specs/plans: `docs/superpowers/{specs,plans}/2026-07-09-mission3-data-aug-baselines*` and
`docs/superpowers/{specs,plans}/2026-07-09-demographic-oversampling-baseline*`.
