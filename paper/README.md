# FAMAIL — KDD manuscript

**This repo is the writing source of truth.** Robert ports completed sections to the
shared Overleaf for Dr. Zhang's review (old Overleaf content is out of scope). Each
`sections/*.tex` file is self-contained plain LaTeX (no custom macros) so it pastes
cleanly.

## Build

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex   # compile gate
    bash lint.sh                                                    # convention lint

Both must pass before every commit. Warnings are tolerated during drafting; errors are not.

## Writing conventions (locked decisions — do not relitigate in prose)

1. **Trim+lift centers ALL reporting.** Trim-only numbers appear ONLY in the
   trim-vs-trim+lift ablation (mark the line `% lint-allow: ablation`).
2. **F_causal keeps its label + associational caveat.** No causality-claim language;
   no F_demo rename (pending PI decision).
3. **The spoken "54%" figure is banned** until grounded. Absolute deltas only
   (+0.0222 SZ / +0.0328 SF).
4. **p = 0.031 never appears without** mean Δ + t-CI + monotone dose-response — it is
   the n=6 Wilcoxon sign-unanimity floor, not an effect size.
5. **SF *reproduces* Shenzhen, never "beats" it** (F_causal is city-specific and
   associational; absolute baselines are not cross-city comparable).
6. **Every load-bearing number carries a provenance comment**: `% src: PAPER/<path>`.
7. **Any single supply number states its accounting tier** — tier-1 (fractional
   presence, optimizer convention) vs tier-2 (distinct-taxi recount from raw GPS).
   See PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md §10.
8. **Three-ring metric firewall** (LIFT_ALGORITHM_REFERENCE.md §13):
   (i) optimized: F_spatial/F_causal/F_fidelity; (ii) design-targeted, not optimized:
   mean(Y|D)/SDR family; (iii) genuinely external: DP, DI, Theil, per-group levels,
   tier-2 recount, channel decomposition. "Improves metrics we never optimized"
   claims ride ring (iii) only.
9. **No product/tool names** anywhere.
10. **Mechanism names (renamed 2026-07-11):** trim's selector is **"demand deficit
    attribution"** (formerly "deficit attribution"); lift's is "supply-gradient
    attribution". Use the full name wherever the mechanism is meant; generic uses
    of "deficit" (e.g. "fairness deficit", "highest-deficit units") stay as-is.

## Layout

`main.tex` (acmart sigconf, anonymous+review) → `sections/01..05` → `refs.bib`
(seeded from PAPER/objective-motivation/REFERENCES.md; T3 human pass pending).
