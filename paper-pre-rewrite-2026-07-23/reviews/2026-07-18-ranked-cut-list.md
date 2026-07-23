# Ranked Cut List — FAMAIL KDD manuscript (2026-07-18, controller synthesis)

Companion to `2026-07-18-cut-inventory.md` (candidate details live there; this file ranks).
**Target: ~330 single-column lines (~3.0 content pages), 12pp → 9pp rendered (8 content + ~1 refs).**

Two corrections to the inventory's framing, applied here:
1. **There is no Overleaf cushion.** Robert verified 2026-07-18: local and Overleaf render
   IDENTICALLY (~12pp). The recon's "local over-counts 0.5–0.75pp" belief is dead; every line
   must come from actual cuts. (Inventory lines 18–19 and 415–416 are stale on this point.)
2. **`teaser-remove` is not on the default plan.** The teaser is Dr. Zhang's requested figure;
   removing (or resizing) it is a Robert/Zhang decision, listed separately at the bottom.

**2026-07-20 addendum — Figure-1 geometry settled (supersedes any teaser-resize speculation):**
- The live Figure 1 is now the **(c)-only** build (Dr. Zhang's request, executed 2026-07-20).
  Measured on acmart harness renders (after Robert's 2026-07-20 FATE-arrow + boxed-legend
  revisions): figure+caption block 15.4 cm (3-strip) → 11.4 cm ((c)-only) = **~4.0 cm ≈ 11
  single-column lines already banked** relative to the geometry this plan was costed
  against. The re-run must re-measure pages from the current build rather than reuse this
  file's totals.
- **Standing liability for the re-run:** Robert's (b)+(c) counter-proposal
  (`figures/figure-1/figure-1-bc.tex`) measures 12.0 cm — only **~0.6 cm ≈ 1–2 lines** more
  than (c)-only. If Dr. Zhang is swayed at this week's meeting, the swap is a one-line
  `\input` change; the 8.0pp endpoint must therefore keep **≥ 2 lines of slack**.

Execution doctrine: **waves with re-measurement** — land a wave, rebuild, count pages, stop
when 8.0 content pages is reached. Prose savings are conservatively counted; rewrites usually
reclaim extra widow/orphan lines, so later waves may prove unnecessary. Gates after every wave
(latexmk + lint + render-QA of touched pages). The appendix skeleton (\appendix + appendix.tex)
is built in Wave 2 when the first RELOCATE lands.

---

## Wave 1 — SAFEST: duplication + appendix-natural detail (all LOW; ~144 lines ≈ 1.3pp)

Ordered by yield. No pillar, headline, or disclosure touched; body keeps interpretation,
appendix gets derivations/protocol.

| # | id | lines | note |
|---|---|---|---|
| 1 | `meth-fcausal-derivation-relocate` | ~24 | keep idempotence half-sentence (dependency) |
| 2 | `exp-tables-merge-A` (ablation+baselines) | ~18 | arguably strengthens the comparison |
| 3 | `meth-editor-impl-relocate` | ~17 | ~5% revert disclosure STAYS in body |
| 4 | `exp-tables-merge-B` (external+channels) | ~13 | shared mean(Y|disadv) row |
| 5 | `meth-weight-dup-compress` | ~8 | pointer target = appendix (couples w/ Wave 2 #1) |
| 6 | `meth-attribution-eq-relocate` (Eq. 4 only) | ~8 | verification note goes with it |
| 7 | `exp-sf-downstream-shorten` | ~8 | n=12 + city-difference stay; tier-2 block untouched |
| 8 | `intro-tier-breakdown` | ~6 | tier numbers live in §4.2 |
| 9 | `concl-restatement-shorten` | ~6 | bounds paragraph untouched |
| 10 | `exp-fourseource-gan-shorten` | ~5 | one-sentence honesty beat survives |
| 11 | `exp-dose-saturation-shorten` | ~5 | "knee, not tuned endpoint" sentence survives |
| 12 | `exp-setup-stats-shorten` | ~4 | floor arithmetic → appendix; n=12 note stays |
| 13 | `exp-setup-instruments-shorten` | ~4 | DP≡gap disclosure stays |
| 14 | `exp-variance-shorten` | ~4 | p=.0039 + magnitude-comparison stay |
| 15 | `exp-provenance-shorten` | ~4 | skip-on-infeasible disclosure stays |
| 16 | `rw-recourse-compress` | ~4 | mechanism lives in §3.5 |
| 17 | `meth-fspatial-gini` variant (a) | ~3 | drop the O(N log N) aside only |
| 18 | `concl-future-shorten` | ~3 | |

**Checkpoint A: rebuild + count. Expected ~10.7 content pages.**

## Wave 2 — The two big relocations (MEDIUM, but evidence lands in the appendix; ~85 lines ≈ 0.8pp)

| # | id | lines | body keeps |
|---|---|---|---|
| 1 | `exp-figure2-relocate` (+§4.6 prose trim) | ~50 | flatness + monotone-decline + criterion sentences |
| 2 | `exp-table6-relocate` | ~35 | tier-2-sig-all-three numbers + most-fair leak DISCLOSURE |

**Checkpoint B: rebuild + count. Expected ~9.9 content pages.**

## Wave 3 — MEDIUM shortens, safest first (~44 lines ≈ 0.4pp). Take only until the count hits 8.0.

| # | id | lines | note |
|---|---|---|---|
| 1 | `exp-fairness-penalty-shorten` | ~9 | verdict + numbers stay; λ-grid → appendix |
| 2 | `exp-baselines-perturbation-note-shorten` | ~8 | 2-sentence honesty version |
| 3 | `meth-attribution-eq-relocate` (Eq. 3) | ~7 | "exact per-unit partition (App.)" stub |
| 4 | `meth-screen-detail-shorten` | ~6 | 80k/95k + nominates-vs-derives stay |
| 5 | `exp-filtering-shorten` | ~5 | one-sentence rebuttal survives |

**STRUCK 2026-07-18 (Dr. Zhang feedback reversal):** `rw-contrast-tighten` (~6) and
`rw-leveling-compress` (~3) are REMOVED from the plan — her feedback asks §2's per-group
limitation summaries to become MORE explicit and systematic, not shorter; §2 grew ~6 lines
accordingly. Re-run the whole cut review after the Zhang-feedback edits settle (Robert,
2026-07-18) before executing any wave.

**Checkpoint C: rebuild + count. Expected ~9.5 content pages → gap ~1.5pp remains if estimates
hold exactly; in practice rewrite-compaction + float re-packing usually beat estimates. Measure.**

## Wave 4 — RESERVE (use only if Checkpoint C still exceeds 8.0; ordered by my safety read)

| id | lines | why held back |
|---|---|---|
| `meth-figure1-resize` | ~15 | Fig-1 is the HOW explainer; shrink hurts the paper's best teacher |
| `intro-contributions-compress` | ~6 | the itemize is the reviewer's claim map |
| `meth-fspatial-gini` variant (b) | ~5 | relocates a scored metric's definition |
| `abstract-tighten` | ~4 | shop window; low yield for the risk |

## Robert/Zhang decisions (NOT ranked; excluded from the plan totals)

- `teaser-resize` (~11) / `teaser-remove` (~33, HIGH): the teaser is Zhang's requested figure.
  If Waves 1–4 fall short, this is the conversation to have — with the render in hand.

## Standing constraints (from the inventory's DO-NOT-CUT register — binding on every wave)

Both pillars with numbers in body; all disclosures main-text (allocation drain both cities,
most-fair leak, leveling-down scoping, SF caveats, §5 bounds, DP≡gap, tier-2/Reading-B, ~5%
revert, oversampling fabrication/placebo); §3.4 structural-diagnosis numbers; the
two-phase-as-control paragraph. Appendix stubs always name what moved and where.
