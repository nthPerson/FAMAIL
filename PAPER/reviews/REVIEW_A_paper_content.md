I'll synthesize the adversarial review. Let me note the data is comprehensive enough to produce the report directly — but one finding mentions the confirmed list was truncated mid-sentence. Let me work with what's verified and the dimension findings.

# ADVERSARIAL REVIEW — COMPONENT A (Paper Content / Data / Figures / Demographics)

## (1) OVERALL VERDICT: Defensible to WRITE, NOT defensible to SUBMIT as currently framed.

The numerical backbone is sound — the sanity gate matches production to 6e-8, and every recomputed headline number (editor Δ+0.0156, L2 null, weighted-BC dose, cleanup magnitudes, feature VIF/Jaccard) reproduces to the digit. So the *data is trustworthy*. What is NOT yet defensible is the **interpretive layer**: three load-bearing framings ("edit-specific," "causal," "robust to demographic choice") overreach the evidence, and the headline figure visually inverts its own conclusion. These are fixable at the writing/presentation level without re-running GPU work, EXCEPT the demographic feature-choice issue, which needs a real decision (see §3). Write the paper, but fix the criticals first — several are cheap edits, and a reviewer will find every one of them.

## (2) CONFIRMED MUST-FIX ISSUES

### CRITICAL (block submission)

**C1 — "Edit-specific" is an overclaim; SELECT is significant, not a null placebo.** The README headline frames the result as edit-specific by lumping `most_fair` (SELECT) with the random placebo. But `most_fair_w10/w30` are 6/6-seed positive at p=0.03125 — the *same* nominal p as edited. Only `random` is genuinely null.
- *Fix:* Replace "edit-specific (most-fair & placebo don't move it)" with "**edit-dominant**: editing yields ~12x the gain of selecting already-fair data (itself weakly significant); random selection is null." Report most_fair's p explicitly. (comparison_3v4.md is already honest here; the README is not — align them.)

**C2 — The headline figure (`fig_dose_response`) visually mislabels the significant SELECT points as "n.s."** The orange Most-fair `*` markers at w10/w30 are correctly computed but rendered *underneath* the gray Random "n.s." annotation (drawn later, larger font, ~same x,y). A reader reads SELECT as null — the exact opposite of the data, and it contradicts the paper's own EDIT≫SELECT story.
- *Fix:* Offset per-series significance markers (horizontal dx or vertical by error-bar height); re-render and confirm both orange `*` are legible. `paper_figures.py` lines 217-230 (no collision avoidance).

**C3 — "Causal" in F_causal is unjustified.** F_causal is 1−R² (partial-R² via FWL) of a *contemporaneous, cross-sectional* OLS of the supply/demand residual on demographics. No instrument, no temporal precedence, no exogeneity test exists anywhere in the codebase. Calling it "causal fairness" and reading edits as removing "demographic-driven inequity" asserts causation from an associational quantity.
- *Fix:* Either (a) rename to **demographic-association / residual-demographic fairness** and drop causal language, or (b) add+defend identification (e.g., a placebo-demographic permutation null). At minimum add a Limitations sentence: "F_causal is associational, not causal." This is a rename-level fix but it touches the paper's central metric name, so decide early.

**C4 — Feature set selected by minimizing before-edit F_causal = circular / metric p-hacking.** The selection criterion ("lower before-edit F_causal = more unfairness captured = better") picks the demographic lens that makes the baseline look maximally unfair — the *same* quantity the editor then optimizes and the paper reports improving. Compounding this: a strictly-dominated-better alternative exists — **h-g-c-logpop** matches F_causal to 3e-5 at lower max VIF (2.87 vs 4.51) and keeps GDP (an interpretable income axis), whereas the shipped set drops GDP for MigrantRatio, which contributes ~0 marginal F_causal (the entire 0.0816 drop is LogPopDensity alone). See §3.
- *Fix:* (a) Justify the four axes on construct/demand-theory grounds FIRST, not by lowest F_causal; (b) rest the robustness claim on **targeting-stability** (editor flags the same trajectories across housing-retaining sets), which the data DO support, not on the magnitude of the F_causal drop (mechanically non-decreasing in added regressors); (c) either switch to h-g-c-logpop or explicitly state MigrantRatio is decorative axis-coverage with ~0 marginal effect.

### IMPORTANT (fix before submission)

**I1 — Numbers transcribed as seed-0 instead of multiseed means.** README line 19 "bc 0.722" should be **0.725** (mean; 0.722 is seed-0). comparison_3v4.md line 18 uses seed-0 for both bc (0.7223→**0.7252**) and gan (0.7385→**0.7369**), mixing single-seed and mean values in one ordering. Note bc mean (0.7252) is tied with raw (0.7253; Δ7e-5) — printing 0.722 spuriously makes bc look ~0.003 below raw.
- *Fix:* Use the 5-seed means consistently; optionally note bc≈raw (n.s.).

**I2 — "p=0.031" is the n=6 Wilcoxon floor (2/2⁶), reported with no disclosure and no multiple-comparison control.** It certifies only "all 6 seeds agree in sign," carries zero magnitude information, yet ~36 arm×metric tests run uncorrected; at the floor, nothing survives Bonferroni.
- *Fix:* (a) Footnote that p=0.03125 is the n=6 floor = unanimous sign; (b) lead with effect sizes + t-CIs (the real discriminator — e.g. edited_w30 CI [0.024,0.031] excludes most_fair's CI, which is the formal basis for EDIT>SELECT); (c) state the multiplicity exposure or pre-register edited_w30 as the single primary endpoint.

**I3 — "Robust to the demographic choice" contradicts the project's own FRAGILE verdict.** The sensitivity module returns **FRAGILE** (F_causal spread 0.1779 vs 0.05 threshold; min Jaccard 0.5636 on drop_housing; min Spearman 0.7987). Robustness holds only within the housing-retaining family. Also: the shipped set's Spearman targeting-stability (0.874) is *below* the module's own ROBUST bar (0.90) and passes only the looser 0.80 gate.
- *Fix:* Scope precisely: "robust across **housing-retaining, low-VIF** sets (Jaccard ≥0.92); F_causal is feature-set-specific and collapses if housing is dropped." Report Spearman alongside Jaccard and reconcile the two internal thresholds.

**I4 — n=5 nulls (L2 transfer, variance) reported as evidence of absence, but Wilcoxon at n=5 cannot reach p<0.05 (floor 2/2⁵=0.0625).** The negative-transfer pillar is "tested" by a test structurally incapable of rejecting.
- *Fix:* For L2, rely on CI/equivalence framing: the 95% CI excludes a weighted-BC-sized (+0.027) effect → "vanilla < weighted" is defensible; "transfer = 0" is not provable at n=5. Use TOST or increase n if absence is to be claimed; replace "n.s." with "CI excludes a weighted-BC-sized effect."

**I5 — L1 "edited is fairest faithful" is partly definitional and does not propagate.** The editor optimizes F_causal (α_causal=0.7), then L1 re-scores on the same metric (std=0, deterministic), so the +0.0156 static gain is partly guaranteed by construction — and it vanishes once trained through a policy (L2 edited 0.7264 < raw 0.7274).
- *Fix:* Frame L1 as "the editor achieves its objective on the static artifact (optimized metric)"; reserve the scientific claim for L2/weighted-BC where the metric is not directly optimized. Don't headline 0.741 as an end-to-end result.

**I6 — Figure distortions undermining the paper's own theses:**
- `fig_l1_data_quality` right panel: Fidelity-A y-axis truncated to [0.80,0.86] makes edited (0.843) look materially less faithful than raw (0.848) when the true spread is only 0.006. *Fix:* axis from 0, or broken-axis marker + caption.
- `fig_feature_robustness`: "every conclusion holds" banner over an editor ΔF_causal that **flips sign** (−0.0004 → +0.0006) and is null (±0.0010 straddles zero), with no CIs drawn. *Fix:* add CIs or soften title; flag the editor-level ΔF_causal as null/sign-unstable.
- `fig_fidb_components`: "preserves trajectory shape" is an overclaim — edited coverage 0.093, RoG 0.102, net_disp 0.127 are non-trivial (net_disp *exceeds* GAN's). *Fix:* "editing concentrates its shift in terminal_cell (0.55); non-terminal components shift modestly but far less than the relocation signal." Restrict "preserves shape" to trajectory length only.

## (3) DEMOGRAPHIC-DEFENSIBILITY VERDICT

**Numerically sound, interpretively the weakest pillar — defensible ONLY if reframed around targeting-stability, not the F_causal magnitude.**

The strongest element: the sanity gate proves the sensitivity sweep is wired to the real metric (recomputed baseline matches production to 6e-8). So the criticism is entirely about **feature-CHOICE justification, not numerical correctness.** But four facts must be confronted honestly:

1. **The whole F_causal signal is LogPopDensity** (carries the full −0.0816); MigrantRatio buys ~0 marginal and is swapped in *over GDP* at a 57% higher VIF cost. The "population-structure axis" is decorative.
2. **A strictly-dominated-better set exists** (h-g-c-logpop): F-identical, lower VIF, keeps interpretable GDP. A reviewer will read the GDP→migrant swap as gratuitous.
3. **Construct validity of LogPopDensity as a *demographic/equity* axis is weak** — it's a demand-scale variable, and it's doing all the work, so the headline rests on the least demographically-defensible feature.
4. **Ecological-fallacy / 10-DOF exposure**: demographics are only 10 distinct district profiles broadcast over ~34,524 cells. All VIFs/correlations rest on 10 effective rows; F_causal applies a district-level lens at cell resolution.

**Recommended stance (already in project memory as a fallback and it is the honest one):** lead with "**the limitation is demographic data resolution (10 districts), not the editor.**" Justify features on demand-theory grounds, then claim robustness via *targeting-stability across housing-retaining sets* — the claim the data actually support. Do NOT use the size of the F_causal drop as evidence of "more real inequity" (mechanically non-decreasing in regressors). Switch to h-g-c-logpop OR explicitly disclose MigrantRatio's ~0 contribution. With these moves, the demographic analysis is defensible; as currently framed, it is not.

## (4) MINOR / OBSERVATIONS

- **Sink-cell coordinate inconsistency** (29,53) vs (28,52) across README/tables — the known +1 offset, but both labels appear unreconciled. Pick the 1-indexed (29,53) convention (matches JSON keys) and state the offset once.
- **README w20 ΔF_causal "+0.026" should be "+0.025"** (source 0.025467).
- **"Dose-responsive" is asserted, never trend-tested**, and saturates (w10→w20 +0.0064, w20→w30 +0.0020). Soften to "monotone-increasing across three weights (saturating)" or fit a Jonckheere/slope. Note most_fair's dose is *non-monotonic* (w20 null) — undercutting any "gradient" framing for SELECT.
- **random_w30 is "null" only on f_causal** — it moves Fidelity-B at the floor p (oversampling artifact). Qualify the placebo claim to the f_causal endpoint (this actually strengthens edit-specificity if stated).
- **Editor improves F_causal but slightly worsens F_spatial (−0.001)** — keep this separate from the +0.021 *cleanup*-driven F_spatial gain so the paper doesn't imply the editor improves both axes.
- **"edited" (w1) stored Wilcoxon p=0.4375 ≠ scipy default (0.3125)** — one diff is exactly 0; pin a single zero-handling mode across the pipeline.
- **L1 "validation gate passed"** is attributed in README to a file (L1v2 JSON) that has no gate field; the gate lives in the L2 metrics file. Point the claim at the right file.
- **Well-supported, no change needed:** GAN Fidelity-B disqualification (0.32 vs 0.15, corroborated by degenerate trip-length ~53 vs ~12); the sanity gate; and all recomputed headline numbers in the "verified-correct" list.

---
**Bottom line for the writer:** The experiments are real and the numbers are right. Before drafting, make four framing decisions — (a) rename or defend "causal," (b) reframe "edit-specific" → "edit-dominant," (c) scope "robust" to housing-retaining sets, (d) decide the demographic feature set (switch to h-g-c-logpop or disclose MigrantRatio's null contribution) — and fix the dose-response figure occlusion. Those five are the reviewer-killers. Everything else is a clean copy-edit pass.

Relevant files: `/home/robert/FAMAIL/PAPER/README.md`, `/home/robert/FAMAIL/PAPER/tables/comparison_3v4.md`, `/home/robert/FAMAIL/PAPER/data/L1v2_4feat_multiseed.json`, `/home/robert/FAMAIL/PAPER/data/weighted_bc_4feat_paired_stats.json`, `/home/robert/FAMAIL/PAPER/data/fcausal_feature_selection.json`, `/home/robert/FAMAIL/famail_temporal/analysis/paper_figures.py` (lines 145-153, 217-230, 314, 401-404, 478-524), `/home/robert/FAMAIL/famail_temporal/fairness/causal.py` (lines 90-158), `/home/robert/FAMAIL/famail_temporal/fairness/fcausal_feature_sensitivity.py` (thresholds lines 467/863).