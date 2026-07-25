# TASK BRIEF T10 — close the page-9 spill to the KDD 8-content-page limit

⚖ Robert authorized (2026-07-24 late) ALL FOUR levers below, full cut round tonight.
KDD 2027 CFP (verified): 8 content pages hard, then references + UNLIMITED appendix.
Current state: ~81 rendered lines of content on p9 before REFERENCES.
**Success = `pdftotext -f 9 -l 9 paper/main.pdf -` opens with the REFERENCES heading.**
Work biggest-first, run gates + re-measure after EVERY lever, stop the moment the
target is met (do not over-cut), and if all four levers still leave a residual, STOP
and report it — no unauthorized cuts (specifically: PI-adopted abstract/intro
sentences, tables, and anything not named below are untouchable).

Governing rule for every lever: **relocation, not deletion.** Load-bearing content
(numbers, disclosures, mechanisms, controls) moves to the appendix, which has no page
limit; only genuine redundancy is deleted outright. `% src:` and `% lint-allow:`
comments travel with their sentences. Every cut/moved block gets a dated in-file
comment naming this task. Appendix insertions must read as prose in the existing
appendix voice (App C "Extended Results" already has per-arm subsections — extend
them naturally, never paste dumps).

## Lever order

1. **§4.4 baselines detail → App C (~20–30 lines).** `04_experiments.tex`: the
   Data-Augmentation Baselines subsection keeps, in the main text: its framing
   opener (editing-quality baselines; fairness expected NOT to improve), Table 2
   (untouched), and a ~10–14 line summary that answers RQ4: the three perturbation
   arms land far below the editor and break adjacency wholesale (one sentence,
   keep the 98.8% king-violation + 0.447-vs-0.187 divergence contrast OR move the
   numbers to App C with a qualitative main-text clause); the random-jitter
   fairness surprise exists but is bought by broken trajectories (keep — it is a
   pre-registered-expectation disclosure); targeted oversampling captures part of
   the gain at 10.5% fabrication while FATE reaches more at zero inflation; the
   untargeted placebo DEGRADES fairness and its DP-gap explosion (+2.8) is the
   §3.3 endogeneity concern made concrete (KEEP this sentence in main — protected
   flavor); the training-side fairness baselines (reweighing moves fairness the
   wrong way; the in-processing penalty is inert-then-destructive) compress to
   ~2 sentences with their numbers, full grids already in App C. Everything
   displaced lands in App C under the existing per-arm/λ-grid material with a
   forward pointer from main ("per-arm detail: Appendix C").
2. **§4.1 protocol trim vs App D + §4.5 first-¶ trim vs App C (~12).** The
   Protocol-and-statistics block keeps: paired seeds, the n=6 sign-unanimity
   reading of p=0.031, the n=12 flagship note, bootstrap scope + understated-
   uncertainty caveat — each as ONE clause; the expanded explanations are already
   in App D verbatim (verify before cutting; if a clause is NOT in App D, move it
   there). §4.5's first paragraph (control rows re-measured deliberately…) trims
   against App C Table 3's caption+notes the same way.
3. **C1–C5 compression (~10).** `02_overview.tex`: each challenge to ~2 tight
   sentences. Every load-bearing clause survives (C1 budget k≪|T| + whole-corpus
   difficulty; C2 read-as-driver + bounded + discriminator-scored-not-gated; C3
   demand-unexplained variation; C4 the demand-only-does-exactly-this empirical
   clause + adds-real-presence; C5 averaged-away + traced-to-edits-not-
   reweighting). Keep the stacked no-itemize format and the lead-in sentence.
   Robert reviews these Saturday — flag the before/after in your report.
4. **§3.3 structural reasons → appendix (~7).** `03_methodology.tex`: the three
   structural reasons + the greedy-search bound compress to one main-text sentence
   ("Three structural facts make this the constrained optimum rather than an
   optimizer artifact — selection, leverage at the demand floor, and supply-side
   inequity a demand editor cannot touch (Appendix~B/C)"); the full versions
   relocate to the appendix (App B editor details or App C — pick where they read
   naturally, likely a short new named paragraph). The 2,455 empirical fact, the
   leveling-down analogy + conservation caveat, the endogeneity paragraph, and the
   one-non-perverse-lever close STAY in the main text untouched.
5. **Fig-2 spacing (only if still short, ~5).** `figures/figure-2/framework.tex`:
   tighten inter-band vertical gaps; the measured-box gates in framework-test.tex
   must still pass; re-render the preview and LOOK at it.

## Gates (after every lever + final)
`cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
plus the p9 check. Zero undefined refs throughout. Final: read the FULL rendered
§4 + §2 + §3.3 via pdftotext for mid-thought sentences (the relocation seams are
where they happen).

## Rules
Files: `02_overview.tex`, `03_methodology.tex`, `04_experiments.tex`,
`appendix.tex`, and (lever 5 only) `figures/figure-2/framework.tex` +
`framework-test.tex`. No git commands. No number changes anywhere — numbers MOVE,
never change. If Write/Edit blocked: exact edits as text, BLOCKED(write-denied).

## Final reply (machine-read, ≤20 lines)
Status; per-lever measured savings + running p9 state; the compressed C1–C5
verbatim; what landed where in the appendix (one line per relocation); final p9
check result; gate results; any residual + what you did NOT touch.
