# TASK BRIEF T1 — adopt Dr. Zhang's abstract (with claim repairs)

Replace the body of `paper/sections/00_abstract.tex` with the PI's new abstract,
repaired where its wording would misstate the algorithm. Her abstract's SUBSTANCE and
vocabulary are authoritative (per her email: "follow the logic of my revised abstract
and introduction"); the repairs below are pre-authorized by Robert's standing rule
that no directive may misrepresent the method.

## Read first
1. `paper/restructure/ALGORITHM_FACTS.md` — ground truth (especially §Validity and
   §Things-FATE-is-NOT).
2. `paper/restructure/meeting/analysis_C_claims.md` §5 (do-not-claim list) + D15.
3. `paper/sections/00_abstract.tex` — the current file: PRESERVE its header comment
   block (the "Edit the abstract HERE, nowhere else" contract + provenance comments),
   appending a new provenance line: adopted from the PI's revision draft 2026-07-25,
   restructure T1.
4. `paper/restructure/ZHANG_DRAFT_DELTA.md` §Authoritative (context for what you are
   installing).

## Her abstract, verbatim (from her revision PDF; the ONLY source you need)

> Human-generated spatial-temporal data (HSTD), such as taxi trajectories, passenger
> trip records, and gig-worker traces, encodes not only human decision-making
> strategies but also demographic disparities in how services are distributed across
> groups and regions. When such data is used for mobility modeling, simulation, and
> decision-making, these disparities can be learned and propagated to downstream
> applications. Existing approaches either modify large portions of the data without
> tracing corpus-level disparity to specific trajectories, or constrain downstream
> models while leaving the biased demonstrations unchanged. We introduce FATE, a
> Fairness-Aware Trajectory Editing framework for budgeted fairness intervention in
> HSTD. Under a limited editing budget, FATE attributes corpus-level disparity to
> influential trajectories and applies bounded local edits with the greatest
> aggregate fairness impact. It distinguishes outcome-side edits, which alter
> measured allocation statistics, from resource-aware edits, which modify the
> underlying service-provision process and direct additional resources toward
> under-served areas. All edits satisfy spatial, continuity, and behavioral-fidelity
> constraints. Because the edited demonstrations constitute only a small fraction of
> the corpus, their fairness signal can be diluted during downstream training. FATE
> therefore couples a limited edit budget with edit-aware weighting to preserve the
> influence of these targeted interventions. We instantiate FATE for taxi mobility
> and evaluate it on real-world HSTD. Results show that editing a small fraction of
> the corpus improves multiple fairness measures and increases distinct-taxi presence
> in under-served areas. Edit-aware weighting further preserves these gains in
> learned policies, demonstrating that sparse, targeted trajectory intervention can
> improve collective service fairness with limited data modification.

## Required repairs (each is one surgical change; keep everything else near-verbatim)
1. **"All edits satisfy spatial, continuity, and behavioral-fidelity constraints."**
   Fidelity is NOT a satisfied-by-construction constraint (D15): the spatial ε-bound
   and continuity ARE enforced; fidelity is a frozen driver-identity guardrail scored
   in the objective. Reword to keep her rhythm, e.g.: "All edits satisfy spatial and
   continuity constraints under a frozen driver-identity guardrail." (Your final
   wording may differ; it must not claim a per-edit fidelity gate.)
2. **"bounded local edits with the greatest aggregate fairness impact"** — soften the
   implied optimality one notch (selection is exact-attribution- and screen-guided,
   not provably optimal): "with the greatest attributed fairness impact" or
   "estimated" — pick one, smallest possible change.
3. Sanity-check every remaining claim against ALGORITHM_FACTS §Headline results —
   they should all pass as written ("improves multiple fairness measures" = ring-iii
   external instruments ✓; "increases distinct-taxi presence in under-served areas" =
   tier-2 supply channel, both cities ✓; "preserves these gains in learned policies" =
   w30 flagship ✓; "small fraction" ≈ a tenth ✓). If anything else fails your check,
   flag it in your reply rather than silently rewording.
4. House style: no em-dashes; "downstream applications" phrasing is fine; HSTD is
   being DEFINED here (first use in the paper) — keep the parenthetical definition
   exactly as she wrote it.

## Gates + checks
- `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- Also compile the standalone abstract doc (it shares this file):
  `latexmk -pdf -interaction=nonstopmode -halt-on-error kdd27-abstract-only.tex` —
  report pass/fail; if it fails for a pre-existing reason unrelated to your edit, say
  so, don't fix it.
- `pdftotext -f 1 -l 1 main.pdf - | head -40` — eyeball the rendered abstract for
  encoding artifacts.

## Rules
- Touch ONLY `paper/sections/00_abstract.tex`. Robert's uncommitted edit in
  `paper/sections/04_experiments.tex` must remain untouched. No `git commit`/`add`.
- If Write/Edit is blocked, return the complete new file contents as text, status
  BLOCKED(write-denied).

## Final reply (machine-read, ≤15 lines)
Status; the exact wording you chose for repairs 1–2; any claim that failed check 3;
gate results (both builds); first-page eyeball verdict.
