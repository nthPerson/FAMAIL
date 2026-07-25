# ZHANG DRAFT DELTA — what in her revision PDF is authoritative vs stale

Her `Zhang_paper_revision.pdf` = her NEW front matter grafted onto an **older snapshot
of our draft** (pre the 07-24 read-aloud batches). Rule of thumb: **her abstract,
introduction, and Figure-1 concept are authoritative; every body section is stale and
our current `paper/sections/*.tex` supersedes it.** Never copy body text from her PDF —
you would silently revert a week of fixes.

## Authoritative in her PDF (adopt, with repairs)

1. **Title** — "Mitigating Demonstration Bias via Fairness-Aware Trajectory Editing".
   Already our title since 07-18. No action.
2. **Abstract** — full rewrite around budgeted fairness intervention. New vocabulary
   introduced here: HSTD (human-generated spatial-temporal data), outcome-side edits,
   resource-aware edits, edit-aware weighting, budgeted fairness intervention.
   Adopt near-verbatim; check every claim against ALGORITHM_FACTS.md (they scan
   accurate: "editing a small fraction … improves multiple fairness measures and
   increases distinct-taxi presence in under-served areas; edit-aware weighting
   preserves these gains in learned policies").
3. **Introduction** — full rewrite. Three italicized challenges (replacing our C1–C5):
   (i) *collective fairness is global, but edits are local*;
   (ii) *reducing disparity does not necessarily improve disadvantaged-group welfare*;
   (iii) *fairness intervention must be effective under a limited edit budget*.
   Three contribution bullets (formulation; FATE mechanism + outcome-vs-resource
   distinction; two-city instantiation + downstream transfer + open code/data).
   Repairs needed:
   - Broken cite `[35? ]` in ¶1 ("mobility modeling, simulation, and downstream
     decision applications") — cGAIL [zhang2019cgail] + second key TBD.
   - Broken cite `[? ]` on the anonymous-link sentence (contribution 3) — swap in the
     anon-repo footnote/URL per Robert's checklist; scope "code and dataset" honestly
     (ledger E14).
   - House style pass: em-dash policy, era numbers, no new coinages beyond her
     established vocabulary.
   - Meeting refinement (confirm in A-report): challenges mentioned BRIEFLY here,
     itemized fully in §2 Overview.
4. **Figure 1 concept** — the two-panel stylized map ("Biased Service in HSTD" → FATE →
   "FATE for Fairer Service") with Advantaged/Disadvantaged columns, Service/Demand
   labels, taxi/passenger icons, edited-trajectory dashes. Convert to TikZ
   (`figures/FIG1_TEASER_SPEC.md`). Her caption: "Collective service disparity emerges
   from the aggregation of local trajectories (left). FATE edits a small set of
   influential trajectories to improve corpus-level fairness."

## Stale in her PDF (ours supersedes — do NOT regress these)

| Area | Her PDF shows | Our current (272bb47) has |
|---|---|---|
| §3.1 | old problem formulation | + downstream-positioning sentence ("The imitation we demonstrate on this corpus (§downstream) is supervised…") |
| §3.2 F_demo opener | "Raw parity across neighborhoods is the wrong target" | recast: "Fairness here cannot mean equal service everywhere…" (Robert explicitly killed the old phrasing as non-human) |
| §3.2 spatial equity | "demand-service ratio", no source cite | **departure-service ratio** + `su2018taxigini` lineage cite (Su et al. 2018) |
| §3.4 opener | attribution-only summary | three-step opener (attribution/trim/lift) + first in-text Figure-2 reference |
| §3.4 trim ¶ | "Because M and I−H are idempotent…" | plain-language exact-split sentence (idempotence justification lives in App A) |
| §3.4 budget ¶ | "Freezing trim keeps its optimization identical… buys two properties…" | plain cause-and-effect recast + §4-ablation forward ref |
| App A | demand-service ratio; cite list order | departure-service ratio + su2018taxigini in both Gini cite lists |
| Bib | 39 entries | 51 entries incl. su2018taxigini; CITATION_PRIORITY_CHECKLIST has 9-row P0 pass |
| §2 Related Work position | still §2 | email moves it to §5 (content itself = ours, unchanged by her) |
| Figure 2 | old 3-panel stylized city | to be REPLACED by framework/sequence diagram (meeting decision) |
| Front matter | ACM ref format + permission block VISIBLE, placeholder venue ("Conference acronym 'XX… 2018") | we suppress the blocks but set REAL venue metadata (KDD '27). Action: un-suppress blocks, KEEP our real venue metadata (ledger E16). |

## Terminology propagation map (her vocabulary → where it lands in our body text)

| Her term | Maps to | Body usage |
|---|---|---|
| HSTD | our "trajectory corpus" framing | define once (abstract/§1), use where natural; don't force into every paragraph |
| outcome-side edit | trim (demand-side phase) | pair on first use: "outcome-side editing (the *trim* operation)" |
| resource-aware edit | lift (supply-side phase) | pair on first use: "resource-aware editing (the *lift* operation)" |
| edit budget K | our k (10,000 SZ / 2,000 SF) | keep lowercase k or adopt K consistently — one symbol everywhere (decide once; her email uses K) |
| collective fairness | corpus-level fairness | interchangeable; prefer "collective" in framing, keep "corpus-level" where already precise |
| fairness surrogate | the F_demo objective | "surrogate" is accurate (differentiable stand-in optimized during editing) |
| edit-aware weighting | our upweighting recipe | adopt her name for the stage; keep "upweighting"/instance-reweighing lineage in the body |
| value-of-resource map | our value-of-presence map | 🟡 pick ONE (hers in definitions with ours parenthesized, or keep ours — decide in spec) |

## Numbers in her PDF to treat with care
- Her intro/abstract cite no numbers — good, nothing to era-check there.
- Her stale body carries the CURRENT era numbers already (+0.0226 etc.) because the
  snapshot postdates the α* campaign — but do not copy body text anyway (see above).
- Her §3.1 says "2,455 trim" in budget ¶ and §4.1 says "2,337 trim after filtering" —
  both correct in their places (selected vs net); preserve that distinction wherever
  the numbers move.
