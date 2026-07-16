# SF fairness framing — DECISION DOC for Meeting 43 (2026-07-16)

**What this is:** the one open PI decision blocking the manuscript's only remaining marker
(`% TODO(PI-framing)`, `paper/sections/04_experiments.tex` §"External Validity: San Francisco").
Both readings below are already drafted side-by-side in the paper; **the decision is which one
LEADS. Nothing is omitted either way — this is emphasis, not omission.**

*For the slide build (Claude Cowork): this doc supersedes/expands MEETING_43_PREP §4 item 2.
The decision block at the bottom is the meeting deliverable.*

---

## The tension, in one paragraph

On San Francisco at α\*, the **supply channel replicates** — lift adds real presence in
under-served areas, positive and significant (**+0.0209**, CI [+0.0122, +0.0300]). But lift also
reroutes *pickups* into those same areas, so **recorded demand there rises**, and the
supply/demand **ratio** for the under-served group falls: total Δmean(Y|disadv.) is
**net-negative** (−0.0324, CI excludes 0; demand channel −0.0533, significant). Shenzhen does not
show this tension (total +0.0529, CI excludes 0). So SF's verdict depends on which quantity you
treat as the claim.

## Reading A — "ratio reading" (conservative)

- **Statement:** the SF lifting-up claim is **withheld**: the pre-registered design-targeted
  quantity (the under-served group's supply/demand ratio) moves the wrong way.
- **Strengths:** maximally conservative; immune to any "post-hoc metric selection" charge;
  the supply-channel replication (+0.0209\*) is still reported as mechanism evidence.
- **Costs:** the abstract/intro external-validity sentence weakens ("conclusions reproduce"
  must be scoped); hands SF to reviewers as a partial failure we declared ourselves.

## Reading B — "external-metrics reading" (demand-endogeneity finding)

- **Statement:** more rides *served* in under-served areas is a demand-side parity gain — **DP,
  DI, and Theil all improve on SF** — and the falling ratio is precisely the demand-endogeneity
  mechanism §3.4 *predicts*: recorded demand was suppressed; serving it raises the denominator.
- **Strengths (all at α\*, all committed):**
  - SF migrant DP significant under **both** grouping conventions (extremes −0.0729\*,
    median-split −0.0370\*);
  - SF weighted-BC recovery reproduces (+0.0332 @ w30, 6/6 seeds; both controls fail);
  - SF four-source table reproduces (edited fairest 0.9067, identity-faithful);
  - the ratio's fall is *explained by the paper's own theory*, not explained away.
- **Costs:** a skeptic can still say the pre-registered ring-(ii) quantity failed and we leaned
  on ring-(iii) to rescue it (reviewer-simulation objection #3 states this verbatim).

## What the reviewer simulation adds (new since the prep brief)

1. **Whichever reading leads, present the tension as a *finding about demand endogeneity*, not
   as an unresolved fork the reader must referee.** The current draft literally says "Two
   readings coexist and we present both" — a reviewer reads that as us not knowing our own
   result. The fix is the same sentence content with a spine: state the mechanism, then the
   scoped claim.
2. **The run that dissolves the tension exists: D1, the SF tier-2 distinct-taxi recount**
   (~1–2 engineering days + ~1h run; `supply_recount.py` currently supports Shenzhen only).
   Shenzhen's most convincing lift-up evidence is tier-2 (+0.0411); giving SF the same recount
   would let SF stand on supply-side evidence **independent of the contaminated ratio** —
   converting the weakness into a replicated mechanism. If D1 is approved and lands, Reading B
   stops being a "rescue" and becomes the natural statement of the result.

## Decision block (fill in at the meeting)

- [ ] **Reading A leads** (ratio; SF lift-up withheld; supply channel reported as mechanism)
- [ ] **Reading B leads** (external metrics; demand-endogeneity finding; ratio disclosed)
- [ ] **Approve D1** (SF tier-2 recount engineering, ~1–2 days) — strengthens either choice,
      near-decisive for B
- Notes: ______________________________________________________________

**Consequence of the choice:** one paragraph of §4.7 gets re-ordered + the abstract's SF clause
scoped to match; the `TODO(PI-framing)` marker is retired. Everything else in the section stays.

*Sources: `PAPER/supply-lift/data/a10/sf12_a10_{metrics,channel_decomposition,external_fairness}.json`;
decision context `MEETING_43_PREP.md` §4 item 2; reviewer analysis
`paper/reviews/2026-07-15-reviewer-sim.md` objection 3.*
