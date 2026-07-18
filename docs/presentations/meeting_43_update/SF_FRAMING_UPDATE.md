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

## Decision block

**⚠️ OUTCOME (2026-07-16, from the meeting transcript): the SF framing was NEVER RAISED in the
meeting — no decision was made.** Robert spoke with Dr. Kash privately afterward (unrecorded);
whether the framing was settled there is an open question for Robert. Until answered, the
`TODO(PI-framing)` marker stands and both readings stay drafted side-by-side.

- [ ] **Reading A leads** (ratio; SF lift-up withheld; supply channel reported as mechanism)
- [x] **Reading B leads** (external metrics; demand-endogeneity finding; ratio disclosed) —
      **DECIDED by Robert 2026-07-16 (post-meeting), PROVISIONAL pending D1.** Rationale:
      (1) SF demographic units are census tracts — far smaller than Shenzhen's districts — so
      the ratio is naturally more sensitive to lift's rerouted pickups (granularity intuition,
      now stated hedged in §4.7); (2) the external-metrics evidence (DP under both groupings,
      DI, Theil, WBC recovery) is strong enough with careful framing of the rerouting finding;
      (3) D1 backstops the choice. **Fallback: if D1's tier-2 recount does not support the
      reading, reassess toward Reading A.**
- [x] **Approve D1** (SF tier-2 recount engineering, ~1–2 days) — approved by implication
      2026-07-16 ("the planned D1 run"); schedule after the fairness-baseline suites.
- Notes: §4.7 rewritten with Reading B leading (demand-endogeneity finding, ratio reading
  disclosed); `TODO(PI-framing)` retired with a provenance comment. PI acknowledgment still
  owed at the next Zhang/Kash touchpoint (Zhang reviews week of Jul 20).

**Consequence of the choice:** one paragraph of §4.7 gets re-ordered + the abstract's SF clause
scoped to match; the `TODO(PI-framing)` marker is retired. Everything else in the section stays.

---

## D1 OUTCOME (2026-07-18) — Reading B CONFIRMED; provisional status retired

The D1 SF tier-2 recount ran gates-first (G-repro MAE exactly 0.0 vs the production
`active_taxis_3d`, all 4,230 active cells; substitution replay 1959/1959 matched) and the
pre-committed decision rule fired on its **upgrade** branch:

- **supply_tier2 = +0.1027, CI [+0.0872, +0.1203] — significant** (the rule's condition;
  largest distinct-taxi supply effect of any corpus: SZ feature sets are +0.0411 / +0.0211 / +0.0771).
- Beyond the rule: **total_tier2 = +0.0493, CI [+0.0185, +0.0790] — significantly POSITIVE**,
  where the tier-1 total was significantly negative (−0.0324). The SF mean(Y|D) tension was a
  tier-1 (fractional-presence) accounting artifact; counted as distinct taxis, the supply
  improvement outweighs the demand-side denominator increase.

§4.7 upgraded accordingly (lower-bound disclosure replaced by the two-tier statement; the
"ratio does not replicate" heading replaced — the ratio tension *dissolves* under tier-2).
The Reading-A fallback is moot. PI acknowledgment of the framing (now with D1 evidence in
hand) still owed at the next Zhang/Kash touchpoint.

*D1 sources: `PAPER/supply-lift/data/a10/sf12_a10_{supply_recount,channel_decomposition}.json`
(ledger D1-RECOUNT / D1-CHAN, 2026-07-18); spec + addendum
`docs/superpowers/specs/2026-07-17-d1-sf-tier2-recount-design.md`.*

*Sources: `PAPER/supply-lift/data/a10/sf12_a10_{metrics,channel_decomposition,external_fairness}.json`;
decision context `MEETING_43_PREP.md` §4 item 2; reviewer analysis
`paper/reviews/2026-07-15-reviewer-sim.md` objection 3.*
