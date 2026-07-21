# Figure-1 design meeting — talking points (week of 2026-07-20)

*Prepared for Robert's meeting with Dr. Zhang. Companion renders:
`preview-3strip-final.png` (the archived 3-strip), `comparison-2026-07-20.png`
(all three variants side by side).*

## Where things stand (open with this)

- Your request is **already executed**: the live manuscript Figure 1 is the
  (c)-only build, committed. Nothing in this meeting blocks the paper — if we
  change nothing, we submit with (c)-only.
- This is a prepared alternative for discussion: **(b)+(c)** — drop only the
  city panel (a), keep the "data worth preserving" panel (b) and the
  two-futures panel (c). It exists as a ready file
  (`figure-1-bc.tex`); adopting it is a one-line `\input` swap plus a caption
  paste, five minutes of work at any point before the deadline.

## The case for (b)+(c)

1. **Figure 1 + Figure 2 should tell the whole project's story.** Figure 1 is
   the WHY, Figure 2 the HOW. With (c)-only, Figure 1 shows only the *stakes*
   (two corpora, two futures). The middle beat — *the recorded human data is
   valuable, so FATE edits it rather than regenerates it* — loses its only
   visual home anywhere in the paper.
2. **That beat is the method's identity.** The title says trajectory
   *editing*; the abstract's closing claim is "without fabricating data."
   Panel (b) is the one place a reader *sees* the rejected door ("regenerate
   the corpus? — realism lost", struck through) next to the chosen one
   ("edit the most-biased slice"). In (c)-only, the edited-corpus chip
   asserts that editing happened, but nothing shows why editing was the
   right verb — the related-work contrast (\S2) loses its figure anchor.
3. **The space concession is essentially free.** Measured on rendered acmart
   pages (table below): (b)+(c) costs only **~0.6 cm ≈ 1--2 text lines** more
   than (c)-only. The reason is structural: nearly all of the savings in
   *both* variants comes from dropping the city panel (a) — which both do —
   and (c)-only has to re-introduce two labeled corpus chips as arrow sources
   (the corpus objects panel (b) used to provide), which claws back most of
   panel (b)'s removal. We are not trading a page for the story beat; we are
   trading two lines.
4. **The page budget does not hinge on those lines.** The ranked cut plan
   reaches 8.0 content pages primarily through appendix relocations (the
   α-Pareto figure and the feature-set table alone free roughly a page);
   Figure 1's final few lines are not load-bearing for the target.

**Measured size of each variant** (figure + caption block, rendered in the
acmart column-width harness at 150 dpi, 2026-07-20; one acmart text line
≈ 0.37 cm):

| Variant | Figure + caption block | Saved vs 3-strip |
|---|---|---|
| A — 3-strip (archived) | 15.4 cm | — |
| B — (b)+(c) counter-proposal | 12.0 cm | 3.4 cm ≈ 9 lines |
| C — (c)-only (live) | 11.4 cm | 4.0 cm ≈ 11 lines |

Difference B vs C: **0.6 cm ≈ 1–2 lines.** (Both variants carry Robert's
2026-07-20 upgrades: the FATE provenance arrow on C, and on both the boxed,
centered legend with a third entry glossing the accent $+$ marks.)

## Anticipated objection

- *"The intro prose already says the data is valuable."* — It does; but the
  service gap is also in the prose and we still give the stakes a visual.
  The beats that define the method are the ones that earn figure space, and
  edit-not-regenerate is the differentiating beat.

## Fallback

If Dr. Zhang still prefers (c)-only: we keep it — it is already the live
figure, no work is lost (the 3-strip is archived, the (b)+(c) variant stays
in the repo as design history).

## Also on the agenda for this meeting

- **PI acknowledgment owed — SF two-tier framing (Reading B / D1):** the
  distinct-taxi tier-2 recount confirmed the supply reading (supply_tier2
  +0.1027, CI-significant; tier-2 total +0.0493, significantly positive) —
  the tier-1 net-negative was a fractional-presence accounting artifact.
  \S4.7 states this; Dr. Zhang has not yet been walked through it.
- **Brighter colors** (promised for both figures either way): direction is
  raising chroma / a second accent while staying grayscale- and CVD-safe —
  can preview once the Figure-1 geometry question is settled.
