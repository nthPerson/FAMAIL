# Meeting 44 — Slide Plan (since Meeting 43 + road to Monday's deadline)

> **This doc is the slide spine** (hand-off to slideshow generation). 9 slides: *status (1) →
> what happened since Meeting 43 (5, incl. the Figure-1 redesign) → the length campaign and the
> new reproducibility record (2) → the remaining TODO list (1).* Each slide gives a **title**,
> the **on-slide content** (terse bullets + at most one table), and a **"the point"** speaker
> note (not on the slide). Numbers are α\*-era and committed; deep provenance lives in
> [`MEETING_44_PREP.md`](MEETING_44_PREP.md) and `PAPER/REPRODUCIBILITY.md`.
>
> **Assets for the deck builder** (paths relative to the repository root):
> - `docs/presentations/meeting_44_update/assets/figure-1-live-cropped-from-manuscript.png` —
>   the live (c)-only Figure 1 **with its manuscript caption**, cropped from the
>   `paper/main.pdf` build of 2026-07-22. (The caption was lightly trimmed on 07-23 in the
>   argument-triage pass; the figure itself is identical — re-crop only if caption fidelity
>   matters for the deck.)
> - `paper/figures/figure-1/design-archive/comparison-2026-07-20.png` — all three Figure-1
>   variants side by side with measured sizes in the headers (landscape sheet).
> - `paper/figures/figure-1/design-archive/preview-3strip-final.png` — the archived original
>   3-strip Figure 1 (deliberately keeps the *old* palette, so it doubles as the color
>   before/after).
> - ⚠️ Do **not** use `paper/figures/figure-1/preview.png` — it predates the redesign (stale).
> - Figure 2 (method overview) has no standalone PNG; if wanted, crop it from the compiled
>   `paper/main.pdf` (it is the only full-width figure, on the methodology pages).
> - All tables below are ready to paste as markdown.
> - House rules: grayscale-safe, one accent color, clean minimal layout.

---

## Slide 1 — FATE: five days from the deadline, nothing left but decisions

**On the slide**
- **FATE — Fairness-Aware Trajectory Editing.** Paper: *"Mitigating Demonstration Bias via
  Fairness-Aware Trajectory Editing"*, KDD 2027 Research Track.
- **Abstract submitted 2026-07-19** (Cycle 1, OpenReview). Full paper due **Monday 2026-07-27,
  23:59**.
- Manuscript state: **content UNDER the strict 8-page limit** (references begin on page 8) +
  structured appendix, 12 pages total; anonymous sigconf build with real venue metadata; **all
  experiments closed — no runs pending or planned**.
- Today: walk through what changed since Meeting 43, ratify the final trims, and close the
  short list of remaining items.

**The point:** the abstract is in, the science is finished, and the manuscript now FITS — this
meeting is about ratifying the argument triage that got it there and the handful of remaining
items, not about new results or open length questions.

---

## Slide 2 — Since Meeting 43 (1/4): submitted and renamed

**On the slide**
- **Abstract re-written and submitted (Jul 19).** Rebuilt to Dr. Zhang's four-beat structure,
  then polished; single-sourced in the repo so the standalone abstract and the manuscript can
  never drift. ("We can always modify the submission.")
- **Title** leads with the problem, per Dr. Zhang: *Mitigating Demonstration Bias via
  Fairness-Aware Trajectory Editing*.
- **Method named FATE** (replaces the FAMAIL working name).
- **F_causal → F_demo** executed everywhere per Dr. Zhang's approval (Jul 20): prose, equations,
  tables, and the regenerated frontier figure; the associational caveat stays. Code and artifact
  keys keep `f_causal` (mapping recorded in the paper README).

**The point:** every naming decision from the Meeting-43 thread is now executed and committed —
the paper's identity (title, method name, fairness symbol) is settled.

---

## Slide 3 — Since Meeting 43 (2/4): the writing follows the outlines

**On the slide**
- **Introduction re-written to the six-beat outline:** gap → why existing interventions fall
  short (new intervention-categories paragraph) → FATE's position → mechanism → results *with
  intervals* (DI +0.0162, DP gap −0.890 (14.199 → 13.309), Theil −0.0087) → contributions.
- **Related work:** each of the four themes now closes with the concrete limitation FATE
  addresses ("state the contrast").
- **Citations:** FairGAN, DECAF, and a §4.4 method-citation pass (iFGSM/FGSM/PGD/oversampling
  class; one new DBLP-verified entry, Madry et al.) are machine-verified against primary
  sources; Robert's human pass per Dr. Kash's mandate is the remaining step (three refs).
- Robert's **read-aloud editing pass** now covers **§1 through §4 complete** — it also drove a
  reader-clarity sweep (tier-1/tier-2 renamed *fractional-presence* / *distinct-taxi*
  accounting; Fidelity-A/B defined at the source; seed-count notation defined; flashy register
  retired). Next: polish §5, check the appendix.

**The point:** both structural requests from the advisor threads (six-beat intro, contrast
closes) are in; citation hygiene is at "machine-verified, human pass pending" — exactly where
Dr. Kash's desk-rejection warning says it must not stay.

---

## Slide 4 — Since Meeting 43 (3/4): four experiments closed the remaining gaps

**On the slide**

| Result (all landed in §4) | Headline |
|---|---|
| **SF two-tier supply recount (D1)** | Counting *distinct vehicles*, the lift-up supply channel is **+0.1027** (CI-sig) and the distinct-taxi total is **+0.0493** (sig-positive) — the earlier net-negative was a fractional-presence accounting artifact. (Paper now says "fractional-presence" / "distinct-taxi"; "tier-1/tier-2" retired as reader-hostile.) |
| **Flagship n=12, both cities** | w30 recovery replicates on 12 paired seeds per city: **p = .00049, 12/12 positive** (SZ +0.0297 ± 0.0029, SF +0.0333 ± 0.0050) |
| **SF n=12 controls** | Random-slice upweighting *degrades* fairness; most-fair-slice selection is ~6× smaller than the edited slice — the effect is **edit-specific** |
| **Penalty-formulation probe** | The fairness-penalty baseline's failure is formulation-independent (absolute-value variant tracks the signed one) |

- ⚠️ **The SF two-tier framing (§4.7) was decided and executed after Meeting 43 — walking Dr.
  Zhang through it is an agenda item for today.**

**The point:** the two-tier recount resolves the one tension in the SF story (the supposed
net-negative was an accounting artifact, not a real harm), and the n=12 replications turn the
flagship transfer result into a p = .00049 claim in both cities. Nothing is left to run.

---

## Slide 5 — Since Meeting 43 (4/4): Figure 1, redesigned and settled

**On the slide**
- *(Main visual: `docs/presentations/meeting_44_update/assets/figure-1-live-cropped-from-manuscript.png`
  — the live figure as it appears in the current build.)*
- Dr. Zhang's call ("only showing (c) would be sufficient") is **executed and kept**: the live
  figure is the (c)-only variant. Robert closed the (b)+(c) counter-proposal on Jul 21 — with
  the 8-page limit binding, "simplicity is a strength." The variants remain archived as design
  history.
- What the redesign added: labeled **raw/edited corpus chips** (with real GPS-trace renderings)
  as the arrow sources, an explicit **FATE provenance arrow** (the edited corpus visibly *comes
  from* the raw corpus), and a boxed legend with a third entry — "+ changed by the edit."
- Size: 15.4 cm (original 3-strip) → **11.4 cm**, ~11 text lines returned to the page budget.
- *(Optional side-by-side: `paper/figures/figure-1/design-archive/comparison-2026-07-20.png`.)*

**The point:** the figure decision from the email thread is fully executed; the redesigned
figure keeps the paper's defining beat (edit, don't regenerate) legible while paying for itself
in page budget.

---

## Slide 6 — Figures: the color refresh (both figures)

**On the slide**
- The "more engaging colors" promised by email are applied to **both** figures: a two-hue
  system — **muted cobalt = added/edited (the FATE intervention)**, **muted amber =
  excess/trimmed**, charcoal neutrals, and pale regional tints marking over-/under-served areas.
- **Verified grayscale-safe and CVD-safe** (blue–amber is the colorblind-safe axis; shapes,
  dashes, and "+" marks carry all semantics without color).
- Figure 2 also gained the glyph vocabulary from the Meeting-43 request: passenger stick figure
  = service pickup, car glyph = taxi presence; panel 1 retitled "(1) Attribute: locate the
  deficit."
- *(Before/after: `paper/figures/figure-1/design-archive/preview-3strip-final.png` (old palette)
  vs the live-figure crop on slide 5.)*
- **Ask: confirm the color direction.**

**The point:** the palette is a deliberate semantic system (one hue for the intervention, one
for the excess it corrects), not decoration — and it survives grayscale printing and colorblind
simulation. Looking for a thumbs-up, not a redesign.

---

## Slide 7 — The length campaign: 10.6 → under 8.0 pages, zero content deleted

**On the slide**
- Robert + Claude ran a measured cut campaign against the strict 8-page submission limit:
  **10.6 → under 8.0 content pages** — eight waves + a §3/§4 restructure (07-21), then a
  final **argument-triage campaign** (07-22/23) with per-item approvals.
- Argument triage, Robert's editorial call — *a simple argument beats pre-emptive defense*:
  the rollout **allocation-boundary** disclosure, the **SF downstream detail**, the
  **distinct-taxi recount mechanics**, and several depth items moved to the appendix (every
  number preserved, origin-labeled, **restorable at camera-ready**: 9 content + 12 total on
  acceptance). Kept deliberately: the random-jitter surprise, the leveling-down refinement,
  Dr. Zhang's §2 contrast cadence and six-beat intro.
- **Zero content deleted:** claims, numbers, and disclosures live in main text or appendix; a
  ~55-number audit of §4 found **0 mismatches**; lint gate tightened 8pt → 5pt.
- ⚠️ Margin is knife-edge (references begin ~95% down page 8) — additions now displace
  something. **Ask: ratify the triage.**

**The point:** the length problem is solved and nothing was destroyed — the submission carries
the simple core argument, the appendix carries the depth, and all of it returns at
camera-ready. The ask is a thumbs-up on what moved, not a decision about what to cut.

---

## Slide 8 — New: the reproducibility record (`PAPER/REPRODUCIBILITY.md`)

**On the slide**
- The Meeting-40 capstone (T17), landed this week: **every headline claim → curated artifact →
  raw results directory → run-ledger row → exact command + environment record.** 39 claim rows
  covering the Shenzhen editor, downstream suites, baselines, feature-set robustness, and the
  SF replication.
- **"Where does this number come from?" is now a one-lookup question** — with per-artifact
  SHA-256 checksums and environment fingerprints.
- Era discipline made mechanical: verify via the artifact's own `config_snapshot` and
  edit-count fingerprints, never a directory name. Name translations recorded once
  (repo "famail" = paper "FATE"; artifact key `f_causal` = paper symbol F_demo).
- Evidence, not just claims: a read-only audit found **zero correctness-critical
  discrepancies**, and the headline corpus **re-derives exactly** end-to-end under clean `main`.
- **PII-free by construction → ready to seed the anonymous artifact repository** (the
  Meeting-43 anonymity workstream).

**The point:** this is the trust document — for reviewers, for the PI, and for the anonymous
repo. It also closed the one reviewer-facing statistics gap (the robustness table's lift-up
cells now carry significance: HGC +0.0594, 4FEAT +0.1461, all intervals excluding zero).

---

## Slide 9 — Still on the TODO list (five days out)

**On the slide**
- ⚠️ **Needs Dr. Zhang today:**
  1. **Ratify the argument triage** (slide 7) — what moved to the appendix and the
     restore-at-camera-ready list. The length problem itself is solved.
  2. **Data-availability / licensing statement** for the datasets — exists nowhere yet; needed
     for the anonymous artifact repo.
  3. **Reading-B / D1 acknowledgment** — walk-through of the SF two-accounting framing
     (slide 4).
  4. **Colors** — confirm the direction (slide 6).
- **Robert, before Monday:**
  5. Polish §5 and check the appendix (the read-aloud pass covers §1–§4) — Claude folds edits
     in batches.
  6. Citation human pass, three refs: FairGAN (IEEE Xplore page — bot-blocked, genuinely needs
     human eyes), DECAF (proceedings-page eyeball), Madry et al./PGD (OpenReview eyeball).
  7. Anonymous artifact repo: scrub repo docs/comments (manuscript-side PII scan is already
     clean); seed with `REPRODUCIBILITY.md`.
  8. Push local commits; final PII check; **submit by Mon 2026-07-27, 23:59.**

**The point:** everything left is either a PI thumbs-up (items 1–4, today) or bounded solo work
with a clear recipe (items 5–8). No experimental, structural, or length risk remains between
here and submission.
