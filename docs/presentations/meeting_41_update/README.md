# Meeting-41 group-update deck — build instructions

This folder specifies a short slide deck for a **~15-minute group research update**. The audience is a
**technical research team that does not know this project's internals** — favor plain language, one
headline per slide, a few bullets, and the key numbers.

## What to build

- **`slides.md`** is the content spec. Build **one slide per `## Slide` / `## …` section, in order:**
  Title → Slide 1 (roadmap) → Slide 2 → Slide 3 → Slide 4 → Slide 5 → Closing overview.
- Text under **ON SLIDE** is the slide's actual content — use it **verbatim** or lightly trimmed.
- **NOTES** are speaker/context notes — keep them **off** the slide (use them as speaker notes only).
- **VISUAL** names an existing image in this repo — place it on that slide using the given path.
  **Do not regenerate or redraw figures**; they are finished artifacts.

## Directives for the presentation agent

- Keep each slide **light**: a headline, ≤ 4 bullets, and at most one small table. No paragraphs.
- **Do not invent, round differently, or add numbers.** Every figure and statistic already appears in
  `slides.md`; if you need a value that isn't there, leave a placeholder rather than guessing.
- **Do not name any specific authoring tool or product** anywhere in the deck.
- Define **F_causal** once, on first appearance, with the one-line gloss given in `slides.md`
  (fairness score; 1 = perfectly fair; higher is fairer). Elsewhere just say "fairness score."
- Lead with the plain-language takeaway on each slide; treat the numbers as support, not the point.
- Tone: confident but candid — the honest caveats (the GAN divergence on the second city; "same result,
  made robust, not new claims") are part of the story, not to be hidden.

## Number crib (authoritative — matches `slides.md`)

Shenzhen primary lens = `{housing, income (comp), migrant}`, cleaned data, seed means:
- Data cleanup: **10** stuck-GPS cells, **9** drivers, **106,677** phantom pickups removed.
- Editor fairness gain: F_causal **0.7988 → 0.8132** (**+0.0144**).
- Three-lens robustness: editor Δ **+0.0144 / +0.0124 / +0.0156**; every conclusion reproduces.
- Edit-vs-controls (upweighted training, @ weight 30): edited **+0.0311** (6/6 runs positive) ≫
  select **+0.0004** (null) > random ≈ 0 (null); editing beats selection **~70×**.
- San Francisco: F_causal **0.8752 → 0.8891** (**+0.0139**); realism score **0.968**; upweighted
  recovery **+0.0387**; both control arms negative; no algorithm change.

## Scope notes (why this deck looks the way it does)

- Deliberately **excludes** the adversarial-review workstream (presenter's call — time).
- The "re-based numbers" point on Slide 2 matters: these replace the values shown at the prior review
  (the data was cleaned and the primary fairness lens re-chosen for construct validity). Present as
  *"same conclusions, cleaner inputs,"* not as a change in findings.
- Deeper backup, if anyone asks in Q&A: the full argument doc-set is in `PAPER/argument/`, and the
  detailed results live in `PAPER/by_feature_set/`, `PAPER/feature_selection/`,
  `PAPER/shared_cleanup/`, and `PAPER/second-dataset/` (see `PAPER/second-dataset/FINDINGS.md`).
