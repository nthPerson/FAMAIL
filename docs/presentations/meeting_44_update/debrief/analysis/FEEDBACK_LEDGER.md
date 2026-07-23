# Meeting 44 — Dr. Zhang Feedback Ledger (itemized)

**Lens:** what exactly did the PI criticize, item by item.
**Meeting:** 2026-07-23, ~2h00m. Robert (RA) + Dr. Xin Zhang (PI), full-draft review.
**Sources:** `meeting_44_transcript.txt` (345 lines, primary), `plaud_summary.md`, `plaud_discussion_summary_raw.txt`, `plaud_highlights_raw.txt` (secondary; verified against transcript).
**Draft pinned to:** `/home/robert/FAMAIL/paper/` at the session's git state (HEAD `c4341fa`).

**Overall verdict (one line):** the paper reads like a *technical report documenting what was done*, is *hard/slow to read*, and must be *rewritten to be problem-driven and reviewer-legible* for KDD. Length is explicitly **not** her concern; clarity/story is.

**Status legend:** AGREED-CHANGE · OPEN (discussed, no decision) · PUSHED-BACK-STANDS (Robert defended, she accepted/moved on) · DROPPED (raised, never resolved).

---

## A. ITEMIZED FEEDBACK (chronological, F1–F54)

### F1 — KDD template: CCS concepts vs keywords
- **[0:09:47]–[0:11:32]**
- Quote: *"you are removing the CCS concepts part and replace it by keywords ... in our previous version ... this part is replaced by keywords ... so the template is wrong but it's okay, I have modified it to the correct version."*
- Target: `main.tex` document preamble (CCS/keywords block; Robert cites `\documentclass[sigconf,anonymous,review]{acmart}` at "line forty seven").
- Problem: the compiled template shipped the ACM CCS-concepts block instead of a keywords block; she wants keywords.
- Robert: explains the one-line template came straight from the KDD authors page; agrees to check his side.
- Status: **AGREED-CHANGE** (she already made the fix in her copy).

### F2 — Figures not rendering in her compile / prefers screenshots
- **[0:12:15]–[0:12:54]**
- Quote: *"for the figures, I'm trying to add it in ... but it doesn't seem to be showing correctly ... I think that is a package conflict issue ... What I usually do is just ... take a screenshot and upload it as screenshot ... it's easier for me to adjust the blank spaces."*
- Target: figure includes / TikZ package stack in `main.tex`; **her local compile environment** (version-discrepancy signal — see §D).
- Problem: her build can't render the TikZ figures, so she is reviewing a differently-rendered PDF.
- Robert: (implicit) uses TikZ for full control.
- Status: **OPEN** (rendering mismatch on her side, not a draft change per se).

### F3 — Too much blank space around figures; make compact
- **[0:13:18]–[0:13:31]**
- Quote: *"there are a lot of blank spaces over here or around over here ... make figures more compact."*
- Target: Figure 1 (`fig:teaser`, `01_introduction.tex` L29–62; TikZ `figures/figure-1/figure-1.tex`) and Figure 2 (`fig:overview`, §3).
- Problem: wasted whitespace inside/around figures.
- Robert: *"Bigger text, smaller gaps is what I'm hearing"*; happy to reduce spacings.
- Status: **AGREED-CHANGE**.

### F4 — Figure text too small / too much text; enlarge figure
- **[0:13:46]**
- Quote: *"there are too many texts in the figure. And they're so small and difficult to read, I think we also need to enlarge this figure."*
- Target: Figure 1 in-figure labels.
- Problem: dense, tiny figure text is unreadable.
- Robert: it was shrunk when the figure was larger; *"I'll just increase the text size there ... shouldn't be a big deal."*
- Status: **AGREED-CHANGE**.

### F5 — Figure text size rule (slightly smaller than caption/body)
- **[0:15:10]**
- Quote: *"We want to make sure that the text in the figure are like slightly smaller than this caption size or than the text in the paper. So that way people are having less difficulty reading."* (uses an ST-iFGSM figure as the reference exemplar)
- Target: typographic sizing of both figures.
- Problem: no consistent sizing convention; figure text should sit just under caption size.
- Robert: agrees.
- Status: **AGREED-CHANGE**.

### F6 — Captions too long / trailing space
- **[0:16:47]**
- Quote: *"for the caption doesn't have to be kind of a long caption. As you can see, there are a lot of space after the caption."*
- Target: Figure 1 caption (`01_introduction.tex` L37–51 — a ~14-line caption) and Figure 2 caption.
- Problem: over-long captions waste space and add reading load.
- Robert: raises the trade-off (explain-everything vs let-audience-infer); current format is "explain everything."
- Status: **AGREED-CHANGE** (see F7).

### F7 — Caption should carry only the most important info
- **[0:17:24]**
- Quote: *"we explain ... because we have limited space is just eight pages. So we just explain the most important information in the caption."*
- Target: both figure captions.
- Problem: captions currently exhaustive; should be triaged to essentials.
- Robert: *"Right now we're under eight pages as far as main content"*; frames it as a stylistic choice ("I like to leave less things up to chance").
- Status: **AGREED-CHANGE** (Robert accepts, mild push on rationale).

### F8 — Main-text figure references must be clear
- **[0:18:22]**
- Quote: *"in the main text where we are referencing the figures, we want to make sure that they are clear."*
- Target: figure `\ref`s in `01_introduction.tex`/`03_methodology.tex`.
- Problem: figure call-outs should be unambiguous.
- Robert: notes references exist but aren't bolded (LaTeX bolding is awkward); **admits he hadn't verified Figure 1 is referenced** ("I didn't check that figure one was referenced ... tables actually was the big one").
- Status: **OPEN** (Robert to verify all figures/tables are cited).

### F9 — Follow the provided template; don't customize captions
- **[0:19:16]**
- Quote: *"we can just follow the templates that Katie gave us. So we don't modify the caption of the figures or anything, and we just leave it as is."*
- Target: caption formatting / template compliance. ("Katie" almost certainly a mis-transcription of "KDD" — see §D.)
- Problem: wants template defaults respected rather than hand-tuned.
- Robert: (no objection).
- Status: **AGREED-CHANGE**.

### F10 — Core verdict: documents "what we do," not "why" (make it problem-driven)
- **[0:19:38]–[0:22:43]**
- Quote: *"the current version is a kind of ... good documentation of what has been done instead of ... a good way of telling the story that what the problem we want to solve and what strategies we have proposed ... it's more like this is what we do instead of why we do it."*
- Target: whole-paper narrative; esp. `01_introduction.tex`.
- Problem: paper is descriptive/procedural, not motivated/problem-first.
- Robert: says the *rewritten* intro already does the "why"; leans on AI for consistency, which she pushes back on ("AI's ... put things in a way that they can understand quickly ... not naturally how ... the human understands it").
- Status: **AGREED-CHANGE** (the governing directive of the meeting; also a VERDICT — see §C).

### F11 — Name the specific challenges that make existing approaches fail
- **[0:23:22]–[0:26:35]**
- Quote: *"what are the specific challenges that makes existing approaches doesn't work ... the major thing is not introducing a new approach, but introducing why we want to invent or create a new approach. So what makes in-processing approaches fail? ... Why can't we directly use in-processing approaches?"*
- Target: `01_introduction.tex` 2nd paragraph (L17–24) + a new Challenges block.
- Problem: intro asserts a new approach without establishing why prior approaches can't be used.
- Robert: *"I'll stand firmly on this. That is exactly what we're doing"* — pushes back that the redone intro already does this.
- Status: **OPEN → AGREED-CHANGE** (Robert defends but ultimately accepts sharpening; the "why they fail" gap is her #1 named missing part, cf. F17).

### F12 — "Why do we need a third position?"
- **[0:25:11]**
- Quote: *"our approach takes a third position. Why we need to take a third position?"*
- Target: `01_introduction.tex` L64 — *"FATE takes a third position."*
- Problem: the "third position" framing isn't self-justifying; reads as novelty-for-its-own-sake.
- Robert: explains pre-/in-/post-processing framing; the "third" = pre-processing/data augmentation; *"it's not necessarily a different approach."*
- Status: **PUSHED-BACK-STANDS** on meaning, but she redirects to "why," i.e. reframe (see F11).

### F13 — "In-processing methods" is jargon she can't parse; prefer "regularization"
- **[0:29:13]–[0:30:38]**
- Quote: *"I cannot understand what is in processing methods ... in processing is kind of a too broad ... so is kind of regularization approaches right instead of in processing approaches."* + *"another thing with AI is that you're usually inventing some phrases that are not like what the domain experts or practitioners use."*
- Target: `01_introduction.tex` L18 ("In-processing methods regularize the model...") and `00_abstract.tex` ("in-processing penalties"); `02_related_work.tex` L9 uses the pre/in/post taxonomy.
- Problem: "in-processing" is an unfamiliar, over-broad, AI-flavored label; she wants the mechanism named (regularization).
- Robert: *"That is exactly what we're doing"*; argues it's a semantics issue, the sentence already says "regularize the model."
- Status: **OPEN → likely AGREED-CHANGE** (Robert defends; she is unconvinced — this recurs at F18/F20/F47).

### F14 — "Realism" vs "fidelity" (first raise)
- **[0:29:13]**
- Quote: *"That's the fidelity, right? ... we use a word that we are familiar with."*
- Target: intro/abstract use of "realism."
- Problem: "realism" ≠ the team's own technical term "fidelity."
- Robert: fidelity is the technical definition of realism.
- Status: **OPEN** (fully argued later at F44).

### F15 — One citation ≠ "a group of work"; cite ≥2 and more recent (2025–26)
- **[0:32:11]–[0:39:15]**
- Quote: *"only having one related work is not enough if we are saying that it's a group of work ... when we're naming like a group of works ... instead of just one work ... we want to excite works that are closer to us, like 2025, 2026"* + *"It doesn't matter, just at least two."*
- Target: `01_introduction.tex` L18–24 — in-processing cites only `\cite{zheng2023}` (one); rebalancing cites only `\cite{kamirancalders2012}` (one). **Confirmed under-cited in current draft.**
- Problem: single-citation groups don't substantiate a "class" of methods; corpus skews pre-2024.
- Robert: pushes back — *"it's usually not that straightforward that we can just pick a 2020 ... paper that supports the exact line"*; the cites are "chosen very strategically."
- Status: **AGREED-CHANGE** (she holds firm on ≥2 per class; Robert to add recent cites — respect `CITATION_PRIORITY_CHECKLIST.md`).

### F16 — Reframe prior work as two lines: trajectory editing vs generation
- **[0:33:24]**
- Quote: *"we can talk about ... trajectory editing approaches or generative approaches ... approaches that are trying to generate different data as a second line ... and the approaches that are trying to modify on the original data ... we can talk about those two lines."*
- Target: intro 2nd paragraph + `02_related_work.tex`.
- Problem: wants the taxonomy re-cut into two intuitive lines rather than the pre/in/post framing.
- Robert: (no objection; discusses citation strategy).
- Status: **AGREED-CHANGE** (framing suggestion).

### F17 — Related work must locate the work + say why prior approaches can't work (= "most important missing part")
- **[0:34:25]**
- Quote: *"by talking about related work ... tell people where our work located ... what is the novelty ... identify the closest related works ... and also identify why those approaches cannot work on our problem? I think that is the most important missing part."*
- Target: `01_introduction.tex` + `02_related_work.tex`.
- Problem: the "why prior work fails on *our* problem" is absent/weak — she calls it the single most important gap.
- Robert: *"I am sorry, I disagree ... I took great care to make sure that ... was done."*
- Status: **PUSHED-BACK-then-CONCEDED** (Robert disagrees; by end accepts clarity must win — see F45/close).

### F18 — "Objective and training signal conflict — what does this mean?"
- **[0:35:24]–[0:35:57]**
- Quote: *"So objective and training signal conflict. What does this mean?"* → her own gloss: *"the problem for those approaches is that they are not specific to the fairness objective."*
- Target: `01_introduction.tex` L19 — *"so objective and training signal conflict."*
- Problem: the phrase is opaque; "objective," "training signal," and their conflict aren't defined.
- Robert: unpacks objective (improve fairness) vs training signal (loss); cites `[37]` (Zhang) as doing the work.
- Status: **AGREED-CHANGE** (phrase to be made explicit).

### F19 — "What is the ONE most important claim?"
- **[0:37:40]**
- Quote: *"what is the most important thing? Or what is the one most important claim we want to make for the paper."*
- Target: intro thesis / abstract.
- Problem: the single core claim isn't surfaced.
- Robert: *"existing approaches can improve fairness, but they don't do so while also retaining realism"* → she reformulates as the **fairness-vs-realism/fidelity trade-off** (shared framing; see P5).
- Status: **AGREED-CHANGE** (converged core claim).

### F20 — Disputes the "reduce realism / shift distribution / obscure source" triad
- **[0:40:48]–[0:41:56]**, echoed **[1:47:46]–[1:48:18]**
- Quote: *"why is reducing realism since their objective is trying to ... mimic how human are making decisions? and why do they shift the distribution since they are actually learning the distribution of real human behaviors? ... for the first two points, I don't agree. for the last point, it might be ... shift what distribution? Are you talking about mode collapse?"*
- Target: `01_introduction.tex` L20–24 and `00_abstract.tex` (same triad: "reduce realism, introduce distribution shift, and obscure the source"). **Present in BOTH abstract and intro.**
- Problem: she rejects two of the three sub-claims about generative methods as unjustified/unclear; wants precise mechanism (e.g. mode collapse) or removal.
- Robert: argues the cited papers (`[34.1]`, `[34.4]`, `[17]`) make it clear "in context."
- Status: **OPEN** (genuine disagreement; she disagrees on substance, not just wording).

### F21 — Sharpen the limitations paragraph (be precise on why prior work fails)
- **[0:46:48]**
- Quote: *"it's only kind of the first paragraph in introduction that is trying to highlight this problem ... in the second paragraph we are trying to say limitations of existing approaches, but we need to sharpen it, make it more precise in terms of why existing approaches cannot work."*
- Target: `01_introduction.tex` L17–24 (paragraph 2).
- Problem: limitations paragraph is vague on the failure mechanism.
- Robert: *"I'll try to fit some more in there ... about three or four lines of additional space."*
- Status: **AGREED-CHANGE**.

### F22 — Paper must be instantly readable; contribution/novelty must land from the intro alone
- **[0:49:07]–[0:50:00]**
- Quote: *"every reviewer is having six seven papers to review ... we want our paper to be easy to read ... by only reading an introduction, we didn't know the contribution, the novelty of this work ... The way of organizing the paper is not kind of in the most optimal way."*
- Target: intro + global organization.
- Problem: a time-boxed reviewer can't extract the contribution from the intro.
- Robert: recounts ~10h/day compaction effort; accepts.
- Status: **AGREED-CHANGE** (also a VERDICT — §C).

### F23 — Intro rewrite plan: motivation + limitations; shrink FATE approach to ONE paragraph; add code link
- **[0:50:46]**
- Quote: *"for the introduction I'm gonna reorganize it so that it's having a very clear motivation and existing works limitation ... we can shrink the FATE's proposed approach part into one paragraph because currently it's taking too much space ... we don't need to introduce in so much details because we are gonna provide more details in the methodology ... we need kind of an anonymous link telling people where the code will be released."*
- Target: `01_introduction.tex` L64–119 (FATE approach currently spans ~4 paragraphs: approach L64–73, trim/lift L74–99, nulls/upweighting L101–119) → collapse to 1. **No code/anon link currently anywhere in abstract/intro/conclusion (grep-confirmed absent).**
- Problem: approach over-detailed in intro (belongs in §3); missing required anonymized code link.
- Robert: (accepts; asks later about not colliding on the rewrite).
- Status: **AGREED-CHANGE** (she will draft it — F51).

### F24 — Add a motivating example (news or early results) showing prior methods amplify unfairness
- **[0:51:43]**
- Quote: *"we may also be needing kind of a motivating example ... existing approaches like GAIL or other approaches are actually amplifying or inheriting the unfairness in the data ... One approach is we are using some news ... Another approach would be if we have any experiment results, we can put that results ... in the first figure to highlight that existing approaches cannot solve the problem."*
- Target: `01_introduction.tex` + Figure 1 (`fig:teaser`).
- Problem: no concrete hook showing baselines fail; Fig 1 could carry early evidence.
- Robert: (discusses Figure 1 next).
- Status: **AGREED-CHANGE / OPEN** (mechanism TBD: news vs experimental result).

### F25 — Figure 1 must show pre-editing fairness score and be self-explanatory
- **[0:53:04]–[0:54:49]**
- Quote: *"Figure one, one thing is not reflected is what is the fairness score before editing? ... we'd better show that information in the figure ... we want the figure itself to be self-explanatory instead of having to read a [caption]."*
- Target: Figure 1 (`fig:teaser`); the `3.0×` lives only in the caption (`01_introduction.tex` L39–40).
- Problem: the key fairness quantity is caption-locked, not visible in the figure.
- Robert: notes an earlier version showed the score; offers to "hover it" in.
- Status: **AGREED-CHANGE**.

### F26 — Label advantaged/disadvantaged districts; keep passenger & vehicle totals constant
- **[1:04:57]–[1:05:32]**, reinforced **[1:18:40]–[1:19:22]**
- Quote: *"it's not clear why the advantage district receives three times the taxi service ... first of all we need to say what are the advantage districts ... just put a text ... this part is advantage and this part is disadvantage ... the total number of passengers and the total number of vehicles should stay the same. Otherwise it's ... assuming that you are eliminating taxis or services or eliminating demands."*
- Target: Figure 1 panels.
- Problem: districts unlabeled; unequal icon counts wrongly imply supply/demand is deleted, not relocated.
- Robert: at first defends *"it is intentional that there are less taxis"*; **then concedes** the figure is stylized, not literal (F31), and agrees to duplicate legends/labels.
- Status: **AGREED-CHANGE** (relocation, not removal; constant totals).

### F27 — City-level view should show less service across all regions
- **[1:06:15]**
- Quote: *"we want to show the figure in the city level. So in the city level, we want less service in general across all the regions."*
- Target: Figure 1 (right/after panel) / Figure 2.
- Problem: aggregate service change should be legible at city scale.
- Robert: pushes back that forcing equal taxi counts would "destroy our actual results" (the service ratio wouldn't improve).
- Status: **OPEN** (tension between literal metric and teaser stylization; resolved conceptually via F31).

### F28 — Can't see a service ratio in the figure; challenges the "3×"
- **[1:07:24]–[1:08:29]**
- Quote: *"currently I do not see a service ratio ... over here I'm looking at infinity supply ... a value of one over four ... this is infinity ... this one is three over two. So explain to me what does a three times means?"*
- Target: Figure 1 icon layout vs the `3.0×` claim.
- Problem: the drawn icon ratios don't compute to 3×; the figure contradicts its own caption number.
- Robert: goes to look up the metric (blanks on "disparate impact"); she cuts it off (F31).
- Status: **AGREED-CHANGE** (make the ratio visually correct or clearly stylized).

### F29 — Figure must be understood *instantly*; yellow reads as parking; "I cannot see what is fairness"
- **[1:09:55]–[1:11:34]**
- Quote: *"people are able to understand the meaning of the figure instantly ... it looks to me like this yellow area is where people park their cars instead of where the taxis are ... I cannot see what is fairness. I can only see certain vehicles towards the left, certain passengers towards the right ... I need to guess what are these vehicles. Are these vehicles taxis or other things?"*
- Target: Figure 1 iconography/color.
- Problem: icons/colors are ambiguous; fairness is not visually conveyed; requires guessing.
- Robert: *"I hear you"* but flags the challenge of conveying it semantically.
- Status: **AGREED-CHANGE**.

### F30 — Concrete fix: draw 6 taxis/2 passengers vs 1 taxi/1 passenger to make 3× visible
- **[1:12:54]**
- Quote: *"you are assigning six taxis over here and you're having two passengers over here. And over here you just have one taxi and one passenger. So ... people would know what are the three times service difference."*
- Target: Figure 1 before-panel.
- Problem: gives an explicit recipe to render the disparity literally.
- Robert: *"that makes sense. But the reality of the method is that it doesn't work like that"* — then concedes stylization (F31).
- Status: **AGREED-CHANGE** (adopt a legible, possibly non-literal ratio depiction).

### F31 — Teaser figure need not be rigorous; convey the idea
- **[1:13:42]**
- Quote: *"you don't need to include all the important details in terms of how we are calculating those service ratios. What most important is you want to convey the idea. The teaching [teaser] figure doesn't need to be rigorous."*
- Target: Figure 1 design philosophy.
- Problem: Robert over-literalized a stylized teaser.
- Robert: *"I misspoke or overemphasized the literal nature of it. It is not literal; it is very much stylized."* (**concession**)
- Status: **AGREED-CHANGE** (shared understanding).

### F32 — Doesn't know the grid is a map; Fig 1 & Fig 2 too far apart; each must stand alone
- **[1:15:22]–[1:16:59]**
- Quote: *"I know it's a grid cell. But I don't know what is a grid cell. Is it a map or is it something? I have no idea ... figure one, figure two, they are so far away from each other. And we want to make sure that each figure is self-explanatory. It's by itself."*
- Target: Figure 1 & Figure 2 placement + legends.
- Problem: the grid isn't identified as a city map; the two figures only make sense jointly, but are pages apart.
- Robert: argues Fig 1 + Fig 2 "in combination" answer the questions; proposes putting the city map behind the grid.
- Status: **AGREED-CHANGE** (make each figure self-contained; duplicate legend/map on both).

### F33 — Adopt Figure 2's style for Figure 1 (map, legend, labeled districts)  ⟵ Plaud highlight
- **[1:17:25]–[1:18:40]**
- Quote: *"the style of the Figure two would work better to replace these two smaller figures in Figure one. Because in this Figure two you have advantaged districts and disadvantaged districts ... having all those neural networks and all the tasks in terms of advantage, disadvantage regions. And also the number of vehicles and the number of taxis."*
- Target: Figure 1 (`fig:teaser`) restyled after Figure 2 (`fig:overview`).
- Problem: Figure 2's labeled, mapped style is legible; Figure 1's is not.
- Robert: *"That's very much heard. I will make sure that the legends and everything are duplicated on both."* (**agreement**)
- Status: **AGREED-CHANGE** (flagged by Plaud as a highlight — the meeting's clearest figure decision).

### F34 — Shrink Related Work to ½ column; move full version to appendix
- **[1:20:02]**
- Quote: *"for the related work, we can shrink it into minimum of a half column ... briefly talk about what are the related approaches and talk about the differences. We can leave a complete version in the appendix."*
- Target: `02_related_work.tex` (currently 5 themed paragraphs, L6–83) → ½ column; full text to `appendix.tex`.
- Problem: related work is over-long for the space it earns.
- Robert: notes derivations already moved to methodology.
- Status: **AGREED-CHANGE** (frees space for motivation/approach).

### F35 — Move the F_demo derivation to the appendix; keep "1 − R²_demo" + meaning in text
- **[1:23:32]–[1:24:34]**
- Quote: *"it's better to keep ideas simple instead of using a very complicated way of representing it. So for equation one, we can directly say that there is [F_demo] that is calculated and the reason why ... and we point readers towards appendix in terms of how it's derived ... in the text we tell people the meaning of one minus R square demo ... and refer people to appendix for more information."*
- Target: `03_methodology.tex` L58–66 (Eq. `eq:fdemo`: the hat-matrix machinery `H = X̃(X̃ᵀX̃)⁻¹X̃ᵀ`, cite `\cite{hoaglinwelsch1978}` L62) → appendix; keep `F_demo = 1 − r²_demo` + interpretation in body. (An `Appendix~\ref{app:derivations}` pointer already exists.)
- Problem: the closed-form hat-matrix derivation is too heavy for the main text.
- Robert: worries moving it loses the seminal hat-matrix reference reviewers need; agrees to keep only `1 − R²_demo`. (Note: Robert calls the seminal work "nineteen seventy," but the actual cite is `hoaglinwelsch1978` — 1978; Plaud calls it "[13]" — see §D.)
- Status: **AGREED-CHANGE**.

### F36 — Rename "Task" → "Problem Formulation/Definition"  ⟵ Plaud highlight
- **[1:25:21]–[1:25:46]**
- Quote: *"for the problem formulation part. There's not a problem formulation. It's just a task. You can just give it kind of problem definition or problem [formulation]."*
- Target: `03_methodology.tex` — subsection is **already** `\subsection{Problem Formulation}` (L3), but a `\textbf{Task.}` run-in header sits inside it at **L29**. She reacted to a "Task" label. (Version nuance — see §D.)
- Problem: the section states a task rather than formulating the problem; the "Task" label should go.
- Robert: *"Yeah, that's I like that better."* (**agreement**)
- Status: **AGREED-CHANGE** (rename/retire the `\textbf{Task.}` run-in; enrich into a real problem definition, cf. F40).

### F37 — Add a "Challenges" block to the introduction
- **[1:25:48]**
- Quote: *"one important thing that we forgot to mention ... in the introduction part is the challenges. What makes this problem challenging ... those challenges are also making existing works not work properly anymore. So what are the challenges?"*
- Target: `01_introduction.tex` — **no "Challenges" block anywhere in abstract/intro/conclusion (grep-confirmed absent).**
- Problem: the difficulty of the problem (and thus why prior work fails) is never stated.
- Robert: *"that I think is a very important thing to include ... goes back to your first comment [what we did vs why]."* (**strong agreement**)
- Status: **AGREED-CHANGE** (ties F10/F11/F17 together).

### F38 — Length is NOT the problem; it's the "least important thing"
- **[1:27:17]–[1:28:17]**
- Quote: *"I don't think length will be a problem for desk rejection ... don't worry at all about length. We can shrink it within eight pages anyways ... The length is the least important thing."*
- Target: Robert's length anxiety / the whole compaction strategy.
- Problem: Robert has over-optimized for length at the cost of clarity; she reprioritizes to story.
- Robert: *"I've worried at great lengths about the length issue ... but shrinking it losslessly is the challenge."* (**push-back**)
- Status: **PUSHED-BACK-STANDS (hers)** — she explicitly de-prioritizes length; Robert's "lossless" concern is reframed by her as a symptom (F39).

### F39 — "Lossless compression is hard" = too much detail; generalize beyond taxis; move 0.01° grid to appendix
- **[1:28:28]–[1:29:27]**
- Quote: *"compressing losslessly is the most challenging part ... means that you're including too much detail ... we don't want to design one approach specifically for this taxi driver data ... this same approach would still work [for transit logs, logistics data] ... we don't need to write in detail about partitioning the map into 0.01 degree grid cells. Those are implementation details that could be pushed into the appendix ... the city can be modeled as a state space."*
- Target: `03_methodology.tex` L5 (`0.01°` grid) + L38 (`ε = 2` grid cells) → appendix; raise abstraction to a state-space framing.
- Problem: methodology is over-specialized to the taxi dataset; implementation constants hurt generality.
- Robert: notes `0.01°` is verbatim from the SIGGRAPH paper's problem formulation; argues the data representation *drives* the F_demo formulation, so it can't be fully removed.
- Status: **AGREED-CHANGE** (move `0.01°`; keep the formulation-relevant abstraction).

### F40 — Methodology = approaches only; add an "Overview" defining trajectories/rewards/states/actions
- **[1:32:12]**
- Quote: *"methodology is only talking about the approaches. And over here we're having our [over]view section ... a brief description about what the data looks like and the problem formulation, where within problem definition we want to define ... the trajectories, the reward functions. Sometimes we also define state or actions because it's kind of a reinforcement learning thing."*
- Target: `03_methodology.tex` §3.1; possibly a new Overview/Preliminaries section.
- Problem: data description and RL formalism are mixed into the method; should be a separate, rigorous problem-definition.
- Robert: (discusses SF using the same partitioning).
- Status: **AGREED-CHANGE / OPEN** (structural split; details TBD).

### F41 — Write methodology *against the challenges*, not as a procedure log (tech report vs paper)
- **[1:36:29]**
- Quote: *"think about what are the challenges ... write or refine the methodology part against the challenges instead of directly telling what we do, because there is a difference about a tech report versus a paper ... when it's a paper, it has to be problem driven."*
- Target: `03_methodology.tex` whole section.
- Problem: methodology narrates steps rather than motivating design choices against difficulties.
- Robert: (accepts).
- Status: **AGREED-CHANGE** (also a VERDICT — §C).

### F42 — Remove duplicate info between problem formulation and experimental setup
- **[1:37:28]**
- Quote: *"make sure that you don't include duplicate information. Because if you define it in the problem formulation ... you want to make sure that those information doesn't appear in your experimental setup."*
- Target: `03_methodology.tex` §3.1 vs `04_experiments.tex` §4.1 (`Experimental Setup`, L11) — e.g. grid-cell resolution stated in both.
- Problem: setup details repeated across sections waste space.
- Robert: acknowledges "strategically duplicated" restatements exist to "jar the memory of the reader." (mild **push-back**)
- Status: **AGREED-CHANGE** (with Robert reserving intentional call-backs).

### F43 — Summary directives: challenges-first, optimize method to challenges, use space wisely, kill blank space
- **[1:39:40]**
- Quote: *"the most important suggestions ... Highlight what are the challenges. And trying to optimize your methodology against your challenges. And use your space wisely. Some formatting issues like those are the blanks you want to avoid."*
- Target: global.
- Problem: her consolidated priority list.
- Robert: asks whether he can pull bullets in / tighten template spacing ("I'm looking for every opportunity").
- Status: **AGREED-CHANGE** (recap of F37/F41/F3).

### F44 — Prefer "fidelity" over "realism"; be consistent; doubts "realism" is domain-standard
- **[1:40:54]–[1:42:14]**, echoed **[1:46:49]–[1:47:46]**, **[1:49:56]–[1:52:14]**
- Quote: *"if you are using realism, make it all realism instead of fidelity. But I wouldn't recommend using realism because realism usually means other things instead of fidelity ... I'm not familiar with people using realism instead of fidelity ... I really doubt [it's from the literature] because you are telling me information that is changing my view for this domain."*
- Target: term "realism" in `00_abstract.tex` (×2), `01_introduction.tex` (×2) vs "fidelity" dominant in `04_experiments.tex` (×20) / `appendix.tex` (×22). **The split she suspected is real: "realism" in the framing sections, "fidelity" in the technical sections.**
- Problem: two words for one concept; she considers "realism" non-standard and ambiguous.
- Robert: insists "realism" is from the literature; flags a tension — "fidelity" has a paper-specific meaning (ST-SiameseNet discriminator; Fidelity-A/B) so using it in the intro over-commits other works to *their* definition. Offers to look up the citation.
- Status: **OPEN** (Robert defends "realism"-from-literature but concedes *"as a simple fix ... just use fidelity"*; unresolved tension between general vs paper-specific meaning; citation not produced in-meeting).

### F45 — Doubts current writing style will survive KDD review; considers a full rewrite
- **[1:43:36]**
- Quote: *"I know how the KDD community usually read papers, and I think it's going to be difficult to work if we are submitting in the current way ... whether to rewrite a more style paper instead of keeping your original writing and see how reviewers like it. What is your idea?"*
- Target: whole-paper voice/style.
- Problem: she predicts KDD reviewers will bounce the current draft.
- Robert: *"My preference would be ... go for what we know ... that would make me sleep better."* (defers to her experience)
- Status: **AGREED-CHANGE** (she will produce a rewrite — F51; also a VERDICT — §C).

### F46 — Find the "skeleton" / most important part of the approach
- **[1:45:36]**
- Quote: *"the most important thing is trying to find what is the skeleton or the most important part of the approach."*
- Target: methodology structure.
- Problem: the core of the method is buried among supporting detail.
- Robert: agrees the intro is the key takeaway ("if you haven't been convinced by the introduction, no one else will be").
- Status: **AGREED-CHANGE**.

### F47 — Reverse the AI workflow: write first, GPT refines; avoid GPT-general language ("rebalancing models")
- **[1:45:55]–[1:47:46]**
- Quote: *"the way of writing a paper is always you write it first, and you ask GPT to refine it. The problem with asking GPT to write it first and then you refine it will be very subtle ... GPT is usually gonna use very general language that is not easily understandable by domain experts. For example, 'rebalancing models' may mean rebalancing data, or model architecture, or hyperparameters ... those general terms make people feel confused and lost."*
- Target: the drafting *process*; concrete example — intro's "data-rebalancing methods."
- Problem: GPT-first drafts smuggle in vague, ambiguous terminology.
- Robert: describes his humanize-after workflow and rule-based prompting; *"I agree."*
- Status: **AGREED-CHANGE** (process directive; also §E AI-writing advice).

### F48 — "Shift the distribution — shift WHAT distribution?" relying on reader inference confuses
- **[1:47:46]–[1:48:44]**
- Quote: *"shift the distribution ... I don't understand why data generation is going to shift the distribution, shift what distribution? ... Are you talking about mode collapse? ... People have different understanding about the extension, that is making people feel confused and lost."*
- Target: `01_introduction.tex` L23 / `00_abstract.tex` ("introduce distribution shift").
- Problem: "distribution" is unqualified; leaving readers to infer ("extension") backfires.
- Robert: *"contextually there, the only distribution is the dataset"* — argues extension is narrow. (**push-back**)
- Status: **OPEN** (she insists inference is unreliable; overlaps F20).

### F49 — Name the exact line of work; "obscure language" ambiguous across method classes
- **[1:49:18]**
- Quote: *"we want to make sure what line of work we're talking about instead of using obscure languages that seems to point to one class of works [but] might also be pointing at another line of approaches."*
- Target: intro/related-work class labels.
- Problem: vague labels map to multiple method families.
- Robert: raises the generality-vs-specificity trade-off (*"the more specific we make the claim, the more narrowed the argument"*); she rejects the trade-off framing (*"It's not a complex balance ... it's application driven"*).
- Status: **PUSHED-BACK-STANDS (hers)** (she overrules the trade-off objection).

### F50 — Abstract + intro: shrink the proposed-approach description; target challenges + how they're solved
- **[1:49:56]–[1:52:14]**
- Quote: *"general suggestion is abstract, introduction, shrink the size of proposal approach description. Target the challenges and describe how those approaches solve those challenges."*
- Target: `00_abstract.tex` + `01_introduction.tex`.
- Problem: both over-describe the method and under-describe the challenges.
- Robert: *"and why we chose the formulation based on the challenges."*
- Status: **AGREED-CHANGE** (consolidates F23/F37/F50).

### F51 — Zhang will write her own rewrite (new .tex file); Robert may adopt it
- **[1:52:42]**, **[2:00:03]**
- Quote: *"if I have time, I'm gonna rewrite a version, and it's up to you whether you want to use the written version or not ... I'm gonna create a new text file, so no worries about [collision]."*
- Target: intro/abstract (primarily).
- Problem: division of labor for the rewrite.
- Robert: *"Oh absolutely, and I'm totally cool with taking your input"*; asks to be told before she works on the intro so he doesn't "roll over" her work.
- Status: **AGREED-CHANGE** (she owns the intro rewrite draft; new file avoids conflict).

### F52 — Don't release raw data; release anonymized data + GitHub link only
- **[1:57:06]–[1:57:24]**, **[1:59:17]**
- Quote: *"we don't release the raw data, we only release some anonymous data, so we don't need to release the data for now. We just give people a GitHub link ... we can just say that we share our code via a GitHub link ... we don't need to say too much about the data."*
- Target: data-availability statement (submission metadata + paper).
- Problem: raw fleet data (from Dr. Yan Hua Li) can't be released; scope the statement to code.
- Robert: asks whether they can claim availability of data they don't own (shareable academic data).
- Status: **AGREED-CHANGE** (code link only; minimal data statement).

### F53 — Defer full reproducibility documentation until after submission
- **[1:58:38]**
- Quote: *"since we are already over time, I need to meet with Manuel. Can we like leave this after the submission deadline?"*
- Target: Robert's `REPRODUCIBILITY.md` claim→artifact map (cf. HEAD commit `c4341fa`).
- Problem: reproducibility doc is post-submission work; meeting is over time.
- Robert: *"probably ... but it affects the one thing related to our submission"* (the artifact pledge).
- Status: **AGREED-CHANGE / DEFERRED** (post-deadline).

### F54 — Empty GitHub at submission is acceptable; artifact pledge proceeds on code availability
- **[1:59:06]**, **[1:59:48]**
- Quote: *"It's okay to GitHub is empty for now ... [artifact pledge] Yeah, it's okay."*
- Target: submission artifact pledge.
- Problem: anon repo need not be populated at submission.
- Robert: notes the artifact pledge "implies that the data and code are available."
- Status: **AGREED-CHANGE** (empty anon repo link suffices for now).

---

## B. POSITIVE — praise / keep-this (don't destroy in the rewrite)

- **P1 [0:49:07] — Effort & results praised.** *"I appreciate your work ... I know you're hardworking and you have done a lot of work ... we have all those results, that is very good."* (She frames all criticism as about *presentation*, not correctness or effort.)
- **P2 [1:03:10] / [1:03:53] — The upweighting/transfer finding is clear once explained.** *"that part I got it ... that part is clear ... we are designing an approach specifically for imitation learning so that the learned policy will replicate human behaviors and provide more fair services. That is a claim."* → Keep the upweighting story; it survives her scrutiny.
- **P3 [1:17:25] — Figure 2's style works.** *"the style of the Figure two would work better."* → Figure 2 is the visual template to preserve and propagate.
- **P4 [1:39:40] — Robert owns the presentation.** *"it's your work. You can have full control over how you want it to be presented."*
- **P5 [0:38:14] — Converged core framing.** She accepts the **fairness-vs-realism/fidelity trade-off** as "the core of this problem" and that `[37]` establishes the trade-off is hard at the model level. → This is the agreed thesis; build the rewrite on it.
- **P6 [1:26:22] — "Challenges" idea lands with both.** Robert calls adding challenges *"a very important thing to include"*; genuine mutual buy-in on the single biggest structural change.

---

## C. VERDICTS — global statements (verbatim; calibrate rewrite depth)

- **V1 [0:21:58]:** *"the current version is a kind of ... good documentation of what has been done instead of ... a good way of telling the story that what the problem we want to solve and what strategies we have proposed to solve this problem. So it's kind of more problem driven and design driven."*
- **V2 [0:22:30]:** *"for current work, it's more like this is what we do instead of why we do it."*
- **V3 [0:39:52]:** *"when I'm reading this paper, it's really taking me a lot of time in terms of understanding ... what is in [-processing] messages? what is objective and training signal conflict? ... I cannot identify why these [approaches] cannot work."*
- **V4 [0:50:00]:** *"The way of organizing the paper is not kind of in the most optimal way, and what we want to do is improve it."*
- **V5 [1:36:29]:** *"there is a difference about kind of a tech report versus a paper ... when it's a paper, it has to be problem driven, and it has to be telling people you are designing this approach because [of] different reasons."*
- **V6 [1:43:36]:** *"I know how the KDD community usually read papers, and I think it's going to be difficult to work if we are submitting in the current way."*
- **V7 [0:50:46]:** *"We'll be doing the rest of the days before the deadline ... rewriting the paper based on what you have already done."* (She commits to co-rewriting, not just commenting.)

**Depth read:** this is a *structural rewrite* verdict (intro + methodology framing + figures + terminology), NOT a copy-edit. But she is emphatic that the *science and results are sound* (P1/P2/P5) and that *length is a non-issue* (F38). The rewrite is about legibility and problem-first framing, executed on top of Robert's existing content.

---

## D. VERSION / PAGE-COUNT / TEMPLATE discrepancies

1. **"~11 pages" is Robert's number about a superseded draft — NOT Dr. Zhang's, and NOT the current PDF.**
   - The only "eleven pages" in the transcript is **Robert** at **[0:48:44]**: *"Landed at about eleven pages, so then ... the hardest part was compacting all of that argument into the eight pages."* He is describing his own *initial, pre-compaction* draft.
   - **Dr. Zhang never states a page count.** On the contrary she says length is *"the least important thing"* (F38).
   - The task brief's premise ("Dr. Zhang refers to the paper as ~11 pages") is **not supported by the transcript.** The Plaud auto-summary ("The manuscript initially exceeds the eight-page limit (~11 pages)", `plaud_summary.md` L53) appears to have absorbed Robert's line and re-attributed it to the manuscript's current state. Current local draft: main content < 8pp, refs start ~95% down p8, 12pp total with appendix. **No evidence any 11-page version was compiled.**

2. **Dr. Zhang was reviewing a differently-compiled / possibly older PDF than Robert's current working tree.** Evidence: she *"just updated"* the template and swapped CCS→keywords in *her* copy (F1); her figures *"doesn't seem to be showing correctly ... package conflict"* so she works from screenshots (F2); she could not locate Figure 2 (*"what do you mean by Figure two? ... the next figure?"* [1:16:37]). Several figure/blank-space complaints (F2–F6) may partly reflect her render, not Robert's source.

3. **"Task" heading discrepancy.** She reacted to a **"Task"** label (F36; Plaud highlight records "rename Task → Problem Formulation"). The current file `03_methodology.tex` **already** titles the subsection `\subsection{Problem Formulation}` (L3) but retains a `\textbf{Task.}` run-in header at **L29** inside it. Either she saw an older heading, or she was pointing at the run-in. The live rename target is the L29 run-in.

4. **Seminal-reference year mismatch (minor).** For the hat matrix, Robert says *"nineteen seventy"* [1:22:40] and Plaud calls it *"ref [13]"* (`plaud_summary.md` L45). The actual citation in `03_methodology.tex` L62 is `\cite{hoaglinwelsch1978}` (**1978**, Hoaglin & Welsch). Robert's recollected year and Plaud's number are both off; the cite itself is fine.

5. **"Katie" = almost certainly "KDD."** [0:19:16] *"follow the templates that Katie gave us"* — no third person was present; reads as a mis-transcription of "KDD" (the template source Robert named earlier).

6. **Deadline reconciliation (logistics, resolved in-meeting).** Zhang initially believed **Sunday**; Robert believed **Monday**. They open OpenReview together and confirm it shows **local time = Mon 2026-07-27, 11:59** ([1:55:52]; Robert says "a.m.", likely a slip for 23:59 AoE per project memory). Both agree to **target Sunday 07-26** to avoid last-minute risk. Abstract was already submitted 07-19. (No draft impact.)

---

## E. MISC — process / AI-writing / logistics / diarization

- **AI-writing doctrine (E1 [1:45:55], F47):** *always human-write-first, then GPT-refine*; never GPT-first. Rationale: GPT injects vague general terms ("rebalancing models", "distribution shift", "realism", "in-processing") that read as non-domain and lose domain experts. This is her single most repeated meta-critique (recurs at F13, F20, F44, F47, F48, F49).
- **Reviewer-load framing (E2 [0:49:07]):** *"every reviewer is having six seven papers to review"* — the justification for the entire ease-of-reading push.
- **"Laura" is a Plaud diarization artifact.** She appears **inside the transcript** at [1:39:40] (*"Yeah Laura is this ... it's your work"*, i.e. Zhang addressing Robert) and as the assigned owner of many action items in `plaud_discussion_summary_raw.txt` and its attendee line ("Speaker 3/Laura"). **Only Robert and Dr. Zhang attended.** Treat every "Laura" owner/date in the Plaud discussion summary as machine-generated, not spoken.
- **Data provenance (E3 [1:56:47]):** source data is from **Dr. Yan Hua Li**; do not release raw data (F52).
- **Robert's stated effort (context, not feedback):** ~10h/day, initial draft ~11pp compacted to 8pp; he repeatedly frames density as forced by "the nature of the problem," which Zhang rejects ("It's a nature of *one* problem", F39).

---

## F. SUMMARY-ONLY CLAIMS (in Plaud, NOT clearly supported by the transcript)

- **"~11 pages" as the manuscript's state** (`plaud_summary.md` L53) — misattributed from Robert's line about his superseded draft (see §D-1).
- **All owner/date assignments** in `plaud_discussion_summary_raw.txt` (e.g. "*Laura* (2026-07-26)", "*Speaker 3* (2026-07-30)") — **none are spoken in the transcript.** The only dates discussed are the submission deadline (Sun 07-26 target / Mon 07-27 hard). "Laura" is an artifact (§E).
- **"Document data licensing constraints from Dr. Yan Hua Li to ensure compliance"** (`plaud_summary.md` L82, L57) — Zhang actually said the *opposite*: *"we don't need to say too much about the data or anything, and we don't need to release the data for now"* [1:59:17]. The "document licensing constraints" action is a Plaud AI-suggestion, not her instruction.
- **"Define and standardize fairness metrics (e.g., disparate impact vs alternatives)"** (`plaud_summary.md` L42, L77) — overstates the transcript. Zhang mentioned "disparate impact" only while *probing* Figure 1's 3× ([1:08:29]); she did not ask for a metric-selection exercise. The real ask is: make the service-ratio **visible and consistent in the figure** (F25/F28).
- **"Background map" for Figure 1** (`plaud_summary.md` L37; `plaud_discussion_summary_raw.txt` L20) — the "map behind the grid" idea is **Robert's** ([1:17:14]); Zhang endorsed adopting *Figure 2's style* (F33). Fair to attribute the map to the joint plan, but not originated by Zhang.
- **Note (supported, for the record):** the "≥2 citations per class" and "2025–2026 recency" asks in Plaud (`plaud_summary.md` L24) **are** in the transcript ([0:32:37], [0:39:15]) — verified, F15.

---

## G. QUICK STATUS TALLY

- **AGREED-CHANGE:** F1, F3, F4, F5, F6, F7, F9, F10, F15, F16, F18, F19, F21, F22, F23, F24, F25, F26, F28, F29, F30, F31, F32, F33, F34, F35, F36, F37, F39, F40, F41, F42, F43, F45, F46, F47, F50, F51, F52, F54 (≈40, several with a defer/OPEN tail)
- **OPEN (no decision / substantive disagreement):** F2, F8, F11(partial), F13(partial), F14, F20, F24(mechanism), F27, F40(structure), F44, F48
- **PUSHED-BACK-STANDS:** F12 (meaning defended), F17 (defended, conceded on clarity), F38 (her de-prioritization stands), F42 (partial), F49 (trade-off overruled)
- **DEFERRED:** F53
- **DROPPED:** none of substance (every raised item got a response)
- Two items are Plaud-flagged **highlights**: **F33** (adopt Figure 2's style) and **F36** (rename Task → Problem Formulation).
