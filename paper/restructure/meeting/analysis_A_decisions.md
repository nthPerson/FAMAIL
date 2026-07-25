# FULL REPORT — Analysis A: Argument & Section Architecture (2026-07-24 meeting, 62 min)

Sources: `meeting/transcript_readable.txt` (authoritative; all 386 lines read), raw JSON `meeting/transcript_2026-07-24.txt` (spot-verified verbatim for load-bearing turns), `plaud_summary.md` + `highlights_raw.html` (topic checklist only). Ground truth: `restructure/ALGORITHM_FACTS.md`. Quotes are ASR text, errors intact ("fade"=FATE, "trajectory planning"=trajectory editing, "SIGL paper"≈cGAIL per [21:58]).

Tags: **CLEAR-DECISION** · **SOFT-PREFERENCE** · **UNRESOLVED**.

## 1. Positioning / the new spine

### 1.1 Why the reframe exists: "editing alone" is indefensible for a two-part method
**CLEAR-DECISION.** [03:38] Zhang: *"I read over your writing, which is kind of uh, it's aligned with what we have discussed. And one thing I have noticed is uh, them. For the current story, uh, It's kind of a little bit difficult to justify to justify that. Um we are only proposing trajectory planning because it's a combination combined strategy in terms of the."* → [04:12] *"Editing and upweighting, So we kind of need kind of a more complete and more tailored story for in that case."*

The restructure's root cause is that FATE is **edit + upweight**, and a paper sold as "trajectory editing" cannot account for the second half. The budget framing is the umbrella making both one claim. Architecture consequence: edit-aware weighting is **not** appendix-able; it is half of what the spine carries.

### 1.2 The spine sentence (adopt framing, not literal wording)
**CLEAR-DECISION (framing) / RISK (wording).** [04:12] *"What I have changed in terms of story is that we are doing a budgeted like um, or budget aware trajectory editing, where the claim is that we don't want to. Modifies too many data. Instead, We want to modify a portion of the data. And even though those portion of data won't have make a great impact on the raw data fairness itself, it is going to be impactful for the downstream models."* [05:09] *"And in a sense like only modifying or. Correcting a small portion of data won't have that like a like an impact. We just need to update it so that it's gonna make sense."*

Two conflicts with ALGORITHM_FACTS: (a) raw-data fairness *does* move measurably (F_demo +0.0226 SZ / +0.0316 SF, plus ring-(iii) DI/DP/Theil) — writing "barely moves" deletes a protected-register headline; (b) downstream impact requires edit-aware weighting — vanilla uniform-weight BC is NULL (+0.0016, n=12, p=0.11), w30 gives +0.0297 (p=.00049 both cities). Accurate spine: *a ~10% edit budget (k=10,000 of ~95k SZ trajectories) produces a corpus-level collective-fairness gain that, preserved by edit-aware weighting, transfers to downstream policies at comparable magnitude — while a uniform-weight learner averages it away.*

### 1.3 Collective fairness is THE differentiator; fidelity trade-off demoted
**CLEAR-DECISION.** [05:09] *"the reason why and your original novelty claim is kind of the trade off between um editing and the fidelity. I think that is a good point, but I think a more important problem that we want to solve is that, And that problem is distinguishing our pro our problem from other approaches, like the fair GAN or or other related works, is that we are looking at a systematic thing like um collective um fairness."* [06:02] *"When, we are looking the whole dataset, instead of like designing a more kind of fair model that won't discriminate, like for example, CV model that is gonna discriminate people of color, that type of thing. So we are kind of a different fairness perspective or metrics. ... And that naturally create the unique challenge that we have been trying to solve, Which is how can we modify a small portion of the trajectories to affect the global fairness."* → [06:55] *"Distinctive part that would differentiate our approach with others."* → [07:01] Robert: *"very much in line with the intent that we have there ... I can definitely make sure that that's emphasized."*

- Fidelity trade-off: retained but demoted; lives as a constraint/guardrail in methodology + the disclosed Fidelity-B cost. Not the intro's novelty sentence; not the lead contribution.
- The **unique challenge** to state: *how can a small portion of edited trajectories move the fairness of the whole dataset?* — the umbrella above the challenge list.
- Differentiation targets named: FairGAN/generative data-side methods, and model-side fairness (her CV example). Axis: collective/global/dataset-level fairness, consistent with ALGORITHM_FACTS ("NOT a per-trajectory fairness fix").

### 1.4 Robert's ground-truth guardrail, stated and unchallenged
[07:48] *"Trying to avoid the situation where we kind of get into like an argumentative gymnastics where, the way we want to tell the story conflicts with what actually happened. Um. So I don't think I see anything uh here that that's gonna do that."* Also [07:23] on drafting outside Overleaf *"because I have the context of the project, So I can make sure that things stay consistent with the implementation um and our results."* No pushback from Zhang.

### 1.5 The snapshot Zhang annotated is being discarded
**CLEAR-DECISION.** [10:10] *"that one that I sent you today is ... about twelve hours old ... heavily revised since then, Especially through actually the all through introduction related work methodology and about ninety percent of the experiments. But one thing I want to avoid is because we're going to be restructuring it. I am basically just going to be ditching that. So, we don't necessarily need to I guess work on it because we're changing direction."* Unopposed. Her Fig-1 and her intro/abstract language carry forward; her line edits on the old snapshot do not.

## 2. Section architecture

**2.0 Top-level order — CLEAR-DECISION.** [39:14] *"we can organize it in that way and uh. Then following related work and then the conclusion."* → 1 Intro · 2 Overview · 3 Methodology · 4 Experiments · 5 Related Work · 6 Conclusion. Related Work after Experiments (email confirmed verbally). No Conclusion content directive.

**2.1 Zhang's names are NOT binding — CLEAR-DECISION (3×).** [28:49] Robert: *"The one part that I I might foresee kind of some challenges is fitting this the exact section names."* → [29:26] *"It doesn't have to be exact section name, But that's just trying to give you kind of idea about what each section is focusing on. Uh, yeah, you can you can like name it in like or different."* [29:39] Robert: *"it doesn't necessarily have to be exactly like this, but itemized? Or do you do you want them to be in this logical shape?"* → [30:07] *"Uh, Doesn't have to be to be because there is one one thing as what i'm thinking it should be written. And there is another thing about the input. When you are writing it, it will become dependencies. So you kind of adjust accordingly."* [32:41/33:12] Robert: *"you want it in this general shape, not necessarily these specific steps."* / *"Exactly. Okay, cool."* (the "Exactly" is plausibly Zhang's, merged by the diarizer) → [33:20] *"Those are all kinds of suggestions."*
Her outline is a content checklist, not a template; explicit license to reorder for logical dependency.

**2.2 The organizing invariant — CLEAR-DECISION (strongest structural constraint).** [30:07] *"The major idea is trying to align it with the challenges we have. So kind of each challenge is, matching one um one bullet point or one subsection or subsubsection."* [33:20] *"when you are writing the methodology, You want to make sure that each methodology kind of have a same [aim] of solving one challenge. And at the same time, it's kind of in a logical order so people can follow along."* Robert accepted [30:37] *"Sweet, No problem I can do that."*
⇒ #methodology blocks = #challenges, each block announcing its challenge. This rule, not any uttered number, decides the challenge count.

**2.3 WHERE THE CHALLENGES LIVE — CLEAR-DECISION, meeting overrides email.**
- [30:46] Robert: *"This one to make it more explicit, what kind of challenges I like kind of went out and did this. Um, Do you want it to be that explicit as in we've actually defined specific challenges and then refer back to them?"*
- [31:00] Zhang: *"Uh, we don't we so that relates to kind of the overview writing. So one style if you, Go to the SIGL paper or the ST-iFGSM paper."*
- [31:30] *"Could you scroll down for the overview? ... As an overview, as you can see that they are having the definitions and following problem. Problem that those definitions of necessary items has been used and then the problem definition. Then follows the challenges. We can define that C one, C two, C three over here by listing or re-listing."*
- [32:12] *"The challenges we have mentioned in the introduction. So in the introduction, we can like briefly tell what are the challenges without kind of itemizing them, and we can itemize them over here."* [Robert: *"Perfect."*] *"Yeah, and due to limited space, we don't need to kind of put them in this itemized like LaTeX environment. We can just stack them all together."*
- [32:41] Robert: *"Perfect. Yep."*

Pinned: (1) Intro = brief prose mention, no itemization → the `\begin{itemize}` C1–C5 block at `01_introduction.tex:112-132` leaves the intro; (2) §2 Overview = where C1,C2,… are defined, after definitions and problem definition; (3) formatting = no itemize/enumerate; stacked bold lead-in sentences in one paragraph, labels kept inline so §3 can reference them; (4) Overview order = definitions → problem definition → challenges; (5) Plaud's *"then challenges and solutions"* (line 42) is **not** in the transcript — optional embellishment.

**2.4 Current §3.1 → §2 Overview — CLEAR-DECISION (both parties).** [34:16] Robert: *"because the problem definition, obviously is going to move out to the overview, yes"*; [34:45] Zhang: *"So basically this section three point one is gonna move to overview."* (HSTD/representation/service-allocation definitions from `03_methodology.tex:3-65` become the Overview's front half.)

**2.5 Methodology leading paragraph — CLEAR-DECISION.** [34:45] *"you're gonna start your section three, which is the methodology right? So before talking about each like be, for example, before talking about the fairness objective, you want to start with kind of leading. Paragraph telling people that we introduce this um fade approach, where um giving people kind of overview of what the framework looks like and refer people to this framework. Figure two, so people know what you are proposing, what you are trying to do. And then um introduce each each parts, each design part uh for within each subsections."* [35:35] *"So that way people will first know what you are trying to do, and then would be able to follow along."* [35:42] Robert read-back accepted, adding: *"I could probably structure the figure similar to the argument, so it'll make sense."* ⇒ Fig-2 stage order should match methodology subsection order (Robert's, unopposed).

**2.6 Ordering of methodology parts — CLEAR-DECISION on principle.** Her example puts the fairness objective first ([34:45]); her skeleton at [20:24]: *"the major skeleton of this approach is that what we do is we do attribution, right? We so from let's say 1000 trajectories, we are selecting Like 10 out of them or 100 out of them. 100 trajectories we have selected out of them. And with these 100 trajectories, we are doing the editing, right? And we are in terms of doing the editing, we are using the trim and the lift to to edit the trajectories. Okay. And did I miss anything?"* → [21:22] Robert: *"No, I think that yeah, I hear you."*
⇒ (1) fairness objective → (2) attribution under budget → (3) trim → (4) lift → (5) edit-aware weighting; one challenge each; compatible with the email's five leaf blocks without its names. ⚠️ Her skeleton omits weighting; Robert had scoped the figure talk to the editor at [13:10] *"This is the algorithm or the editing part of the algorithm, separating it from any kind of model evaluation or pass through to a model."* See U3.

**2.7 Abstraction level; "tricks" → appendix — CLEAR-DECISION.** [19:28] *"You are like thinking it in it like, in a very complicated way. ... One way is that we stick to what our implementation looks like and reflect our implementation. ... but when it's kind of we are explaining to people what we have done step by step, it's gonna be complicated to understand for the people. Instead, We want to abstract our approach in a way that we can put it into a more simple story."* [20:24] *"Some very complicated implementation details can be some tricks that we have used to make this approach work better. And that part could be put into the appendix."* Re-applied to text at [56:27]. ⇒ main text = abstracted skeleton + objective + weighting; mechanism detail (soft cell assignment, tapered w_j, backward-reachability repair, hat-matrix/FWL, O(N) identity) → appendix.

**2.8 Intro register — CLEAR-DECISION.** [36:50] *"It's mainly about the introduction, the style of the introduction. And when it comes to methodology, it's gonna be whatever makes sense for us. But when it's introduction, you don't want to lose audience because they don't understand the language."* [37:09] Robert: *"first version was definitely lose everybody like right from the beginning."* ⇒ no formalism in the intro (no F_demo definition/hat matrix/residual notation); technical register starts §2.

**2.9 Heading terminology free — CLEAR-DECISION.** [36:28] Robert: *"Fairness surrogate, just to make sure we're on the same page"* → [36:31] *"It's a fairness objective that one you you're referring to. I'm just using my language."* → [36:40] *"Not necessarily like you can just use whatever language that makes sense."* ⇒ "Collective Fairness Surrogate" is her paraphrase; "surrogate" not required.

## 3. How many challenges, and which

**UNRESOLVED (count) / CLEAR (the deciding rule).** [31:30] *"We can define that C one, C two, C three over here"* is illustrative of the labeling device (she was reading another paper's overview on screen), immediately followed by [32:12] *"The challenges we have mentioned in the introduction"* — i.e. ours, whatever the number. No count was mandated. Her revised draft has 3; the paper has 5. §2.2's rule + the five main-text blocks of §2.6 ⇒ **five maps 1:1**; three forces merges.
Mapping against current C1–C5 (`01_introduction.tex:112-132`): C3 (equal service is wrong target) → objective; **C1 (data is scarce) is the one needing rewrite** — under the new spine the operative constraint is *budget* ("we may touch only a small share, and it must be the share that matters") → attribution-under-budget; C4 (level up not down) → lift; C2 (fidelity) → trim/validity+fidelity constraints, where the demoted trade-off now lives; C5 (survive training) → edit-aware weighting. Recommendation (not a decision): keep five, relabel C1 as the budget challenge, put the [06:02] "unique challenge" sentence above the list rather than in it.

## 4. Experiments architecture

**4.1 Leading paragraph — CLEAR-DECISION.** [39:14] *"when we are writing kind of experiments part, we usually start with kind of. In the experiment, What we want to show kind of having a leading paragraph telling people with what we want to show. Um. What is the experiment?"*

**4.2 RQ framing endorsed, count unfixed.** [43:05] *"we're having several questions we want to answer within evaluation section. The first one may be um the fairness. Whether our approach is able to improve fairness."* [44:03] *"For fairness, still we want to answer these questions."* The email's "5 research questions" was never restated.

**4.3 Shenzhen-main + SF-transferability — CLEAR (direction) / SOFT (mechanics).** Strategy A [43:05]: *"we can directly put The experiment results for Shenzhen and San Francisco side by side, And we can just use one paragraph summarizing what we have observed in Shenzhen and San Francisco ... for each questions ... side by side. That is one strategy."* Strategy B [44:03]: *"For fairness question, we just show the Shenzhen. Yes, I think the second way is what you are using right now. And for other questions, you also want to use only the Shenzhen data. And finally, You are saying making the claim that this approach is also transferable to San Francisco. For other or or to taxi data in other cities, and then you are listing the San Francisco result."* → [44:52] *"I personally would prefer this way of writing because it's gonna save us some space."* Also [42:36] *"So you have created okay, a distinct section for for."* ⇒ existing §4.7 architecture survives; tighten, re-label as **transferability**, phrase the claim as "to taxi data in other cities". Robert's framing at [40:03] (*"the Shenzhen experiments and then partial reproduction of all of those experiments on San Francisco"*) went uncorrected.

**4.4 Redundancy policy — CLEAR-DECISION.** [46:31] *"if we observe, we have the same observation in Shenzhen and San Francisco, we can either combine it, Say that we have the same observation for Shenzhen and San Francisco, and illustrate the observation in one paragraph, in the same paragraph or paragraphs. Yeah. And we don't need to kind of say the same thing for two times."*

**4.5 Baselines asymmetry accepted; no remedial work.** [44:58] Robert: *"we also have baselines that we have to report results for or experiment results, or else we lose the motivating, you know, or the isolating factors."* → [45:27] *"Yeah, so we have the results for both cities, right?"* → [45:31] *"in the fairness and the propagation that we have uh basically parity between San, Francisco and Shenzhen. But when it comes to the baselines, uh much of the baselines are, Have only been implemented or like, um, juxtaposed against uh Shenzhen, because computational yeah issues"* → [46:08] *"Okay so."* ⇒ SF carries fairness + propagation; baselines remain Shenzhen-only. Plaud "AI suggestion" #4 (minimum SF baseline set) is the tool's question, not Zhang's.

**4.6 NO new experiment; specifically no k-sweep — CLEAR by exhaustive absence.** "budget" occurs exactly once in the transcript (line 9, [04:12], positioning). No budget analysis, k-sweep, or sensitivity study mentioned; the email's "budget analysis if possible" never came up, and Zhang spent the back half removing scope ([37:48] *"we can just submit whatever we have for now to the conference"*; [53:20] *"still don't be too stressful"*). ALGORITHM_FACTS L152: no k-sweep exists; new compute required; flag any text implying one. Honest substitutes already in hand: the ~10%-of-corpus edited-slice figure; existing dose-response sweeps (w10–w50, oversampling, α).

**4.7 Optional case-study figure — SOFT-PREFERENCE.** [52:23] *"initially, I was thinking that this is a figure for case study,"* → [52:33] *"So in experiment, We usually what we usually do is to give people a more concrete idea about the editing results. We show a case study of what a real editing looks like. So in this way, I thought this one is a case study."* → [52:57] *"So I think if we have time, if you have time, it's good to have a case study telling people, What a successful editing looks like if we, no time is totally okay to, um, have like to put in the paper whatever we already have and reorganize it."* → [53:20] *"Yes. So it's up to you"*. Illustration of existing results, not a new experiment; natural salvage path for the retired Fig-2 city panels.

## 5. Appendix policy — CLEAR-DECISION
[40:30] *"It is okay. So, we don't need to worry too much about the reproducibility for now because we still have time to modify the code. What, we want to do is we want to make sure that in the appendix, we have enough detail about this implementation."* Plus [20:24] (tricks → appendix).
⇒ (1) reproducibility is not a deliverable this round; (2) appendix's job = "enough detail about this implementation" (keep A–E; absorb displaced mechanism detail, derivations, hyperparameters, ε/king-move repair, weight sweep); (3) per repo protected register, content is **relocated, never deleted** (Fidelity-B 0.187, supply-channel tier accounting, ecological caveat, vanilla-BC null must survive somewhere).

## 6. Abstract / title / contributions
- **Title — UNRESOLVED by silence.** Zero mentions in 62 min. Adopted title lacks "budget-aware"; if her revision PDF differs, hers is more recent and she does the final pass.
- **Abstract — UNRESOLVED by silence.** Zero mentions; her draft supplies one; must not assert raw fairness barely moves nor attribute downstream gains to edits alone (§1.2).
- **Contributions — UNRESOLVED but mechanically forced.** Current list (`01_introduction.tex:157-175`) cites C1–C5, which are no longer itemized in the intro (must point forward to §2 or be reworded); its first bullet leads with editor+fidelity, whereas §1.3 says the collective/dataset-level fairness contribution should lead and fidelity should read as a constraint. Keeping the list in the Intro was never questioned.

## 7. Robert's pushbacks / Zhang's softenings

| # | Item | Robert | Outcome |
|---|---|---|---|
| P1 | [28:49]–[30:07] exact section names/shape | trouble fitting work into her names/blocks | constraint released; only challenge-alignment binding. **Robert won.** |
| P2 | [30:46]–[32:12] explicit challenges + back-refs | keep the C-item device | device kept, itemization relocated to §2, itemize env banned. **Compromise; Zhang set location.** |
| P3 | [17:57]–[19:27] figure fidelity to algorithm (districts matter) | districts/grid carry method content | overridden: abstract it, details→appendix. **Zhang won.** |
| P4 | [25:09]–[28:19], [54:57]–[56:27] TikZ vs Keynote | consistency/craft; "uncanny ring" invites scrutiny | [56:27] *"My suggestion is put it into the least priority ... what is really important is the content"*; [57:24] *"as long as the information in that figure are are preserved"*. **Softened: tool is Robert's choice, content first.** Parallel-window plan [57:01] unopposed. |
| P5 | [47:09]–[47:46] distinct SF section for differences | *"There were some like, uh, slight differences between how uh Shenzhen and San Francisco behaved, that I think as far as like, defensibility um kind of need to be like at least recognized and that's why, Um we have like a separate section"*; *"A very astute reviewer might look at some of the tables and see some numbers that lead to questions"* | [48:31] *"Mhm Okay. So I think that's pretty much what I have."* → accepted. Redundancy policy trims duplicated *agreements*; *differences* stay explicit. **Robert won.** |
| P6 | [44:58]–[46:08] baselines are the isolating controls | must report or isolation collapses; SF partial for compute | asymmetry accepted. **Robert won.** |
| P7 | [10:10]–[10:42] ditch her annotated snapshot | restructure supersedes | unopposed. **Robert won.** |
| P8 | [37:09]–[38:19], [61:09] scope/quality vs deadline | reviewer-foothold worry | [37:48] *"we can just submit whatever we have for now"*; [38:19] *"it's okay that is is not perfect for this time"*; [53:29] *"we want a sustainable way of working instead of just work for this submission"*. **Zhang lowered the bar.** |

## 8. MEETING vs EMAIL deltas

| # | Email / revised draft | Meeting | TS | Status |
|---|---|---|---|---|
| D1 | Challenges itemized in the **Introduction** (3 items) | Intro = brief prose, **not itemized**; labeled list defined in **§2 Overview** after problem definition | [31:00],[31:30],[32:12] | **CLEAR — supersedes email** |
| D2 | — | In Overview, **no LaTeX itemize env**; "just stack them all together" for space | [32:12] | **CLEAR — new** |
| D3 | Fully named outline (2 Overview / 3.1 Surrogate / 3.2.1-3 / 3.3) | *"It doesn't have to be exact section name"*; shape flexible; adjust for dependencies | [29:26],[30:07] | **CLEAR — non-binding** |
| D4 | "Collective Fairness Surrogate" | *"It's a fairness objective ... I'm just using my language"*; *"whatever language that makes sense"* | [36:31],[36:40] | **CLEAR — non-binding** |
| D5 | Structure = her outline | Real invariant: **each challenge ↔ one subsection**, logical order | [30:07],[33:20] | **CLEAR — the binding rule** |
| D6 | **5 research questions** | RQ framing yes, **count never stated** | [43:05],[44:03] | **SOFT / count UNRESOLVED** |
| D7 | **"budget analysis if possible"** | **Never mentioned**; no new experiment anywhere; scope reduced | absent; [37:48],[53:20] | **DROPPED — no k-sweep (ALGORITHM_FACTS L152)** |
| D8 | (silent on SZ/SF split) | **SZ answers all questions; SF appended as transferability**, for space; our draft already does this | [43:05],[44:03],[44:52],[42:36] | **CLEAR (direction)/SOFT (mechanics)** |
| D9 | (silent) | Same observation both cities → say once; real differences → still called out | [46:31],[47:09],[48:31] | **CLEAR** |
| D10 | (silent) | **Reproducibility not a deliverable**; appendix needs "enough detail about this implementation" | [40:30] | **CLEAR — new** |
| D11 | Derivations → appendix | Broadened: **any complicated implementation detail / "tricks"** → appendix; abstracted main text | [19:28],[20:24] | **CLEAR — broader** |
| D12 | Novelty inherited: editing↔fidelity trade-off | **Demoted**; collective/global fairness vs FairGAN/model-side is *"the most distinctive part"* | [05:09],[06:02],[06:55] | **CLEAR — biggest positioning delta** |
| D13 | Overview para names **two stages** (editing; edit-aware weighting) | Verbal skeleton = attribution→trim→lift only (figure scoped to editor) | [13:10],[20:24] | **UNRESOLVED (U3) — keep both; email more complete** |
| D14 | "additional figure" (read as possible case study) | It **is the framework figure**; a **case study** is separate and optional | [52:07],[52:57],[53:20] | **CLEAR / SOFT** |
| D15 | Outline numbering typo (3.4 vs 3.3) | Moot — numbering non-binding | [29:26] | **Resolved by D3** |
| D16 | (silent) | §3 opens with leading paragraph naming FATE + **cross-ref Figure 2**; §4 opens with purpose+organization | [34:45],[39:14] | **CLEAR — new** |
| D17 | (silent) | **Intro must be non-expert readable**; methodology register ours | [36:50] | **CLEAR — new** |
| D18 | Related Work §5 after Experiments | *"Then following related work and then the conclusion"* | [39:14] | **CLEAR — confirmed** |

## 9. UNRESOLVED (11)
- **U1 Challenge count.** Never fixed; "C one, C two, C three" [31:30] illustrative; §2.2 rule ⇒ 5 blocks ⇒ 5 challenges is self-consistent. Robert's call.
- **U2 Fate of current §3.3 "Why Demand-Only Editing Cannot Help the Under-Served"** (`03_methodology.tex:168`). Never mentioned; email demotes to a paragraph; ALGORITHM_FACTS says it must survive. Recommend keeping as the block answering C4.
- **U3 Does Fig-2 / the skeleton include downstream edit-aware weighting?** Figure scoped to editor [13:10]; skeleton stops at lift [20:24]; but the reframe exists because of edit+upweight and the spine claim is downstream. Recommend a final weighting/BC band.
- **U4 Where the demoted fidelity trade-off is argued.** Retained but unlocated (candidates: constraint paragraph in trim/validity block; Fidelity-B disclosure in §4).
- **U5 RQ count/labels for §4.** "Several"; only Q1 named.
- **U6 Whether each Overview challenge carries a paired solution pointer** ("and solutions" is Plaud's, not Zhang's).
- **U7 Title.** Never mentioned; "budget-aware" absent from adopted title; her revision may differ.
- **U8 Abstract.** Never mentioned; hers supplies one; must avoid the §1.2 literal claim.
- **U9 Contributions list.** Never discussed; forced to change by D1 and D12.
- **U10 Conclusion content/length.** Only position settled (current §5 spills ~7 lines onto p9).
- **U11 cGAIL-style boldfaced defined terms in §2.** Email yes; meeting cited that paper only for ordering. Harmless to keep.

## 10. Machine-summary errors to not propagate
1. `highlights_raw.html` highlight #2 auto-title inverts the decision (claims the editing/fidelity trade-off is the more important problem); its own body and the transcript ([05:09]–[06:55]) say the opposite. `plaud_summary.md` lines 15-17 are correct.
2. `plaud_summary.md` line 42 adds *"then challenges and solutions"*; the transcript ([31:30]) says definitions → problem definition → challenges, with no solutions step.
