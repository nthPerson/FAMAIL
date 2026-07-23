# Meeting 44 — Narrative / Argument-Strategy Analysis

> ⚠️ **POST-DEBRIEF CORRECTIONS (Robert, 2026-07-23) — parts of this document are superseded.**
> 1. Dr. Zhang WAS reviewing the CURRENT paper content (Robert transferred it to Overleaf pre-meeting; she views and will edit there). Retract every "stale copy / older version / discount-as-render-artifact" inference in this file — ALL her feedback binds against the current text.
> 2. Template: per Robert, conform to her direction to NOT use the `\keywords{...}` block (the "corrected template adds keywords" reading is superseded); verify against KDD template standards.
> 3. Raw data: releasable IF 100% anonymous (not a flat no); in-paper data references must not leak identifying information.
> 4. Hat-matrix citation: stays in the main body (derivation content still moves to the appendix).
> Authoritative record: ../MEETING_44_DEBRIEF.md (§2, §3, §6, §7).

**Lens:** why the paper fails to persuade, and exactly what story structure the PI mandated instead.
**Sources:** `meeting_44_transcript.txt` (346 lines, primary), `plaud_summary.md`, `plaud_discussion_summary_raw.txt` (secondary), current `paper/sections/{00_abstract,01_introduction,02_related_work,03_methodology,04_experiments}.tex`.
**Date:** 2026-07-23. Paper due Mon 2026-07-27 23:59 AoE (Hawaii); meeting targets a **Sunday 07-26** submit.

> **North star (the one claim the paper must sell).** Zhang forced Robert to name the single most important claim, and they converged:
> — Robert [0:37:51]: *"existing approaches can improve fairness, but they don't do so while also retaining realism. And so, That realistic usability aspect is the core thing or the issue."*
> — Zhang [0:38:14]: *"So okay, so the fairness versus realism tradeoff is a major thing we want to solve. So that is kind of core of this problem, right?"*
> Everything below is downstream of this: the whole rewrite exists to make a reviewer *feel* this trade-off and *believe* existing methods cannot resolve it, within the first column.

---

## Part 1 — THE MANDATE (quote-anchored)

### 1.1 Core-problem framing: the fairness-vs-fidelity trade-off

The paper's spine is a trade-off: **existing approaches improve fairness but lose realism/fidelity; FATE achieves both.** Zhang anchors the *difficulty* of this trade-off to the model-level regularization literature (ref "[37]", which Robert maps to `\cite{zheng2023}` at [0:35:29]):

- Zhang [0:38:14]: *"this literature is saying that this balance is different. 37 is saying that this balance is difficult to be, to be reached right."*
- Zhang [0:38:47]: *"So you can see at this model level. Approaches so model level approaches, and you are adding more ... citations supporting it."*

So the framing is: *at the model level the fairness/fidelity balance is provably hard to reach (anchored on [37]); data-side generation buys fairness by sacrificing fidelity; FATE claims both because it edits real data under a fidelity guardrail.*

**Terminology nuance — "fidelity" vs "realism" (which term wins, and the paper-specific trap).**
Zhang wants the **general** term to be **fidelity**, not "realism":
- Zhang [0:29:13]: *"That's the fidelity, right? ... So we use a word that we are familiar with. So I think another thing with AI is that you're usually inventing some phrases that are not like what ... domain experts or practitioners use in this domain."*
- Zhang [1:41:47]: *"if you are using realism, make it a[ll] of realism instead of ... fidelity. But I wouldn't recommend using realism because realism usually means other things instead of fidelity."*
- Zhang [1:47:46]: *"I'm not familiar with people using realism instead of ... Fidelity in terms of trajectory generation."*

Robert raises a real collision: in **this** paper "fidelity" is already a *paper-specific measured quantity*, so using it as the general word in the intro risks implying other works meet FATE's definition:
- Robert [1:49:56]/[1:50:33]: *"the problem with using fidelity in the context of the introduction is ... we define fidelity in our paper as it has a specific meaning. And in the introduction, it's a general meaning we're trying to apply it to other works. So there is a tension there."*
- Robert [1:51:04]: *"we have fidelity A and fidelity B. Fidelity A being the discriminator that we're familiar with, fidelity B being a distributional similarity."*
- Zhang [1:51:04] draws the distinction herself: *"fidelity score is not fidelity"* — i.e. the ST-SiameseNet **fidelity score** (Fidelity-A) is a paper-specific instrument, distinct from *fidelity* the general property.

**Resolution (per Plaud + her firmness):** replace "realism" → **"fidelity" as the general term**, but **typographically/nominally separate** the paper-specific senses (the ST-SiameseNet discriminator = *Fidelity-A*; distributional similarity = *Fidelity-B*) from the general notion. The residual worry — that intro-level "fidelity" implies other works satisfy FATE's exact definition — was **left unresolved** in the meeting (see Open Questions).

### 1.2 The problem-driven-story requirement (tech report → paper)

Her central diagnosis: the draft documents *what was done*, not *why*.

- Zhang [0:21:57]/[0:22:00]: *"the main issue is not humanizing. It is ... how the story has been like combined together and how explanation went. ... a lot of times, AI's are just gonna like put things in a way that they can understand quickly, and it's kind of not naturally how we are kind of trying to [convey]."*
- Zhang [0:22:00]: *"the current version is a kind of ... good documentation of what has been done instead of ... a good way of telling the story that what the problem we want to solve and what strategies we have proposed to solve this problem. So it's kind of more problem driven and design driven. ... We are designing approaches accordingly, but for current work, it's more like this is what we do instead of why we do it."*
- Zhang [1:36:29]: *"there is a difference about kind of a tech report versus a paper because when it's paper, It has to be problem driven, and it has to be kind of telling people you are designing this approach because [of] different reasons."*

**KDD reviewer conventions (why this matters procedurally):**
- Zhang [0:49:07]/[0:50:00]: *"when we are evaluating and reviewing papers, We want to make sure the paper is easy to read, because every reviewer is having six seven papers ... we want our paper to be easy to read for them. So that they can know ... what problem we want to solve and what motivation ... and why existing approaches don't work. Instantly, by only reading ... an introduction, so [otherwise] we didn't know the contribution, the novelty of this work."*
- Zhang [1:43:36]: *"I know how the KDD community usually read papers, and then I think it's going to be difficult ... if we are submitting in the current way."* (She is candidly torn — [1:43:36] *"I'm defeating myself"* — between rewriting to KDD-idiom and preserving Robert's style; Robert [1:44:15] defers to her reviewer experience.)

### 1.3 The Challenges section — what it contains, where it goes, how it links to the method

A **Challenges** element is the single biggest *missing* structural piece. It belongs **in the introduction**, states **why the problem is hard AND why existing methods fail** (she treats these as the same list), and the **methodology must be organized to answer each challenge**.

- Zhang [0:23:18]/[0:23:22]: *"what are the specific challenges. What are specific challenges that makes existing approaches doesn't work."*
- Zhang [0:26:14]/[0:26:35]: *"the major thing is not introducing a new approach, but introducing why we want to invent or create a new approach. So what makes in-processing approaches fail? ... Why can't we directly use in-processing approaches?"*
- Zhang [1:25:46]: *"one important thing that we forgot to mention in the challenges part is, oh, sorry, in the introduction part is the challenges. What makes this problem challenging and those ... challenges are also making existing works ... doesn't work properly anymore. So what are the challenges?"*
- **Link to method** — Zhang [1:36:29]/[1:39:40]: *"you can just think about what are the challenges. And ... write or refine the methodology part against the challenges instead of ... directly telling what we do."* / *"Highlight what are the challenges. And trying to optimize your methodology against your challenges."*
- Robert [1:26:22] accepts and connects it back: *"that ... goes back to ... your first comments, which was we very much just show what we did ... and that would be adding the why we did it, because this was a hard fought ... formulation."*

The Plaud raw summary logs it as a concrete deliverable owned by Robert: *"Add a 'Challenges' subsection to the introduction explaining why the problem is difficult and why existing methods fail."*

### 1.4 The motivating example + early experimental evidence in Figure 1

Zhang wants an explicit **motivating example that existing imitation learners inherit/amplify unfairness**, and — if the evidence exists — **early experimental proof in Figure 1** that existing approaches cannot fix fairness.

- Zhang [0:51:43]: *"in this introduction, we may also be needing kind of a motivating example ... and I think this figure would do part of the work. Which is we want an example that is saying that existing approaches like GAIL or other approaches are actually ... amplifying or inheriting the unfairness in the data."*
- Two options offered — Zhang [0:51:43]: *"One approach is we are kind of using some news to try to highlight that problem. Another approach would be if we have any experiment results, We can put that results ... in the first figure to highlight that existing approaches ... cannot solve the problem we have. And that's in Figure one."*
- The specific ask for Figure 1 — Zhang [0:53:04]: *"what is the fairness score before editing? ... we can directly show one value over here. Like ... fairness score for this data and fairness score for this data. And then when we are doing imitation learning, we are able to generate something ... with like this unfair service. This is more fair service."*

**Firmness:** the *before* fairness score shown in Figure 1 is a **firm** ask (she repeats it). The **early-evidence-that-existing-methods-fail** is **conditional/firm-if-available** ("if we have any experiment results"); the news-anchored version is the explicit fallback. This is a narrative-strategy lever, not just figure polish: it turns Figure 1 from "here's our pipeline" into "here's the problem existing methods can't solve."

### 1.5 Confusion at the current intro + the precise-language rule

Zhang could not parse the current §1 paragraph-2 vocabulary — this is the concrete evidence the story fails to persuade its most sympathetic reader:

- Zhang [0:39:42]: *"No, I don't know what is in processing methods."*
- Zhang [0:39:38]: *"it's really taking me a lot of time in terms of understanding ... what is in[-processing] methods? what is ... objective and training signal conflict? what is objective? what is training signal? ... and I cannot identify why these in[-processing] methods cannot work ... and we have to create a new approach."*
- Zhang [0:40:48] on the data-generation sentence: *"why is reducing realism since their objective is trying to ... mimic[] how human are making decisions[?] and, why do they shift the distribution since they are actually learning the distribution of real human behaviors? And why are they obscuring the source of fairness[?] gain for sure, but ... for the first two points, I don't agree."*

**Her rule for precise language** (three concrete demands):
1. **Name the works, not a vague class.** Zhang [0:29:13]: *"you're usually inventing some phrases that are not like what ... domain experts or practitioners use."* [0:29:39]: *"we can do a lot of literature survey ... based on the paper itself, We ... extract some keywords ... [that] best explain the approach proposed by those papers."* [1:49:18]: *"we want to make sure ... what line of work we're talking about instead of using ... obscure languages that seems to point to one class of works, [when] it might also be pointing at another line of approaches."*
2. **Specify the level of intervention (data / model / hyperparameters).** Zhang [1:46:49]: *"Rebalancing models may be meaning ... you are rebalancing data ... or rebalancing model architecture or hyperparameters ... it's not by itself clear what you are actually rebalancing."*
3. **Specify WHICH distribution.** Zhang [1:47:46]: *"trajectory generation shift the distribution ... I don't understand why data generation is going to shift the distribution, shift what distribution?"* [0:41:36] adds she *disagrees* on the merits, not just the wording, for two of the three failure reasons — so precision here is also a **correctness** fix, not only clarity.

### 1.6 Citations — ≥2 per category, and recent (2025–2026)

- **≥2 per group** — Zhang [0:31:46]: *"only having one related work is not enough if ... we are saying that it's a group of work."* [0:32:37]: *"it's just one for the in-processing method. It doesn't make it a kind of a group ... When we're naming ... a group of works, We want to make sure that it's just a group of work instead of just one work."* [0:39:07]/[0:39:12]: *"if we are saying kind of a group of classes, you don't just have one paper ... at [at] least two."*
- **Recent works** — Zhang [0:32:37]: *"for all the related works, we want to make sure that they are more ... current, we want to [cite] works that are closer to us, like 2025, 2026."* (Robert [0:33:11] pushes back that a recent paper rarely supports the *exact* line; Zhang [0:33:24] concedes it's acceptable but still wants recency where possible.)
- **Her preferred re-categorization of the data side** — Zhang [0:33:24]/[0:33:57]: *"we can talk about ... trajectory editing approaches or generative approaches ... approaches that are trying to generate different data as a second line ... and the approaches that are trying to modify on the original data for certain purposes. So ... trajectory editing approaches and ... trajectory generation approach. So we can talk about those two lines."*

**Which categories need the ≥2 fix (concretely):** the intro's **in-processing** claim currently rides on a **single** cite (`\cite{zheng2023}`); the **data-side** claim should be split into her two lines (**trajectory editing** vs **trajectory generation**), each with ≥2 cites, at least one recent.

### 1.7 Related Work → half a column (full version to appendix)

- Zhang [1:20:02]: *"For the related work, we can shrink it into minimum of a half column. So, we can just briefly talk about what are the related approaches and talk about the differences. We can leave a complete version in the appendix. So we have more spaces to talk about what our problem looks like and what our approach looks like."*

**Firmness:** firm ("a half column"). **What must survive in the half-column:** only *(a)* what the related approaches are and *(b)* the differences (FATE's positioning/contrast). The current five-theme, full-column §2 moves wholesale to the appendix; the freed space is reinvested in problem + approach.

### 1.8 "Task" → "Problem Formulation" + rigorous definitions

- Zhang [1:25:21]: *"look at this one, the task. ... for the problem formulation part. There's not a problem formulation. It's just a task. You can just give it kind of problem[] definition or problem [formulation]."* Robert [1:25:46]: *"Formulation. Yeah, that's I like that better."*
- What "rigorously define" means to her — Zhang [1:32:12]: *"within problem definition, we want to define several things ... Like the trajectories, the reward functions. Sometimes we also define state or actions because it's kind of reinforcement learning thing."*

> **VERSION-SKEW FINDING (verify before writing).** The **local** `03_methodology.tex` **already** titles the subsection `\subsection{Problem Formulation}` (line 3) — but it still contains a `\textbf{Task.}` lead-in paragraph (line 29) that defines the task as a 3-part goal (i/ii/iii) **without** RL-style definitions of *state*, *action*, or *reward*. Either Zhang reviewed an older Overleaf version titled "Task," or she was reacting to the `\textbf{Task.}` lead-in. **The heading rename may already be done; the substantive ask (rigorous trajectory / state / action / reward definitions) is only partly met** — trajectory is defined ("a sequence of passenger-seeking states ending in a pickup", line 24), but there is no explicit reward function or state/action formalism.

### 1.9 Main-text vs appendix policy

**F_demo / R²_demo derivation → appendix; keep the collapsed equation + meaning + importance + the seminal hat-matrix reference in main.**
- Zhang [1:23:32]: *"for the equation one, we can directly say that there is f[_demo], That is calculated and the reason why it's calculated, and we point readers towards appendix in terms of how it's derived."*
- Robert [1:24:00]/Zhang [1:24:14]: *"move all of this to the appendix? ... And then just keep the one minus R square demo?"* — *"Yes, exactly. And also in the text we tell people why ... the meaning of one minus R square demo. And why it makes it, and we just ... point people to appendix for more information."*
- **Hat-matrix ref stays in main** — Robert [1:22:40]: *"It's the reference to use ... use the seminal work if there is a seminal work is the rule."* Zhang [1:22:40]: *"It's okay."* (The ref is `\cite{hoaglinwelsch1978}`, currently at methodology line 62.)
- **Concretely:** keep in main → Eq. (1) collapsed to `F_demo = 1 − r²_demo`, its meaning ("share of demand-adjusted service inequity attributable to demographics"), its importance ("this is what we optimize"), and the `hoaglinwelsch1978` pointer. Move to appendix → the `H = X̃(X̃ᵀX̃)⁻¹X̃ᵀ` / `M` construction, the RSS/TSS closed form, Frisch–Waugh–Lovell, and the `O(N)` identity (methodology lines 58–76). The appendix `\section{Derivations}` already exists to receive it (it already back-references `eq:fdemo`).

**Grid-cell / implementation details → appendix; the state-space abstraction stays.**
- Zhang [1:29:27]: *"we don't need to ... write in the detail about ... partitioning the map into ... 0.01 degree grid cells. Those are ... implementation details that could be pushed into the appendix. What we want to do is ... the city can be modeled as a kind of a state space ... the state are defined as spatial[-]temporal grid cells."*
- Zhang [1:30:24]: *"this is an implementation detail that doesn't need to go within methodology."*
- **Robert's real tension** [1:31:16]: *"the representation of the data impacts the formulation in such a way that if we were to just discretize the grid in any other way, our formulation would change."* Resolution in-meeting: the specific value **0.01°** is a config detail that can go; the abstraction (*state space = spatio-temporal grid cells*) stays. Motivation: generalizability — Zhang [1:28:52] *"we don't want to design one approach specifically for this taxi driver data ... When our data has been changed to ... transit logs ... logistics data, this same approach would still work."*

**Remove duplication between Problem Formulation and Experiments.**
- Zhang [1:37:28]: *"if you define it in the problem formulation ... [that] you want to create the map into 0.01 degree grid cells, ... make sure that those information doesn't appear in your experimental setup."*
- **CONFIRMED duplication:** `0.01°` grid appears in **both** `03_methodology.tex:5` and `04_experiments.tex:13` (`$48{\times}90$ grid of $0.01^{\circ}$ cells`). Robert [1:37:54] acknowledges only *strategic* intro↔method restatements ("jars the memory of the reader"); the grid-cell repeat is the non-strategic kind she wants gone.

### 1.10 AI-writing guidance

- **Workflow: write first, refine with AI after (never AI-first).** Zhang [1:45:55]: *"the way of writing a paper is always, you write it first, and you ask GPT to refine it. ... The problem with asking GPT to write it first, and then you refine it will be very, very, very subtle ... GPT is usually gonna use very general language that is not ... easily understandable by domain experts."* (Robert's current workflow is the inverse — [0:19:43] *"first draft, I'll have the AI write and then I go through and I make things sound more human"* — which Zhang [0:21:19] identifies as the root cause: *"the main issue is not humanizing."*)
- **Flagged GPT-vagueness examples:** *"rebalancing models"* [1:46:49] (rebalancing *what* — data/architecture/hyperparameters?); *"data generation ... shifts the distribution"* [1:47:46] (*which* distribution? she asks if it means mode collapse [0:41:36]); *"in-processing methods"* [0:39:42] and *"objective and training signal conflict"* [0:35:00] (unparseable to a domain expert).
- The fix is precision (§1.5), not tone-smoothing.

---

## Part 2 — GAP ANALYSIS (mandate vs current §1 / §2)

### 2.1 Current §1 `01_introduction.tex`, paragraph by paragraph

The file header comment names the structure as a **six-beat** design (a structure that was **Zhang's own earlier ask** — see §2.3 collisions).

| # | Current beat (lines) | First words | Serves / Contradicts / Orphaned |
|---|---|---|---|
| P1 | Hook / motivation (7–15) | *"Urban mobility policies are increasingly learned by imitation..."* | **SERVES.** This is the problem beat Zhang endorses — [0:46:32] *"it's only kind of the first paragraph in introduction that is trying to highlight this problem."* Keep; it is the motivating-example seed (imitation reproduces + re-enacts inequity, `\cite{ensign2018,lumisaac2016}`). |
| P2 | Limitations of existing approaches (17–27) | *"Interventions typically target one end of the pipeline or the other. In-processing methods regularize the model while leaving the demonstrations biased `\cite{zheng2023}`, so objective and training signal conflict. Data-generation and data-rebalancing methods ... — but this can reduce realism, shift the distribution, and obscure the source of any fairness gain."* | **SERVES the slot but UNDER-DELIVERS — this is the paragraph she spent ~20 min rejecting.** Failures: (a) "in-processing methods" opaque [0:39:42]; (b) "objective and training signal conflict" opaque [0:35:00]; (c) **one** cite for in-processing, needs ≥2 [0:32:37]; (d) failure reasons *asserted not explained*, and she *disagrees* with two [0:41:36]; (e) "realism"→"fidelity" [1:41:47]; (f) needs her editing-vs-generation split [0:33:57]. Zhang [0:46:32]: *"we need to sharpen it, make it more precise in terms of ... why existing approaches cannot work."* |
| P3 | Approach summary — "third position" (64–73) | *"FATE takes a third position. As data augmentation, it keeps the real corpus..."* | **PARTIALLY SERVES; framing CONTRADICTS the mandate.** The "third position" framing is exactly what Zhang challenged — [0:25:11] *"Why we need to take a third position?"*; [0:26:14] *"the major thing is not introducing a new approach, but introducing why we want to invent ... a new approach."* The approach should be summarized in **one** paragraph *tied to the challenges*, not slotted as "a third option." |
| P4 | Trim/lift mechanism + leveling-down + results (75–99) | *"FATE's first editing phase, trim, moves only demand..."* | **CONTRADICTS length/altitude mandate.** This is methodology-grade detail (two-cell radius, `+0.0226`, `+0.0316`, leveling-down analogue) living in the intro. Zhang [0:50:46]: *"we can shrink the FATE's proposed approach part into one paragraph ... we don't need to introduce in so much details because we are gonna provide more details in the methodology part."* Collapse; push detail to §3/§4. |
| P5 | Transfer + controls + external metrics + baseline (101–119) | *"Neither edit survives training on its own..."* | **CONTRADICTS same mandate.** More results/method detail (null, upweighting dose-response, DP/DI/Theil, oversampling `10.5\%`). Belongs in the one-paragraph summary at most as a one-line promise; evidence stays in §4. |
| P6 | Contributions (122–136) | *"\textbf{Contributions.}"* | **SERVES** (KDD convention). Keep, but re-point each bullet at a Challenge (C1..Cn) so contributions read as *challenge → solution*. |

**Orphaned / missing against the mandate:**
- **No Challenges beat at all** — the single largest structural gap [1:25:46]. Nothing in §1 states *why the problem is hard*.
- **No explicit "existing imitation learners inherit/amplify unfairness" example** as a distinct motivating beat with an early-evidence hook [0:51:43]; P1 gestures at feedback loops but does not stage GAIL-as-failure.
- **No anonymous code link** — Zhang [0:50:46]: *"we need kind of an anonymous link telling people ... where ... the code will be released."*
- **"realism" is load-bearing in §1** (lines 21, 24, 68/72 "realism is inherited from the data") and in the **abstract** (lines 12, 27, 37) — all must migrate to "fidelity" with the paper-specific senses disambiguated (§1.1).

### 2.2 Current §2 `02_related_work.tex`, theme by theme

The file header names it **five themes, each closing on a one-sentence FATE contrast** (also **Zhang's own earlier ask** — §2.3).

| Theme (lines) | Content | Fate under the mandate |
|---|---|---|
| T1 Fairness interventions in ML (6–21) | pre/in/post taxonomy; `barocas2023, kamirancalders2012, feldman2015, corbettdavies2017, vermarubin2018` | Half-column keeps a **compressed** version (the pre/in/post framing is where "in-processing" is properly *defined* — that definition should arguably surface into §1's limitations beat). Full text → appendix. |
| T2 Fairness in urban transportation (23–35) | Gini/Theil; `zheng2023` as closest neighbor (0.361→0.084) | The `zheng2023` contrast is the **model-level anchor [37]** the intro leans on; its one-line differentiator ("lives inside one model's loss, does not transfer") should survive in the half-column. Rest → appendix. |
| T3 Imitation learning + mobility-as-identity (37–50) | cGAIL/xGAIL/`feng2020`; TULER/TULVAE/DeepTUL; `ren2020stsiamese` | Supplies the **GAIL motivating-example** citations Zhang wants in §1 [0:51:43] and the ST-SiameseNet guardrail lineage. Pull the essential contrast up; full survey → appendix. |
| T4 Adversarial perturbation & recourse (52–65) | FGSM/iFGSM/ST-iFGSM; STE/Gumbel; recourse | Method-lineage; least load-bearing for the *persuasion* story. Compress hardest / mostly → appendix. |
| T5 Leveling down & feedback loops (67–83) | `parfit1997, mittelstadt2024, zietlow2022, ensign2018, lumisaac2016` | Conceptually tied to P1's feedback loop and to §3's leveling analysis. One sentence may survive; rest → appendix. |

**Whole-section verdict:** the five-theme, ~full-column §2 is **superseded at the main-text level** — it becomes the appendix's complete related work; the main paper keeps a **half-column of approaches + differences only** [1:20:02]. The recurring *"none of these operate on sequential demonstrations / measures spatial service fairness"* contrasts are the sentences worth rescuing.

### 2.3 Where today's mandate SUPERSEDES / COLLIDES with Zhang's OWN earlier structures

*(Exposed, not adjudicated — Robert must decide.)*

1. **§1 six-beat structure (her earlier ask) vs today's "motivation → sharpened limitations → Challenges → one-para approach → contributions."** Today's mandate **inserts a Challenges beat** and **collapses the two detailed method/results beats (P4, P5) into one**. The six beats do not map one-to-one onto the new shape; beats P4/P5 largely *leave* §1. This is a direct supersession of the earlier six-beat design.
2. **§2 "five themes each closing on a FATE contrast" (her earlier ask) vs today's "half a column, full version in appendix."** A half-column cannot carry five themes each with its own closing contrast; the earlier pattern **survives only in the appendix**. Main-text §2 keeps at most the differentiators, not the five-theme scaffold.
3. **"realism," literature-backed (earlier posture) vs "fidelity," domain-convention (today).** Robert states "realism" was chosen *because the literature uses it* [1:41:59]/[1:42:14] and that all language is literature-backed per her *own prior* email rule [1:42:14]. Today she overrides toward "fidelity" [1:41:47]. Two of her own rules now collide (use-domain-conventions vs be-literature-faithful); Robert was still hunting the "realism" citation when the topic closed [1:43:07]. **Unresolved.**
4. **Scope: "general downstream data augmentation" (earlier framing, still in abstract/§1) vs "data augmentation *for imitation learning / behavior cloning*" (today).** Zhang repeatedly narrows the claim — [1:00:06] *"we are more like a data augmentation approach specifically for imitation learning"*; [1:03:10] *"we are designing an approach specifically for imitation learning."* Robert reframes as a **dual finding** rather than a narrowing [1:03:53]. The current abstract/§1 keep the broader "keeps the real corpus, generates nothing" framing. **Collision on how wide the top-line claim is.**

---

## Part 3 — PROPOSED EXECUTION SHAPE

Every element tagged with the mandating timestamp; `[inference]` marks my extrapolation beyond her explicit words.

### 3.1 Target §1 outline (paragraph by paragraph)

- **¶1 — Hook + motivating example (the trade-off is real).** Keep current P1's imitation-reproduces-inequity spine [0:22:47 = "the original data is not fair ... the bias will be propagated"], but **stage an explicit example that an existing imitation learner (GAIL) inherits/amplifies the unfairness** [0:51:43]. End on the fairness-vs-fidelity trade-off as the paper's core problem [0:38:14]. `[inference]`: fold the `ensign2018/lumisaac2016` feedback-loop cites here as the "why it compounds" clause.
- **¶2 — Limitations of existing approaches, sharpened (the trade-off is unsolved).** Rewrite current P2. Split into Zhang's **two lines**: *trajectory-editing/model-side* vs *trajectory-generation* [0:33:57]. For each: name the specific works (≥2, ≥1 recent 2025–26) [0:32:37], state the **level of intervention** (data/model/hyperparameter) [1:46:49], and explain the failure *mechanistically* — model-level regularization leaves demonstrations biased and the fairness/fidelity balance is hard to reach (anchor [37]=`zheng2023`) [0:38:14]; generation buys fairness by degrading fidelity / mode issues [0:41:36]. Replace "realism"→"fidelity" [1:41:47]. Retire "in-processing methods" and "objective and training signal conflict" as bare labels [0:39:42].
- **¶3 — Challenges C1..Cn (why it's hard = why prior methods fail).** NEW [1:25:46]. `[inference]` candidate challenges, each stating why it defeats a prior class and which §3 subsection answers it:
  - **C1 Fidelity under editing** — fairness edits must stay recognizable as real human trajectories (defeats generation) → answered by the ST-SiameseNet guardrail (§3.2 `F_fidelity`).
  - **C2 Demand-adjusted fairness target** — raw parity is wrong; must measure only the demographic-unexplained component (defeats naive rebalancing) → `F_demo` (§3.2).
  - **C3 Level-up, not level-down** — demand-only editing can only reduce over-service (defeats trim-alone / any demand-only editor) → the *lift* supply channel (§3.3, §3 leveling analysis).
  - **C4 Survival through training** — a 10% edit is averaged away by vanilla BC (defeats "just edit the data") → upweighted imitation (§3 editor/downstream).
  This directly executes her *"optimize your methodology against your challenges"* [1:39:40] — each Cx maps to a §3 subsection and a contribution bullet.
- **¶4 — One-paragraph approach summary, tied to the challenges.** Collapse current P3+P4+P5 into **one** paragraph [0:50:46]: FATE edits a small attribution-chosen slice of real trajectories under a frozen identity guardrail and upweights it — one clause per challenge, **no** two-cell-radius / `+0.0226` / leveling-down detail (that goes to §3/§4) [1:52:14]. Drop the "third position" framing [0:26:14].
- **¶5 — Contributions + anonymous code link.** Keep the four-bullet list [current 122–136] but re-tie each bullet to a Cx; add the **anonymous GitHub link** [0:50:46]/[1:57:12] (empty repo acceptable [1:59:06]).
- **Figure 1 (narrative role).** Re-purpose as *motivation*, not pipeline: show the **before fairness score** on raw data and that an existing imitation learner reproduces the gap [0:53:04]; keep totals of taxis/passengers constant (relocation, not removal) [1:18:40]; add advantaged/disadvantaged labels + legend + map background, adopting Figure 2's style [1:17:25]. *(Figure mechanics are the Figures-analyst's lane; flagged here only because early-evidence-in-Fig-1 is a narrative lever.)*

### 3.2 Half-column §2 (what survives, sentence-level)

Target ≈ half a column [1:20:02]. Keep only approaches + differences:
- One sentence: fairness interventions group into pre/in/post; FATE is pre-processing on *demonstrations* [T1] — this is also where "in-processing" gets its one honest definition `[inference]`.
- One sentence: the closest transportation-fairness neighbor regularizes inside one model's loss and does not transfer (`zheng2023`) [T2].
- One sentence: imitation-learning-for-mobility optimizes fidelity and never audits demographic fairness (cGAIL/xGAIL); FATE edits the demonstrations such models consume (`ren2020stsiamese` guardrail) [T3].
- (Optional, if space) one clause: editing machinery descends from bounded adversarial perturbation / recourse [T4]; one clause: leveling-down framing [T5].
- **Full five-theme text → appendix related work** [1:20:02].

### 3.3 Knock-on moves (§3 / §4 → appendix, and §3 internal)

- **Rename + rigorously formalize Problem Formulation** [1:25:21]: confirm heading is "Problem Formulation" (already local); add explicit **trajectory / state / action / reward** definitions [1:32:12] to the `\textbf{Task.}` block (currently RL-underspecified).
- **F_demo derivation → appendix; keep collapsed Eq. (1) + meaning + importance + `hoaglinwelsch1978`** in main [1:23:32]/[1:24:14]. (Methodology lines 58–76 move; the collapsed `F_demo = 1 − r²_demo` + its two paragraphs of interpretation stay.)
- **Grid-cell `0.01°` → appendix**; keep the *state-space = spatio-temporal grid cells* abstraction in §3 [1:29:27]. `[inference]`: keep a one-line "we discretize space-time into cells; the specific resolution is an appendix config" bridge to preserve Robert's formulation-depends-on-discretization point [1:31:16].
- **De-duplicate:** remove the `0.01°`/`48×90` grid restatement from `04_experiments.tex:13` once it lives in §3+appendix [1:37:28] (confirmed duplicate).
- **Rewrite the abstract** to the same problem→challenge→approach shape and swap "realism"→"fidelity" [1:52:14] (raw summary: *"Rewrite the abstract and introduction"*).

### 3.4 OPEN QUESTIONS Robert must decide before writing

1. **Who owns the intro rewrite?** Zhang [0:50:46] *"I'm gonna reorganize it"* and [1:52:42] *"if I have time, I'm gonna rewrite a version, And it's up to you whether you want to use [it]"*, [2:00:03] *"I'm gonna create a new text file."* Robert [2:00:03] wants to avoid clobbering. **Decide: wait for her draft, write in parallel, or divide by section — and how to merge.**
2. **"fidelity" in the intro — general or paper-specific?** Adopting "fidelity" as the general term risks implying other works meet FATE's *Fidelity-A/B* definitions [1:49:56]. **Decide the disambiguation device** (e.g. "fidelity" general vs "Fidelity-A/B" small-caps instruments) — unresolved in-meeting.
3. **Top-line scope: general data augmentation vs BC-specific?** Zhang narrows to imitation-learning/BC [1:03:10]; Robert wants the "dual finding" width [1:03:53]. **Decide what the abstract/¶1 claim, and whether the future-work-beyond-BC caveat carries the width.**
4. **How many Challenges (C1..Cn), and their exact wording?** The C1–C4 above are `[inference]`; the set and phrasing are Robert's call, constrained by "each must kill a prior class and map to a §3 subsection."
5. **Early evidence in Figure 1 — which experiment (or news)?** [0:51:43] offers "results" or "news." **Decide whether a runnable existing-method-fails number/panel goes in Fig 1, or the news-anchored fallback** (the raw-editing-doesn't-survive null [§4] is a candidate).
6. **The `zheng2023`/[37] load.** Robert concedes it is "probably the most important work in this space" doing "a lot of work ... one reference" [0:36:43] and a *weakness* if it stands alone. **Decide the ≥2-cite companions for the model-level claim.**
7. **Recency sourcing.** Zhang wants 2025–26 cites [0:32:37]; Robert flags that recent papers rarely support the exact line [0:33:11]. **Decide which claims get recent companions vs keep seminal-only** (the "use the seminal work" rule [1:22:40] still applies to hat-matrix-type anchors).

### 3.5 TENSIONS (do not smooth over)

- **Mandate vs the 8-page knife-edge.** Robert: *"we have about three or four lines of additional space"* [0:47:22], compression is *"the number one struggle"* [0:31:27], lossless shrink is the hard part [1:28:17]. Zhang insists *"length is the least important thing ... We can always shrink it within eight pages"* [1:28:07] and blames a *"way too long related work"* [1:27:22]. **She is betting the half-column §2 buys the room the Challenges section + sharpened limitations will spend. That bet is unverified** — if Challenges + ≥2-cites-per-class + rigorous Problem Formulation out-grow the §2 savings, something in §4 must also give. Robert's fear of a length-driven desk reject [1:27:08] vs her *"length will [not] be a problem for desk rejection"* [1:27:17].
- **New Challenges section vs length.** Adding a whole intro beat while under an 8-page cap is in direct tension with 3–4 free lines. `[inference]`: net-neutral only if P4/P5 detail genuinely leaves §1 for §3/§4 (which itself is near-full).
- **Early-evidence-in-Figure-1 vs figure real estate.** Adding before-fairness scores, labels, legend, map background, and constant taxi/passenger totals [0:53:04]/[1:17:25] competes with the two-panel pipeline the figure already carries — the figure may need to *drop* the intermediate application layer Robert defended [0:57:50] to fit motivation content, but Zhang conceded that layer stays [1:03:10]. **Fig 1 cannot be both a full pipeline and a clean motivation panel; something gives.**
- **Precision vs generalizability.** Robert's recurring worry [1:48:44]/[1:48:31]: *"the more specific we may make the claim, the more narrowed the argument is ... we ... distance ourselves from the general problem."* Zhang [1:49:18]: *"It's not a complex balance ... we want to make sure what line of work we're talking about."* The mandate resolves toward precision, but the generalizability claim (transit/logistics [1:28:52]) still has to survive the specific language.
- **Two of Zhang's own rules collide:** use-domain-conventions ("fidelity") vs be-literature-faithful ("realism, from the literature"). Unresolved [1:42:14].

---

## Appendix — quick reference: mandated main-text vs appendix split

| Element | Main text | Appendix |
|---|---|---|
| `F_demo` [1:23:32] | `F_demo = 1 − r²_demo` + meaning + importance + `hoaglinwelsch1978` ref | `H`,`M` construction, RSS/TSS, FWL, `O(N)` (currently §3 ll.58–76) |
| Grid resolution [1:29:27] | *state space = spatio-temporal grid cells* abstraction | `0.01°`, `48×90`/`32×30` specifics |
| Related work [1:20:02] | half-column: approaches + differences | full five-theme survey |
| Duplicated grid [1:37:28] | define once (in §3) | — (remove restatement from §4 setup) |
| Approach detail [0:50:46] | one paragraph in §1 | trim/lift mechanics live in §3 (not appendix), out of §1 |
