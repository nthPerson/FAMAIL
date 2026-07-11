# FAMAIL Meeting 42 — Transcript Extraction (Ground Truth)

- **Source page:** "FAMAIL Meeting 42"
- **Page URL:** https://app.notion.com/p/398eb30651108020a026c9c50dee1e86
- **Parent:** Research Dashboard → Meetings → Meetings Database
- **Meeting date (page property "Scheduled Time & Date"):** 2026-07-09 (page created 2026-07-09T17:51Z; last edited ~2026-07-11)
- **Attendees (per page property):** Dr. Xin Zhang + Robert Ashe (2-person meeting)
- **Extraction date:** 2026-07-10
- **Transcript length processed:** full transcript, ~2,700 words / ~95 spoken blocks, read start ("Hi Robert.") to end ("See you next week. Bye."). **Confirmed read in full — single fetch returned the entire `<transcript>` block; no truncation.**
- **Diarization caveat:** The Notion transcript is NOT speaker-labeled. Robert delivers a long presentation-style update; Dr. Zhang interjects short acknowledgements ("Mm-hmm", "Yes", "Yeah") plus a handful of substantive sentences. Speaker attributions below are inferred from context and flagged where uncertain.

---

## 1. TODO / Action Items (each with supporting verbatim quote)

| # | Action item | Owner | Priority / timing | Verbatim support |
|---|-------------|-------|-------------------|------------------|
| T1 | Finish the still-running GPU evaluation of whether fairness propagates through the **trained behavioral cloning models** (the last piece of the previous eval stage). | Robert (GPU running) | Results "this afternoon" | "there's still one part of the previous evaluation that I'm still going through and running through the GPU and that's understanding if this the fairness propagates through the trained behavioral cloning models. So that's currently running on the GPU. It should land sometime this afternoon." |
| T2 | Implement the **other data-augmentation baselines** discussed last week (still not done as of this meeting). | Robert | "next steps" / while GPU runs | "the next steps, because last week we also talked about implementing other data augmentation baselines. So I'm going to be investigating that. I've got a little time while the GPU is running to actually do that right now. So those things are still on my list and I definitely intend to do them." |
| T3 | **Human-review all AI-assisted literature references** before putting them in the paper (verify each source actually says what is claimed). | Robert | before writing/citing | "I'm going to take a human review to make sure that these actually do say what they do... the last thing I want is to go making a claim that's not actually real. So that's literal nightmare fuel for me." |
| T4 | Motivate the **new attribution variant** ("new flavor of attribution") introduced for the trim+lift algorithm, same as the other objective-function terms. | Robert | with the objective-function write-up | "there's a new flavor of attribution that we implement for the new algorithm... So I'll make sure that that's motivated properly and all that stuff too." |
| T5 | **Start writing the paper — begin with the methodology section** (Dr. Zhang's directive; introduction can come later). | Robert (per PI direction) | now / near-term | Zhang: "maybe we can start like kind of working on the paper, at least have kind of for introduction or not necessarily introduction, but the methodology part." |
| T6 | Deliver a **draft abstract to Dr. Zhang by next week** (ahead of the abstract deadline so she can review). | Robert | by next week | "I know the abstract deadline is like a week before the paper deadline. So I'm hoping to have an abstract to you by next week." |

> Note: The Notion summary also lists "Begin assembling the methodology section" and "prepare a draft abstract" as action items (matches T5/T6). No additional action items were voiced beyond T1–T6.

---

## 2. PI Decisions (Dr. Xin Zhang) — verbatim

1. **Directs paper-writing to begin now, methodology first:**
   > "I think now for now, I think we can, I know you have documented a lot of things, so maybe we can start like kind of working on the paper, at least have kind of for introduction or not necessarily introduction, but the methodology part."

2. **Endorses the ablation study as necessary to the argument** (spoken right after the trim/lift explanation; attribution: Zhang):
   > "I think one thing, yeah, I think we also have the kind of the ablation study to say that this kind of design is really necessary."

3. **Approves the abstract serving as a placeholder** (can be refined later, only needs to be on-topic):
   > "That is sometimes like a placeholder, as long as it is related to what we are doing, that is okay. We can always come back and modify it."

4. **General endorsement of the new, more solid approach + wait-and-see on the BC/GAIL propagation result:**
   > "It's good that we kind of have a more solid approach, kind of improving all those metrics and we can wait and see how it's gonna perform like EC and Gale. Result." (garbled ASR; "EC and Gale" ≈ cGAIL / the imitation-learning models being rolled out.)

5. **Timeline confidence:**
   > "Sounds good. We still have some time, and hopefully we can catch the deadline and have it published."

> There was **no naming decision** (no F_causal→F_demo discussion), **no rejection**, and **no framing veto** in this meeting. Dr. Zhang's role was largely to acknowledge, endorse the direction, and direct writing to start.

---

## 3. Topic-by-Topic Discussion Record

### 3a. External fairness metrics (not optimized for)
- Robert implemented **three external metrics**: **disparate impact**, **demographic parity gap**, and the **Theil index** (ASR renders it "TEAL index").
  > "I implemented disparate impact, a demographic parity gap, and the TEAL index. And across the board, improvements in fairness."
- Purpose: prove fairness gains aren't just an artifact of optimizing the team's own metric.
  > "we can show that the improvement isn't just a byproduct of us optimizing for our own fairness metrics."
- **Caveat (the leveling-down problem), stated plainly:** the external-metric gains came from **removing service from over-served / advantaged areas** (higher income, higher property prices, lower migrant ratio), which conflicts with the "maintain service while improving fairness" goal and invites reviewer attack.
  > "The improvement that we achieved in these external metrics was achieved by removing service from over-served areas... higher income, higher property prices, lower migrant ratio... with just removing advantaged service, we're not really maintaining that good service."
- This external-metrics caveat is what **motivated the new algorithm** (below).

### 3b. New algorithm — "Trim & Lift" (the headline of the meeting)
- **Root cause of leveling-down:** in the old algorithm only **demand** was differentiable in Y = supply / demand, so it could only move the pickup (demand) and only ever reduced advantaged-area service.
  > "in the current algorithm, demand is the only differentiable part of the outcome. So we have the y equals the supply divided by the demand. Currently, demand is the differentiable part of that, and that's how we manipulate to the fairness."
- **Fix:** make **supply also differentiable** by reusing the **Gaussian softmax smoothing** already used for demand — applied to the 5×5-grid soft counts of unique active taxis (a taxi passing through the 5×5 grid within a time unit = one unit of supply; time unit = one hour).
  > "before, we used soft cell counts to use Gaussian smoothing to enable us to differentiate the demand... So I extend that to the supply, do the Gaussian softmax, and basically count taxis over a smooth region."
- **Two phases:**
  - **Trim** — manipulate **demand** by perturbing the final pickup state (the previous algorithm).
  - **Lift** — manipulate **supply** by modifying the **last ~4 states of the trajectory** *prior to* the pickup, pushing taxis toward underserved areas.
  > "instead of just optimizing that last pickup state, we also modify the... the last four trajectories prior to that pickup state... One is the trim side... And then we have the lift side or the lift phase where we manipulate those last few states in the trajectory."
- **Origin of the lift idea — credited to Dr. Cash** (about the "lower half" of trajectories — worst-offenders vs. also-improvable milder cases). *[This attribution is in the transcript but omitted from the Notion summary — see §6.]*
  > "last week we had a conversation with Dr. Cash and Dr. Cash mentioned this like lower half of the trajectories... those other trajectories that maybe weren't as unfair, but could also be improved."
- **King-moves constraint enforced:** trajectories restricted to **one-cell moves** (including diagonals), consistent with the preprocessing realism rule from the "Seagale" (cGAIL/SEA-GAIL) paper. The **old algorithm violated this** by allowing two-cell (ε=2) jumps; the new algorithm adds a final king-move enforcement step.
  > "king moves means that we can move one move in any direction... in our previous algorithm, we actually modified trajectories to move more than one space away, which violates our own pre-processing... With the new... algorithm, we also take the step of enforcing that king move in the final... phase of the algorithm."
- **Argumentative payoff:** edits are now more **realistic** and close the "you're just cutting advantaged service" reviewer criticism.
  > "doesn't give a reviewer the chance to just tee off on us and say that, hey, you're just reducing service, so that makes it more fair. So now it's a much more well-rounded argument."
- Robert framed the decision to build lift as high-risk/high-reward: "one of the hardest things I've ever done my whole life was making a decision to implement this new part of the algorithm... the risk paid off."

### 3c. Results of Trim & Lift
- **F-causal improved by "a little bit over 54%"** — described as a massive gain. *(This is the only quantified headline spoken; the absolute deltas SZ +0.0222 / SF +0.0328 were NOT said aloud.)*
  > "this improves F-causal by a little bit over 54%. So a massive gain in fairness."
- Gains **propagate to the external fairness metrics** too.
  > "the fairness improvements also propagate to those... external fairness metrics."
- **Holds on both datasets** — Shenzhen and San Francisco.
  > "we have the Shenzhen dataset and the San Francisco dataset, and the same is true for both datasets."
- **Attribution coverage grew from ~2,400 → over 7,500 trajectories**, capped at a **~10% of dataset** limit — enabling the "small fraction of data, large fairness gain" arc.
  > "the original algorithm... selected about 2,400 trajectories. We now have over 7,500 trajectories with the new attribution... it's actually at a limit that I've set, which is at about 10% of the size of the data set."
- **Ablation recorded:** trim-only vs. trim+lift.
  > "I've basically recorded everything, so we have a clear ablation between just the trim and the trim and lift."

### 3d. Pending evaluation (behavioral-cloning propagation)
- GPU rolling out trajectories for **"60 or 80 different model combinations"** across the two datasets (~40 models each) to test whether fairness propagates through trained BC models; results expected same afternoon, wrapping the prior analysis stage.
  > "the GPU is working on rolling out trajectories for our 60 or 80 different model combinations because we now have two data sets which each have like 40 or so models."

### 3e. Literature review / objective-function motivation
- Two-stage literature review to **motivate the objective function** and each fairness term / attribution / design choice; references passed **adversarial AI review**, with a **human-review gate still to come** before citing (T3).
  > "I've been going through and finding references that we can use to motivate that... these have all... gone through some rigorous adversarial AI review... I'm going to take a human review to make sure that these actually do say what they do."
- Preliminary argument drafted for each fairness term and for attribution.

### 3f. Paper writing / process
- Robert has kept a **running argument document for ~1.5 months**; every algorithm change propagates into paper content, so writing is "assembly."
  > "for about the last month and a half, I've had like a running argument... every change that I make actually propagates through to... content for the paper. So the paper writing process is going to be more of a paper assembly process."
- Zhang directs starting with **methodology** (T5); abstract as placeholder OK (PI decision #3); Robert wants the abstract to Zhang early (T6) to leave review time.

### 3g. Timeline
- **Abstract deadline ≈ one week before the paper deadline**; Robert targets an abstract "by next week."
  > "I know the abstract deadline is like a week before the paper deadline."

---

## 4. Numbers Spoken (verbatim, as said aloud)

- External metrics: **three** — disparate impact, demographic parity gap, Theil ("TEAL") index.
- **F-causal improvement: "a little bit over 54%"** (only quantified result spoken).
- Grid / neighborhood: **5×5 grid**; ε = **2** (perturb "at most two grid spaces away") in the OLD algorithm; king move = **one** cell (new constraint).
- Lift edits: **last ~4 trajectory states** prior to pickup ("the last four trajectories prior to that pickup state").
- Supply time unit: **one hour**.
- Attribution coverage: **~2,400 → over 7,500 trajectories**.
- Edit cap: **~10% of the dataset size**.
- BC eval: **"60 or 80 different model combinations"**, **two datasets**, **~40 models each**.
- Datasets: **Shenzhen** and **San Francisco**.
- Running-argument doc maintained: **~1.5 months** ("month and a half").
- Abstract deadline: **~1 week before** the paper deadline; abstract target: **next week**.

**Explicitly NOT spoken aloud (despite being in project notes):** the absolute headline deltas SZ +0.0222 / SF +0.0328; F_spatial numbers; α-weight / Pareto sweep; any per-arm baseline names (ST-iFGSM / FGSM / random / demographic-oversampling); "rollout-allocation drain"; "demand endogeneity"; F_demo renaming. None of these terms appear in the transcript.

---

## 5. Open Questions Left Unresolved at Meeting End

1. **Does fairness propagate through the trained BC models under trim+lift?** — GPU still running; result pending "this afternoon" (T1). Left open.
2. **Data-augmentation baselines** — acknowledged as still-owed work (T2); not yet designed/run in this meeting.
3. **Human verification of AI-found citations** — flagged as a required gate not yet performed (T3).
4. **Which sections to draft first / paper structure** — Zhang said methodology; introduction deferred ("not necessarily introduction, but the methodology part"). Section plan otherwise open.
5. No discussion of α-weight sensitivity, F_causal naming, or the specific supply-lift absolute-delta reporting — these remained untouched (not raised, so neither resolved nor rejected).

---

## 6. Discrepancies & Omissions vs the Notion AI Summary

The Meeting-42 summary is **substantially more faithful than Meeting-41's** (no invented framework). But it has real errors and omissions:

1. **[OMISSION — substantive] Dr. Cash's contribution is erased.** The transcript explicitly credits the lift idea to a conversation with **Dr. Cash** ("Dr. Cash mentioned this like lower half of the trajectories... those other trajectories that... could also be improved"). The summary presents the trim/lift motivation with **no mention of Dr. Cash** at all. This drops the actual intellectual provenance of the algorithm change.

2. **[FACTUAL — checkbox states wrong] Two action items are marked done `[x]` that the transcript shows were NOT done at meeting time:**
   - "Robert to implement data augmentation baselines" is checked `[x]`, but Robert said it is future work: "those things are still on my list and I definitely intend to do them" / "I'm going to be investigating that." → should be open.
   - "Wait for GPU results on fairness propagation... (expected same afternoon)" is checked `[x]`, but at meeting time it was "currently running on the GPU. It should land sometime this afternoon." → not complete at the meeting.
   (These may have been auto-checked post-hoc, but as a record of the meeting they misstate status.)

3. **[NUMBER — false precision] "~80 (estimated) model combinations."** Robert actually said **"60 or 80"** ("rolling out trajectories for our 60 or 80 different model combinations"). The summary collapses the spoken range to a single upper figure.

4. **[OMISSION] The king-moves realism rule's source paper is dropped.** Transcript ties king moves to the "**Seagale**" (cGAIL/SEA-GAIL) imitation-learning paper's preprocessing; the summary states the constraint but omits that it derives from that paper's convention.

5. **[OMISSION — minor] The ">54%" is real but the summary drops the hedge.** Transcript: "a little bit over 54%"; summary: ">54%." Directionally fine, but note the number is an approximation Robert voiced, not a precise reported figure, and the summary presents no absolute deltas (correctly — none were spoken).

6. **[TONE/FRAMING — minor] The summary omits the risk framing** that Robert emphasized ("one of the hardest things I've ever done my whole life was making a decision to implement this new part of the algorithm... the risk paid off"). Not load-bearing for the plan, but it's the meeting's emotional/decision core and explains why lift was non-obvious.

**No fabrications detected** in the Meeting-42 summary (unlike Meeting 41). The metrics named (disparate impact / demographic parity gap / Theil), the two-phase trim/lift description, the 2,400→7,500 / 10% figures, the Gaussian-softmax-on-supply mechanism, and the abstract-as-placeholder note all match the transcript.
