# Analysis C — terminology, claims, and misrepresentation risk (2026-07-24 meeting)

Lane: (a) the exact language Dr. Zhang wants, (b) every place her mental model — or
Robert's own conversational shorthand — diverges from what FATE actually does.
Ground truth: `ALGORITHM_FACTS.md` (cited as **AF**). Transcript:
`transcript_readable.txt` (all 386 lines read). Sibling reports cover
decisions/architecture (A) and action items/figures (B).

Two framing facts that govern everything below:

1. **[36:50] Zhang scopes her language mandate to the Introduction.** "I think it's
   mainly about the introduction, the style of the introduction. And when it comes to
   methodology, it's gonna be **whatever makes sense for us**. But when it's
   introduction, you don't want to lose audience because they don't understand the
   language." → the new vocabulary is binding in title/abstract/intro (and Fig-1/Fig-2
   captions, which reviewers read like intro text). In §2–§3 Robert picks the precise
   technical terms.
2. **[37:48] and [57:24] confirm the prime directive.** "we can just submit whatever we
   have for now… it's okay that the current version is not perfect"; "it's okay, **as
   long as the information in that figure are preserved**." She is optimizing for a
   legible story, not for specific sentences. Nothing in 62 minutes asks for a claim to
   be strengthened beyond the evidence.

---

## §1 Endorsed vocabulary

| Term / phrase | Timestamp | Verbatim source | Replaces / coexists | Notes on accuracy |
|---|---|---|---|---|
| **budget-aware trajectory editing** (also "**budgeted** … trajectory editing") | [04:12] | "we are doing a budgeted like um, or budget aware trajectory editing" | **REPLACES** "fairness-versus-fidelity trade-off" as the spine (currently declared the central problem at `01_introduction.tex:86`). Coexists with the title, which Zhang already fixed. | Accurate: `k` is a real, configured edit budget (SZ k=10,000; SF k=2,000; AF §Budget). **Hazard:** the name invites a budget *analysis*; there is no k-sweep (AF: "There is NO edit-budget (k) sweep"). See D16. |
| **collective fairness** | [05:09], [06:02] | "we are looking at a systematic thing like um collective um fairness" | **NEW** — "collective" appears nowhere in the current paper. Coexists with `F_demo`, "corpus-level". | Accurate and load-bearing: AF "fairness is collective (corpus-level), measured over the aggregate service allocation". Define on first use. |
| **global fairness** (of the whole dataset) | [06:02] | "what the global fairness looks like when we are editing those trajectories"; "how can we modify a small portion of the trajectories to affect the global fairness" | **NEW**; a synonym for "collective fairness" in her usage. | Accurate if "global" = corpus-level / city-wide aggregate. Pick ONE of collective/global as the paper's term (recommend **collective**, with "corpus-level" as the technical gloss) — using both interchangeably invites a reviewer to ask what the difference is. |
| **the whole dataset** (unit of analysis) | [06:02] | "When we are looking the whole dataset, instead of like designing a more kind of fair model" | Coexists. | Accurate. This is the FairGAN/in-processing contrast axis. |
| **a different fairness perspective or metrics** | [06:02] | verbatim | Coexists. | 🟡 Careful: we do NOT invent a new fairness notion. We measure a demand-adjusted dependence (F_demo) *and* report standard instruments (DP, DI, Theil). Wording must be "a different level at which fairness is measured", not "a different definition of fairness". |
| **the most distinctive part that would differentiate our approach** | [06:55] | verbatim | Positioning directive. | See C-3. |
| **systematic** (as opposed to per-instance) | [05:09] | "a systematic thing" | Coexists with "collective". | Fine as an adjective; do not let it become "systemic bias" (a different, causal-sounding claim). |
| **modify a (small) portion of the data** / "we don't want to modify too many data" | [04:12], [05:09], [06:02] | verbatim | **NEW** framing of the budget. | Accurate: 9,882 applied edits (2,337 net trim + 7,545 lift) of ≈95,297 SZ trajectories ≈ 10%. Use "about a tenth of the corpus" — already the intro's internal comment (`01_introduction.tex:53`). |
| **sparse edits** (her email §3.3 title) | — (email only; meeting silent) | "Preserving the Influence of Sparse Edits via Edit-Aware Weighting" | Coexists. | Accurate and useful: names exactly why the downstream stage exists. |
| **edit-aware weighting** (her email; meeting: "editing and upweighting") | [03:38–04:12] | "it's a combination combined strategy in terms of the. Editing and upweighting" | **REPLACES** the current §3.4 label "downstream recipe" if desired. | Accurate (Kamiran–Calders instance reweighing on demonstrations). |
| **framework figure / solution framework / sequence diagram** | [12:55], [23:36], [52:07] | "for the figure two, I'm looking into … a more framework or or procedure like thing"; "telling people, What is the input of this algorithm? What are the different stages … and what is the output? So that is a framework figure" | **REPLACES** the current 3-panel stylized-city Figure 2. | Terminology only. Claim constraints in D12. |
| **case study** ("what a real editing looks like") | [52:33], [52:57] | "We show a case study of what a real editing looks like" | **NEW**, conditional ("if we have time"). | 🔴 Only usable over a REAL edit drawn from artifacts. See C-8/D12. |
| **attribution / trim / lift** (the three step names) | [13:10], [20:24] | Robert names them; Zhang repeats "we do attribution … we are using the trim and the lift" | **SURVIVES** unchanged. | Zhang adopted our vocabulary here — keep the names exactly. |
| **"attribute / locate the deficit"** | [14:02], [14:20] | "The attribute locate the deficit, like what what makes the location happen?" | Coexists (current Fig-2 panel-1 label). | Accurate at unit level; see D3/D13 for the trajectory-level slip. |
| **challenges C1, C2, C3 …** itemized in the Overview, mentioned briefly in the Intro | [31:00–32:41] | "in the introduction, we can like briefly tell what are the challenges without kind of itemizing them, and we can itemize them over here"; "due to limited space, we don't need to kind of put them in this itemized LaTeX environment. We can just stack them all together" | Structure change; C-labels survive. | She said "C one, C two, C three" — illustrative, not a mandate to cut C1–C5 to three. Each challenge maps to one component ([30:07], [33:20]). |
| ~~**surrogate**~~ / "collective fairness surrogate" | **[36:28–36:45]** | Robert: "Fairness surrogate, just to make sure we're on the same page." Zhang: "**It's a fairness objective** that one you're referring to. **I'm just using my language.**" Robert: "do you want that language included?" Zhang: "**Not necessarily** like you can just use whatever language that makes sense." | **RELEASED** — her email's §3.1 title word is explicitly NOT required. "Fairness objective" (our current `sec:objective`) is approved by her own restatement. | See §4.1. If "surrogate" is kept anywhere, define it as a differentiable corpus-level proxy the editor can optimize — never as a surrogate for a causal quantity. |
| **HSTD**, **outcome-side**, **resource-aware/resource-side**, **"attribution is the budget-allocation mechanism, not post-hoc explanation"** | *never spoken in the meeting* | — | Email-only; coexist with our terms; section names explicitly non-binding ([29:26] "It doesn't have to be exact section name"; [32:41]; [33:12]). | "outcome-side"=trim and "resource-aware"=lift are accurate paraphrases (AF §Editor). **HSTD** is an unverified acronym (Zhang's PDFs are not yet in `restructure/zhang/`): if adopted, define on first use and make sure it names exactly our object — a corpus `T` of per-driver passenger-seeking trajectories, with D, S, Y as deterministic aggregations. Do not introduce an acronym implying a data model we do not have. |
| **"Collective service disparity emerges from the aggregation"** (her Fig-1 headline) | [10:42], [11:19], [11:21] | Robert: "your edit introduces a new like. Collective service disparity emerges from the aggregation. Do you want this figure one to be used?" Zhang: "**I think that is more clear**." Robert: "Me too." | **REPLACES** the current teaser headline. | Accurate: fairness is defined over the aggregate allocation, and trim conserves totals. Do not let it imply individual trajectories are individually unbiased — it is a statement about where fairness is *measured*. |

**Terms Zhang used that must NOT migrate into FATE's description:** "attacks",
"perturbations", "perturbed dataset" ([22:45–23:18], describing ST-iFGSM's own figure —
one of our *baselines*); "tricks" ([20:24], for implementation detail). See D10, D11.

---

## §2 Claim-by-claim analysis

### C-1 (THE RISKIEST). "Small corrections won't much change raw-data fairness but will be impactful downstream"

**Exchange, verbatim.**
> **[04:12] Zhang:** "What I have changed in terms of story is that we are doing a
> budgeted like um, or budget aware trajectory editing, where the claim is that we don't
> want to. Modifies too many data. Instead, We want to modify a portion of the data. And
> **even though those portion of data won't have make a great impact on the raw data
> fairness itself, it is going to be impactful for the downstream models.** And in a
> sense like only modifying or."
> **[05:09] Zhang:** "Correcting a small portion of data won't have that like a like an
> impact. **We just need to update it so that it's gonna make sense.**"

Note "we just need to update **it**" = update the *story/writing*, not run new
experiments. This is a framing request, and it is the one framing request that is
factually wrong in both halves.

**What is actually true (AF §Headline results, §Downstream stage).**
- Raw/data-level fairness **does** move, measurably: SZ F_demo 0.7988→0.8214 (+0.0226,
  i.e. demographic dependence 20.1%→17.9%, an 11% relative reduction); SF
  0.8752→0.9067 (+0.0316). More importantly for claim discipline, every *external*
  instrument — the class-(iii) ring the objective never optimizes — moves toward
  fairness with **bootstrap CIs excluding zero**: DI +0.0162 [+0.0136,+0.0189], DP gap
  −0.890 [−0.992,−0.785], Theil −0.0087 [−0.0097,−0.0076]; and the gap closes from both
  ends. Saying the raw data barely changes **discards our own strongest evidence**.
- The downstream half is not automatic. Uniform-weight BC on the edited corpus is
  **null**: ΔF_demo +0.0016, n=12, 7/12 positive, p=0.11 (`04_experiments.tex:207`).
  Only with edit-aware upweighting does it move: +0.0217 (w10), +0.0267 (w20),
  **+0.0297 ± 0.0029 (w30, adopted)**, 12/12 positive, exact Wilcoxon p=.00049, in both
  cities (SF w30 +0.0333).
- Magnitudes: policy-level +0.0297 vs corpus-level +0.0226 — **comparable, not
  dramatically larger**. There is no evidence for "small data change → large model
  change". The defensible statement is that the corpus-level gain is *not diluted* by
  training once the edits are upweighted.

**Severity if written as she said it:** WOULD-MISREPRESENT twice over (D1: understates
data-level and contradicts our abstract's own "improves metrics it never optimizes"
claim; D2: implies editing alone propagates, which our own null refutes).

**Her intent, which is entirely salvageable.** She wants: *(i)* the budget to be the
story, *(ii)* the downstream model to be where the payoff lands, *(iii)* an explanation of
why editing a *small* portion is a hard problem worth a paper. All three are true. The
correct causal chain is the **reverse** of hers: the portion is small **which is why**
naive training averages it away, **which is why** edit-aware weighting is part of the
method. That reversal makes the two-stage design necessary rather than bolted-on — exactly
the gap she opened the meeting with ([03:38] "difficult to justify that we are only
proposing trajectory [editing] because it's a combination combined strategy").

**Proposed accurate wordings.**

*Abstract / intro spine (replaces the "fairness-versus-fidelity … is the central
problem" sentence at `01_introduction.tex:86`):*
> FATE works under an explicit edit budget: it changes about a tenth of the recorded
> trajectories and leaves the rest exactly as recorded. Spending that budget where an
> exact attribution of the fairness deficit says it counts most makes the service
> allocation the corpus induces measurably fairer on established measures the objective
> never optimizes, and upweighting the edited demonstrations carries that improvement
> into the trained policy. Training on the edited corpus at uniform weight does not:
> a tenth of the corpus is too small a share of the training signal to survive
> averaging, and that is why the weighting stage is part of the method.

*One-sentence version for the contributions list:*
> We show that a budgeted edit to about a tenth of a real trajectory corpus is enough to
> shift the corpus's collective fairness on measures the editor never optimizes, and
> that the shift reaches the trained policy only when the edited demonstrations are
> upweighted.

*If a sentence in her voice is wanted (keeps "small portion", stays true):*
> Correcting a small portion of the data changes the corpus's measured fairness by a
> modest but statistically reliable margin; the same small portion has no effect on a
> policy trained at uniform weight, and a decisive one once the edits are weighted for
> what they are.

*Never write:* "the edits barely change data fairness", "editing a small portion has
little effect on the data but a large effect on models", or any sentence implying the
data-level effect is negligible or the downstream effect follows from editing alone.

---

### C-2. "Trajectory editing alone is hard to justify — the method is a combined strategy"

> **[03:38] Zhang:** "For the current story … It's kind of a little bit difficult to
> justify to justify that. Um we are only proposing trajectory [editing] because it's a
> combination combined strategy in terms of the. **[04:12]** Editing and upweighting, So
> we kind of need kind of a more complete and more tailored story."

**ACCURATE — adopt as-is.** AF: FATE = bounded editing + edit-aware upweighting; the
abstract already says both. This is a diagnosis of a *presentation* gap, not of the
algorithm. Consequence: the FATE-overview paragraph must present **two stages** as one
method (her E3), and no sentence may describe the contribution as "trajectory editing"
full stop.

*Proposed:* "FATE has two stages: a budgeted, constrained edit to the recorded corpus,
and an edit-aware weighting of the edited demonstrations during imitation. Neither stage
is sufficient alone — the edit is what makes the corpus fairer, and the weighting is what
keeps that fairness from being averaged away in training."

---

### C-3. Differentiation from FairGAN and model-side fairness, via collective fairness

> **[05:09] Zhang:** "your original novelty claim is kind of the trade off between um
> editing and the fidelity. I think that is a good point, but I think a more important
> problem … is distinguishing our problem from other approaches, like the **fair GAN** or
> or other related works, is that we are looking at a systematic thing like um collective
> um fairness. **[06:02]** When we are looking the whole dataset, instead of like
> designing a more kind of fair model that won't discriminate, like for example, CV model
> that is gonna discriminate people of color … So we are kind of a different fairness
> perspective or metrics. **[06:55]** that is the most distinctive part that would
> differentiate our approach with others."

**Largely ACCURATE and already supported** by `01_introduction.tex:89–108` (in-processing
leaves the corpus biased; generation pays fidelity; reweighing/repair cannot create
service where the fleet never went). Two wording constraints:

1. **"different fairness perspective or metrics"** → say "a different *level* at which
   fairness is measured and repaired: the service allocation the whole corpus induces,
   rather than one model's decisions". We do not propose a new fairness *definition*; we
   report DP/DI/Theil precisely so a reviewer can check us against the standard ones.
2. **The CV / "people of color" analogy is a conversational illustration only.** Do not
   transplant it. Our protected signal is a **district-level** demographic profile
   (~10 districts) with a mandatory associational + ecological-fallacy caveat (AF
   §Setting). An individual-level protected-class analogy in the intro would set up
   exactly the misreading the caveat exists to prevent (D17).

*Proposed differentiation sentence:*
> Model-side methods make one model behave more fairly while the demonstrations stay as
> recorded; generative methods make a fairer dataset that is no longer a record of human
> driving. FATE instead asks what the *existing* corpus makes collectively true about
> service across the city, and changes a bounded portion of it so that the aggregate
> allocation is less predictable from demographics — leaving every trajectory a real,
> recorded one.

---

### C-4. "The unique challenge: modify a small portion of trajectories to affect global fairness"

> **[06:02] Zhang:** "that naturally create the unique challenge that we have been trying
> to solve, Which is how can we modify a small portion of the trajectories to affect the
> global fairness."

**ACCURATE as a challenge statement** (unlike C-1's answer to it). This is the honest
headline for the new spine and maps cleanly onto components: attribution answers *which*
small portion; the budget k answers *how small*; ε/validity/fidelity answer *how much per
edit*; edit-aware weighting answers *how it survives training*.

**One hazard:** it must not slide into "we characterize how fairness varies with the
budget". No k-sweep exists (D16). Frame k as a *configured* budget, with the
budget-adjacent evidence we do have: the trim-vs-trim+lift composition ablation, the
upweighting dose-response (w10–w50), the oversampling dose (d2.5k–d10k), the α sweep, and
the oracle ceiling. A k-sweep is future work.

---

### C-5. Status of the editing–fidelity trade-off contribution

> **[05:09] Zhang:** "your original novelty claim is kind of the trade off between um
> editing and the fidelity. **I think that is a good point**, but I think a more important
> problem…"

**Retained, demoted — not deleted.** Nothing in the meeting asks for the fidelity material
to be removed, and [57:24] ("as long as the information … are preserved") plus the
protected register require it to survive. Under the new spine the trade-off becomes a
*constraint on the budget* rather than the spine: each edit is bounded (ε = 2 cells),
scored by a frozen identity discriminator at every iteration, and the corpus-level
distributional cost is disclosed (Fidelity-A 0.844 vs raw 0.848; **Fidelity-B 0.187**).

*Proposed framing sentence:* "Because the edits must remain records of the same drivers'
behavior, the budget is spent under constraints rather than freely: every edit stays
within two grid cells of its recorded location, a frozen driver-identity discriminator
scores the edited tail in the objective at every iteration, and we report the
distributional cost this incurs rather than netting it out."

*Do not claim* we "solve" or "resolve" the fairness–fidelity trade-off. We bound and
disclose it.

---

### C-6. "Attribution is scoring each trajectory('s contribution to unfairness)"

> **[16:50] Robert:** "the attribution itself takes the fairness metric, and then it
> determines. **What each trajectory's influence is on that fairness metric.** And then
> we basically normalize that or **z-score normalize** that … those individual scores and
> we get a positive and negative. **And positive are considered overserved, negative are
> considered underserved.**"
> **[17:28] Zhang:** "So essentially what are you attributing? So you are scoring, **is
> attribution essentially scoring each trajectory?** [Robert: Yes.] So in that sense,
> what you should show is that you have a bunch of trajectories. And from this bunch of
> trajectories, you are able to tell, okay, **this trajectory is contributing to
> unfairness. This is contributing less to unfairness.**"

Three separate problems (D3, D4, D5, D6). What is actually true (AF §Editor; App A
Eq. 4–5; `PAPER/argument/03_fairness_theory.md`):
- The exact partition is over **active units** (cell,hour), not trajectories:
  r²_demo = Σ_i [(MR)_i² − ((I−H)R)_i²] / RᵀMR. Trajectories are then selected because
  their **pickups land in the highest-deficit units**.
- The only normalization is that denominator: each unit carries a *share* of the total
  deficit and the shares sum to the whole. **There is no z-scoring of attribution scores
  anywhere** in the code or the paper. z-scoring appears in two unrelated places: the
  demographic design matrix X (App A) and the 11-dim driver profile features for the
  identity model. The over/under-served split comes from the **signed variant, which
  multiplies by sign((HR)_i)**.
- There are **two** attribution mechanisms, one per phase. Lift's score is
  v_i = ∂L/∂S_i (a signed value-of-presence map, explicitly *not* bounded to [0,1]),
  turned into trajectory candidates by a linearized-offset screen that ranks each
  trajectory by its **best bounded tail translation** and only **nominates**.
- Sign semantics therefore differ between phases: on the trim side a positive signed
  deficit marks an **over-served** unit; on the lift side a **positive screen score**
  marks a trajectory worth editing *because it can help the under-served* ("lift fills
  the remaining budget with positive-score nominees", `03_methodology.tex:355`).
  Publishing Robert's meeting gloss verbatim would put two contradictory sign conventions
  in one paper.
- "This trajectory is contributing to unfairness" is loosely defensible for trim (its
  pickups sit in over-served, high-deficit units) and **wrong for lift**, where the score
  is improvement potential, not blame.

**Proposed accurate wording (main text, and the replacement Fig-2 caption):**
> Attribution scores where the fairness deficit lives and which trajectories can move it.
> The demographic term admits an exact decomposition across active units: each
> (cell, hour) carries a defined share of the deficit, the shares sum to the whole, and a
> signed variant separates over- from under-served units. Trim takes the trajectories
> whose recorded pickups land in the highest-deficit units. Lift asks a different
> question of every unit — how much fairer the city's service would be with marginally
> more taxi presence there — and scores each candidate trajectory by the best bounded
> reroute of its final seeking states into high-value units, keeping only those whose
> score is positive. In both phases a score answers "how much fairness would an
> admissible bounded edit here buy", not "how unfair is this trajectory".

**Caption-length version:** "Attribution ranks candidate trajectories by the fairness gain
a bounded, admissible edit to them would produce, and selects the top ones until the
budget is spent."

---

### C-7. Zhang's skeleton: "select 100 of 1000, then edit them with trim and lift"

> **[20:24] Zhang:** "the major skeleton of this approach is that what we do is we do
> attribution, right? We so from let's say 1000 trajectories, we are selecting Like 10
> out of them or 100 out of them. 100 trajectories we have selected out of them. And with
> these 100 trajectories, we are doing the editing, right? And we are in terms of doing
> the editing, we are using the trim and the lift to to edit the trajectories. Okay. **And
> did I miss anything?**" — **[21:22] Robert: "No, I think that yeah, I hear you."**

The **ratio is right** (≈10% of the corpus; k=10,000 of ≈95k). The **flow is not**: this
describes one ranked selection feeding both editors. Reality (AF §Phase order): trim takes
its own selection by demand-deficit attribution (2,455, of which 118 revert); **then** the
supply gradient is computed on the **post-trim** state and lift fills the remaining budget
(7,545) with positive-score nominees; lift never alters trim's edits, which is what makes
trim identical to the demand-only editor and the trim-vs-trim+lift ablation a clean
isolation of lift.

Robert answered "did I miss anything?" with agreement — fine for the *figure abstraction*.
But the framework figure and the FATE-overview paragraph must not encode
one-selection-then-both-editors, or the paper loses its own scientific control.

*Proposed (accurate and still simple enough for the framework figure):*
> Stage 1 — Attribute & Trim: score units by their share of the fairness deficit; relocate
> the pickups of the highest-deficit trajectories within a two-cell bound.
> Stage 2 — Re-score & Lift: recompute the value of added taxi presence on the edited
> corpus; reroute the final seeking states of the best-scoring remaining trajectories into
> under-served areas until the budget is spent.
> Stage 3 — Weight: upweight the edited demonstrations during imitation.

Three labeled bands, no more complex than her ST-iFGSM template, and it does not lie about
the order.

---

### C-8. "A case study of what a real editing looks like"

> **[52:23] Zhang:** "looking at this figure, so initially, I was thinking that this is a
> figure for case study … **[52:33]** in experiment, we usually … give people a more
> concrete idea about the editing results. **We show a case study of what a real editing
> looks like.** So in this way, I thought this one is a case study." — **[52:54] Robert:
> "It kind of is in that regard because Yes"** — **[52:57] Zhang:** "if you have time,
> it's good to have a case study telling people, What a successful editing looks like if
> we, no time is totally okay."

Zhang **mistook the stylized three-panel Figure 2 for a case study**, and Robert's "it kind
of is" half-conceded it. Figure 2 is schematic ("FATE on a stylized city",
`03_methodology.tex:225`) — it depicts no real trajectory. Captioning or describing a
schematic as a case study of a real edit is a straightforward misrepresentation, and it is
the kind reviewers check.

**Two acceptable outcomes:** (a) build a real case study from actual artifacts (a real
edited trajectory: before/after geometry, the unit's deficit share, the resulting ΔS/ΔY)
and label it as one; or (b) skip it — she explicitly said "no time is totally okay". Do not
relabel the schematic. Whatever the new Figure 2 becomes, its caption must say
"schematic"/"illustrative" if no real data is plotted.

---

### C-9 / C-10. Transferability and cross-city parity

> **[44:03] Zhang:** "finally, You are saying making the claim that this approach is also
> **transferable to San Francisco. For other or or to taxi data in other cities**, and
> then you are listing the San Francisco result … **[44:52]** I personally would prefer
> this way of writing because it's gonna save us some space."
> **[45:31] Robert:** "in the fairness and the propagation that we have uh basically
> **parity** between San Francisco and Shenzhen. But when it comes to the baselines …
> much of the baselines have only been implemented … against Shenzhen, because
> computational … issues."
> **[46:31] Zhang:** "if we observe, we have the same observation in Shenzhen and San
> Francisco, we can either combine it … And we don't need to kind of say the same thing
> for two times."

Two wording constraints:
- **"transferable to other cities" is an n=2 generalization.** House rule already: SF
  "reproduces", never "beats" — extend it: "the protocol reproduces on a second city with
  different geography, sampling, and fleet size", not "generalizes to cities".
- **"Parity" must be scoped.** True for direction and significance of the data-level
  fairness deltas and the downstream dose-response (SF w30 +0.0333, p=.00049). NOT true for
  magnitudes or supply accounting: SF supply tier-2 +0.1027 vs SZ +0.0411, and **SF total
  under tier-1 accounting is net-negative (−0.0324)**, read as demand endogeneity in the
  wild. Every single supply number states its accounting tier (AF). Robert defended keeping
  the differences visible at [47:09] ("slight differences … as far as defensibility need to
  be at least recognized") and [47:46] ("a very astute reviewer might look at some of the
  tables and see some numbers that lead to questions"); Zhang did not object — her
  instruction was only *don't say the same thing twice*. So: merge the redundant prose, keep
  the divergence disclosure.

---

### C-11. Borrowing the ST-iFGSM framework figure's vocabulary

> **[22:45–23:18] Zhang:** "the input is a training data … from k drivers … their stage
> one, which is **adversarial attacks** to human models. So given the raw data, they are
> putting on **attacks which are perturbations**. To generate a **perturbed dataset**."

Accurate about *their* paper; ST-iFGSM is one of **our baselines**. FATE reuses bounded
signed-gradient tooling **constructively** (`03_methodology.tex:264–270`,
algorithmic-recourse framing) and its output is an edited corpus of real trajectories,
repaired for physical validity. Do not let "attack", "perturbation", or "perturbed
dataset" describe FATE's stages when the figure style is borrowed. Also note the symbol
clash: their **k = number of drivers**; our **k = edit budget** (her outline uses K for the
budget and ε for the per-edit bound) — pick one convention and state it once.

### C-12. "Complicated implementation details can be some tricks … put into the appendix"

> **[19:28] Zhang:** "you are like thinking it in a very complicated way … One way is that
> we stick to what our implementation looks like and reflect our implementation … but when
> we are explaining to people what we have done step by step, it's gonna be complicated to
> understand … Instead, We want to abstract our approach in a way that we can put it into a
> more simple story. **[20:24]** Some very complicated implementation details can be
> **some tricks that we have used to make this approach work better**. And that part could
> be put into the appendix."

**Accept the instruction, reject the label.** Abstraction is legitimate: it may *omit*, it
may not *assert something false* — and Robert already conceded the visual point ([21:22]
"We don't necessarily need to perfectly replicate the algorithm in visuals"). But several
items she would class as "tricks" are correctness conditions, not performance tuning: the
ε = 2-cell cumulative clip (what makes an edit an edit rather than a fabrication), the
exact backward-reachability **king-move repair** and the skip/revert of infeasible edits
(what makes the edited corpus physically valid), endogenous supply during lift (what makes
the lift claim meaningful), and the fidelity term in L. The main text may compress them
into one clause each; the appendix carries the detail (her [40:30]: "we want to make sure
that in the appendix, we have enough detail about this implementation"). Main text must not
call them optional tricks, and the protected register (associational caveat,
demand-endogeneity bound, leveling-down analogy, accounting tiers, Fidelity-B, p=0.031
discipline) must survive somewhere.

### C-13. What the Figure-1/attribution panel asserts

> **[14:20] Robert:** "Locating the deficit is basically … what I would be hoping the
> audience would see is looking at the grid. We see advantaged district overserved,
> disadvantaged district underserved." **[16:36] Zhang:** "the attribution you mentioned is
> actually that we are able to identify that this region is orange. And this region is
> blue."

Accurate *if* the panel shows attribution over **units** (colored by deficit share) and the
advantaged/disadvantaged reading is presented as the demographic interpretation. Constraint
from AF §Why-demand-only: all 2,455 SZ trim pickups originated in **advantaged** cells and
none landed in a disadvantaged cell. A figure that shows trim editing disadvantaged-district
trajectories, or demand being moved *into* the disadvantaged district, contradicts our own
empirical section. Lift is the only mechanism that touches the under-served side, and it
does so by adding presence.

### C-14. Reproducibility / availability statements

> **[40:30] Zhang:** "we don't need to worry too much about the reproducibility for now
> because we still have time to modify the code."

No claim risk in the meeting itself, but it intersects E14: her draft's intro says code
**and dataset** are available via an anonymous link. Any availability sentence must describe
what the anonymous repo actually contains on Sunday. Do not claim a dataset release that has
not passed the PII/anonymity pass.

---

## §3 Divergence catalog

Severity key: **WM** = WOULD-MISREPRESENT if written into the paper · **HS** =
HARMLESS-SHORTHAND in conversation (with the in-text guard noted).

| # | Time | Who | Quote (short) | What is actually true (AF) | Severity |
|---|---|---|---|---|---|
| D1 | 04:12, 05:09 | Zhang | "those portion of data won't have make a great impact on the raw data fairness itself" | Data-level F_demo +0.0226 SZ / +0.0316 SF; all class-(iii) external instruments move with bootstrap CIs excluding zero (DI +0.0162, DP gap −0.890, Theil −0.0087); gap closes from both ends. Modest in magnitude, not negligible, and it is our headline evidence. | **WM** |
| D2 | 04:12 | Zhang | "it is going to be impactful for the downstream models" (as a consequence of editing) | Uniform-weight BC on the edited corpus is NULL (+0.0016, n=12, 7/12, p=0.11). The downstream gain requires edit-aware upweighting (+0.0297±0.0029 at w30, 12/12, p=.00049). And it is *comparable to*, not larger than, the corpus-level gain. | **WM** |
| D3 | 16:50 | Robert | "attribution … determines what each trajectory's influence is on that fairness metric" | The exact partition is over **active units** (App A Eq. 4); trajectories are selected because their pickups land in high-deficit units. Same looseness already exists in the current Fig-2 caption (`03_methodology.tex:230`, `:240`) — fix it in the replacement. | HS in talk / **WM** in text |
| D4 | 16:50 | Robert | "we basically normalize that or **z-score normalize** that … those individual scores" | No z-scoring of attribution scores exists in the paper or the code (verified: z-scoring hits are the demographic design matrix X and the 11-dim driver profiles only). Normalization = division by RᵀMR (shares of the deficit summing to the whole). Dangerous because Plaud flagged "describe normalization/scoring in the main text". | **WM** |
| D5 | 16:50 | Robert | "positive are considered overserved, negative are considered underserved" | Over/under-served separation comes from the signed deficit variant, sign((HR)_i) — trim side only. On the lift side, positive screen scores mark trajectories worth editing *to help the under-served*. Two different sign semantics; stating only Robert's gloss creates an internal contradiction with §3.4. | **WM** |
| D6 | 17:28 | Zhang | "this trajectory is contributing to unfairness. This is contributing less to unfairness" | Defensible for trim; wrong for lift, where the score is improvement potential, not blame. Unified accurate frame: "how much fairness an admissible bounded edit here would buy". | HS for the trim panel / **WM** if generalized |
| D7 | 20:24 | Zhang (unchallenged) | "from 1000 trajectories we are selecting 100 … and with these 100 trajectories we are doing the editing … using the trim and the lift" | Per-phase selection: trim's deficit selection (2,455; 118 reverted) → supply gradient recomputed on the **post-trim** state → lift fills the remainder (7,545). Lift never alters trim's edits (the scientific control). Ratio ≈10% is correct. | **WM** |
| D8 | 18:30 | Robert | "the districts really play a big role in what makes **a trajectory** fair or not" | Fairness is collective; there is no per-trajectory fairness quantity (AF "Things FATE is NOT"). Districts matter because F_demo regresses the demand-adjusted residual on district-level demographics. | HS (never write per-trajectory fairness) |
| D9 | 13:10 | Robert | "it works in three steps … But we can still kind of **merge those together**" | Merging trim+lift **visually** is fine. Merging them **in prose** destroys the two-phase control (trim identical to the demand-only editor; trim-vs-trim+lift isolates lift) and the fact that lift's gradient is computed post-trim. | HS for the figure / **WM** in prose |
| D10 | 22:45–23:18 | Zhang | "attacks which are perturbations … to generate a perturbed dataset" | Accurate about ST-iFGSM (our baseline). FATE reuses the tooling constructively; its output is real trajectories, validity-repaired. Also k-symbol clash (their k = drivers; our k = budget). | HS about their paper / **WM** if applied to FATE |
| D11 | 20:24 | Zhang | complicated details are "tricks that we have used to make this approach work better" | ε-clip, king-move repair + skip/revert, endogenous supply, and the fidelity term are correctness conditions, not tuning tricks. Appendix placement is fine; "trick" framing in the main text is not. | HS as editorial instruction / **WM** if the text adopts the label |
| D12 | 52:23–52:54 | Zhang, then Robert's "it kind of is" | the stylized figure "is a figure for case study"; "a case study of what a real editing looks like" | Current Figure 2 is schematic ("FATE on a stylized city") and plots no real trajectory. Either build a real case study from artifacts or do not use the term. | **WM** |
| D13 | 44:03 | Zhang | "this approach is also transferable to San Francisco. For other or to taxi data in other cities" | n=2 cities. House rule: SF "reproduces". Claim external validity, not generalization. | **WM** |
| D14 | 45:31 | Robert | "basically **parity** between San Francisco and Shenzhen" | Parity holds for direction/significance of fairness deltas and downstream dose-response; NOT for magnitudes or supply accounting (SF tier-2 +0.1027 vs SZ +0.0411; SF tier-1 total −0.0324). Baselines are SZ-only for compute reasons — as Robert said in the same breath. | HS (scope it in text; label accounting tiers) |
| D15 | pre-existing text activated by her §3.2.3 outline | paper | abstract: a discriminator "**verifies** that an edited trajectory still resembles the original driver"; C2: "bounded and **gated by** a frozen driver-identity discriminator"; contribution: "fidelity **enforced by**" | AF §Validity: **fidelity is NOT a per-edit accept/reject threshold.** It is (a) a weighted term in L every iteration and (b) an evaluation-time corpus gate (Fidelity-A). Accept/skip/revert is driven by king-move **validity**. Her outline's step "evaluate fidelity → accept, skip, or revert" would harden this error. | **WM** (fix during the rewrite) |
| D16 | 04:12 (name) + email E10 | Zhang | "budget aware"; email: "add or emphasize an analysis across different edit budgets **if possible**" | No k-sweep exists. Any sentence implying a budget–fairness curve, "we vary the budget", or "budget analysis" would describe compute we never ran. Existing budget-adjacent evidence: trim-vs-trim+lift composition, upweighting dose w10–w50, oversampling dose, α sweep, oracle ceiling. | **WM** if written |
| D17 | 06:02 | Zhang | "CV model that is gonna discriminate people of color" | Illustration of *model-side* fairness, not our setting. Our protected signal is a ~10-district demographic profile; associational + ecological caveat mandatory. An individual-protected-class analogy in the intro invites exactly the over-reading the caveat guards. | HS in talk / **WM** if transplanted |

**Count: 13 divergences are WOULD-MISREPRESENT if written into the paper** (D1, D2, D3, D4,
D5, D6, D7, D9, D10, D12, D13, D15, D16 — several conditionally, as noted), plus 4
harmless-shorthand items that need a guard in the text (D8, D11, D14, D17).

Nothing in the meeting asks Robert to overstate a result. Every WM item is either a
paraphrase of the algorithm that drifted, or a framing sentence whose *intent* is
satisfiable by accurate wording (see §2). **No directive needs to be refused outright** —
C-1 needs inverting, C-7 needs re-ordering, C-8 needs a real artifact or a different label,
D16 needs a "future work" sentence.

---

## §4 Metric-description directives (F_demo, the "surrogate", attribution scoring)

### 4.1 "Surrogate" is released; "fairness objective" is approved
[36:28–36:45] is explicit: "collective fairness surrogate" is *her* language, she is not
asking for it, and Robert may "use whatever language makes sense". Recommended:
**"collective fairness objective"** in the §3.1 heading (keeps her adjective, uses her own
restatement of the noun) and `F_demo` as the symbol. If "surrogate" appears anywhere, define
it narrowly: *a differentiable, corpus-level proxy the editor can optimize, with the
external instruments (DP, DI, Theil) as the check it never sees.* Never "a surrogate for the
causal effect of demographics".

### 4.2 The normalization/scoring paragraph Plaud flagged (AI-suggestion #3)
This paragraph is the single highest-risk new text in the restructure — it is where D3, D4,
D5, D6 would land. Accurate content, in order:

1. **Which metric drives attribution:** the demographic term. The deficit is
   r²_demo = 1 − F_demo; F_spatial (a demographic-independent spatial regularizer) and the
   fidelity term enter the *edit optimization* through L but do not define the deficit
   partition. Lift's map is the gradient of the **full** objective L w.r.t. supply.
2. **Trim's scoring:** exact per-unit decomposition of r²_demo across active units,
   denominator RᵀMR, so each unit's score is a share of the total deficit and the shares sum
   to the whole — "an exact partition, not a heuristic weight". Signed variant: multiply by
   sign((HR)_i) to separate over- from under-served units. Trajectory selection = pickups
   landing in the highest-deficit units.
3. **Lift's scoring:** v_i = ∂L/∂S_i at ΔS = 0, one backward pass, closed form for the
   F_demo component (App A Eq. 5), autograd-verified. Signed real values, **not** rescaled to
   [0,1]; large where presence would repair fairness, small or negative where presence is
   already abundant. Then a linearized-offset screen ranks trajectories by best bounded tail
   translation; only positive-score nominees are taken; the screen only nominates and each
   nominee is re-optimized under the full objective.
4. **Say the negative explicitly if a normalization sentence is needed:** the scores are
   *shares of an exact decomposition* (trim) and *marginal fairness values* (lift). Do not
   write that scores are z-scored or standardized.
5. **F_demo reading:** F_demo = Rᵀ(I−H)R / RᵀMR = 1 − r²_demo; 1 = fairest; higher = less
   demographic dependence. Keep the teaser's reading (demographics explain 20% of the
   demand-adjusted imbalance → 17.9% after editing, an 11% relative reduction).
6. **H is constant during editing** (demographics fixed), so every edit step implicitly
   re-fits the demographic regression exactly — this is what makes the measure an *objective*
   rather than an audit. Keep this sentence: it answers "isn't your metric gameable by
   refitting?".
7. **Keep the three-ring claim discipline** in the new §4 (optimized / design-targeted /
   genuinely external) and keep "improves measures it never optimizes" riding ring (iii)
   only. Under the new spine the *collective-fairness* claim should be evidenced with ring
   (iii); the class-(i) F_demo delta keeps its current honest label, "reported for
   completeness rather than as evidence".
8. **Protected caveats stay** even though Zhang never mentioned them: associational, not
   causal; ~10 district-level degrees of freedom with ecological-fallacy exposure; demand
   endogeneity bounds both the metric and the editor.
9. **Fidelity wording:** "guardrail" and "scored in the objective at every iteration" +
   "evaluation-time gate (Fidelity-A)". Not "gates each edit", not "verifies each edit", not
   "enforces". Fix the three existing instances (D15).
10. **F_spatial naming:** DSR = **departure**-service ratio per `su2018taxigini` (renamed
    2026-07-24); ASR = arrival-service ratio.

---

## §5 Do-not-claim list

**From this meeting**
1. Do **not** claim the edits barely move raw/corpus-level fairness (D1). They move it
   modestly and reliably; that is the evidence, not an embarrassment.
2. Do **not** claim downstream fairness improves because of editing alone (D2).
   Uniform-weight BC is null; the weighting stage is load-bearing. State the null — it is our
   own prediction confirmed, and it is what makes the two-stage design necessary.
3. Do **not** claim a per-trajectory fairness score, per-trajectory fairness, or "this
   trajectory is unfair" (D3, D6, D8). Scores are unit-level deficit shares and edit-value
   estimates.
4. Do **not** claim any z-scoring / standardization of attribution scores (D4).
5. Do **not** state one sign convention for both phases (D5).
6. Do **not** describe one selection feeding both editors, or merge trim and lift in prose
   (D7, D9).
7. Do **not** claim a fidelity accept/reject gate per edit; accept/skip/revert is validity
   (D15).
8. Do **not** imply an edit-budget sweep or a budget–fairness curve exists (D16); a k-sweep
   is future work.
9. Do **not** call FATE's edits attacks or perturbations, or its output a perturbed dataset
   (D10); do not call FATE generative or an imitation-learning method.
10. Do **not** label a schematic figure a case study (D12).
11. Do **not** claim transferability to cities beyond the two tested (D13); "reproduces on a
    second city".
12. Do **not** claim full cross-city parity; label every supply number's accounting tier and
    keep the SF tier-1 net-negative total disclosed (D14). Robert's defensibility argument at
    [47:09]/[47:46] stands and Zhang did not overrule it — she only asked not to repeat
    identical observations.
13. Do **not** transplant the individual-protected-class analogy (D17).
14. Do **not** claim data or code availability beyond what the anonymous repo holds at
    submission (C-14 / E14).
15. Do **not** describe correctness machinery as optional tricks (D11).

**Standing (protected register) — unchanged by anything said in the meeting**
16. No causal or individual-level demographic claim; F_demo is associational (partial R² over
    ~10 district profiles) with ecological-fallacy exposure.
17. Leveling-down is an **analogy only**: trim relocates under conservation; nothing is
    destroyed. Keep the empirical fact that all 2,455 SZ trim pickups originated in advantaged
    cells and none landed in a disadvantaged cell — it is what motivates lift.
18. Demand endogeneity is disclosed as bounding **both** the metric (under-detects inequity
    where service was thin) and the editor (nothing to move in under-served areas). Under a
    "small portion" spine a reviewer will ask "why not edit more?" — the honest answer is
    budget + ε-bound + validity + fidelity + this structural limit, not that more editing was
    tried.
19. No realism guarantee: identity-level behavioral fidelity only; Fidelity-B 0.187 disclosed
    as a by-design distributional cost.
20. Do **not** claim the trade-off is solved (C-5) and do **not** claim the service gap is
    closed — DI moves 0.3325 → 0.3487; it is a partial repair.
21. p = 0.031 is the n=6 sign-unanimity floor, never an effect size.
22. Era numbers only: α* = (0.1, 0.8, 0.1); SZ 2,455 selected → 2,337 net + 7,545 lift; SF
    1,330 + 629; headline Δ +0.0226 SZ / +0.0316 SF.
