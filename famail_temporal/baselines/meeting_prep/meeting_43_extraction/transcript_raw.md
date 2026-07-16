# Meeting 43 — Full Raw Transcript (ground truth)

- **Meeting:** FAMAIL Group Bi-Weekly Summer 4 (= project **Meeting 43**)
- **Date held:** 2026-07-16
- **Source:** Notion page `39feb306-5110-8001-9bce-cab390801cc2` (https://app.notion.com/p/39feb306511080019bcecab390801cc2), fetched 2026-07-16 via Notion MCP with `include_transcript: true`
- **Attendees (from context):** Robert Ashe (presenter), Dr. Xin Zhang (PI), Dr. Kash (transcribed variously as "Dr. Cash"/"Dr. Kash"); possibly others (Manu mentioned but no update given).
- **Transcription caveats:** The recording tool does not label speakers; paragraph breaks below are as delivered by Notion. Known garbles: "Dr. Chong" = Dr. Zhang; "Dr. Cash" = Dr. Kash; "TEAL index"/"TEO index" = Theil index; "ST-FGSM"/"STFJSM"/"ST-IFGSM" = ST-iFGSM; "dye line" = deadline; "word leaf" = Overleaf; "cetacean" = citation; "GPT-0" = GPTZero; "auto information" = author information; "reproductibility" = reproducibility.
- The Notion page also embeds a `<mention-date start="2025-10-12"/>` on the meeting-notes block — that date is wrong (Notion artifact); the meeting was held 2026-07-16 per the page's Scheduled Time & Date property.

---

## Transcript (verbatim, in order)

if it's kind of related to a paper. So let's see. How the books... Thank you. Mm-hmm.

Yes.

So I'll just kind of roll right into it. So to kind of update you guys, me and Dr.

Chong met last week and where the entire group left off is that I developed a method. We found that it was working and the method being we want to develop a framework where we can modify trajectories to improve global fairness. be used to train behavioral models. So I left off and the task that I was I set out on was to find a set of external metrics that we can use to measure if we're improving fairness on metrics that we have not optimized for. And I did that. And Dr.

Cash, I have to credit you with inspiring a pretty impactful idea and the idea if you remember from last last couple weeks was I How can we reach those trajectories that might not be the worst trajectories, but we can still improve them? And I didn't quite directly answer that, but that idea spurred something that turned out to be really, really impactful. So this is a, I'm not going to go through this whole presentation because Dr. Zhang's seen this and we have bigger things to get to, but really, really fast. The external metrics actually are the external metrics, disparate impact, demographic parity, and the TEAL index.

I was able to show that on metrics that we did not optimize for, we have improved fairness through the trajectory modification approach that we've applied. So really awesome news and I'm sleeping better since. So the catch though is that this improvement I found that We improve fairness by leveling down and this leveling down terminology you might be familiar with but it's in the fairness literature where instead of lifting up the disadvantaged group we just reduce the service to the advantaged group and unfortunately that was what my original approach was doing and this is where the inspiration from Dr.

Cash comes in big time because I tried to find ways that maybe because the trajectories that I'd identified or attributed to harming fairness most were those over-served trajectories, hence the leveling down and therefore not lifting up the disadvantaged group. So very key realization from Dr. Cash there. Thank you. And without getting into details as far as why, the reason why is it's actually structural. I kind of just alluded to it with the attribution. The fix, though, is manipulating supply because the demand is something that is a lever that can really only touch the over-served.

So by manipulating supply, meaning where taxis actually are physically present in the environment, And therefore provide that taxi supply to the underserved and therefore lift up the underserved population. So the way that we do this is by adding this lift aspect to the algorithm. The original thing I've been talking about this whole like year so essentially was trim algorithm where we trim the oversupply.

The lift is how we actually lift up the underserved and provide them with additional supply. So The current trim work basically moves taxi service around and it's only able to stay within kind of those over-served areas. This new addition, we're actually able to reroute taxi service into the under-served regions and therefore lifting up the under-served population. Now, I'd love to go into further detail into this, and I'm of course open for questions, I have a different presentation here. This is the one that I...

or I'm planning on presenting That's my current progress. So, just again, I give you guys a little sneak peek here. This is a paper that was written by me about the, or me and my friend Claude Fable, because it's 2026. You guys know how it works. So, we have, and it wasn't written over just the last couple weeks. It was assembled because... I'll get into why but I've been building the argument over the last few months and preparing for this so is more just kind of plug-and-play so The end the message I want to convey is that we are very close I still have quite a bit of work to do but we've made huge progress over the last week and that's what I'm about to talk about and So...

One of the important aspects of the work is, excuse me for one second, I'm going to roll down to find this. This is our objective function. This is what we're optimizing through the trajectory modification algorithm. And this is just a standard weighted objective function. And we need to motivate why we've chosen the weights we did. And the weights that I was using were,I have a not very structured.

selection of weights, just kind of trial and error. Hey, these weights have produced the best ones. Move to the next, see if it works. Never anything structured that we can put in a table. So this last week, one of my goals was to run a true grid search over reasonable weights. And I I actually came to find that the weights that I had been using were not optimal. They were optimal in the objective, or they were optimal in improving the metrics that we optimized for, but in the external metrics, the external fairness metrics, demographic parity, disparate impact. I didn't take into those account and with our end goal being improving fairness, I wanted to include those in the choice for the optimal parameters.

So I found that there was a better, more optimal set of parameters and that's what I've structured the paper around at this point. So, Finding those optimal weights, forced me to rerun basically the entire experiment stack And that resulted in about a week's worth of GPU time. I think I started the rerun last Thursday, and one of the final runs ended this morning. So we've been basically churning, making sure that we have all the results, metrics that we need to kind of stand up to review scrutiny.

And I've actually, just so you guys understand my workflow, I've leveraged adversarial review, AI adversarial review, to find holes in the argument to motivate some of the baselines and comparisons that will make the argument stronger, which has been, I think, extremely important, especially since we're trying to submit to KDD, a pretty serious group that I want to treat as such in the development of the paper.

Over the next few weeks one thing I will kind of hope is that you guys will give me your very human review because of course that is the most important part because we got to pass it through some humans and So without getting into a whole lot of detail as far as results, The general idea is that we're not just trying to motivate optimal weights, but we're trying to motivate the optimal choice of demographic features, which is what we actually or the fairness ideas that we're trying to motivate for.

So this results in a huge grid of different experimental runs that I've taken the time to make sure that are documented carefully. So reproducibility is going to be a big thing. Anyways, so this alpha sweep resulted in finding a more optimal weight, or set of weights, which then spurred rerunning of basically the entire experimental stack. As I just mentioned, We... I think an important part of doing quality research is making sure that the research is reproducible.

So because we have so many experimental results, I've been tracking every single run down to the exact command that's used to run it so reviewers could even just hit go on a single script and reproduce all of our results. So really trying to emphasize passing reviewer scrutiny when the time comes. So all of that is taking place. The most important thing I think we achieved last week was in our meeting, or me and Dr.

Zhang's meeting last week, I said that I really wanted to get an abstract ready by this week and it ended up being a little more mechanical than I had predicted in assembling the results. So I was actually able to get all of the prose written for the actual paper and now It's going to be a matter of, of course, checking citations, shrinking the actual length of the paper because KDD gives us a maximum length of eight main content and right now I'm sitting at about nine pages so we just got to shrink the length of the paper a little bit in addition to actually you know like validating the argument and stuff and I've been doing that over the last week but since the final results have landed it's now or I'm now able to Do that with a little more direction, I guess.

you Thank you. you you Okay.

Thank you. you Thank you. you you - you youHmmum youOkay, sounds good.

That was not very well said, but you might understand what I mean.

So part of the work over the last week and one of the things I feel strongly about is developing a strong figure one, something that encapsulates your argument, really allows the audience to understand what you're trying to do. So one of the things that I'd like to ask you guys is to give me your opinion on what you think of this figure one. And without my... Really explaining too much of it because I don't want to bias your opinions because people that have never read this never had experience with the work will need to to encounter this figure and actually understand what it means.

So that's something I'm hoping I'll send out to you guys very soon, but I've put a lot of work into this, and I'll have some specific things that I'd like you guys to scrutinize besides the obvious, the metric, or the message itself, I mean. So we'll get to that. And... The argument that we're making in the paper, I've taken a lot of time to make sure that it's reasonable, logical, and will be well structured in the paper so first time audience will understand what we're trying to do so we don't just get desk rejected out of this thing doesn't make sense and it's AI slop.

So I'm taking a lot of time to make sure that it actually makes sense and the The core motivation is that human demonstrations encode inequality or bias and the behavioral models that we train with that data, they not only inherit that bias but sometimes they can actually amplify it. So that motivates our, or that leaves us with a task of finding a way to remove that bias so we can train better behavioral models that, you know, don't harm society.

So we instead of ditching all of that biased data, we want to edit it to try to preserve as much human gen or human human source data as possible. We have baselines and tests that motivate these things by the way if you guys are wondering I'm and I'm trying to be quick so hopefully I'll kind of get into explaining how we do that and Because we now have a two-part algorithm, we must, by ablation, show that it's necessary to not only level down the over-served but also lift up the under-served. That's going to be a major part of the argument.

And then, of course, the kind of capstone is that the fairness doesn't just improve on metrics that we've optimized for, community as this is what fairness means um so we can show that the work that we're or the editing work that we actually do does have the intended effect of improving fairness. So These are kind of further details that I don't want to leave many without some time on so we have alright I would like to go into exactly how we are the exact results that we have but I'm just Just kind of high level, we're going to present results that show a data level fairness, which these are some of the external metrics I was talking about, like disparate impact, demographic parity, the TEO index, and then very importantly, the average service outcome or the average outcome for the disadvantaged group because we want to show that we're not just tearing down service for the over-served, we also are improving service, really improving fairness and that outcome-wise our service, the taxi service essentially.

So that is going to be one of the external metrics that we quantify for this to evaluate the approach. And Also, to kind of just slip it in here, we have not just the Shenzhen dataset, which is our core primary dataset, but we also show these similar and sometimes better results on a dataset generated in San Francisco.

So we have two datasets, lots of generalizability, cool stuff to put in the paper.

So another set of or part of the results that we'll present are why this second half, the rerouting of taxi supply is necessary. We want to show that just trimming service for the over-served gets us some amount of fairness improvement, but lifting up the under-served gets us a greater amount of fairness improvement. And this nice little AI-generated bar chart here kind of encapsulates it pretty well.

is our original um just trim only um the red is trim and lift bigger bar good stuff um So none of this is the, or there's no There are, um... I... There are aspects of our results that we'll need to, or I guess not everything is. perfect as in it's just this nice linear increasing line but I don't know what I'm trying to say here. There are going to be things that, or there are results that aren't exactly what we expect. And that's going to be, I think, a good or a realism improvement for the paper.

I don't know. That was kind of a weird thing to say, but you guys might understand what I'm saying. I'm going to move on.

So another big or large aspect of the results is that We don't just want to improve the fairness in a data set because can't really do a lot with just a data set. We want to make sure that the fairness actually survives training. So the way that we do that is if we just stick with standard maximum likelihood estimation and just train policies, we find that the small section of trajectories that we edit are actually, or the fairness improvement in those small section of trajectories that we edit data set.

So if we make the behavioral model training fairness aware, trying to stay on brand, By just simply up weighting those particular edited trajectories, we can show that fairness propagates through the train models and in a linear fashion as we increase the weight for these trajectories.

So W equals 10 is a 10x weight, W equals 30, 10x weight for the edited trajectories. Nice line, so it's monotonic in increasing fashion.

SoUh, here's the thing. Huge part or huge part of the paper itself is we don't want to just show that we can produce the results that we hope to set out for. We also need to motivate the specific aspects of our approach. So the way that we do that is through the iterative FGSM and then FGSM. Also, random jitter approach.

And then very interesting baseline demographic oversampling. And I wish I had more time to go over this, but this is...

Uh... There's another work in this space that was published within the last couple years that took the approach to improving fairness by oversampling fair samples in their data sets. So I want to adapt that same approach and let's just oversample trajectories from our underserved or our more fair trajectories in the over-served regions. And we actually show that is a viable approach, but it's not as good as our trajectory editing approach. It's pretty good, though. Surprisingly good.

I was like kind of hoping that we could just, you know, blast this thing out of the water, but it wasn't quite that straightforward.

And that's kind of one of those things that not everything turned out exactly how I had thought it was going to. So there are other approaches that we can improve fairness with, but we have in the current work the optimal approach. Each of these baselines is meant to motivate a specific aspect like The actual gradient guided perturbation where we use an adaptation of the ST-IFGSM algorithm to take a fairness aware approach to editing trajectories.

Then we also want to make sure that or we want to motivate the iterative approach because what if you could just take a single measurement and understand and and find the optimal direction to perturb a trajectory and do it all in one shot. So we motivate the iterative approach. Why do we bound the trajectories? And the answer to that is because we need the edited trajectories to stay realistic to the original experts. So we motivate that.

And also the interesting one, demographic oversampling, we want to find out if instead of Editing trajectories, is there other recent approaches that can give us a run for our money so to speak? So these are some of the results from our baselines and each of the baselines is evaluated in the same way as possible as our trajectory editing approach. So everything is compared apples to apples. And all along the way, we are gathering this data in a very presentable way because I don't want to make reviewers sift through this massive code repository and find our results.

I want to hand everything nice with a little bow on top of it.

And AI is making this extremely possible to make replication kind of a benefit of the submission. providing reviewers with the data set.

I've heard that that's a major bonus for KDD reviewers, so I want to take advantage of that.

And by evaluating the baselines as we're evaluating the FAML trim and lift, we can show very clearly that we have the most effective approach to improving fairness in these types of taxi GPS trajectories. So what I'm planning on doing over the next couple weeks, because this Sunday we have an abstract deadline.

The following Sunday we have the actual paper deadline.

So I am being very mindful about what I have left to do in that amount of time. I'm going to be getting an abstract for Dr. Zhang to review by tomorrow. Obviously, I already have an abstract, but I want to make sure it's, that you have a final say on the one that we submit. I'll get you that polished version tomorrow or by tomorrow. It might happen today.

We got to reduce the the length of the paper and then I also want to provide it to you guys where you can make edits because currently I'm modifying the actual paper content within my development environment. I'll get that on Overleaf so you guys can touch it and play with it and Apparently we might have less of a problem in paper length because over leaf apparently the fonts are just a little bit smaller so it might work out in our favor.

And then I'm going to continue leveraging AI adversarial audits to make sure that not just human things are found, but we can also see things that are possibly superhuman or subhuman, depending on how you think of them. And then, uh... One important aspect of the submission is going to be not just the paper quality, but we also want to pledge our data set with it, which means we want to make the data resources that we have available.

I'll handle all that stuff, giving them a document that they can use to reproduce the results and everything. So all of that's on the list too. And then hopefully taking in some feedback from you guys on figure one and if I could ask you to get through the whole paper it'd be great but I know you guys are all busy and all that stuff it's a big thing. And then just making sure citations check out and all of that stuff so have a lot of stuff planned sorry I took so long.

But that's what I got.

So Robert, yeah, so for the submission, it's gonna be anonymous. So no auto information in the paper. So you can just send me, you can just start like, copy pasting the abstract initiated submission on the old word leaf.

Yes.

Yes.

I'm sorry, open review and our one to the author list.

And I'll be start reviewing your paper sometime next week. And so we have some time to kind of polish a paper. before the end of the the dye line and Yeah.

Yes.

Got the anonymous thing. It's a little one line comment out.

Okay.

So.

We're ready. I got you. Yep.

Yeah.

Mm-hmm.

So yeah, I would like to kind of take my own first pass through the paper because there are things that I definitely want to change to emphasize, minimize, stuff like that, so I don't waste your guys' time. So if I could have a couple days and I'll have the Overleaf ready as soon as possible.

Okay, sounds good.

Yeah.

Great idea.

And also maybe like one kind of a teasing figure, other one introduction to try to highlight what's the problem we are trying to solve or what the overall claim looks like. You can refer to the other KDDT paper, like I said, STFJSM for kind of some idea about what to do with that. And that will be... start looking at the paper in terms of its organization and everything. next week.

Cool.

And in terms of the reproductibility, in terms of your applying, so the reproductibility, you can just for, I'm not so sure,Mm.

Love it. Thanks for the suggestion there. Really, really happy. Okay.

So basically when we are submitting the code, we can just give them kind of anonymous link to the GitHub repository where we put our code over there. Make sure that your code doesn't have any kind of personal information like a password that has your name on it. And the repository, the code, it doesn't need to be perfect itself. I think the priority would be having all the paper contents ready Having all these experiments ready and the data set as well as the code can be cleaned up after the submission.

So we have more time focusing on the paper itself.

Yes, and I have, I'm, I'm, looking for it as I'm talking but I have a All of the data logged, so it's kind of easy and a huge stack of results.

And that's actually something else that I'd I'd like to kind of get your... input on is we have a huge, huge set of results here.

Each one of these directories is an experimental run. And they all come with results. So I don't know if you have any... input on how you might want us to organize this but the way that I'm currently doing it is I have documents that point to these these results directories each of them has like the data sets that goes with them any results and stuff but any of the data that's actually used in the paper itself that'll all be self-contained in a single little thing but those are kind of details like a couple weeks from now I guess Yes.

Yeah, yes, those can be finalized after the submission deadline.

And there's an ins... Yes, we may decide what details we want to, what information we want to make public or not. And yes.

Yeah.

Thanks for reading between the lines there.

And I don't think...

Yeah, sometimes, man.

We have much time for Manu to update.

I have a few comments before we move on now. Yeah. So it's good if it's on your roadmap, but I think it can't be emphasized enough that citations are getting so much more scrutiny than they used to. I have seen multiple papers just get essentially desk rejected because of citation issues.

Yes.

from all roles in the root process. So yeah, we cannot let any AI citations get through. They all need to be-so don't accept any citation that AI gave you. Make sure that you've pulled all of the citations that are in the paper manually and ideally from a reputable source. So don't just take what Google Scholar gives you. I've actually seen papers get rejected because of essentially including garbage that Google Scholar returns.

4.

Yes.

for the thing. So yeah, don't trust it. Seek the authoritative source for the citation, be it ACM, whatever will give you So a proper digital repository will give you a proper clean citation for everything that's in there.

Okay.

Okay. We have a pretty...

Um...

It's not a massive list of references, but it's suiting for this...

This paper it's not insane, but.

I don't know how you guys feel about bribes, but it's out there. I'm just gonna say this. I don't know what that how how explicit I can get about this But you guys can read I know that so Just putting it out there.

um Yeah, I think also to add on that, kind of you can So basically you want to make sure that recitation is not has to be originated by AI and then one could kind of strategy is put it into GPT-0 or use the AI tool to help you to track hallucinated cetacean.

Okay.

Yes.

And that's actually-I've taken-or I've had-I think two or three AI reviews that are specific to this and leveraging the fancy new Fable model and stuff, which is really quite impressive and a little scary as far as its capabilities. So, and basically having the AI work in pairs, one of them basically finds a resource, checks it, verifies it, and then the other AI opens that resource and verifies that the claim that we're making actually does exist in it.

pretty good machine verification, but As Dr.

Cash, I think, very rightly emphasized, and thank you for the heads up on things getting desk rejected on this, because I'm... Not that I wasn't taking it seriously, but maybe I'll just, like, make sure, an extra make sure, but I think that you can't beat having human eyes on it, so... But I've been, like, kind of trying to play the CYA, if you guys know what I mean. Um...

But, We'll get there. And there's a couple other experimental results that I'm popping out. Things that basically just round out tables, make sure that everything's all nice.

Because I really want this thing to work. I think we've been working way too long on this for it to not work. It's gonna be great. It's gonna be fine.

So then more kind of subtly on the content itself.

So on your figure one, on uh, Question. I think the point here is, is an important one. I don't know that I would use-Mm-hmm. quite the leveling down framing, though, in the sense like what's going on here is not leveling down in the classic sense. Like I can sort of see what's going on, but like, Like the classic version of leveling down is like, you know, we have a classifier. It's giving out loans to people.

We discover it's giving loans to an advantage group much more than disadvantage group. So what's our solution? Just give fewer loans to the advantage group. Right. And then like so like the the you know, the effect is there's just fewer loans being given out overall. Right. And that's a sense that like there's an economic problem. We're like, I'm like, lost there. We've just taken away loans that would otherwise happen. But if I understand, but like...

like you're not actually eliminating pickups in this picture right you're just kind of relocating those pickups so like there's a conservation of pickups going on here right so we're not actually destroying the valuable pickup so it's not like leveling down in that classic sense so like like i think there's an important point that yeah like like what it's like you It's moving pickups that...

Yes.

Hmm. I see what you mean. No.

from like the most advantaged place to like a...

Mm-hmm.

moderately less advantaged place Right?

Which like, Shares the important feature with like the leveling up leveling down that it's not helping the truly disadvantaged at all. Right, and I think that's the point to be emphasizing there But it's not you know, it's not kind of destruct like truly destructive in in that wayYeah, I would eliminate leveling down as the description of what's going on.

Excellent.

Okay. So, and okay, just to make sure I get this right.

So not leveling down in the classic sense. Is that something where you would suggest we remove the leveling down language entirely or qualify it as something like pseudo leveling down?

What do you think?

I might keep leveling down as sort of an analogy when explaining what's going on. But yeah, but I'd be careful not to describe what's actually happening as like Yeah, I wouldn't even label the effect as quasi-Ah. Uh. Uh. Yeah. Well, you know, like you could keep... Yeah, you could...

I hear you.

Okay.

I think you can keep the trim and lift. terms, right, which are kind of your own bespoke thing But yeah, I wouldn't use that terminology as even like the primary way you describe what's going on.

Cool.

Yeah, and I've had...

I, I, I'm not all that familiar or wasn't all that familiar with leveling down as this widely used terminology.

I've learned that since. But the just the effect that it has as I read the paper, I'm like, it's like it doesn't like.

It feels like I'm kind of over explaining the actual effect anyways, but now with the I guess the more trained explanation I can feel a little more confident in my Just the intuition that I had wasn't really well founded, but now I feel pretty justified so Thank you.

Yeah, so that's awesome.

Yeah, so that's good.

And maybe when I send you guys figure one, because what I'll do is I'll kind of get a little email or a Slack thing. I'll put a figure one in there. And if you guys wouldn't mind taking the time to give that scrutiny to our abstract two, because we got that deadline coming up. I love you forever.

Yes.

But for the abstract, we can always modify before the paper submission deadline.

And title too.

So yeah. We can also modify the title a little bit as long as it's not far off.

I think title. It was kind of an important. Yes. Yes.

So currently we just want to put a meaningful abstract, a meaningful title as a placeholder for our paper. Mm-hmm. Okay, sounds good.

Yes. All right.

Thanks for your time and help guys.

If there are no more questions, thank you all for your time. And Dr. Cash, can I talk to you for a couple of minutes? Yep, sure. Okay, sounds good.

Yeah, just give me like one minute and then I'll be right back and we can chat.

Yeah, okay. Thank you.

See ya.
