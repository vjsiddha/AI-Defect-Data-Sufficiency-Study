# How Much Real Defect Data Does AI Inspection Actually Need?

A Quality Engineering case study testing whether VLM-guided synthetic defect images can substitute for scarce real defect data — and what they actually change about an inspection system when they can't.

---

## Project Overview

AI-based visual inspection has a data problem that doesn't get talked about enough: the rarer a defect is on your line — which is exactly the kind of defect a high-performing process produces — the less data you have to teach a model to catch it. That's backwards from how most computer vision problems work, where more data is just a matter of collecting more images. Here, the data you most need is the data your process is actively trying not to produce.

Generative AI has made an attractive-sounding fix possible: instead of waiting for real defects to occur, manufacture synthetic ones. A vision-language model can describe what a real defect looks like, and a diffusion model can paint a plausible version of it onto a clean part. This is genuinely interesting technology, and it's why I wanted to dig into it rather than wait for someone else to apply it to a manufacturing question I cared about.

But almost every evaluation of this technique I could find was written for a computer vision audience — reporting mean average precision or F1 score, the metrics that matter if you're comparing model architectures against each other. None of that maps directly onto the decision a Quality Engineer actually has to make: if I deploy this, how many bad parts reach the customer, and how much good product gets scrapped for nothing? I wanted to answer the question in those terms instead, the same way I'd evaluate any new measurement tool before trusting it on a production line.

So this project isn't a computer vision benchmark with a manufacturing example bolted on. It's a Quality Engineering investigation that happens to use computer vision — vision-language models, diffusion image generation, classifier training — as the tools, not the point.

---

## The Problem

Modern manufacturing quality control has a structural irony built into it: the better you get at preventing defects, the less data you have to train an AI system to catch the defects that do slip through. A defect rate of a few parts per million is a quality engineering win — and a data scientist's nightmare. There simply aren't enough real failure examples sitting around to train a model on, especially for a newly launched product or process where no defect history exists at all yet.

Collecting more real defects on purpose isn't a real option either. You can't deliberately run a process out of spec to manufacture training data — that defeats the entire purpose of quality control, and in a lot of contexts (safety-critical parts, expensive raw material, contractual quality commitments) it's simply not something a plant manager would ever sign off on. So the realistic alternative is waiting — running production and hoping enough real defects occur naturally over time. That's slow, and it leaves an inspection system under-trained and under-validated during exactly the window — a new launch, a process change — when defects are most likely and inspection coverage matters most.

This is the gap synthetic defect generation is trying to fill. If a generative model can produce convincing artificial defects, a manufacturer doesn't have to wait. The promise is real. What's missing from most discussions of it is whether that promise actually holds up once you stop measuring it like a computer vision leaderboard and start measuring it like a quality engineer would.

---

## Why I Built This

Most synthetic defect generation papers optimize for benchmark accuracy — mean average precision, F1 score. Very few ask the question a Quality Engineer actually has to answer before approving an AI inspection system: how does this change escape risk and scrap cost? I wanted to evaluate synthetic defect generation the way I'd evaluate any new inspection tool — through escape rate and false rejection rate, repeated trials, and a healthy amount of skepticism — rather than as another computer vision benchmark.

---

## Why These Metrics?

This study deliberately tracks four things, not one. Accuracy alone is the metric every standard evaluation defaults to, and it's the least useful one here, because it blends two completely different kinds of mistake into a single number that hides which one you're actually making.

**Escape rate** — the share of real defective parts the system lets through as "good." This is the number that matters most, because it's the one that reaches the customer. A field failure, a warranty claim, a safety incident — they all start with an escape. A Quality Engineer cares about this above everything else.

**False rejection rate** — the share of real good parts the system wrongly flags as defective. This one costs money and time (scrap, rework, slower throughput) but never reaches the customer. It's a real cost, just a different kind of cost than an escape.

**Accuracy** — included because it's the number everyone expects to see, but treated as secondary throughout. A model can post a great accuracy score while quietly making one of the two errors above at an unacceptable rate — accuracy alone can't tell you which error you're trading for which.

**Run-to-run stability** — whether a result holds up across repeated training, or whether it was a lucky (or unlucky) roll of the dice. This isn't a standard ML metric at all, but it's exactly what a Quality Engineer would ask about any new measurement system before trusting it: does it give you the same answer reliably, or does it wander?

Evaluating with only the first of these four would have produced a confident, misleading headline. The other three are what actually let you make a deployment decision.

---

## What Was Tested

A vision-language model (Qwen2-VL) described real scratch defects on a metal fastener (MVTec AD `metal_nut`). Those descriptions guided a diffusion model to generate 60 synthetic scratch images.

**Why scratches only, and not all four defect types in the dataset.** Generating every defect type at once would have diluted the experiment and stretched the generative model across types it wasn't being carefully validated for. Scratches are the cleanest surface-texture defect for diffusion-based inpainting to attempt, and focusing on one type meant the rest of the defect data (bent, color, flip) could stay completely unchanged across every condition — isolating the one thing actually under test.

**Why nine conditions instead of one.** A single "real vs. synthetic" comparison wouldn't tell you anything about *where* synthetic data helps or hurts. Varying the real-data count from 0 up to the full available set, with and without synthetic data layered on top, makes it possible to ask a sharper question: at this specific amount of real data, what does adding synthetic data actually buy you? That's the question a manufacturer in the middle of a launch would actually be asking, not "does synthetic data work in general."

**Why the same test set, every time.** If the evaluation images changed between conditions, you'd never know whether a result came from the training data or from an easier or harder test set. Fixing one real-image test set at the very start and reusing it, unchanged, across all nine conditions is what makes every comparison in this study apples-to-apples.

**Why five seeds, not one.** A classifier trained on a handful of images can converge to a degenerate shortcut by chance — for instance, a model that always predicts "defective," which trivially produces a 0% escape rate while learning nothing real. A single training run can't tell a genuinely good result from a lucky one. Each of the nine conditions was trained five times under different random seeds — 45 classifiers in total — specifically so the results below reflect a real pattern, not one favorable roll of the dice. This is the same logic behind a Gage R&R study, where a single measurement is never accepted as evidence.

That caution paid off directly: an early version of this study had roughly 40% of runs silently collapse into predicting one class regardless of input. That instability had to be found and fixed before any of the results below could be trusted — and finding it at all was only possible because the study wasn't relying on a single run per condition.

**Full implementation:** [`notebooks/01_Baseline_Training.ipynb`](notebooks/01_Baseline_Training.ipynb) (baseline), [`notebooks/02_VLM_Guided_Synthetic_Generation.ipynb`](notebooks/02_VLM_Guided_Synthetic_Generation.ipynb) (generation), [`notebooks/03_Data_Sufficiency_Study.ipynb`](notebooks/03_Data_Sufficiency_Study.ipynb) (the experiment itself).

---

## What Happened

| Real Scratches Used | What Synthetic Data Did |
|---|---|
| 2 | Caught more real defects, but flagged noticeably more good parts as false alarms |
| 5 | No meaningful change in defect detection; false alarms increased |
| 10 | Caught more real defects, but flagged noticeably more good parts as false alarms |
| 18 (full real set) | No meaningful change in defect detection; false alarms increased the most here |

No condition beat training on real data alone for overall accuracy. But the direction of the tradeoff never flipped — at every real-data level, synthetic data held escape rate steady or improved it, while consistently increasing false alarms.

> **In plain terms:** synthetic data made the inspector more cautious. It didn't make it smarter.

What stood out to me wasn't any single number — it was that the *same* tradeoff showed up whether there were only 2 real examples to work with or the full set of 18. I went in expecting synthetic data to matter most when real data was scarcest, and to fade out as more real data became available. That's not what happened. The pattern held at every level I tested, including the condition with the most real data already in hand — which suggests this isn't really a "filling a data gap" effect at all. It looks more like synthetic data is teaching the model a specific, consistent bias toward caution, independent of how much real signal it already has.

The one thing that stayed genuinely consistent across every single condition was the *direction* of the false-rejection increase — it never once reversed. That consistency is what makes me trust the finding. A result that only shows up at one data level could easily be noise; a result that shows up the same way four times, at four very different amounts of real data, is a real effect.

---

## Key Findings

- **Synthetic data did not replace real defect data at any tested level** — real-only training was never beaten on accuracy.
- **It consistently shifted inspection behavior, not performance** — the same tradeoff pattern repeated regardless of how much real data was available.
- **Escape rate held steady or improved** at every real-data level tested — it never got worse.
- **False rejection rate increased consistently** — more good parts flagged as defective, every time.
- **Multi-seed evaluation caught something a single run would have hidden** — the early training collapse mentioned above, which a one-shot evaluation would have missed entirely.

---

## What This Means for Quality Engineers

| If you're... | Synthetic data is... |
|---|---|
| Launching with zero real defects | A temporary bridge — not a long-term fix |
| In early ramp-up, tolerant of re-inspection | Worth trying |
| Running a mature line with defect history already | Not adding much |
| In a safety-critical program | Not a substitute for real validation |
| Highly sensitive to scrap cost | Probably the wrong tool here |

The reasoning behind each row follows directly from the tradeoff above, not from a general impression of the technology:

**A new launch with zero real defects** is the scenario synthetic data is actually marketed for, and it's a defensible use case *as a bridge* — but only because the cost of extra false rejections (a part gets re-inspected, maybe scrapped) is usually tolerable during a low-volume ramp-up, while the cost of an escape during launch (when you have the least track record to fall back on) is high. It's a reasonable trade to accept temporarily, not a reason to stop collecting real defect data.

**Early ramp-up with re-inspection tolerance** is the best-fit scenario in this study, for the same reason — you're deliberately choosing to absorb more false alarms in exchange for not missing real defects while your real-data collection is still catching up.

**A mature line** already has the real defect history that synthetic data is meant to substitute for. Adding synthetic data here mostly just adds false rejections without a meaningful escape-rate benefit, because the thing synthetic data is good at — nudging the model toward caution — has diminishing value once the model already has enough real examples to be confident.

**Safety-critical programs** can't accept an unverified bias toward caution as a substitute for real-world validation. The escape-rate improvement here was real but modest, and it was measured across five seeds, not five hundred — solid evidence for a directional pattern, not the kind of statistical confidence a safety case needs.

**High scrap-cost sensitivity** is the scenario this study's findings argue against directly. False rejection rate got worse in every single condition tested. If scrap cost is your dominant concern, this technique pushes in exactly the wrong direction.

---

## Where the Detail Lives

- **`notebooks/`** — full, runnable implementation: data setup, VLM prompting, image generation, and the 45-run experiment
- **`outputs/`** — the aggregated CSVs behind every number above (`aggregated_results.csv`, `pairwise_comparisons.csv`, `variance_analysis.csv`)
- **`report/Research_Report.md`** — the complete write-up: methodology, limitations, and the reasoning behind why the results likely look this way
- **`report/figures/`** — uncertainty plots and a real-vs-synthetic defect comparison

---

## Limitations, Briefly

**Scratches only, one part.** Everything here applies to one defect type on one fastener. A different defect — something with more 3D structure, like a dent or a crack — might respond completely differently to synthetic generation. Extending this to the other defect types in the same dataset would be the natural next step before generalizing the finding any further.

**Five seeds, not five hundred.** Five independent runs per condition is enough to see a consistent direction, which is what this study is built around — but it isn't enough for the kind of statistical confidence a safety-critical decision would need. More seeds would tighten the error bars; they probably wouldn't change the direction of the result, but I can't claim that with certainty from this data alone.

**Simplified defect placement.** Synthetic defects were placed using randomized mask shapes, not the precise, pixel-level ground-truth defect masks the reference methodology (SynSur) uses. A more careful placement approach might change how convincing the synthetic defects are, which could shift the magnitude of the effect — worth testing before drawing conclusions for a defect type with more spatial complexity than a scratch.

Full discussion in `report/Research_Report.md`.

---

## Lessons Learned

**Accuracy alone is a poor measure of inspection quality.** I knew this going in, in the abstract. Actually watching a condition post a *better* accuracy score while quietly making more dangerous escapes — or the reverse — made it concrete in a way I didn't fully appreciate until I saw it in my own numbers.

**Multi-seed evaluation found something a single run would have missed completely.** The training collapse wasn't a hypothetical risk I was guarding against — it actually happened, in close to half of all runs in an earlier version of this study. If I'd evaluated each condition once, I would have published a confident, wrong conclusion, and had no way of knowing it was wrong.

**Synthetic data changes behavior more than it changes capability.** Going in, I expected a "does it help or not" answer. What I found instead was more specific and, I think, more useful: the model didn't get better at telling defective parts from good ones — it got more willing to call something defective. That's a real, usable finding, but it's a different finding than the one I set out looking for.

**Real defect data didn't become optional at any point in this study.** Even at the highest real-data level I tested, synthetic data still didn't outperform real data alone. I went in open to the idea that synthetic data might eventually substitute for real data once enough real examples were already in place. It didn't — not at any level I tried.

**The metric you choose decides the conclusion you're allowed to reach.** If I had evaluated this study on accuracy alone, the headline would have been "synthetic data doesn't really help." Evaluating on escape rate and false rejection rate separately surfaced a real, specific, actionable tradeoff that the single-number version of this study would have hidden entirely.

---

## Reproducing This

Three notebooks, run in order, on a free Colab GPU:

1. `notebooks/01_Baseline_Training.ipynb`
2. `notebooks/02_VLM_Guided_Synthetic_Generation.ipynb`
3. `notebooks/03_Data_Sufficiency_Study.ipynb`

Each produces files the next notebook needs — download and re-upload between sessions if running on Colab. Expect roughly 2 hours total, most of it in the 45-run experiment in Notebook 3. See `requirements.txt` for dependencies.
