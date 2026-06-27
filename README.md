# How Does VLM-Guided Synthetic Defect Augmentation Change AI Inspection Performance Under Limited Real Defect Data?

**A Quality Engineering investigation into whether AI-generated synthetic defect images can responsibly substitute for scarce real defect data — and what they actually change about an inspection system's behavior when they can't. Evaluated against escape rate, false rejection rate, and run-to-run stability, with trust in the resulting system as the underlying question.**

---

## Executive Summary

Production lines that run well don't generate enough real defective parts to train an AI inspection model — a problem manufacturers are increasingly trying to solve with AI-generated synthetic defects. This study tested that approach directly: a Vision-Language Model (VLM) described real scratch defects on a metal fastener, and those descriptions drove a diffusion-based pipeline to generate synthetic defect images.

Across nine experimental conditions, each trained five times (45 models total), the study measured what synthetic data actually changes — not just accuracy, but the two numbers a Quality Engineer is accountable for: **escape rate** (undetected real defects) and **false rejection rate** (good parts incorrectly scrapped).

**The finding:** within the range of real-data availability tested (2 to 18 real examples), synthetic data did not substitute for real defect data at any level. Instead, it consistently shifted the decision boundary toward a more conservative operating point — escape rate improved or held steady at every level, while false rejection rate worsened at every level. Synthetic data did not make the system more accurate; it changed *what kind of mistakes it makes* — and that consistent distinction is this study's central finding.

---

## The Manufacturing Problem

[FIGURE PLACEHOLDER: diagram illustrating the "data paradox" — a well-run production line producing very few defective parts, with a sparse defect timeline]

A core contradiction sits underneath every AI-assisted quality inspection rollout: the inspection system needs real examples of defects to learn what to catch, but a well-controlled, low-defect-rate production line — which is the entire goal of quality engineering — generates very few of them. This is especially acute during a new product launch, when a line has no defect history at all, but inspection coverage is most needed.

The proposed industry fix is synthetic defect generation: using generative AI to manufacture artificial defect images that can supplement, or in theory replace, scarce real data. This is an active, rapidly developing area — most recently formalized in academic work such as the SynSur pipeline (Kühn et al., 2026), which combines a vision-language model with diffusion-based image generation to produce synthetic industrial surface defects.

It is worth distinguishing this approach from traditional data augmentation (rotation, flipping, cropping, brightness adjustment), which has long been standard practice in computer vision. Traditional augmentation increases the *quantity* of training images derived from existing examples, but it does not create new defect *morphology* — a rotated image of the same scratch is still, fundamentally, the same scratch. It cannot help a model learn to recognize a defect shape, scale, or severity it has never seen, which is exactly the gap that matters when real defect examples are scarce to begin with. VLM-guided synthetic generation is being investigated specifically because it offers something traditional augmentation cannot: the ability to generate plausible new defect *instances*, not just new views of existing ones.

What is largely missing from this conversation is a Quality Engineering framing of the result. Most published evaluations of synthetic defect data report standard machine learning metrics — mean average precision, F1 score — which do not map cleanly onto the operational questions a quality organization actually has to answer: *if we deploy this, how many bad parts reach the customer, and how much good product do we needlessly scrap?*

---

## Why This Matters to a Quality Engineer

This study was deliberately structured around the two numbers that matter operationally in inspection, rather than a single blended accuracy score:

- **Escape Rate** — the percentage of real defective parts the system incorrectly passes as good. This is the number that drives customer complaints, warranty claims, and field failures. It is the most safety- and reputation-critical metric in the study.
- **False Rejection Rate** — the percentage of real good parts the system incorrectly flags as defective. This is a cost and throughput problem: unnecessary scrap, rework, and lost production time.

A model can post excellent overall accuracy while making either of these errors at an unacceptable rate. Treating accuracy as the headline metric — as most computer vision evaluations do — can hide exactly the tradeoff a production quality organization needs to see before approving a system for deployment.

---

## Why This Is Different From a Standard Computer Vision Project

This is not a defect-detection accuracy benchmark, and it was not designed to produce the best-performing model. It was designed as a **measurement systems validation study**, applying the same skepticism a Quality Engineer would apply before trusting any new inspection tool — real or AI — in production:

- Every condition was run **five times with different random seeds**, because a single training run is not sufficient evidence for a decision that affects defect escape risk. This mirrors the logic of a Gage R&R study: one measurement is an anecdote, repeated measurement is evidence.
- Results are reported as **mean ± standard deviation**, not a single number, because run-to-run consistency is itself part of whether a system can be trusted.
- An explicit **collapse criterion** was tracked throughout the study — instances where a model degenerated into predicting only one class — because a model that is "accurate" by accident, through a degenerate shortcut, is not a trustworthy measurement system regardless of its reported score.
- The question driving the analysis was never "is this accurate," but **"holding the amount of real defect data fixed, what does adding synthetic data actually change about how the system fails?"**

---

## Research Question

**How does VLM-guided synthetic defect augmentation affect the trustworthiness of AI visual inspection systems when only limited real defect data is available?**

This study deliberately does not ask "does synthetic data improve accuracy." It asks a more operationally honest question: at each level of real-data scarcity, does synthetic augmentation make the system more trustworthy — in the specific senses of escape risk, false rejection cost, and run-to-run stability — or does it simply change the system's behavior without making it more deployable?

---

## Study Objectives

1. Generate synthetic surface defect images using a Vision-Language Model to describe real defect characteristics, paired with diffusion-based inpainting to render them onto real "good" parts.
2. Quantify how much real defect data is required before an AI inspection classifier reaches stable, trustworthy performance.
3. Determine whether synthetic data narrows that requirement — and if so, at what cost to escape rate and false rejection rate.
4. Evaluate not just average performance but **run-to-run consistency**, since a system that is right on average but unpredictable run to run is not a system a quality organization can certify.

---

## Methodology Overview

[FIGURE PLACEHOLDER: three-phase study workflow diagram — Phase 1 (Baseline: real data → classifier → fixed test set) → Phase 2 (VLM description → diffusion inpainting → synthetic defects) → Phase 3 (nine mixed-data conditions × five seeds → evaluation against the fixed test set from Phase 1)]
*Figure: High-level study workflow. Each phase builds on the output of the previous one — the test set fixed in Phase 1 is reused unchanged through Phase 3, and the synthetic images generated in Phase 2 are the sole independent variable introduced in Phase 3.*

[FIGURE PLACEHOLDER: end-to-end pipeline diagram — real defect images → VLM description → diffusion inpainting → synthetic defect images → mixed training sets → classifier training → evaluation against fixed real test set]

The study used the MVTec AD `metal_nut` category as its physical part, focusing specifically on **scratch defects** as the target defect type for synthetic generation. All experiments were evaluated against a single, fixed, held-out test set of real images — never synthetic — established once at the start of the study and reused identically across every subsequent experiment, so that every comparison in this study is measured against the same real-world ground truth.

### Three Study Phases

**Phase 1 — Baseline Establishment**
A defect classifier (ResNet18, transfer-learned) was trained on the full available set of real defect images, across all four real defect types present in the dataset (scratch, bent, color, flip). This established the reference point every later experiment is compared against, and fixed the test set used throughout the rest of the study.

**Phase 2 — VLM-Guided Synthetic Defect Generation**
A Vision-Language Model (Qwen2-VL) was shown real scratch defect images and asked to describe their visual characteristics in structured terms (direction, width, depth, length, contrast). These descriptions were aggregated into representative prompts and used to drive a Stable Diffusion inpainting pipeline, which painted synthetic scratch defects onto real "good" part images. Sixty synthetic scratch images were generated. A direct visual comparison against real scratches was conducted before any quantitative use of the synthetic data, following the same principle a Quality Engineer applies before trusting any new measurement source: inspect it before you rely on it.

**Phase 3 — Controlled Mixed-Data Experiments**
Nine training conditions were defined, varying the number of real scratch examples (0, 2, 5, 10, or 18 — the full available set) and whether synthetic scratch images were added on top. Every condition also included the same constant pool of real bent, color, and flip defects, isolating the variable under study to scratch data specifically. Each of the nine conditions was trained five times under different random seeds, for 45 total trained models, all evaluated against the same fixed real-image test set established in Phase 1.

---

## Experimental Design

| Condition | Real Scratches | Synthetic Scratches | Purpose |
|---|---|---|---|
| A | 0 | 60 | Cold-start: synthetic data only, no real signal |
| G | 2 | 0 | Real-data floor at minimal sample size |
| B | 2 | 58 | Same real count as G, synthetic added |
| H | 5 | 0 | Real-data floor at small sample size |
| C | 5 | 55 | Same real count as H, synthetic added |
| I | 10 | 0 | Real-data floor at moderate sample size |
| D | 10 | 50 | Same real count as I, synthetic added |
| E | 18 | 0 | Full real-data baseline |
| F | 18 | 60 | Full real data, synthetic added on top |

Pairing each real-only condition (G, H, I, E) against its matched real-plus-synthetic counterpart (B, C, D, F) isolates exactly what synthetic data adds, or costs, at a fixed amount of real data — rather than conflating "more data" with "synthetic data helps."

It is worth being explicit about a deliberate design choice: the *total* amount of scratch training data is not held constant between a real-only condition and its matched real-plus-synthetic counterpart. Condition G (2 real, 0 synthetic) trains on far fewer total images than condition B (2 real, 58 synthetic). This is intentional, not an oversight. The research question being tested is the practical, real-world decision a quality organization actually faces: *given the real defect data I have, does adding synthetic data on top of it help?* — not the more academic question of whether synthetic images are exactly as informative as real ones on a strict per-image basis. Total dataset size was deliberately allowed to vary because that is how this technique would actually be used in practice; isolating dataset size as a separate variable was outside the scope of this study.

A methodological note on model stability: an initial version of this study trained each condition with a fully fine-tuned classifier, which produced a high rate of training collapse — models degenerating to predicting a single class regardless of input. This was diagnosed as a known failure mode of fine-tuning a large pretrained network on a very small dataset, and was resolved by freezing the pretrained backbone and training only the final classification layer at a reduced learning rate. This fix was verified on a pilot subset before the full study was run, reducing collapse from 70% of pilot runs to 0%. All results reported below are from the stabilized configuration, with zero collapsed runs across all 45 trained models.

---

## Results

[FIGURE PLACEHOLDER: uncertainty plots — mean ± standard deviation for accuracy, escape rate, and false rejection rate, real-only vs. real+synthetic, across all four real-data levels]

[TABLE PLACEHOLDER: full aggregated results table — mean ± std for accuracy, escape rate, false rejection rate, and collapse count, all nine conditions]

The pairwise comparison between matched real-only and real-plus-synthetic conditions, at each real-data level, showed a consistent pattern. Throughout this study, a change is labeled **improved** or **worsened** only if it exceeds roughly half the average standard deviation observed between the two conditions being compared (with a minimum threshold of 2 percentage points); smaller changes are labeled **negligible**, since they fall within the range of normal run-to-run variation already observed across the five seeds and should not be over-interpreted as a real effect.

| Real Scratches | Escape Rate Change | False Rejection Rate Change | Accuracy Change |
|---|---|---|---|
| 2 | −6.7 pts (improved) | +5.8 pts (worsened) | −2.4 pts (worsened) |
| 5 | −5.2 pts (negligible) | +4.4 pts (worsened) | −1.8 pts (negligible) |
| 10 | −7.4 pts (improved) | +6.4 pts (worsened) | −2.6 pts (worsened) |
| 18 (full) | −3.0 pts (negligible) | +6.9 pts (worsened) | −4.2 pts (worsened) |

Across every real-data level tested, adding synthetic scratch data moved the system in the same direction: escape rate held steady or improved, false rejection rate worsened, and overall accuracy did not improve. This same pattern repeating at four independent real-data levels — rather than appearing once — is what separates this from a single noisy result.

The variance analysis showed an equally consistent pattern in run-to-run stability: synthetic data **reduced escape rate variance at every real-data level tested**, meaning the system's defect-catching behavior became more *predictable* run to run. At the same time, it **increased accuracy variance at every level**, and increased false rejection rate variance at the full real-data level specifically. Synthetic data made the system's safety-relevant behavior (escape rate) more consistent, while making its overall performance less consistent.

---

## Discussion

The consistent direction of the escape-rate and false-rejection-rate changes is consistent with a specific underlying mechanism, though this study did not directly test it: synthetic scratch data may have shifted the model's decision boundary to be **more conservative** — more willing to call a part defective when uncertain. One plausible explanation, grounded in an independent observation from Phase 2, is as follows. During visual inspection of the generated images, the synthetic scratches were observed to be visually smoother and lower-contrast than real scratches — described at the time as "airbrushed" relative to the sharper, more irregular real defects. If a model trained partly on these softer synthetic examples became more sensitive to subtle surface irregularities in general, that could account for both observed effects simultaneously: catching slightly more real defects, while also flagging more good parts that have minor, non-defective surface variation.

This explanation should be read as a plausible hypothesis consistent with the available evidence, not as an experimentally verified mechanism. The study measured *what* changed in the model's error profile, consistently, across four independent real-data levels — it did not include an experiment specifically designed to isolate *why*. The visual-softness observation from Phase 2 and the behavioral shift observed in Phase 3 are correlated in time and in plausible direction, but no causal test (for example, deliberately varying synthetic image contrast and re-measuring the effect) was conducted to confirm the link. A reader should treat this section as the study's best informed interpretation, not as a separately demonstrated finding.

The escape-rate variance reduction is the most operationally interesting secondary observation. A production quality organization deciding whether to deploy an AI inspection system cares not only about average escape rate, but about whether that escape rate is reliable from one production run to the next. The data is consistent with synthetic augmentation having value specifically as a *stabilizer* of detection behavior, even in conditions where it did not improve, and in some cases worsened, overall accuracy — though, as above, this study observed and measured that pattern without isolating the specific mechanism that produces it.

---

## Key Findings

1. **Synthetic data did not substitute for real data at any tested level.** At every real-data count studied (2, 5, 10, and 18 real examples), adding synthetic scratches did not produce a model that outperformed its real-only counterpart on accuracy.

2. **Synthetic data consistently shifted — rather than improved — the error profile.** The same tradeoff (escape rate flat-to-improved, false rejection rate worsened) appeared at every real-data level, indicating a systematic behavioral shift rather than noise.

3. **Synthetic data reduced escape rate variability at every real-data level**, suggesting a stabilizing effect on the specific behavior most relevant to customer-facing quality risk, independent of its effect on overall accuracy.

4. **Synthetic data increased accuracy variability at every real-data level**, meaning its net effect on overall model reliability was mixed rather than uniformly positive.

5. **No amount of real data tested eliminated the tradeoff.** Even at the full available real-data count (18 examples), adding synthetic data still increased false rejection rate by the largest margin observed in the study (+6.9 points) — synthetic data's cost did not diminish as real data became more plentiful within the range tested.

---

## Practical Manufacturing Implications

For a quality organization evaluating whether to use synthetic defect data to accelerate AI inspection deployment, this study suggests:

- Within the range of real-data availability tested in this study, synthetic data should not be treated as a way to *reduce* the amount of real defect data that must be collected before deployment — at no tested level did it close the performance gap to the matched real-only condition.
- Synthetic data may be appropriate where a more conservative inspection posture is an acceptable tradeoff — for example, a new product launch phase where the cost of escapes is judged to outweigh the cost of additional manual re-inspection of flagged parts.
- Based on the pattern observed across this study, synthetic augmentation appears better suited as a **deployment-risk management tool** than as a replacement for collecting real manufacturing defect data — a way to deliberately bias an inspection system toward caution during a specific window (such as a launch ramp-up) while real defect data continues to be collected, rather than a way to avoid collecting that real data at all.
- The decision to use synthetic augmentation should be evaluated explicitly against escape rate and false rejection rate separately, not against a single accuracy figure, since this study's accuracy numbers alone would have obscured the actual tradeoff being made.
- Any organization evaluating an AI inspection vendor's synthetic-data claims should ask for run-to-run variance, not just a single reported performance number — a single favorable run in this study would not have been representative of the condition's true behavior.

---

## Engineering Decision Framework

**When should a manufacturer actually consider using VLM-guided synthetic defect data?** The recommendations below are grounded directly in this study's findings and should be read as a starting point for evaluation, not a validated deployment policy.

| Manufacturing Scenario | Recommendation | Rationale (from this study) |
|---|---|---|
| **No real defects available yet** (new part, no failure history) | Use with caution, as a temporary bridge only | This study did not find synthetic-only training (0 real examples) to be a reliable substitute for real data; treat it as a stopgap while real data is collected, not a long-term solution |
| **Early product launch / ramp-up** | Reasonable candidate use case | A more escape-averse, false-rejection-tolerant posture — the pattern observed in this study — may be an acceptable tradeoff when escape cost is judged higher than the cost of manual re-inspection |
| **Mature production with an established defect history** | Limited additional value observed | At the highest real-data level tested (18 examples), synthetic augmentation still increased false rejection rate without improving accuracy — its benefit did not grow as real data became more available within the tested range |
| **Safety-critical manufacturing** | Do not substitute for real validation data | This study's escape-rate improvements were modest and not formally statistically confirmed (five seeds per condition); a safety-critical program should not rely on synthetic augmentation in place of rigorous real-world validation |
| **High-cost scrap / low tolerance for false rejection** | Approach with caution | False rejection rate worsened at every real-data level tested in this study; an environment where scrap cost is the dominant concern is the scenario least supported by these findings |

---

## Limitations

- **Single defect type.** Synthetic generation and all conclusions in this study apply specifically to scratch defects on one part category. The findings should not be assumed to generalize to other defect types (bent, color, flip) or other part geometries without independent testing.
- **Five seeds per condition.** This is sufficient to observe consistent directional patterns and is a substantial improvement over a single-run evaluation, but it is not a large enough sample for formal statistical significance testing. Findings are reported as directional and consistent, not statistically definitive.
- **Simplified defect mask placement.** Synthetic defect placement used randomized geometric masks rather than the pixel-precise ground-truth defect masks used in the reference academic methodology (SynSur), which may understate how well-targeted synthetic generation could perform with more precise masking.
- **Small absolute dataset size.** The underlying real dataset (MVTec `metal_nut`) is a controlled academic benchmark with a limited number of real defect images available in any category, which constrains how large the "full real data" condition in this study could be.
- **One classifier architecture.** All conclusions are specific to a frozen-backbone ResNet18 classifier. Behavior may differ for other architectures, including the kind of zero-shot VLM-based inspection approaches discussed in this study's motivating literature.

---

## Future Work

- Extend the same controlled methodology to the remaining defect types in the dataset (bent, color, flip) to test whether the observed conservative-shift pattern is specific to scratch-type surface defects or generalizes across defect categories.
- Repeat the study using pixel-precise ground-truth defect masks for synthetic generation, to test whether more accurate defect placement changes the magnitude or direction of the observed tradeoff.
- Test whether the synthetic-induced conservative shift can be deliberately tuned — for example, by adjusting the proportion of synthetic to real data — to hit a specific target escape rate or false rejection rate, rather than treating the shift as a fixed side effect.
- Investigate classifier decision-threshold tuning directly, rather than relying solely on the default classification threshold used in this study. A manufacturer may intentionally select a different operating point along the escape-rate/false-rejection-rate tradeoff curve depending on the acceptable cost of an escape versus the acceptable cost of a false rejection for a given part or program — and synthetic data's effect at a deliberately chosen threshold may differ from its effect at the default threshold evaluated here.
- Evaluate a zero-shot VLM-based inspection approach directly (rather than a fine-tuned classifier) under the same multi-seed, escape-rate-and-false-rejection-rate framework, to compare against the fine-tuning-based results in this study.

---

## What This Study Demonstrates

Synthetic defect data, as generated and tested in this study, did not simply make the inspection system better or worse. It systematically changed *how* the system fails. Across every real-data level investigated:

- Escape rate was reduced or held steady — never worsened
- False rejection rate increased consistently
- Escape-rate variability across training runs was reduced, making that specific behavior more predictable
- Overall accuracy variability increased, and overall accuracy itself did not improve, and in most conditions slightly worsened

Taken together, these results indicate that VLM-guided synthetic defect augmentation, within the scope of this study, **shifted the operating point of the inspection system** — trading false rejections for a more conservative, escape-averse posture — rather than replacing the need for additional real manufacturing defect data. This is the central takeaway: synthetic data acted as a behavioral lever on an existing system, not as a substitute for the real-world evidence that system was missing.

---

## Key Contributions

This study is best understood as an applied extension of existing synthetic-defect-generation research (notably the SynSur pipeline), not an independent invention of the underlying generation technique. Its original contribution is in *how* that technique was evaluated:

- **Reframing synthetic defect generation as a Quality Engineering question rather than a computer vision benchmark.** Published evaluations of synthetic defect data typically report mean average precision or F1 score. This study instead evaluates escape rate and false rejection rate separately — the two metrics that map directly onto manufacturing risk and cost, and that a single blended accuracy score can obscure.
- **Introducing repeated, multi-seed evaluation to a question usually answered with a single training run.** Running every condition five times surfaced a training-stability problem (Section: Experimental Design) that a single-run evaluation would have hidden entirely, and made it possible to report run-to-run variance as a finding in its own right, not just an average result.
- **Demonstrating that synthetic augmentation changes inspection behavior rather than replacing real defect data.** The consistent escape-rate/false-rejection-rate tradeoff observed across four independent real-data levels is a more specific and more operationally useful finding than a simple "works" or "doesn't work" verdict.
- **Translating the result into a practical engineering decision framework** (introduced earlier, following the Practical Manufacturing Implications section), rather than leaving the reader with a set of metrics and no guidance on when, if ever, this technique is appropriate to use.

---

## Repository Structure

[PLACEHOLDER — to be finalized in a later deliverable]

---

## Reproducing the Study

[PLACEHOLDER — to be finalized in a later deliverable]

---

## References

- Kühn, P. J., Pommeranz, M., Kuijper, A., & Sinha, S. N. (2026). *SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection.* arXiv:2604.26633.
- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

---

*This study was conducted as an applied Quality Engineering research project investigating the operational implications of AI-generated synthetic training data for manufacturing visual inspection systems.*
