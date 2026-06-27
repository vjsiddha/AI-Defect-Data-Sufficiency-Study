# How Does VLM-Guided Synthetic Defect Augmentation Change AI Inspection Performance Under Limited Real Defect Data?

### A Quality Engineering Research Study on Synthetic Defect Generation for Manufacturing Visual Inspection

---

## Abstract

Manufacturers adopting AI-assisted visual inspection face a structural data problem: the better a production line performs, the fewer real defective parts it produces, leaving little data to train the very system meant to catch defects. Synthetic defect generation — using generative AI to manufacture artificial defect images — has been proposed as a solution, and recent academic work (notably the SynSur pipeline) has demonstrated that vision-language models (VLMs) can guide diffusion-based image generation to produce plausible industrial surface defects. However, published evaluations of this technique are framed in machine learning terms (mean average precision, F1 score) that do not directly answer the question a Quality Engineer must answer before approving such a system for production: *does this change how many defective parts reach the customer, and how much good product gets needlessly scrapped?*

This study investigated that question directly. A Vision-Language Model (Qwen2-VL) was used to describe real scratch defects on a metal fastener (MVTec AD `metal_nut` category), and those descriptions were used to drive a diffusion-based inpainting pipeline that generated sixty synthetic scratch defect images. Nine experimental conditions were defined, independently varying the amount of real scratch data (0, 2, 5, 10, or 18 examples) and whether synthetic scratch data was added on top, with every condition also including a constant pool of real bent, color, and flip defects. Each condition was trained five times under different random seeds — forty-five total trained classifiers — and evaluated against a single, fixed, real-image test set established once at the start of the study.

Within the range of real-data availability tested, synthetic augmentation did not function as a substitute for real defect data at any level: no real-plus-synthetic condition outperformed its matched real-only counterpart on accuracy. Instead, a consistent behavioral pattern emerged across every real-data level studied: escape rate (real defects missed) was reduced or held steady, false rejection rate (good parts wrongly flagged) increased, and run-to-run escape-rate variability decreased even as overall accuracy variability increased. The interpretation offered in this report is that synthetic augmentation shifted the classifier's decision boundary toward a more conservative, escape-averse operating point — a change in *what kind of mistakes the system makes*, not an improvement in its underlying ability to discriminate defective from non-defective parts. This finding has direct, practical implications for how a manufacturing quality organization should — and should not — use synthetic defect data when planning an AI inspection deployment.

---

## 1. Introduction

The integration of artificial intelligence into manufacturing quality inspection has accelerated rapidly, driven by advances in computer vision and, more recently, by multimodal models capable of combining visual perception with natural language reasoning. Much of the public discussion of this technology — in vendor marketing, conference talks, and increasingly in peer-reviewed literature — centers on a single number: how accurate is the model. This report argues, and demonstrates empirically, that accuracy alone is an insufficient and at times misleading basis for a manufacturing organization to decide whether and how to deploy an AI-assisted inspection system, particularly when that system has been trained partly on synthetic data.

The investigation documented here was conducted as an applied Quality Engineering research project. It treats an AI inspection classifier the way a Quality Engineer would treat any new measurement instrument introduced to a production line: as a system whose output cannot be trusted until its behavior has been characterized under repeated, controlled conditions, and whose accuracy figure — however favorable — is not by itself sufficient evidence of trustworthiness. This framing is the throughline of the entire study and is reflected in every methodological choice described in Section 9 onward.

---

## 2. Manufacturing Background and Motivation

### 2.1 The Defect Data Scarcity Problem

A central operational goal of quality engineering is to minimize the defect rate of a production process. This goal is, by its nature, in direct tension with any data-driven approach to defect detection: a well-controlled process produces few defective parts, and few defective parts means little data with which to train a model intended to recognize them. This tension is sharpest during a new product launch or process change, when historical defect data does not yet exist at all but the need for reliable inspection coverage is highest — precisely the moment a quality organization can least afford an inspection system with unknown or unproven behavior.

### 2.2 Why Traditional Data Augmentation Does Not Solve This Problem

Computer vision practice has long relied on data augmentation — geometric and photometric transformations such as rotation, flipping, cropping, and brightness adjustment — to expand the effective size of a training set. These techniques are well understood, computationally inexpensive, and standard practice. However, they operate only on the *quantity* of training images derived from existing examples; they do not introduce new defect *morphology*. A rotated image of a given scratch remains, in every meaningful sense, the same scratch viewed from a different angle. Traditional augmentation cannot help a model learn to recognize a defect shape, severity, or spatial pattern it has not already been shown in some form. When the underlying problem is that too few *distinct* real defect examples exist — as is the case in the scenario this study investigates — traditional augmentation does not address the actual gap.

### 2.3 The Proposed Solution: Generative Synthetic Defect Data

Generative AI offers, in principle, a different kind of solution: the ability to synthesize plausible new defect *instances*, not merely new views of existing ones. This is an active area of both academic and applied research. The SynSur pipeline (Kühn et al., 2026) is among the most directly relevant recent examples, combining a vision-language model — used to extract structured descriptions of real defect characteristics — with a diffusion-based generative model that paints synthetic defects onto clean part images guided by those descriptions. This study adopts a simplified version of the same conceptual pipeline, described in full in Section 10, and is best understood as an applied replication and quality-engineering-focused extension of that line of work, rather than as an independent invention of the underlying technique.

---

## 3. Related Work

Three pieces of existing work directly motivate this study's design. The MVTec AD dataset (Bergmann et al., 2019) is the standard benchmark for industrial defect detection research and is the sole source of real data used here, specifically its `metal_nut` category. Earlier synthetic defect generation methods, such as Cut-Paste-style approaches that directly overlay image patches, are computationally simple but prone to visually unnatural results — unrealistic borders, inconsistent lighting, texture mismatches with the surrounding material. The SynSur pipeline (Kühn et al., 2026) addresses this by pairing a vision-language model — used to extract structured, human-readable descriptions of real defect characteristics — with diffusion-based inpainting conditioned on those descriptions, aiming for synthetic defects that are visually and texturally consistent with their surroundings. This study adopts a simplified version of that same VLM-plus-diffusion pipeline, described in full in Section 10.

---

## 4. Research Gap

The published evaluations of synthetic defect generation methods that motivated this study, including SynSur, report results using standard machine learning detection metrics: mean average precision (mAP), F1 score, and related measures. These are appropriate metrics for comparing model architectures and generation methods against one another, but they do not directly correspond to the operational decision a manufacturing quality organization must make. Specifically, they do not separate the two distinct, asymmetric costs that an inspection error can produce:

- A **missed defect** (escape) reaches the customer, with consequences ranging from a warranty claim to, in safety-relevant applications, a field failure.
- A **false alarm** (false rejection) discards or reworks a good part, with consequences limited to cost and throughput.

A single blended metric can report a favorable score while concealing an unacceptable rate of the more severe error type. This is the specific gap this study addresses: not whether synthetic defect data can improve a detection metric, but what it does, specifically and separately, to escape rate and false rejection rate — the two numbers that actually determine whether a quality organization should trust a given inspection configuration in production.

---

## 5. Research Question

**How does VLM-guided synthetic defect augmentation change AI inspection performance — specifically escape rate, false rejection rate, and run-to-run stability — when only limited real defect data is available?**

This study deliberately frames its central question around behavioral change rather than performance improvement. It does not ask whether synthetic data makes a classifier more accurate; it asks what synthetic data changes about how that classifier fails, and whether that change is consistent enough, across different amounts of available real data, to draw a practical conclusion from.

---

## 6. Study Objectives

1. Construct a VLM-guided synthetic defect generation pipeline for a real manufacturing defect type (surface scratches on a metal fastener), and assess the visual plausibility of its output before any quantitative use.
2. Establish a fixed, real-image evaluation baseline, ensuring every subsequent experimental comparison in the study is measured against the same ground truth.
3. Systematically vary the amount of real defect data available for training, both with and without synthetic augmentation, to isolate the specific effect of adding synthetic data at each level of real-data scarcity.
4. Evaluate every experimental condition using repeated, independently seeded training runs, reporting mean and variability rather than a single result, consistent with a measurement-systems-validation approach to evaluating a new inspection tool.
5. Interpret the resulting evidence specifically in terms of escape rate and false rejection rate — the two metrics that map directly onto manufacturing quality risk and cost — rather than accuracy alone.

---

## 7. Methodology

### 7.1 Overview

The study was conducted in three sequential phases, each building directly on the output of the one before it. Phase 1 established the real-data baseline and the fixed evaluation set used throughout the remainder of the study. Phase 2 produced the synthetic defect images used as the study's independent variable. Phase 3 conducted the controlled experiment itself: training and evaluating classifiers under nine combinations of real and synthetic data, each repeated across five random seeds.

[FIGURE PLACEHOLDER: full study workflow diagram, annotated with phase boundaries and key artifacts passed between phases — the fixed test set from Phase 1, and the synthetic image set from Phase 2]

### 7.2 Dataset

All real images were drawn from the MVTec AD `metal_nut` category, which includes a "good" (non-defective) image set and four defect types: scratch, bent, color, and flip. This study's synthetic generation effort, and its primary experimental manipulation, focused specifically on the scratch defect type. The bent, color, and flip defect images were retained as a constant, unmodified component of the training data across every experimental condition, so that the manipulation under study — varying real and synthetic *scratch* data — would not be confounded with changes to the rest of the defect distribution.

### 7.3 Evaluation Metrics

Three metrics were tracked throughout the study, with an explicit priority ordering reflecting Quality Engineering practice rather than machine learning convention:

- **Escape Rate** (highest priority): the proportion of real defective test images incorrectly classified as good. This is the metric most directly tied to customer-facing quality risk.
- **False Rejection Rate** (second priority): the proportion of real good test images incorrectly classified as defective. This is the metric most directly tied to manufacturing cost and throughput.
- **Accuracy** (reported for completeness, treated as secondary): the overall proportion of correct classifications across both classes. Reported throughout, but not treated as the basis for any conclusion in isolation, since it can mask the asymmetric cost structure between the two error types above.

A fourth, binary diagnostic — **training collapse** — was introduced after an early version of the study revealed a stability problem in the training process itself, described in Section 13.4. A run was flagged as collapsed if the resulting model predicted one class for nearly the entirety of the test set (operationally defined as fewer than 5% of predictions falling to the other class), since such a model has not meaningfully learned to discriminate between classes regardless of the accuracy figure it happens to produce.

---

## 8. Experimental Design

### 8.1 Conditions

Nine experimental conditions were defined, organized into four matched pairs (plus one unpaired cold-start condition) so that the specific effect of adding synthetic data could be isolated at each fixed level of real-data availability:

| Condition | Real Scratches | Synthetic Scratches | Role |
|---|---|---|---|
| A | 0 | 60 | Cold-start: synthetic data only |
| G | 2 | 0 | Real-only floor, paired with B |
| B | 2 | 58 | Real (2) + synthetic, paired with G |
| H | 5 | 0 | Real-only floor, paired with C |
| C | 5 | 55 | Real (5) + synthetic, paired with H |
| I | 10 | 0 | Real-only floor, paired with D |
| D | 10 | 50 | Real (10) + synthetic, paired with I |
| E | 18 | 0 | Full real-data baseline, paired with F |
| F | 18 | 60 | Full real data + synthetic, paired with E |

Every condition's defective training set also included the full, unchanged pool of real bent, color, and flip defects (48 images), so the only variable manipulated between conditions was the composition of the scratch-defect portion of the training data.

### 8.2 A Note on Total Dataset Size

It is important to state explicitly that the *total* amount of scratch-defect training data was not held constant between a real-only condition and its matched real-plus-synthetic counterpart — for example, Condition G trains on 2 total scratch images, while Condition B trains on 60. This was a deliberate design choice, not an oversight. The research question under investigation is the practical decision actually faced by a manufacturing quality organization: *given the real defect data already in hand, does adding synthetic data on top of it change the system's behavior, and if so, how?* This is distinct from, and was prioritized over, the more academic question of whether synthetic images are exactly as informative as real ones on a strict per-image basis — a question this study's design does not directly answer, since it would require holding total dataset size constant while varying only the real/synthetic ratio, which was outside this study's scope.

### 8.3 Repeated Measurement Design

Each of the nine conditions was trained five times, using five fixed random seeds (42, 7, 123, 456, 999) controlling both model weight initialization and training-data shuffle order. This produced forty-five independently trained models in total. Results are reported as the mean and standard deviation across these five runs per condition, rather than as a single value, for the reason stated in Section 6, Objective 4: a single training run does not constitute sufficient evidence for a decision with escape-risk implications, in the same way a single measurement does not constitute a valid Gage R&R study.

---

## 9. Phase 1 — Baseline Dataset and Classifier

### 9.1 Establishing the Fixed Test Set

A held-out test set was constructed once, at the outset of the study, by randomly partitioning the available real "good" and real defective images (approximately 30% reserved for testing, with a minimum floor applied to ensure adequate test set size for the smaller defect categories). This partition was fixed using a defined random seed and was never altered, retrained on, or regenerated for any subsequent experiment. Every result reported anywhere in this study — across all three phases — is measured against this same fixed set of real images, which were never used to generate synthetic data and never used in training under any condition.

### 9.2 Baseline Classifier

An initial classifier was trained using the complete pool of available real defect images (all four defect types) to establish a reference point and to validate the overall training and evaluation pipeline before any synthetic data was introduced. This baseline used a ResNet18 architecture, transfer-learned from ImageNet-pretrained weights, and is referred to throughout this report as the basis for the Phase 3 architecture choice, though the Phase 3 experiments use a modified training configuration described in Section 11.4.

---

## 10. Phase 2 — VLM-Guided Synthetic Defect Generation

### 10.1 Defect Description via Vision-Language Model

A pretrained Vision-Language Model (Qwen2-VL, 2B-parameter instruction-tuned variant) was used to generate structured natural-language descriptions of real scratch defect images. The model was prompted to describe each defect's direction, width, depth, length, and contrast, rather than to produce a free-form description, in order to obtain consistent, mergeable output across multiple images.

An initial round of this process, conducted on eight images with a prompt that included a worked example of the desired output format, produced eight nearly identical descriptions — a known failure mode in which a small VLM echoes the example provided in its prompt rather than independently characterizing each input image. The prompt was revised to remove the example output and instead instruct the model to identify what was *distinctive* about each specific image, and the number of described images was increased to eighteen. This revised approach produced five distinct description patterns across the eighteen images, which were found, on inspection, to correspond to two coherent underlying clusters of real scratch characteristics — a "narrow, shallow" cluster (the majority of images) and a "moderate width and depth" cluster (a minority). This clustering was treated as a genuine signal about the underlying real data, consistent with the dataset's defects having been introduced through a controlled process during the dataset's original creation, rather than as a continuing prompting failure.

### 10.2 Prompt Construction

Rather than averaging all eighteen descriptions into a single prompt — which an initial attempt showed could produce internally contradictory instructions (for example, specifying both "narrow" and "moderate" width simultaneously) — two separate prompts were constructed, one per identified cluster, each internally consistent and built from the majority attribute values observed within that cluster. During image generation, each synthetic image was produced using one of the two prompts, selected with a probability matching that cluster's proportion in the real data (approximately 78% "shallow" cluster, 22% "moderate" cluster), so the synthetic dataset's overall composition would reflect the same proportions observed in the real defect population.

### 10.3 Diffusion-Based Image Generation

Sixty synthetic scratch images were generated using a Stable Diffusion inpainting pipeline. Each synthetic image was produced by taking a real "good" (non-defective) part image, applying a randomized elliptical mask approximating a plausible scratch location and orientation, and inpainting a defect into the masked region guided by the appropriate cluster prompt from Section 10.2. This represents a deliberate simplification relative to the reference SynSur methodology, which derives defect placement from pixel-precise ground-truth annotation masks; this study instead used randomized geometric masks, a limitation discussed further in Section 16.

### 10.4 Visual Validation

Before any synthetic image was used in a quantitative experiment, a direct side-by-side visual comparison was conducted between real and synthetic scratch examples. This step is treated, throughout this study, as methodologically equivalent to a Quality Engineer's practice of inspecting a new measurement source before trusting its output — synthetic data, like any new instrument, was not assumed valid by default.

[FIGURE PLACEHOLDER: side-by-side panel of real vs. synthetic scratch images]

This inspection identified a consistent qualitative difference: synthetic scratches were judged to follow plausible machining patterns and lighting, with overall color and texture distribution closely matching real examples, but were also consistently described as visually "smoother," more uniform, and lower-contrast than real scratches — appearing to blend into the surrounding surface texture rather than visually disrupting it the way real scratches did. Real scratches, by contrast, displayed more irregular width, sharper onset and termination points, and greater local contrast against the surrounding material. This observation — made independently of, and prior to, any of the Phase 3 quantitative results — is referenced directly in this report's interpretation of the Phase 3 findings (Section 14).

---

## 11. Phase 3 — Data Sufficiency Experiments

### 11.1 Training Procedure

Each of the forty-five training runs (nine conditions × five seeds) used the same procedure: a ResNet18 classifier, transfer-learned from ImageNet-pretrained weights, trained for fifteen epochs to distinguish "good" from "defective" parts, then evaluated once against the fixed test set established in Phase 1.

### 11.2 Initial Stability Problem

An initial execution of the full Phase 3 experiment, using a fully fine-tuned ResNet18 (all layers trainable) with standard unweighted cross-entropy loss, produced results that on inspection were not usable as evidence. Across the forty-five runs, a substantial proportion — including runs within the same experimental condition, differing only in random seed — collapsed into a degenerate state in which the model predicted only one class for nearly every test image, regardless of input. This was diagnosed as a consequence of fine-tuning an eleven-million-parameter network on training sets as small as fifty defective images: the optimizer had sufficient freedom to find a trivial, always-predict-one-class shortcut that minimized loss without learning any real discriminative signal, particularly under class imbalance between the "good" and "defective" categories.

A first remediation attempt — introducing class-weighted loss to penalize the trivial shortcut — was tested and found insufficient; the collapse rate across the full forty-five-run study remained at approximately 40%, and several conditions collapsed at a *higher* rate than before weighting was introduced. This result is reported here because it is itself informative: it demonstrates that the underlying instability was not primarily an artifact of class imbalance, but of the model's parameter count relative to the available training data.

### 11.3 Stabilization

The training configuration was revised to freeze the pretrained ResNet18 backbone entirely, training only the final classification layer — reducing the trainable parameter count from approximately eleven million to approximately one thousand — combined with a reduced learning rate (0.0003, down from 0.001). This change was first verified on a ten-run pilot subset (two of the conditions previously most prone to collapse, across all five seeds) before being applied to the full study, in order to confirm the fix before committing further computation to it. The pilot showed zero collapsed runs, compared to seven of ten in the equivalent unweighted, unfrozen configuration. The full forty-five-run study was then re-executed under the stabilized configuration and likewise produced zero collapsed runs across all nine conditions. All results reported in Section 12 are from this stabilized configuration.

### 11.4 Implication of the Stability Finding

This stability problem, and its resolution, is itself a finding relevant to the study's broader Quality Engineering framing, and is treated as such rather than as a methodological footnote to be set aside. A model evaluated through a single training run — the typical practice in many reported AI inspection benchmarks — provides no visibility into whether that run's result reflects stable, repeatable learning or a degenerate shortcut that happened to score well by chance. The forty percent collapse rate observed in this study's initial configuration demonstrates that this is not a hypothetical concern.

---

## 12. Experimental Results

[TABLE PLACEHOLDER: full aggregated results — mean ± standard deviation for accuracy, escape rate, and false rejection rate, all nine conditions, plus collapse count per condition]

[FIGURE PLACEHOLDER: uncertainty plots — mean ± standard deviation for accuracy, escape rate, and false rejection rate, real-only vs. real-plus-synthetic, plotted against real-data count]

### 12.1 Pairwise Comparison

Comparing each real-only condition against its matched real-plus-synthetic counterpart, holding real-data count fixed, produced the following pattern. A change is labeled **improved** or **worsened** only when it exceeds roughly half the average standard deviation observed between the two compared conditions (with a minimum threshold of two percentage points); smaller changes are labeled **negligible**, reflecting that they fall within the range of normal run-to-run variation already observed.

| Real Scratches | Escape Rate Change | False Rejection Rate Change | Accuracy Change |
|---|---|---|---|
| 2 | −6.7 pts (improved) | +5.8 pts (worsened) | −2.4 pts (worsened) |
| 5 | −5.2 pts (negligible) | +4.4 pts (worsened) | −1.8 pts (negligible) |
| 10 | −7.4 pts (improved) | +6.4 pts (worsened) | −2.6 pts (worsened) |
| 18 (full) | −3.0 pts (negligible) | +6.9 pts (worsened) | −4.2 pts (worsened) |

Two patterns hold consistently across all four real-data levels: false rejection rate worsened at every level, and escape rate never worsened at any level (improving at two of the four levels and showing a negligible change at the other two). Accuracy did not improve at any level, and worsened at three of the four.

### 12.2 Variance Analysis

Comparing the standard deviation of each metric, rather than its mean, between matched real-only and real-plus-synthetic conditions revealed a second consistent pattern:

- **Escape rate variance was reduced by the addition of synthetic data at every one of the four real-data levels tested.** The system's defect-catching behavior became more consistent from run to run.
- **Accuracy variance was increased by the addition of synthetic data at every one of the four real-data levels tested.** Overall classification performance became less consistent from run to run.
- **False rejection rate variance showed no meaningful change at three of the four real-data levels**, and increased at the fourth (the full real-data condition).

### 12.3 Summary of the Quantitative Pattern

No single result in this study, taken in isolation, would justify a strong conclusion. The basis for this report's central finding is the *consistency* of the pattern across four independently tested real-data levels and forty-five independently trained models: synthetic augmentation did not improve the classifier's underlying ability to discriminate defective from non-defective parts, but it did consistently and repeatedly change the *type* of error the classifier made, and did so in the same direction every time it was tested.

---

## 13. Discussion

### 13.1 Interpreting the Behavioral Shift

The consistent direction of the escape-rate and false-rejection-rate changes is compatible with a specific underlying mechanism, though this study did not design an experiment to directly test that mechanism in isolation. The interpretation offered here — and it should be read explicitly as an interpretation, not a separately confirmed finding — is that synthetic scratch data shifted the trained classifier's decision boundary toward a more conservative operating point: more willing to flag a part as defective under visual uncertainty.

This interpretation is grounded in the independent visual observation made in Section 10.4, prior to and separate from any quantitative result: the synthetic scratches were judged to be smoother, lower-contrast, and less disruptive to surrounding texture than real scratches. If a classifier trained partly on these visually subtler synthetic examples became correspondingly more sensitive to subtle surface variation in general, this would account for both observed effects simultaneously — a modest improvement in catching real defects, paired with a corresponding increase in false alarms on good parts that exhibit minor, non-defective surface variation. This explanation is plausible and consistent with the available evidence, but it has not been isolated as a causal mechanism; no experiment in this study deliberately varied synthetic image contrast or sharpness and re-measured the effect, which would be required to confirm rather than merely support this interpretation.

### 13.2 The Variance Finding as a Distinct Observation

The reduction in escape-rate variance is reported as a separate observation from the shift in escape-rate mean, because the two are not the same claim and do not necessarily share the same explanation. A plausible account, consistent with but not separately verified by this study's design, is that the additional sixty synthetic images — generated from only two underlying prompt templates — introduced a degree of redundancy into the harder-to-learn minority class (scratch defects) that partially dampened the run-to-run sensitivity of the classifier's decision boundary to the specific, more limited set of real examples available in the smaller real-data conditions. This account would also be consistent with the corresponding increase in accuracy variance, if the same redundancy made the model's behavior on the *majority*, "good"-class examples less consistent in a way that the escape-rate-specific metric does not capture. As with Section 13.1, this is offered as the most plausible interpretation available, not as a demonstrated mechanism.

### 13.3 Why the Pattern's Consistency Matters More Than Any Single Number

The Quality Engineering value of this study's design lies specifically in the repetition of the same directional pattern across four independent real-data levels. Had only one real-data level been tested, an escape-rate improvement of seven percentage points might plausibly be attributed to noise, to the specific seeds drawn, or to an idiosyncrasy of that particular sample size. The same pattern recurring at 2, 10, and (in attenuated form) 5 and 18 real examples — across forty-five total independently trained models — is the basis for treating the behavioral shift as a real, structural effect of the synthetic data rather than an artifact of any single experimental run.

### 13.4 On the Training Instability Finding

The training collapse problem documented in Section 11.2 and its resolution in Section 11.3 are discussed here, separately from the main escape-rate and false-rejection-rate findings, because they constitute an independent contribution of this study. A forty percent collapse rate under a standard, unweighted, fully fine-tuned training configuration — on a dataset and architecture combination not unusual in applied industrial inspection projects — demonstrates concretely why single-run evaluation of a small-sample AI inspection classifier is an insufficient basis for a deployment decision. This finding is offered as a methodological caution applicable beyond the specific synthetic-data question this study investigates.

---

## 14. Practical Manufacturing Implications

The findings of this study suggest the following practical guidance for a quality organization evaluating synthetic defect data as part of an AI inspection deployment strategy:

**Synthetic data should not be treated as a way to reduce the amount of real defect data that must be collected.** At no real-data level tested in this study did adding synthetic data close the performance gap to the matched real-only condition on accuracy. An organization planning to substitute synthetic data for a real-data collection effort, on the expectation that doing so will produce an equally capable system, would not be supported by this study's results.

**Synthetic data appears better suited as a deployment-risk management tool than as a data-collection substitute.** The consistent shift toward a more escape-averse, false-rejection-tolerant operating point suggests a specific, narrower use case: deliberately biasing an inspection system toward caution during a defined window — such as the ramp-up period of a new product launch, when escape risk is judged to be the dominant concern and additional manual re-inspection of flagged parts is operationally tolerable — while real defect data continues to be collected in parallel for longer-term system refinement. This is a materially different recommendation than using synthetic data to avoid collecting real data altogether.

**Evaluation of any AI inspection system, synthetic-data-augmented or otherwise, should report escape rate and false rejection rate separately, not as a single blended accuracy figure.** This study's own accuracy numbers, taken alone, would have obscured the entire tradeoff documented in Section 12 — in several conditions, accuracy actually decreased while escape rate simultaneously improved, a pattern that a single-metric evaluation would report as a net negative despite representing, from a customer-risk perspective, an improvement on the more safety-critical metric.

**Vendor or internal claims about synthetic-data-augmented AI inspection performance should be accompanied by run-to-run variance, not a single reported result.** This study's initial, unstabilized configuration would have produced a single favorable-looking run roughly six times out of ten — a result indistinguishable, without repetition, from a genuinely reliable system.

---

## 15. Engineering Decision Framework

The implications above translate into the following practical guidance for when VLM-guided synthetic defect data is, and is not, an appropriate tool. These recommendations are grounded directly in the findings reported in Section 12 and should be read as a starting point for an organization's own evaluation, not as a validated deployment policy.

| Manufacturing Scenario | Recommendation | Rationale (from this study) |
|---|---|---|
| No real defects available yet (new part, no failure history) | Use with caution, as a temporary bridge only | Synthetic-only training (0 real examples, Condition A) was not found to be a reliable substitute for real data within this study; it is better treated as a stopgap while real data is collected than as a long-term solution |
| Early product launch / ramp-up | Reasonable candidate use case | The more escape-averse, false-rejection-tolerant posture observed in this study may be an acceptable tradeoff when escape cost is judged higher than the cost of manual re-inspection during a launch window |
| Mature production with an established defect history | Limited additional value observed | At the highest real-data level tested (18 examples, Condition F), synthetic augmentation still increased false rejection rate without improving accuracy — its apparent cost did not diminish as real data became more available within the range tested |
| Safety-critical manufacturing | Do not substitute for real validation data | The escape-rate improvements observed in this study were modest in magnitude and were not confirmed with formal statistical testing (Section 16); a safety-critical program should not rely on synthetic augmentation in place of rigorous real-world validation |
| High-cost scrap / low tolerance for false rejection | Approach with caution | False rejection rate worsened at every real-data level tested in this study; an environment where scrap cost is the dominant operational concern is the scenario least supported by these findings |

---

## 16. Limitations

This study's conclusions should be read within the following explicit boundaries:

- **Single defect type.** All synthetic generation and all Phase 3 experimental conclusions concern scratch defects specifically, on a single part category (MVTec AD `metal_nut`). No claim in this report should be assumed to extend to the bent, color, or flip defect types present in the same dataset, or to other part geometries, without independent testing.
- **Five seeds per condition.** Five independently seeded training runs per condition is a substantial improvement over the single-run evaluation common in much applied AI inspection work, and was sufficient to observe a directionally consistent pattern across four real-data levels. It is not, however, a sample size sufficient for formal statistical significance testing, and this report's findings are presented as directionally consistent observations rather than statistically confirmed effects.
- **Simplified synthetic defect placement.** Synthetic defects were placed using randomized geometric masks rather than the pixel-precise, ground-truth-derived defect masks used in the reference SynSur methodology. This is a meaningful simplification that may understate the quality achievable with more precise defect placement, and the visual softness noted in Section 10.4 may be partially attributable to this simplification rather than solely to the underlying generative model.
- **Small absolute dataset size.** MVTec AD is a controlled academic benchmark with a limited number of real defect images available per category; the study's "full real data" condition (18 scratch examples) reflects the actual ceiling of available real data in this dataset, not a deliberately chosen experimental limit, and may not be representative of real-data availability in other manufacturing contexts.
- **Single classifier architecture.** All Phase 3 conclusions are specific to a frozen-backbone ResNet18 classifier under the specific training configuration described in Section 11.3. The behavior documented here may not generalize to other architectures, including the zero-shot VLM-based inspection approaches referenced in this study's motivating literature, which operate on a fundamentally different principle (in-context visual reasoning rather than learned, fixed decision boundaries) and were not directly tested in this study.

---

## 17. Threats to Validity

**Internal validity.** The primary internal validity concern is the training instability documented in Section 11.2, which was identified and addressed prior to the collection of any result reported in Section 12; all reported results derive from the stabilized configuration with a verified zero collapse rate. A secondary internal validity consideration is the construction of the two synthetic-image generation prompts from clustering eighteen VLM-generated descriptions (Section 10.1) — this clustering was based on a single descriptive attribute (depth) chosen because it showed the clearest two-group separation in the data, and an alternative clustering choice might have produced different synthetic image characteristics and, potentially, different downstream results.

**Construct validity.** Escape rate and false rejection rate, as measured in this study, are proxies for real-world customer risk and scrap cost respectively; the actual operational cost of an escape or a false rejection depends on factors outside this study's scope (part criticality, downstream inspection redundancy, rework cost) and was not modeled. The "trustworthiness" framing used in this report's title and throughout is a qualitative interpretation of the escape-rate, false-rejection-rate, and variance evidence collected — this study did not directly survey or measure human trust in the resulting systems, and the term should be understood as shorthand for this evidentiary basis, not as an independently measured construct.

**External validity.** The MVTec AD benchmark, while widely used in industrial anomaly detection research, is a curated academic dataset; the controlled nature of its defect introduction (noted in Section 10.1 as a likely explanation for the low diversity observed in real scratch descriptions) may not be representative of the more variable defect populations found on an active production line. Findings should be treated as evidence from a controlled, bounded test case, not as a general claim about synthetic data's behavior across manufacturing contexts.

---

## 18. Future Work

- **Extend to additional defect types.** Repeating this study's methodology on the bent, color, and flip defect types present in the same dataset would establish whether the conservative-shift pattern observed here is specific to scratch-type surface defects or reflects a more general property of VLM-guided synthetic augmentation.
- **Use pixel-precise defect masks.** Repeating the synthetic generation process using ground-truth-derived defect masks, rather than the randomized geometric masks used in this study, would test whether more precise defect placement changes the magnitude or direction of the observed escape-rate/false-rejection-rate tradeoff.
- **Tune the synthetic-to-real data ratio deliberately.** Rather than treating the conservative shift as a fixed side effect, future work could test whether the magnitude of the shift can be deliberately controlled by varying the proportion of synthetic to real data, with the goal of hitting a specific target escape rate or false rejection rate chosen by a quality organization.
- **Investigate classifier decision-threshold tuning.** This study evaluated each trained classifier at a single default decision threshold. A manufacturer may instead choose to operate at a different point along the escape-rate/false-rejection-rate tradeoff curve, depending on the relative cost of an escape versus a false rejection for a specific part or program; synthetic data's effect at a deliberately chosen, non-default threshold may differ from the effect documented in this report and warrants direct investigation.
- **Evaluate zero-shot VLM-based inspection directly.** This study's Phase 3 experiments concern a fine-tuned classifier. A direct evaluation of a zero-shot VLM-based inspection approach — prompting a vision-language model to make pass/fail judgments directly, without a separate fine-tuning step — under the same multi-seed, escape-rate-and-false-rejection-rate framework would provide a useful point of comparison against this study's fine-tuning-based results.

---

## 19. Lessons Learned

The points below are presented as lessons drawn from the experience of conducting this investigation, distinct from the Future Work items above, which describe additional experiments this study did not perform. These are conclusions about *how to evaluate* AI-assisted inspection systems, learned in the course of generating the evidence reported in Section 12.

- **Accuracy alone was not sufficient to evaluate the inspection systems in this study.** Several of this study's most operationally relevant findings — most notably, escape rate improving while accuracy simultaneously worsened — would have been invisible to an evaluation that reported only a single blended accuracy figure.
- **Multi-seed evaluation revealed behavior that a single training run would have hidden entirely.** The training collapse documented in Section 11.2 affected a substantial fraction of runs under the initial configuration; a single-run evaluation could easily have reported either a falsely reassuring or a falsely alarming result, depending on which seed happened to be used.
- **Synthetic defect data changed inspection behavior; it did not simply improve or degrade performance.** Treating the central question as "does it work" rather than "what does it change" was necessary to arrive at this study's actual finding — a single pass/fail framing would have forced an artificially binary conclusion onto a result that was genuinely a tradeoff.
- **Model stability had to be validated before any result could be meaningfully interpreted.** The decision to pause the original experimental plan and investigate the cause of the training collapse, rather than proceeding to interpret results from an unstable configuration, was necessary for every subsequent finding in this report to be trustworthy.
- **Synthetic data, in this study, complemented rather than replaced real manufacturing defect data.** This was not the study's starting assumption; it is a conclusion reached only after observing the same pattern recur across four independent real-data levels.

---

## 20. Conclusion

This study set out to answer a specific, practically motivated question: when real defect data is scarce, does VLM-guided synthetic defect augmentation make an AI visual inspection system more trustworthy, or does it simply change the system's behavior? The evidence collected across forty-five independently trained classifiers, spanning four levels of real-data availability, supports a clear answer within the scope tested: within the experimental conditions evaluated, synthetic augmentation did not substitute for the information provided by real manufacturing defect data, but it did produce a consistent, repeatable shift in the classifier's operating behavior — reducing or maintaining escape rate, increasing false rejection rate, stabilizing escape-rate variance, and destabilizing overall accuracy variance, at every real-data level studied within this investigation.

This is not the same finding as "synthetic data improves AI inspection," nor is it the same finding as "synthetic data does not work." It is a more specific and, this report argues, more operationally useful finding: synthetic defect data acted as a lever on an existing system's decision boundary, with a direction and consistency clear enough to be characterized, but a magnitude and underlying mechanism that this study's design can describe but not fully explain. A manufacturing quality organization considering this technology should evaluate it accordingly — not as a replacement for the real-world evidence an inspection system needs, but as a tool that can deliberately, and predictably, bias that system's behavior in a chosen direction once the real-data foundation it modifies is already in place.

---

## References

1. Kühn, P. J., Pommeranz, M., Kuijper, A., & Sinha, S. N. (2026). *SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection.* arXiv:2604.26633.
2. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

---

*This report is the second deliverable of an applied Quality Engineering research project investigating the operational implications of AI-generated synthetic training data for manufacturing visual inspection systems. It expands on, and should be read alongside, the project's README.md, which presents the same study in a shorter, repository-facing format.*
