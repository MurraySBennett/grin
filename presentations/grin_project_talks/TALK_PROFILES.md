# GRIN Talk Profiles

This file describes the actual rhetorical routes. These should become route
presets in the eventual HTML deck.

## 1. Mathematical Psychology Conference

Likely audience: GRT/SFT/modeling people, quantitatively fluent, skeptical of
black-box inference but sympathetic to formal tools.

Primary job: show that GRIN is a valid amortized inference workflow for a known
model class, and that the paper understands identifiability.

Tone: technical, precise, no hype.

What to emphasize:

- Gaussian GRT parameterization and identified 12-parameter structure.
- The simulator and training prior.
- Recovery split by parameter family.
- Calibration and the need for held-out scale correction.
- Fairness and limitations of the MLE/R package comparisons.
- Correlation identifiability frontier.
- DS non-identifiability as inherited from GRT, not discovered by GRIN.

What to avoid:

- "AI solves GRT."
- Treating DS robustness as a new result.
- Overselling real-data examples.

5-minute version:

1. Claim: amortized inference for Gaussian GRT.
2. Why this matters: optimization cost and convergence.
3. What GRIN returns.
4. Recovery/calibration headline.
5. Speed/comparison headline.
6. Limits: rho and DS.
7. Takeaway: reusable inference layer for existing GRT.

10-minute version:

1. Title and claim.
2. GRT task and constructs.
3. Identification: 12 proportions, 12 identified parameters, uneven information.
4. Existing fitting costs.
5. Simulator and network outputs.
6. Recovery result.
7. Calibration result.
8. Comparison to MLE/mdsdt/grtools.
9. Design envelope/adaptive stopping.
10. Limits and future work.

15-minute version:

Use the 10-minute route plus:

- More detail on the training prior and construct heads.
- Complete-case comparison caveat.
- DS teaching branch if the room is likely to care.
- Future work on richer posterior families, RTs, and group structure.

## 2. Psychonomics / General Cognitive Psychology

Likely audience: cognitive psychologists, mixed modeling fluency, interested in
what the tool lets them do with experiments.

Primary job: show that GRIN makes a useful model practically usable, and explain
why the validation matters without burying the room in parameterization.

Tone: scientific, concrete, application-forward.

What to emphasize:

- Same accuracy can hide different perceptual/decisional explanations.
- Existing tools are useful but costly for intervals, simulations, and adaptive
  use.
- GRIN returns uncertainty and construct probabilities immediately.
- Validation shows strong recovery where the task is informative and honest
  limits where it is not.
- Adaptive stopping and design guidance are the new workflow payoffs.

What to avoid:

- Long derivations.
- Too many package internals.
- DS non-identifiability unless asked.

5-minute version:

1. Same accuracy, different mechanisms.
2. Why GRT is useful but hard to fit repeatedly.
3. GRIN in one sentence.
4. Headline recovery/comparison result.
5. Adaptive stopping payoff.
6. Honest limits.
7. Availability/software close.

10-minute version:

1. Opening vignette.
2. GRT constructs in plain language.
3. Bottleneck of current fitting.
4. GRIN workflow.
5. Recovery and calibration.
6. Comparison to existing tools.
7. Accuracy/design envelope.
8. Adaptive stopping.
9. Real-data compatibility.
10. Limits/takeaway.

15-minute version:

Use the 10-minute route plus:

- More calibration explanation.
- More real-data examples.
- More practical package/web workflow.
- Short future-work slide.

## 3. University-wide Workshop / Across Colleges

Likely audience: intelligent non-specialists, mixed disciplinary backgrounds,
probably interested in AI, measurement, and practical scientific workflows.

Primary job: make the work legible as AI-assisted scientific measurement without
making it sound like a generic AI demo.

Tone: accessible, restrained, outcome-focused.

What to emphasize:

- The general problem: the same observed score can come from different hidden
  processes.
- Scientific models can separate explanations, but they can be expensive to fit.
- Simulation lets us train a fast inverse for a trusted model.
- The tool reports uncertainty, not just a best guess.
- This enables faster design iteration and adaptive data collection.

What to avoid:

- Abbreviations without explanation.
- Parameter names.
- Package comparison details beyond "benchmarked against established tools."
- DS except as a backup question.

5-minute version:

1. Same behavior, different causes.
2. Model-based measurement.
3. Train on simulations, infer quickly on real data.
4. Validation: accurate where informative, uncertain where not.
5. Adaptive stopping as a broad example.
6. Takeaway: AI as an accelerator for transparent scientific models.

10-minute version:

1. Opening example.
2. What GRT gives cognitive scientists.
3. Why traditional fitting is a bottleneck.
4. What amortized inference means.
5. What GRIN returns.
6. Validation in accessible terms.
7. Adaptive stopping/design payoff.
8. Software availability.
9. Limits.
10. Broader lesson.

15-minute version:

Use the 10-minute route plus:

- A gentler walkthrough of the confusion matrix.
- One real-data example.
- A short explanation of why uncertainty calibration matters.

## 4. Job Talk Segment

Likely audience: hiring committee with mixed expertise. GRIN may be one project
inside a broader research narrative.

Primary job: show intellectual ownership, technical range, and future trajectory.

Tone: strategic and mature.

What to emphasize:

- You identify bottlenecks in formal cognitive measurement.
- You build tools that preserve interpretability while gaining scalability.
- You validate carefully rather than treating ML as magic.
- The work opens a research program in adaptive perceptual-expertise training:
  real-time model fits can diagnose a learner's perceptual representation and
  select the next stimulus, difficulty level, or feedback target.
- Melanoma identification is one motivating case, but the arc applies to any
  domain where expertise depends on learning multidimensional perceptual
  structure.

What to avoid:

- Too much package minutiae.
- Making the talk about software engineering alone.
- Letting limitations sound like project failure.

Possible 8-12 minute segment:

1. Bridge from broader research question to measurement bottleneck.
2. GRT example: same accuracy, different cognitive explanations.
3. Existing inference limits.
4. GRIN solution.
5. Validation headline.
6. What it enables: adaptive design.
7. Limitations as next research directions.
8. Future program slide.

Job-talk future-program slide:

- Adaptive cognitive measurement.
- Formal models plus amortized inference.
- Adaptive training that moves learners toward expert perceptual
  representations.
- Design-based identification for decisional/perceptual ambiguity.
- Open tools that make model-based methods easier for empirical labs.

## 5. Postdoc / Lab 20-45 Minute Talk

Likely audience: lab members or department colleagues who can tolerate process,
detours, and lessons learned.

Primary job: tell the full project story: motivation, implementation, validation,
problems discovered, and where it goes next.

Tone: candid, detailed, useful.

What to emphasize:

- Why this was worth doing during the postdoc.
- What the initial technical bet was.
- What validation changed about the story.
- Where the project is robust and where it is intentionally limited.
- How the package/web/manuscript pieces fit together.
- Open problems and collaboration hooks.

20-minute version:

1. Why GRT and why inference speed matters.
2. Vignette and constructs.
3. Simulator and prior.
4. Network outputs and package workflow.
5. Recovery.
6. Calibration.
7. Baseline comparison.
8. Design envelope.
9. Adaptive stopping.
10. Real data.
11. Limitations.
12. Current manuscript/software status.

45-minute version:

Use the 20-minute route plus:

- More simulator and parameterization detail.
- More recovery/error-map detail.
- More baseline failure-subset detail.
- DS non-identifiability teaching branch.
- Package/web demo.
- Future directions and group discussion.

## 6. Casual Lab Meeting

Likely audience: familiar colleagues, interruptions likely, exploratory tone.

Primary job: support discussion rather than deliver a polished performance.

Tone: conversational and modular.

What to emphasize:

- What problem the project is solving.
- What results surprised or did not surprise us.
- What should stay out of the manuscript.
- Where reviewers may push.
- What future analyses are worth doing versus scope creep.

Suggested structure:

1. Two-slide setup: problem and GRIN idea.
2. Pick-your-own result modules: recovery, calibration, baselines, adaptive
   stopping, real data.
3. Optional DS module.
4. Discussion slide: what belongs in paper, supplement, teaching, or next
   project.

Useful live branches:

- "Show me the DS issue."
- "How does it compare to grtools?"
- "What happens at low trial counts?"
- "What would a richer posterior head buy us?"
- "What is the next publishable extension?"

## Open Questions For Murray

These should be answered before building final HTML slides:

1. Do you want the full deck to include live code/web-app demo placeholders, or
   should it be slide-only?
2. Should the Math Psych route use British spelling from the manuscript
   ("amortised") or American spelling for US conference slides?
