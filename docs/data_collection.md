# Getting data from your experiment platform into GRIN

Both packages accept a 2x2 confusion matrix however you produced it. This page is
about the step before that: getting from whatever ran your identification task
into that matrix. Three audiences in practice ask about this, so it's split
accordingly: [PsychoPy](#psychopy), [browser/JavaScript platforms](#online-javascript-platforms-jspsych-labjs-psychojspavlovia-gorilla-)
(jsPsych, lab.js, PsychoJS/Pavlovia, Gorilla, ...), and [where `grin` (R) fits](#where-does-grin-the-r-package-fit)
in a pipeline that usually starts in one of the other two.

## The short version

Both packages accept trial-level ("long format") data directly — you do not need
to hand-tally a 4x4 matrix yourself. If your experiment logs one row per trial
with a stimulus label and a response label (an optional `count` column if you've
already aggregated), hand the whole log to `to_confusion(..., long=True)`
(`grintools`) or `grin_to_confusion(..., long = TRUE)` (`grin`). This is the
shape a PsychoPy per-trial data file and a typical jsPsych/lab.js/Pavlovia/Gorilla
export already come in, so for most people this is the only new thing on this
page: everything else (fitting, plotting, stopping rules) is the same API
documented in each package's README regardless of what collected the data.

## PsychoPy

PsychoPy's per-trial data file is already one row per trial — the only work is
telling GRIN which columns are the stimulus and which is the response, since
those column names come from your own Builder routine or `conditions.xlsx` and
GRIN can't guess them.

```python
import pandas as pd
import grintools as gt

trials = pd.read_csv("participant_01_2026-08-13.csv")

# whatever your condition/response columns are actually called, spell out both
# dimension levels of the item shown and of the observer's judgment:
trials["stimulus"] = trials["age_cond"] + "/" + trials["emotion_cond"]
trials["response"] = trials["age_resp"] + "/" + trials["emotion_resp"]

ci = gt.to_confusion(trials, factor_a=("Old", "Young"), factor_b=("Neg", "Pos"), long=True)
result, constructs = gt.infer(ci.counts, ci.trials)
```

Build the `stimulus`/`response` columns after the fact as above, or add them once
during the routine with a Code Component's `addData()` call if you'd rather they
already exist in the exported CSV. Either way, `factor_a`/`factor_b` are your own
two levels per dimension — see the ordering contract in the `grintools` README for
what they're doing.

**Live, adaptive use.** Because PsychoPy is plain Python, `grintools` can run
inside the same process as the trial loop — no separate analysis step. Evaluate a
stopping criterion between trials (not inside a frame-refresh callback) and end
the loop early once it's met:

```python
crit = gt.Criterion([gt.Target.probability("PS_A", at_least=0.90)], combine="any")
trial_log = []  # append (stimulus_label, response_label) after each trial

for this_trial in trials_handler:
    ...  # present stimulus, collect response
    trial_log.append((stim_label, resp_label))
    if len(trial_log) >= 8:  # let a few trials land per cell before evaluating
        ci = gt.to_confusion(trial_log, factor_a=("Old", "Young"), factor_b=("Neg", "Pos"), long=True)
        result, constructs = gt.infer(ci.counts, ci.trials)
        if crit.evaluate(result, constructs).stop:
            trials_handler.finished = True
            break
```

A fit is a few milliseconds, so re-running it every trial or every few trials is
cheap; just call it between trials rather than anywhere timing-sensitive.

## Online / JavaScript platforms (jsPsych, lab.js, PsychoJS/Pavlovia, Gorilla, ...)

Two paths, depending on when you want the answer.

### Offline: export, then analyse (the common case)

Most of these platforms export one record per trial already — a CSV from
`jsPsych.data.get().csv()`, a Pavlovia/PsychoJS results CSV, a Gorilla data
export. Read the export into a data frame and follow the same recipe as
PsychoPy above: build `stimulus`/`response` columns from whatever your platform
called the condition and response variables, then call `to_confusion(...,
long=True)`. This is identical whether you analyse in `grintools` or `grin` —
platform-side export format doesn't care which package reads it downstream.

### Live, in-browser: worked example, not a package

If you want a running estimate or an adaptive stop condition inside the browser
itself, GRIN's own web app already does exactly this:
[`web/assets/js/grin-model.js`](../web/assets/js/grin-model.js) loads the
trained network as an ONNX graph and runs it client-side via
`onnxruntime-web`, sub-millisecond per fit, alongside its manifest-driven
`web/assets/models/` directories and vendored `web/assets/vendor/ort/` runtime.

This is **not published as a versioned npm package** — there is no equivalent of
the `pip`/CRAN release contract here. Treat it as example code to copy into your
own experiment and adapt: pin to a commit, expect to touch it if the model
manifest format changes, and re-test after pulling updates.

The pattern:

```javascript
import { loadModel } from "./grin-model.js";  // copied alongside its models/ + vendor/ort/ directories

const model = await loadModel("./assets/models/cm");  // counts-only network; "./assets/models/cmrt" for +RT

// maintain a running canonical-order 4x4 counts matrix and length-4 trial totals,
// incrementing the appropriate cell after each response
const result = await model.predict({ counts, trials });
if (result.sep.A >= 0.90) { /* stop the trial loop */ }
```

`counts`/`trials` use the same canonical-order shape (A1B1, A1B2, A2B1, A2B2)
both packages use elsewhere on this page — decide the stimulus/response ordering
once, the same way `to_confusion()` requires it, and increment cells directly
rather than accumulating a trial log and re-parsing it every call, since a JS
experiment loop doesn't have a long-format DataFrame helper standing by.

## Where does `grin` (the R package) fit?

R is a fine place to do the actual GRT analysis — nothing about the plotting or
stopping-rule API differs between the two packages — but R is rarely the
language an identification task actually ran in. In practice `grin` mostly
enters *after* data collection is over: export from PsychoPy, jsPsych,
Pavlovia, or Gorilla, read the export into R, `grin_to_confusion(..., long =
TRUE)`, and go straight to per-participant and group-level reporting. If you
specifically need an adaptive/live decision inside R rather than in whatever
ran the experiment, `grin_infer()` is native `torch` and fast enough in
principle; it's just an unusual choice compared to keeping the adaptive logic
in the same process as the trial loop (PsychoPy: `grintools` in-process,
browser: the worked example above). Treat `grin` as the post-hoc analysis and
reporting package, and the other two as where the live/adaptive path lives.
