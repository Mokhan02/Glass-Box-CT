# Devpost draft: Glass-Box CT

Track: Best Apps and Agents. Everything in [brackets] is for you to confirm or fill before submitting; nothing below claims a Nebius result that has not happened yet.

## Tagline
Don't just measure the tumour; measure whether to trust the measurement. An open trust layer for oncology imaging AI: per-scan and per-lesion verdicts, an audit record, and the evidence behind every warning.

## Inspiration
AI segmentation models are now good enough that their measurements get used: tumour size changes decide whether a treatment is working. But a model that is right on average can fail in a consistent, invisible direction. We took NVIDIA's NV-Segment-CT (VISTA-3D), a widely used CT model, measured it on public held-out scans, and found it under-measures liver tumours by about a fifth, almost never the other way. Instead of reporting a number, we wanted to know why, and to build a tool that shows a user the failure as it happens.

## What it does
- **Trust verdicts:** every scan and every lesion is Trusted, Review or Unreliable, with plain reasons drawn from tool results (quality flag, contrast, contrast phase, size, fragments). A **contrast-phase heuristic** (literature thresholds fixed before it was checked) warns when a scan matches the model's known bright-tumour blind spot. A downloadable **audit record** (JSON and PDF) captures model checkpoint hash, thresholds, phase, flag, default vs recalibrated values, the agent's report and the tool-call log. A **two-timepoint check** shows when a RECIST response category depends on calibration, using clearly labelled **simulated** pairs from a fitted bias model (no real follow-ups exist in the data).
- **Five real research scans**, chosen by fixed rules (a good case, a bright tumour the model largely misses, a tumour a recalibrated threshold recovers, a counter-example, and a false alarm of our own quality flag).
- For each: **measurements** (largest lesion, sum of diameters, tumour-minus-liver contrast, muscle at L3), a **verdict** in plain words ("Tumour likely under-segmented, don't trust these measurements"), and the **prediction drawn against the dataset's own reference annotation**.
- A **default vs recalibrated (research) threshold** toggle. Recalibration removes most of the bias (median error -19.2% to +0.6%), and the app states the cost beside it: it over-estimates by 20% or more in 13.5% of cases. The default stays primary.
- A **Nemotron agent** that calls tools (segmentation, body composition, lesion measurement, quality audit) and writes a short report. **Every number is checked against the tool results and must carry its unit**; a report that fails is rewritten, then replaced by a template, never shown unchecked.
- **Ask about this scan:** questions go to the agent with that scan's tool results; the tool steps stream; the answer passes the same number check; clinical advice, diagnosis and treatment questions are refused before any model is called.
- An **"Evidence" page** with the research behind it, every figure traceable to the project's summary.

## The finding
The model **represents tumours it fails to segment**. Its misses **depend causally on the tumour being brighter than the liver**. And a single threshold recalibration removes most of the one-sided under-measurement, at a cost we state. In numbers (all on held-out public HCC scans, 95% patient-bootstrap intervals; see `results/SUMMARY.md` for cohorts and which analyses were pre-registered):
- Sum of diameters median error **-19% [-28%, -13%]** (75 scans); detected lesions 11.4 mm too small; about 1 in 5 lesions largely missed.
- A read-out of the mid-encoder finds **0.587 [0.551, 0.656]** of the missed tissue, against 0.079 [0.060, 0.090] for an untrained network and 0.114 [0.050, 0.146] for raw brightness: the information is there and **lost late**.
- Editing a dark tumour to be bright, nothing else changed, reduces how much of it is found by 0.208 (95% CI 0.098 to 0.284), and 56 of 62 tumours move as predicted.
- Erasing the one tumour-vs-liver direction at the mid-encoder drops lesion Dice by **0.33 [0.29, 0.38]**; erasing a random direction drops it by 0.002; liver segmentation is unchanged. Pasting the dark version's mid-encoder state back into the tumour restores **0.79 [0.71, 0.85]** of the loss.
- We also tried to do better than a threshold: a 48-weight re-fit of the output layer and steering the internal direction **did not beat a single threshold shift**. We report that as it is.

## How we built it
- **Research:** pre-registered experiments (the rule and numbers committed before the result), a held-out cohort converted and re-verified from DICOM (including finding and fixing a series-mismatch in our own conversion), probes with random-init and raw-brightness controls, paired counterfactual edits, linear concept erasure (LEACE), activation patching, steering, and direct logit attribution. Deviations are logged with their timing.
- **Model:** NVIDIA NV-Segment-CT (VISTA-3D, `vista3d132`), frozen. Compute on a rented A100.
- **Agent:** an OpenAI-compatible client reading four environment variables, so Nemotron on NVIDIA's hosted API (development) and Nebius Token Factory (submission) are a config change. A grounding checker validates every number and unit; a rewrite loop and template fallback keep unchecked text off the page.
- **App:** FastAPI + a React, Tailwind and Framer Motion front end with a NiiVue viewer. A slim Docker image carries the five pre-computed scans and their reports (no GPU needed). Verified in a real browser with Playwright, which found several real bugs (mis-aligned viewer coordinates, stale cached reports, a unit-check false failure).

## Challenges
- The model's "segment everything" mode **silently omits tumour classes**, so an automatic run on a liver CT shows no tumour whether or not one exists; we found this by reading the shipped code, and the tools now always prompt the tumour class explicitly.
- Our own data pipeline: some labels referenced a different acquisition than the one we first converted. We rebuilt a clean cohort and re-ran the headline on it rather than trust the first result.
- Making an LLM report trustworthy: models write a typographic minus, restate a difference without its sign, or round; the checker, the retry loop and the fallback are the product of that.
- Honest negatives: re-fitting and steering did not beat a threshold; two analyses are exploratory and labelled so.

## Accomplishments
A finding we can defend (with controls and causal tests), a tool that shows it, and a number-checked agent, shipped as a container that runs without a GPU. 68 automated tests, including one that ties every number on the explanation page to the research summary.

## What we learned
Averages hide direction. A model can know more than it says, and the failure can be located and partly repaired, but the repair has a price that should sit next to it, not in a footnote.

## What's next
Other cohorts and organs; a second model; contrast-phase labels (stripped from this dataset); letting a user upload a scan on a GPU backend (the code path exists, the demo server has no GPU).

## Built with
NVIDIA NV-Segment-CT (VISTA-3D), NVIDIA Nemotron, Nebius Token Factory [confirm], Nebius AI Cloud [confirm], MONAI, PyTorch, FastAPI, React, Tailwind CSS, Framer Motion, NiiVue, Docker, Playwright, Python.

## Try it
Live demo: [URL] &middot; Code: https://github.com/Mokhan02/Glass-Box-CT (MIT) &middot; Video: [URL]

## Data and safety
HCC-TACE-Seg from The Cancer Imaging Archive (CC BY 4.0; citations in the README and the app). Research use only, not a medical device, not for clinical decisions, no patient data.

---

# Tool feedback (for the "feedback on the tools you used" field)

**NVIDIA NV-Segment-CT / VISTA-3D.** *Worked well:* strong multi-organ results out of the box on an A100; the MONAI bundle is easy to run and to hook (we recorded encoder and decoder activations with plain forward hooks). *Hurt us:* (1) the paper describes the 127-class release but the shipped network is `vista3d132`, so class tables and counts differ from the paper; (2) the "segment everything" mode excludes the tumour and lesion classes without any warning, which makes an automatic liver run look like "no tumour"; (3) classes 129 and 130 cannot be prompted by label; (4) the foreground threshold is fixed at logit > 0 with no exposed probability output, which is exactly the knob our research says matters for tumours; (5) instance normalisation in the decoder makes logits depend on the 128-cubed window position. *Suggestions:* warn or list tumour classes in auto mode; expose a per-class threshold and calibrated probabilities; document the class table of the shipped bundle.

**NVIDIA Nemotron (hosted API on build.nvidia.com).** *Worked well:* OpenAI-compatible tool calling worked on the first try, a free tier was enough to develop, and the reasoning model followed grounding rules once the prompt asked for a short summary. *Hurt us:* it writes negative numbers with a typographic minus or en dash and sometimes restates a difference without its sign ("14.6 HU lower"), which broke naive number checking; a report with a 120B reasoning model plus a tool loop took about 35 seconds, so we cache and pre-compute. *Suggestion:* document sign and unit conventions for structured outputs.

**Nebius Token Factory.** [FILL AFTER USE: setup time from sign-up to first call; whether a Nemotron model was available and its ID; whether tool calling worked on the model you chose; latency for a report; anything that differed from the OpenAI client assumptions; rate limits you hit.] What the docs gave us before use: an OpenAI-compatible base URL, bearer keys from the console, and `GET /v1/models` for IDs, which made the swap a four-variable config change.

**Nebius AI Cloud (Serverless AI endpoints, Container Registry).** [FILL AFTER USE.] From the docs: the CPU endpoint quickstart was clear; we could not find, in the pages we read, how to expose an endpoint publicly without a Bearer token (a browser demo needs that) or the flag for environment variables.

**MONAI.** *Worked well:* `SlidingWindowInfererAdapt`, bundle transforms and the network classes made a faithful reproduction of the published pipeline quick. *Hurt us:* the hook points for window coordinates are not part of a stable API (we read them from the call arguments).

**NiiVue.** *Worked well:* a fast WebGL viewer that loads NIfTI directly and blends overlays. *Hurt us:* no native outline or dashed rendering for overlay masks (we pre-bake outline volumes), and the canvas needs an explicit resize observer inside a sticky, full-height layout.

**Lambda (A100).** *Worked well:* quick GPUs and a persistent filesystem. *Hurt us:* anything installed outside the filesystem is lost on termination (our shared virtual environment pointed at an interpreter on local disk), so a relaunch needs a short restore step.

**Playwright.** Driving the real app in a browser found bugs unit tests could not (viewer coordinates on the wrong grid, stale caches, a grounding false failure). We would start with it earlier.
