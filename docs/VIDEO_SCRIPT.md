# Video script: Glass-Box CT (target 2:45)

Record at 1440x900, dark system theme, browser zoom 100%, no bookmarks bar. Use scan 2 (bright tumour the model misses) as the hero, scan 4 (counter-example) only if time allows. Voice-over is about 400 words at a calm pace (about 150 words per minute). Numbers are exactly those in `results/SUMMARY.md` and the app.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00-0:12 | Landing page, slow scroll over the five scan cards (Good / Missed / Flagged chips) | "This AI measures liver tumours on CT. On held-out research scans it makes them look about a fifth smaller than they are, and almost never errs the other way. We wanted to know why, and to build something that shows it." |
| 0:12-0:38 | Click scan 2. Viewer fills the left: teal prediction small, dashed white reference large. Right: verdict card "Tumour likely under-segmented, don't trust these measurements"; the three big numbers count up; point at "reference +1 HU" line | "Here is a real scan. The teal outline is what NVIDIA's NV-Segment-CT predicts. The dashed outline is the dataset's own reference. The model found a small part of this tumour, and the quality flag knows: it says don't trust these measurements, and explains why in one sentence." |
| 0:38-0:58 | Click "Recalibrated (research)". Amber overlay grows to cover the reference; sum of diameters bar animates 82.7 to 146.3 mm; cost box visible | "A single lower threshold recovers most of it. But look at the cost, stated right here: across held-out scans it over-estimates by twenty percent or more in thirteen point five percent of cases. The default stays primary." |
| 0:58-1:25 | Agent steps tick off (or Ask box): type "Why was this flagged?"; tool step streams; grounded answer appears. Then type "Should we treat this with chemotherapy?" and show the refusal. Briefly show the page's provider line (About page, or terminal start-up line) | "The agent is Nemotron, called through Nebius Token Factory [confirm]. It calls tools, then answers, and every number it writes is checked against the tool results and must carry its unit. Ask it about treatment, and it declines: this is a research instrument, not a clinician." |
| 1:25-2:10 | "Why it fails" page, scroll: finding 1 (-19%), finding 2 (bars: 0.587 vs 0.079 vs 0.114), finding 3 (counterfactual), finding 4 (erasure), finding 5 (patching), then the causal chain | "Why does it fail? Our research, pre-registered before we ran it. The model does represent the missed tissue: a read-out of its middle layers finds about six tenths of it, an untrained network finds under a tenth. Edit a dark tumour to be bright, change nothing else, and the model loses it, in fifty-six of sixty-two cases. Erase one direction inside the network and it loses a third of the tumours it used to find, while the liver is untouched; erase a random direction and nothing happens. Paste the dark version's internal state back into the tumour and about four fifths of the loss comes back. Brightness weakens one internal feature, and the tumour drops below the threshold." |
| 2:10-2:30 | About page, then the README "Research summary" table | "We also tried to beat a threshold, by re-fitting the output layer and by steering that internal direction. Neither did better than one threshold shift, and we report that. Every claim is labelled pre-registered or exploratory." |
| 2:30-2:45 | Landing page with the headline; footer visible with attribution and disclaimer | "Glass-Box CT. See where the model fails, and why. Research use only, not for clinical decisions. Scans from the Cancer Imaging Archive, code open source." |

## Shot list and cues
1. Pre-load every page once (so the viewer is warm); the reports are cached and load instantly.
2. Keep the cursor still while numbers count up (600 ms); do not talk over the count-up.
3. For the Ask shot, paste the question rather than typing; the answer takes about 9 seconds with a live model: cut the wait or speed it up, but keep the streamed tool step in frame.
4. For the provider shot, show the server start-up line (`Glass-Box CT: language model provider=... model=...`) or `/api/health` for two seconds. **This is the runtime proof of the Token Factory call; record it only after the key is switched** (`docs/TOKEN_FACTORY.md`).
5. End card: repository URL and the TCIA citation line.

## Checks before recording
- All five reports say "number check: ok" (Evidence section) and were built with the final provider (`python scripts/precompute_reports.py data/demo`).
- `python scripts/smoke_deployed.py <url>` passes against the deployed server.
- Replace every `[confirm]` in the voice-over with what is true on the day.
