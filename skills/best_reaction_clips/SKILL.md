---
name: best_reaction_clips
description: Turns a folder of raw wedding-live-illustrator reaction clips (guests seeing their live portrait for the first time) into a batch of SEPARATE individual clip files — one per guest reaction — instead of one compiled highlight video. Use this skill when the user wants the "best moments," "high energy moments," "funniest reactions," "best reactions," or "confused/curious reactions" pulled out of a `raw` folder and saved as their own standalone clips, not stitched together — e.g. "give me each good reaction as its own clip," "export every high-energy moment separately," "pull out all the funny/confused reactions into individual files." If the user instead wants everything combined into one finished reel/TikTok/compilation video, use short_form_edit instead.
---

# Best reaction clip exporter

## What this produces

A batch of short, individually-cropped vertical (9:16) .mp4 clips — one file
per surviving candidate reaction — sorted into subfolders by mood:

- `output/excited/` — laughing, gasping, hand to face/mouth, big smile,
  exclaiming: the "high energy" / "funniest" / "best reaction" moments.
- `output/curious/` — calm, contemplative "what is this" energy: staring,
  furrowed brow, head tilt, quiet close reading of the portrait, before any
  laugh or gasp.

This is NOT a compiled reel — there's no stitching, no manifest-wide pacing
decisions, no fades between clips. Each output file stands alone, trimmed
to just that moment and cropped tight on the reacting face, ready to be
scrolled through, picked from, or dropped into a separate edit later. If the
user wants a single finished video instead, that's `short_form_edit`, not
this skill.

## Why the two-step approach

Detecting "high energy" automatically and deciding what's actually worth
keeping are separate steps on purpose. The detection script
(`analyze_clips.py`) is deterministic — it scores moments using audio
loudness and smile detection — but it can't tell a genuine happy reaction
from a loud clatter, a posed phone photo, or a stranger walking through
frame smiling at something unrelated. So it surfaces *candidates* with
thumbnails, and you look at the thumbnails before deciding which ones are
worth exporting as final clips.

## Step 1 — Locate the folders and check dependencies

Find the raw clips folder (the user will usually say "raw folder" or you'll
see a `raw/` directory) and where they want the exported clips to land
(often `output/` right next to it).

Confirm the Python packages the analysis script needs are available:

```bash
python3 -c "import cv2, numpy" || pip install opencv-python-headless numpy --break-system-packages
```

`ffmpeg`/`ffprobe` are assumed to be present (they're standard in this
environment).

## Step 2 — Run the detector

Run this via `scan_pipeline.py` (see "One-command pipeline" below) whenever
the request asks to search *all* footage. The manual form is only a partial
audio-led diagnostic pass; it cannot discover a quiet facial reaction
outside an audio-selected window:

```bash
python3 scripts/analyze_clips.py <raw_dir> --out-dir <workdir>/analysis
```

This scans every clip in `<raw_dir>`, extracts audio RMS energy, samples
frames for face + smile detection (OpenCV Haar cascades — no model download
needed), treats the two signals as independent evidence, merges nearby
high-scoring moments into candidate segments, and writes:

- `<workdir>/analysis/analysis.json` — ranked candidate segments (file,
  start, end, score)
- `<workdir>/analysis/thumbnails/*.jpg` — a time-labeled nine-frame contact
  sheet spanning the candidate plus 4s pre-roll and 4s post-roll

`analysis.json` is not a coverage guarantee. Haar cascades commonly return
0 for small/profile faces and open-mouth laughter or screams, and quiet
expressions may never enter an audio-selected window. A failed face score
must not suppress a strong audio event, and a missing audio event must not
suppress visual review. The full-timeline dense and motion sweeps are
therefore required for comprehensive work; lowering `--percentile` cannot
recover a moment that was never eligible.

Since this skill exports every worthwhile candidate rather than curating
down to a fixed ~30-45s runtime, lean toward a looser `--percentile` and a
higher `--top-n` than you would for a compiled reel — the goal is
comprehensive coverage of good moments, not a tight cut.

Useful knobs if the first pass looks off (rerun with adjusted flags rather
than trying to hand-fix `analysis.json`):
- `--percentile` (manual-detector default 85) — raise it to be pickier, lower it to surface more candidates
- `--audio-weight` / `--smile-weight` (default 0.55 / 0.45) — shift emphasis if one signal is clearly more reliable for this footage (e.g. windy outdoor audio → lean more on smile detection)
- `--min-segment` (default 1.2s) — minimum candidate length
- `--top-n` (default 25) — how many candidates to keep; raise this well above the reel default since nothing here gets cut for runtime
- `--review-preroll` / `--review-postroll` (defaults 4s / 4s) — context
  shown around each detected window; increase when reactions visibly build
  before the sound or continue after it

## One-command pipeline (recommended — saves a lot of back-and-forth)

Running the detector, then separately extracting dense frames, then
tiling them into grids, one ffmpeg call at a time, burns a large number of
tool-call turns for what is ultimately mechanical work with no judgment
involved. `scan_pipeline.py` does all of it — audio+smile detection *and*
a dense fixed-interval frame sweep across every file, tiled into grid
montages sized for one Read call each — in a single invocation:

```bash
python3 scripts/scan_pipeline.py <raw_dir> <workdir> \
  --percentile 30 --top-n 300 --max-per-file 30 \
  --review-postroll 4
```

This is slow in wall-clock time (it's decoding/seeking every file twice),
so run it with `run_in_background` or `nohup ... &` and check back rather
than waiting on it synchronously. Output layout:

- `<workdir>/analysis/analysis.json` + `thumbnails/*.jpg` — same as Step 2;
  every thumbnail is a multi-frame candidate contact sheet
- `<workdir>/dense/manifest.json` — `[{idx, file, t}, ...]` for every dense-sweep frame
- `<workdir>/dense/frames/fNNNN.jpg` — the raw extracted frames
- `<workdir>/dense/grids/gridNNN.jpg` — those frames tiled `cols x rows`
  (default 4x3) per image, in order, so reviewing the whole sweep is one
  Read call per grid instead of one per frame

To trace an interesting grid cell back to its source: flat index =
`grid_number * cols*rows + row*cols + col` (0-indexed row/col within the
grid), then look that index up in `dense/manifest.json` for the exact file
and timestamp, and `ffmpeg -ss <t> -i <file> -frames:v 1 ...` a closer look
before trusting it — a low-res tiled thumbnail reads pointing-shaped motion
blur as a pointing gesture more often than it should (in practice this
happens most with the illustrator's own drawing arm at the portrait table
— always confirm the "pointer" is a guest, not the artist, before adding a
candidate).

Useful flags beyond the ones shared with `analyze_clips.py`:
- `--dense-interval` (default 1.0s) — this is intentionally denser than the
  old 3s default: a 2s facial reaction can fall entirely between 3s samples.
  Use sub-second sampling only for a targeted clip/window after this pass.
- `--skip-files` / `--skip-substrings` — exclude files already covered by
  an earlier deep-scan (comma-separated), or filter out known duplicate
  copies (defaults to filtering `video 2.mp4`-style dupes)
- `--grid-cols` / `--grid-rows` — tile size; smaller grids are easier to
  inspect closely but mean more Read calls overall
- `--skip-audio-detection` / `--skip-dense-sweep` — run just one half of
  the pipeline (e.g. rerunning only the dense sweep with a tighter interval
  after the audio pass already looked complete)

The looser audio settings in the command are deliberate for standalone
exports. Normal-volume praise can rank just below a global top-150 cutoff,
especially in a long, busy file. Do not tighten them merely because the
candidate set is large: comprehensive reaction export values recall over a
short review queue. The dense and motion sweeps still need to be reviewed,
but neither can reveal what a subtly smiling guest actually said.

This still doesn't replace Step 3 — someone (or Claude) still has to look
at the thumbnails and grids and judge what's genuine. It just collapses
the mechanical extraction work into one process instead of dozens of
separate tool calls.

## Motion scan (catches what the dense sweep structurally misses)

The dense sweep above has two blind spots that no amount of careful review
fixes: a gesture shorter than `--dense-interval` can fall entirely between
samples (a ~0.5s thumbs up, a quick point), and a real but subtle reaction
(quietly holding a portrait up, raising a phone to photograph it) can read
as indistinguishable from generic crowd milling even in the larger 4x3
review grids. Both of these caused real misses in practice —
confirmed by user-supplied timestamps that the dense sweep + review had
walked right past.

`scan_pipeline.py`'s motion stage (on by default, `--skip-motion-scan` to
disable) targets exactly that gap:

1. Decodes each file once, continuously, at `--motion-fps` (default 3.0,
   ~0.33s resolution, gapless) — with `--motion-hwaccel videotoolbox`
   (default, macOS) this is ~7x faster than software decode on 4K source,
   which is what makes dense continuous sampling affordable at all.
2. Scores frame-to-frame motion per sample as the single hottest block in
   an N×N grid (`--motion-blocks`, default 16), not a whole-frame average —
   a hand gesture only changes a small patch of a crowded wide shot, and
   averaging over the whole frame buries it under ambient crowd motion.
3. Ranks candidates by how far they stand out from *that file's own*
   baseline (robust z-score), not pooled globally — otherwise a file with
   a lot of camera panning dominates and a quiet file's one real spike
   never surfaces. `--motion-max-per-file` (default 8) caps how many
   slots any single file can take for the same reason.
4. Renders only the top `--motion-top-n` (default 300) candidates
   individually at `--motion-out-size` (default 560px, much larger than
   dense thumbs), tiled few-per-grid (`--motion-grid-cols/rows`, default
   3x3) so each one stays legible.

Raw per-frame scores are cached to `<workdir>/motion/all_scores.json`; add
`--motion-reuse-scores` to re-tune `--motion-top-n` / `--motion-min-gap` /
`--motion-max-per-file` without re-paying the decode cost (the expensive
part) on a rerun.

**Known limits, disclosed rather than papered over:** this is a motion
detector, not a reaction detector — a lot of what surfaces is still posed
photos and generic milling that need the same Step 3 judgment as anything
else, and files with near-continuous handheld camera shake throughout can
have their one real gesture buried under an already-high, uniform motion
baseline (a genuinely brief gesture in an otherwise-shaky clip may not
outscore the file's own noise floor even with all of the above). Treat a
"nothing here" result on a shaky file as inconclusive, not as confirmation
there's no reaction in it — that's still a case where a direct human tip
beats the algorithm.

## Screaming/squealing reactions must survive a failed smile cascade

An open-mouth scream or gasp often scores 0.0-0.1 in the Haar smile
cascade. `analyze_clips.py` therefore ranks each audio-selected candidate
by the better of its audio-only score and its audio+smile score; face
detection is positive evidence, never a penalty. Review the whole contact
sheet even when `smile_score` is 0.

Do not use a fixed loudness threshold such as 2.2σ as a coverage gate.
Real screams on a consistently loud clip can peak much lower relative to
that clip's baseline. Audio scores surface candidates; the full-timeline
visual sweep provides coverage.

## Spoken delight needs a full-timeline language pass

RMS loudness and facial cascades both miss a common keeper: a guest speaking
enthusiastically at normal conversational volume while looking down, in
profile, or smiling too subtly for Haar detection. Expressions such as
"that's us," "nice," "incredible," "I love it," "this is amazing," "this is
so cute," and "can we take it?" are strong reactions even without a scream,
large motion spike, or detected smile.

For a comprehensive run, transcribe the full audio timeline of every source
file as a discovery pass, not only audio-ranked finalists. Search the
timestamped result for positive/surprised reaction language (the examples
above are illustrative, not an exhaustive whitelist), then inspect the
aligned face and several seconds of surrounding context. Treat a transcript
hit as an independent candidate source: it must survive even when its RMS
rank misses `analysis.json` or its face score is 0.0-0.1.

ASR is noisy in crowded venues. Use one fast full-timeline pass to find
possible hits, then validate each hit by listening or by retranscribing that
short window from both raw audio and a cleaned copy (`highpass=f=120,
afftdn=nf=-25,dynaudnorm`). Agreement is useful evidence, but the visible
speaker/reaction and intelligible source audio are decisive. Never discard a
user-supplied phrase merely because ASR paraphrases it.

## Don't truncate a long reaction to the detector's merged window

`merge_windows` stops a segment the moment the score drops below
threshold, which can slice a single continuous reaction into a much
shorter clip than the reaction actually was — a guest who screams, then
keeps talking with her hand on her chest, then laughs again a few seconds
later is one continuous moment, not a 2-second one. Before locking in a
candidate's `start`/`end`, check what's actually happening just before and
after the merged window at 0.25-0.5s spacing and extend the clip to cover
the full arc.

If the reaction has multiple genuinely distinct beats — e.g. it dips back
to calm conversation for 15-20s and then spikes again — treat those as
**separate clips** from the same source file rather than one clip spanning
the dead air between them, even when it's the same guest both times. Two
clean, distinct clips read better than one clip with a long lull in the
middle. A useful rule of thumb: if the audio stays elevated (or the visual
energy stays clearly up) the whole way through a gap, it's one continuous
reaction and should be one (potentially long, 8-12s+) clip; if it drops
back to baseline in between, it's two.

## A candidate contact sheet can still hide a very short reaction

`analyze_clips.py` writes nine-frame contact sheets spanning 4s before and
4s after each detected window. This is essential because the face can react
before the loudest word, laugh, or clap that triggered detection. Use the
sheet's `review_start`/`review_end`, not only the candidate's `start`/`end`,
when judging and finding the true clip boundary. A sub-second expression
can still fall between samples; if a sheet has a strong audio score but no
visible explanation, inspect its context at 0.25-0.5s spacing instead of
discarding it. Open-mouth screams are especially likely to have a
`smile_score` of 0.

Before finishing Step 3, pull every candidate with `duration >= 5.0` and
`smile_score <= 0.1` and inspect between the contact-sheet samples at
0.25-0.5s spacing. Treat low smile as "cascade inconclusive," not as
evidence of low emotion.

## Step 3 — Review candidates visually and audibly

Read `analysis.json`, then inspect every multi-frame candidate sheet — not
just the top 15-20. Review every grid in both `motion/grids/` and
`dense/grids/`, keeping a running list of any (grid, row, col) worth a
closer look, then batch the flat-index lookups + zoom-in verification.
Maintain a per-source-file coverage ledger and do not finalize until every
source file appears in the dense manifest and every grid has been reviewed.
Detector ranking is triage, never permission to ignore unranked footage.
For each moment, judge:

- Does this look like a genuine reaction — open smile, laughing, hand-to-mouth
  surprise, animated gesture, or (for the curious bucket) real staring/
  furrowed-brow "what is this" body language — not just an open mouth
  mid-sentence or a random bystander?
- Is the frame usable (subject visible, not badly blurred, not mid-blink)?
- Is this a candid reaction to the portrait, not a posed photo? If a guest
  is facing/looking toward a phone or camera lens with a static held smile
  (or someone nearby is visibly holding up a phone pointed at them), that's
  a posed photo moment, not a reaction — discard it even if the audio/smile
  score is high. Candid reactions have guests looking at the tablet/portrait
  or each other, with more spontaneous, in-motion body language.

Review the audio content too; frames and RMS scores cannot reveal meaningful
words. Audition each audio-led candidate from `review_start` to `review_end`
before discarding it. Short, clear exclamations such as "oh my God," "wow,"
or "no way" are excited reactions even when the smile cascade returns 0.
If direct listening is unavailable, extract raw and cleaned audio and use
agreement between two transcription attempts as a discovery aid; never
trust one Whisper pass. Extend the final segment to include the complete
phrase and its corresponding facial reaction.

Maintain a separate speech-hit ledger alongside the visual coverage ledger.
Before finalizing, every phrase hit must be marked kept or rejected with a
reason. This prevents a visually subtle but verbally excellent reaction from
being silently lost during final curation after the detector successfully
surfaced its neighborhood.

Discard candidates that don't hold up. Unlike a compiled reel, there's no
minimum total runtime to hit here — it's fine to end up with anywhere from a
handful to a few dozen clips depending on how much genuine reaction footage
exists in `raw/`. If almost nothing survives, that's a signal to loosen
`--percentile` and rerun Step 2, not to keep weak candidates.

## Reference example (optional)

If a `reference/` folder exists alongside `raw/`, it holds an example of the
kind of reaction moment being looked for. Watch/view it with the Read tool
before Step 4 and use it to calibrate what "genuine reaction" looks like for
this shoot — but note it's less directly applicable here than for
`short_form_edit`, since there's no pacing/rhythm decision to match, just a
bar for what counts as a keeper.

## Step 4 — Sort survivors into buckets and build the manifest

Sort the surviving candidates by feel, not by score:

- **`curious`** — calm, contemplative, "what is this" energy: staring,
  furrowed brow, head tilt, quiet close reading of the portrait. No laugh or
  gasp yet.
- **`excited`** — the actual reaction: laughing, gasping, hand to face/mouth,
  big smile, exclaiming. This bucket covers "high energy," "funniest," and
  "best reaction" phrasing alike — those are overlapping descriptions of the
  same kind of moment, not separate categories to split further.

Every reaction clip must land cleanly in exactly one bucket — re-judge it
*after* cropping tight on the face (Step 4b), not just from the wide
establishing shot, since a wide shot can hide a subject's actual expression
that only shows once you push in.

Build a single manifest listing every clip to export, tagged with its
bucket as `category`:

```json
[
  {"file": "clip_04.mp4", "start": 12.4, "end": 15.1, "category": "excited"},
  {"file": "clip_11.mov", "start": 40.0, "end": 43.8, "category": "curious",
   "zoom": 1.8, "crop_x": 0.3, "crop_y": 0.4}
]
```

(`file` can be just the filename — it's resolved against `--raw-dir` — or an
absolute path. Order in the manifest doesn't matter here — unlike a reel
manifest, there's no playback sequence, each entry becomes its own file.
`zoom`/`crop_x`/`crop_y` are optional — see Step 4b.)

Trim segment `start`/`end` slightly if the merged window includes obvious
dead air before/after the reaction — a tighter trim makes each standalone
clip read as one clean beat instead of a padded moment.

If the venue's couple appears among the candidates, it's fine to include
them here (unlike the reel, there's no "letting them dominate runtime"
concern since every clip is independent) — just still apply the same
genuine-reaction bar as any other candidate.

## Step 4b — Frame each clip on the face

The default crop just fills the 9:16 frame from the center, which on wide
venue-cam footage leaves a lot of background and a small, distant face. Push
in tighter so reactions read clearly on a small phone screen:

1. Grab a representative frame from the segment (mid-point is usually fine)
   with `ffmpeg -ss <t> -i <file> -frames:v 1 <tmp>.jpg` and look at it.
2. Estimate where the reacting face sits as a fraction of the frame, `fx`
   horizontally and `fy` vertically (0,0 = top-left, 1,1 = bottom-right).
3. Pick a `zoom` — how much tighter than a plain fill-crop to push in.
   1.5-1.8 works for a single close subject; 2.0-2.2 for a distant/small
   face; use something looser like 1.3-1.5 for a small group so no one gets
   cropped out. When in doubt, prefer looser — a face with more headroom
   than intended is a minor cosmetic issue, a cropped-out face is a broken
   clip.
4. Convert `fx`/`fy` to a starting guess for `crop_x`/`crop_y` with:
   `crop = (f * zoom - 0.5) / (zoom - 1)`, clamped to 0-1. Treat this as a
   starting point, not a guarantee — it assumes the scaled frame is exactly
   `width*zoom` × `height*zoom`, but ffmpeg's `force_original_aspect_ratio=
   increase` only guarantees *at least* that big, and actually overflows
   one dimension depending on the source clip's aspect ratio (nearly always
   the horizontal dimension for typical landscape/square venue-cam footage
   against a 9:16 target). The formula is usually close enough to start
   from, but always verify per step 5 below rather than trusting it blind.

**Verify every clip, not just one or two** — a face being in frame at the
segment's midpoint doesn't mean it stays in frame for the whole segment; a
subject who turns, bends, or walks partway through can end up cropped out
or out of frame entirely even with a good `crop_x`/`crop_y`. For each clip
in the manifest, render the same `scale`+`crop` filter string
`export_clips.py` uses at 4+ points spread across the segment (e.g.
fractions 0.05, 0.35, 0.65, 0.95 of its duration) and look at all of them.
If the subject leaves frame for part of the window, either loosen the zoom,
or — usually better — trim `start`/`end` down to the sub-window where
they're actually framed well, rather than trying to crop-track movement
with a static crop.

## Step 5 — Export the clips

```bash
python3 scripts/export_clips.py <workdir>/manifest.json <output_dir> --raw-dir <raw_dir>
```

This trims each segment, scales/crops it to fill a 1080x1920 vertical frame
per the `zoom`/`crop_x`/`crop_y` in the manifest (cropping to fill rather
than letterboxing — standard for Reels/TikTok), re-encodes it to a
consistent format, applies a short audio fade in/out, and writes it as a
standalone file under `<output_dir>/<category>/`. It also forces 8-bit
color output (`-pix_fmt yuv420p`) regardless of source: some cameras
(action cams/gimbals especially) shoot 10-bit HDR footage, and libx264
silently preserves that into a "High 10" profile if not told otherwise —
which most phones, browsers, and Reels/TikTok itself can't decode, and
shows as a black screen instead of an error.

If `raw/` has many long/high-resolution candidates, rendering all at once
can be slow — use `--only-index <n>` to render one clip at a time across
several shorter command calls instead. Defaults (`--width`, `--height`,
`--fps`, `--fade`) rarely need changing unless the user asks for a different
aspect ratio.

## Step 6 — Verify before handing off

Don't just assume it worked — spot-check the outputs:

```bash
find <output_dir> -name '*.mp4' | wc -l
ffprobe -v error -show_entries format=duration -show_entries stream=width,height,pix_fmt,profile -of default=noprint_wrappers=1 <one_output_clip>.mp4
```

Confirm the clip count matches the manifest length, resolution is 1080x1920
(or whatever was requested), and `pix_fmt`/`profile` read `yuv420p` /
`High` (or `Main`/`Baseline`), never `yuv420p10le` / `High 10` — see the
10-bit note in Step 5, since that plays as a black screen in most players.
Pull a frame from a handful of clips across both categories (e.g. via
`ffmpeg -ss <t> -i clip.mp4 -frames:v 1 preview.jpg`) and look at them with
the Read tool as a final sanity check that the tighter crop from Step 4b
didn't cut off faces.

If a clip looks wrong, fix its entry in the manifest and re-run Step 5 for
just that one with `--only-index` rather than re-running the whole
detection pass.

## Notes

- Source clips can be any common format ffmpeg reads (mp4, mov, mkv, etc.).
- This is tuned for the "one main subject reacting to camera" framing typical
  of live-illustrator guest reaction footage. If a clip has multiple people
  in frame, the detector scores based on the largest face, which is usually
  the intended subject.
- Nothing here uploads footage anywhere or requires internet access — the
  Haar cascades are bundled with OpenCV.
- Portrait/detail insert shots (of the artwork itself, not a guest reacting)
  aren't relevant here — those exist to punctuate a compiled reel
  (`short_form_edit`'s Step 3b), not as standalone reaction clips, so this
  skill doesn't look for them.
