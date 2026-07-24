---
name: short_form_edit
description: Turns a folder of raw wedding-live-illustrator reaction clips (guests seeing their live portrait for the first time) into a single short-form highlight video ready for Instagram Reels / TikTok. Use this skill whenever the user asks to make a "reel," "highlight video," "TikTok," "compilation," or "edit" out of raw footage sitting in a folder (commonly named `raw`), asks to find the "best moments," "high energy moments," "funniest reactions," or "best reactions" across a set of clips, or mentions stitching/combining guest reaction clips into one video. Also trigger if the user references a `raw` folder of video clips alongside an `output` folder and wants a finished .mp4. Don't wait for the user to spell out the exact algorithm — this skill already knows how to detect and assemble the highlights.
---

# Short-form reaction highlight editor

## What this produces

Two vertical (9:16) .mp4 outputs by default, both built from the same pool of
curated candidates but cut differently:

1. **Curious-open version** — opens with exactly 3 clips of 3 *different*
   guests showing curiosity/confusion (staring, furrowed eyebrows, frowning,
   angled head) at the table, then continues into the best excitement clips
   (laughing, gasping, big smile, excited exclamation), same as this skill's
   original single-cut structure.
2. **Excited-only version** — no curiosity/confusion clips at all; every
   reaction clip is a positive/excited one. This pool is usually thinner
   than the curious pool (genuine big reactions are rarer than people just
   looking at a screen), so this version coming out shorter than the other
   is expected — don't pad it with weak or borderline clips just to hit a
   duration target.

Each is roughly 30-45 seconds (the excited-only version may run shorter —
see above), made of the best guest-reaction moments cut together. This is a
reactions highlight reel, not a full recap: it should feel punchy, not
padded. If the user asks for just one cut instead of both, that's a fine
deviation from this default.

If `raw/` also has clips of the portraits/illustrations themselves (not
guests reacting — just the artwork), a few of these can be interleaved as
short cutaway/insert shots between reaction beats, giving the viewer a
look at what's being reacted to. Keep these as connective tissue, not
padding — reactions are still the majority of runtime. These insert shots
aren't "reaction clips," so they're fine in both versions above, including
the excited-only one.

## Why the two-step approach

Detecting "high energy" automatically and building the final video are separate
steps on purpose. The detection script (`analyze_clips.py`) is deterministic —
it scores moments using audio loudness and smile detection — but it can't tell
a genuine happy reaction from a loud clatter or a stranger walking through frame
smiling at something unrelated. So it surfaces *candidates* with thumbnails,
and you look at the thumbnails before committing to a final cut. This human
(well, Claude) judgment step is what keeps the output from feeling like a
random supercut.

## Step 1 — Locate the folders and check dependencies

Find the raw clips folder (the user will usually say "raw folder" or you'll
see a `raw/` directory near an `output/` directory) and the output folder
(often `output/` right next to it, or wherever the user wants the mp4 saved).

Confirm the Python packages the analysis script needs are available:

```bash
python3 -c "import cv2, numpy" || pip install opencv-python-headless numpy --break-system-packages
```

`ffmpeg`/`ffprobe` are assumed to be present (they're standard in this
environment).

## Step 2 — Run the detector

```bash
python3 scripts/analyze_clips.py <raw_dir> --out-dir <workdir>/analysis
```

This scans every clip in `<raw_dir>`, extracts audio RMS energy, samples
frames for face + smile detection (OpenCV Haar cascades — no model download
needed), combines the two signals into a per-moment score, merges nearby
high-scoring moments into candidate segments, and writes:

- `<workdir>/analysis/analysis.json` — ranked candidate segments (file,
  start, end, score)
- `<workdir>/analysis/thumbnails/*.jpg` — one representative frame per
  candidate, for visual QC

If a clip has no audible reaction and no visible face reacting, it simply
won't produce candidates — that's expected, not a bug.

Useful knobs if the first pass looks off (rerun with adjusted flags rather
than trying to hand-fix `analysis.json`):
- `--percentile` (default 75) — raise it to be pickier, lower it to surface more candidates
- `--audio-weight` / `--smile-weight` (default 0.55 / 0.45) — shift emphasis if one signal is clearly more reliable for this footage (e.g. windy outdoor audio → lean more on smile detection)
- `--min-segment` (default 1.2s) — minimum candidate length
- `--top-n` (default 25) — how many candidates to keep

## Step 3 — Review candidates visually

Read `analysis.json`, then use the Read tool on a good chunk of the
thumbnails (at least the top 15-20 by score, more if the reel needs more
material) to actually look at them. For each, judge:

- Does this look like a genuine positive reaction — open smile, laughing,
  hand-to-mouth surprise, animated gesture — not just an open mouth mid-sentence
  or a random bystander?
- Is the frame usable (subject visible, not badly blurred, not mid-blink)?
- Is this a candid reaction to the portrait, not a posed photo? If a guest
  is facing/looking toward a phone or camera lens with a static held smile
  (or someone nearby is visibly holding up a phone pointed at them), that's
  a posed photo moment, not a reaction — discard it even if the audio/smile
  score is high. Candid reactions have guests looking at the tablet/portrait
  or each other, with more spontaneous, in-motion body language.

Discard candidates that don't hold up. If too few strong candidates survive
to reach ~30-45 seconds total, go back to Step 2 and loosen `--percentile`
or re-weight the signals, rather than padding with weak moments.

## Step 3b — Find portrait/detail insert shots (if present)

The detector only scores guest reactions (audio + smile), so it won't
surface clips that are just shots of the finished portrait/illustration
itself — no face reacting, so both signals stay flat. If `raw/` has these,
find them manually:

- Grab a representative frame from clips you suspect are portrait/detail
  shots with `ffmpeg -ss <t> -i <file> -frames:v 1 <tmp>.jpg` and look at
  it with the Read tool, the same way you QC'd reaction thumbnails above.
- Pick a small number (roughly 2-5, fewer than the reaction count) of the
  cleanest ones — steady, in focus, portrait clearly visible — each
  trimmed to about 1-2s.
- Note their `file`/`start`/`end` next to the surviving reaction candidates
  so they can be interleaved in Step 4.

## Reference example (optional)

If a `reference/` folder exists alongside `raw/`, it holds an example finished
video (e.g. downloaded from an Instagram post) showing the target look for
this edit. Watch it with the Read tool before Step 4 and use it to steer
curation — cut rhythm/pacing, how quickly it opens with a strong moment,
roughly how many segments it uses, and overall energy. Match the style, not
the literal content.

## Step 4 — Curate two cuts

Sort the surviving candidates into two buckets by feel, not by score:

- **Curious/confused** — calm, contemplative, "what is this" energy: staring,
  furrowed brow, head tilt, quiet close reading of the portrait. No laugh or
  gasp yet.
- **Excited/positive** — the actual reaction: laughing, gasping, hand to
  face/mouth, big smile, exclaiming.

Then build two manifests:

- **`manifest_curious_open.json`** — exactly 3 clips from 3 *different*
  guests from the curious/confused bucket, in order, followed by the best
  excited/positive clips (roughly 5-12 more segments) the same way the
  single-cut version used to work.
- **`manifest_excited_only.json`** — only excited/positive clips, no
  curious/confused ones at all. Reuse the same excited segments (and their
  crop framing from Step 4b) from the first manifest rather than
  re-curating from scratch.

When choosing segments for either manifest:

- Prefer variety — different guests/clips over repeatedly picking several
  moments from the same person if options exist. If the venue's couple
  appears in the candidates, use them sparingly (e.g. once, for the single
  strongest moment) rather than letting them dominate — this is a guest
  reaction reel, not a couple portrait video.
- Lead with one of the strongest reactions to hook a viewer in the first
  couple seconds; don't save the best moment for last on a Reel.
- Trim segment `start`/`end` slightly if the merged window includes obvious
  dead air before/after the reaction — tighter cuts read as more energetic.
- If portrait/detail shots were found in Step 3b, interleave them in both
  manifests rather than blocking them together — e.g. a portrait shot right
  before or after the reaction it belongs to, punctuating the rhythm the way
  the reference example does. They aren't "reaction clips," so they belong
  in the excited-only manifest too.
- Every reaction clip must land cleanly in exactly one bucket — curious/
  confused, or excited/positive. Don't let an ambiguous one sit in the
  curious opening just because it's not laughing/gasping; if once tightly
  cropped (Step 4b) it reads as a clear smile/laugh, it's excited, not
  curious. Re-judge the bucket *after* cropping tight on the face, not just
  from the wide establishing shot — a wide shot can hide a subject's actual
  expression that only shows once you push in.

Manifest format (a plain list, in playback order):

```json
[
  {"file": "clip_04.mp4", "start": 12.4, "end": 15.1},
  {"file": "clip_11.mov", "start": 40.0, "end": 43.8, "zoom": 1.8, "crop_x": 0.3, "crop_y": 0.4}
]
```

(`file` can be just the filename — it's resolved against `--raw-dir` — or an
absolute path. `zoom`/`crop_x`/`crop_y` are optional — see Step 4b.)

## Step 4b — Frame each reaction clip on the face

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
5. Portrait/detail insert shots don't need this — they're already
   well-framed stills of the artwork, so leave `zoom`/`crop_x`/`crop_y` at
   their defaults (1.0/0.5/0.5) for those.

**Verify every reaction clip, not just one or two** — a face being in frame
at the segment's midpoint doesn't mean it stays in frame for the whole
segment; a subject who turns, bends, or walks partway through can end up
cropped out or out of frame entirely even with a good `crop_x`/`crop_y`.
For each reaction clip in the manifest, render the same `scale`+`crop`
filter string `build_reel.py` uses at 4+ points spread across the segment
(e.g. fractions 0.05, 0.35, 0.65, 0.95 of its duration) and look at all of
them. If the subject leaves frame for part of the window, either loosen
the zoom, or — usually better — trim `start`/`end` down to the sub-window
where they're actually framed well, rather than trying to crop-track
movement with a static crop.

## Step 5 — Build both videos

```bash
python3 scripts/build_reel.py <workdir>/manifest_curious_open.json <output_dir>/highlight_reel_curious_open.mp4 --raw-dir <raw_dir>
python3 scripts/build_reel.py <workdir>/manifest_excited_only.json <output_dir>/highlight_reel_excited_only.mp4 --raw-dir <raw_dir>
```

Each run trims its segments, scales/crops them to fill a 1080x1920 vertical
frame per the `zoom`/`crop_x`/`crop_y` in the manifest (cropping to fill
rather than letterboxing — standard for Reels/TikTok), re-encodes everything
to a consistent format, concatenates in the manifest's order through a
single continuous encode (not a stream-copy concat — stream-copying
independently-encoded segments back together causes audio drift and
stutter at the cuts), and applies a short audio fade in/out. It also forces
8-bit color output (`-pix_fmt yuv420p`) regardless of source: some cameras
(action cams/gimbals especially) shoot 10-bit HDR footage, and libx264
silently preserves that into a "High 10" profile if not told otherwise —
which most phones, browsers, and Reels/TikTok itself can't decode, and
shows as a black screen instead of an error. Defaults (`--width`,
`--height`, `--fps`, `--fade`) rarely need changing unless the user asks
for a different aspect ratio or a longer fade.

## Step 6 — Verify before handing off

Don't just assume it worked — check both outputs:

```bash
ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of default=noprint_wrappers=1 <output_dir>/highlight_reel_curious_open.mp4
ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of default=noprint_wrappers=1 <output_dir>/highlight_reel_excited_only.mp4
```

Confirm resolution is 1080x1920 (or whatever was requested) and duration is
in the ~30-45s range for the curious-open version (the excited-only version
may legitimately be shorter — see "What this produces"). Also check
`ffprobe -show_entries stream=codec_type,duration` — video and audio stream
durations should match to within a fraction of a second; a growing gap
between them means the concat re-encode didn't happen correctly. Also
check `ffprobe -show_entries stream=pix_fmt,profile` — this should read
`yuv420p` / `High` (or `Main`/`Baseline`), never `yuv420p10le` / `High 10`
(see the 10-bit note in Step 5); the latter plays as a black screen in most
players. Pull a couple of frames from each result (e.g. via `ffmpeg -ss <t>
-i out.mp4 -frames:v 1 preview.jpg`) and look at them with the Read tool as
a final sanity check that the tighter crop from Step 4b didn't cut off
faces.

If the duration is way off target, adjust the manifest (add/remove/trim
segments) and rerun Step 5 rather than re-running the whole detection pass.

## Notes

- Source clips can be any common format ffmpeg reads (mp4, mov, mkv, etc.).
- This is tuned for the "one main subject reacting to camera" framing typical
  of live-illustrator guest reaction footage. If a clip has multiple people
  in frame, the detector scores based on the largest face, which is usually
  the intended subject.
- Nothing here uploads footage anywhere or requires internet access — the
  Haar cascades are bundled with OpenCV.
