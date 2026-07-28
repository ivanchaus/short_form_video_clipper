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

Since this skill exports every worthwhile candidate rather than curating
down to a fixed ~30-45s runtime, lean toward a looser `--percentile` and a
higher `--top-n` than you would for a compiled reel — the goal is
comprehensive coverage of good moments, not a tight cut.

Useful knobs if the first pass looks off (rerun with adjusted flags rather
than trying to hand-fix `analysis.json`):
- `--percentile` (default 75) — raise it to be pickier, lower it to surface more candidates
- `--audio-weight` / `--smile-weight` (default 0.55 / 0.45) — shift emphasis if one signal is clearly more reliable for this footage (e.g. windy outdoor audio → lean more on smile detection)
- `--min-segment` (default 1.2s) — minimum candidate length
- `--top-n` (default 25) — how many candidates to keep; raise this well above the reel default since nothing here gets cut for runtime

## Step 3 — Review candidates visually

Read `analysis.json`, then use the Read tool on the thumbnails — for this
skill, look at *all* of them, not just the top 15-20, since the goal is to
export every good moment rather than curate a short list. For each, judge:

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
