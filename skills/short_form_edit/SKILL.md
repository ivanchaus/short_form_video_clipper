---
name: short_form_edit
description: Turns raw wedding-live-illustrator reaction footage into vertical highlight reels and a clean all-selected compilation for Instagram Reels / TikTok. Use when the user asks for a reel, highlight video, TikTok, compilation, best reactions, or stitched reaction edit from a footage folder.
---

# Short-form reaction highlight editor

## What this produces

Three vertical (9:16) .mp4 outputs by default, built from the same pool of
curated candidates for different review and publishing needs:

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
3. **All-selected no-text compilation** — every accepted curious and excited
   reaction exactly once, normally with the curious clips first and the
   excited clips after them to create a natural build. It has no hook,
   captions, titles, or other text overlays. This is intentionally longer
   than the two highlight edits and serves as a clean master compilation.

The first two are roughly 30-45 seconds (the excited-only version may run
shorter — see above), made of the best guest-reaction moments cut together.
They are highlight reels, not full recaps: they should feel punchy, not
padded. The all-selected compilation has no target duration because complete
coverage is its purpose. If the user asks for fewer outputs, follow that
request instead of forcing all three.

**The first two versions never share a clip.** Every reaction clip is used in
curious-open *or* excited-only, never both — the two outputs need to feel
like distinct edits, not the same video with a different intro stapled on.
This means the excited/positive clip pool has to stretch across both
videos, which pushes both durations toward the shorter end — that's an
expected tradeoff for distinct cuts, not a bug to fix by padding.

The all-selected no-text compilation is the deliberate exception: it combines
the complete accepted pool from both categories, so it necessarily contains
the clips assigned to both highlight edits.

By default, each of the two short highlight versions opens with a bold
on-screen hook line (the
"POV: guests didn't know what 'live portraits' actually meant" style seen
on this kind of Reel) over the first clip, and any reaction clip with a
clear, worth-quoting audible reaction gets the guest's actual words shown
on screen as a caption. See Step 4c.

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

For any request that asks to search all footage, use the comprehensive
pipeline supplied by the sibling `best_reaction_clips` skill:

```bash
python3 <skills_dir>/best_reaction_clips/scripts/scan_pipeline.py <raw_dir> <workdir> \
  --percentile 45 --top-n 150
```

Running only `scripts/analyze_clips.py` is a partial audio-led diagnostic;
it cannot discover quiet facial reactions outside audio-selected windows.

This scans every clip in `<raw_dir>`, extracts audio RMS energy, samples
frames for face + smile detection (OpenCV Haar cascades — no model download
needed), treats the two signals as independent evidence, merges nearby
high-scoring moments into candidate segments, and writes:

- `<workdir>/analysis/analysis.json` — ranked candidate segments (file,
  start, end, score)
- `<workdir>/analysis/thumbnails/*.jpg` — a time-labeled nine-frame contact
  sheet spanning the candidate plus 4s pre-roll and 2s post-roll
- `<workdir>/dense/` — 1s full-timeline visual coverage for quiet faces
- `<workdir>/motion/` — gapless motion-ranked candidates for brief gestures

`analysis.json` is not a coverage guarantee. Haar cascades commonly return
0 for small/profile faces and open-mouth laughter or screams, and quiet
expressions may never enter an audio-selected window. The dense and motion
sweeps are required; lowering `--percentile` cannot recover a moment that
was never eligible.

Useful knobs if the first pass looks off (rerun with adjusted flags rather
than trying to hand-fix `analysis.json`):
- `--percentile` (default 75) — raise it to be pickier, lower it to surface more candidates
- `--audio-weight` / `--smile-weight` (default 0.55 / 0.45) — shift emphasis if one signal is clearly more reliable for this footage (e.g. windy outdoor audio → lean more on smile detection)
- `--min-segment` (default 1.2s) — minimum candidate length
- `--top-n` (default 25) — how many candidates to keep
- `--review-preroll` / `--review-postroll` (defaults 4s / 2s) — context
  around each detected audio window so earlier/later facial beats remain visible

Face/smile detection is positive evidence, never a gate or penalty. The
analyzer ranks each audio-selected segment by the better of its audio-only
and audio+smile scores, so a failed smile cascade cannot demote a scream.
Still inspect the actual before/after extent: a long, multi-beat reaction
condensed into one short segment throws away the strongest story beat.

## Step 3 — Review candidates visually and audibly

Inspect every multi-frame candidate sheet and every grid in both
`motion/grids/` and `dense/grids/`; do not stop once the reel has enough
runtime. Maintain a per-source-file coverage ledger and do not curate the
reels until every source appears in the dense manifest and every grid has
been reviewed. Ranking is triage, not permission to ignore unranked
footage. For each, judge:

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

Review audio content before curation, not only later when adding captions.
Audition every audio-led candidate across `review_start` to `review_end`;
short clear exclamations such as "oh my God," "wow," or "no way" can make a
clip a keeper even when the face cascade returns 0. If direct listening is
unavailable, require agreement between raw-audio and cleaned-audio
transcription attempts before treating words as real. Extend the selected
segment to contain the complete phrase and its associated facial reaction.

Discard candidates that don't hold up. For any candidate at least 5s long
with `smile_score <= 0.1`, inspect between the nine context-sheet frames at
0.25-0.5s spacing before discarding it; open-mouth screams are a common
zero-smile false negative. If too few strong candidates survive to reach
~30-45 seconds total, rerun the comprehensive pipeline with denser targeted
sampling rather than padding with weak moments.

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

## Step 4 — Curate two highlight cuts and one clean compilation

Sort the surviving candidates into two buckets by feel, not by score:

- **Curious/confused** — calm, contemplative, "what is this" energy: staring,
  furrowed brow, head tilt, quiet close reading of the portrait. No laugh or
  gasp yet.
- **Excited/positive** — the actual reaction: laughing, gasping, hand to
  face/mouth, big smile, exclaiming.

First build the two short highlight manifests:

- **`manifest_curious_open.json`** — exactly 3 clips from 3 *different*
  guests from the curious/confused bucket, in order, followed by excited/
  positive clips the same way the single-cut version used to work.
- **`manifest_excited_only.json`** — only excited/positive clips, no
  curious/confused ones at all.

Also build **`manifest_all_reactions_no_text.json`** from the complete accepted
pool, not merely the subset used in the two shorter reels. Put every curious
clip first and every excited clip second unless chronology or the user's
requested ordering calls for something else. Include each accepted reaction
once. Preserve its verified `start`, `end`, `zoom`, `crop_x`, and `crop_y`, but
omit `hook_text` and every `caption` field.

Split the excited/positive pool between the two manifests — don't put the
same clip in both (see "What this produces"). Decide the split by how many
curious clips vs. excited clips each version needs to feel complete: since
curious-open already has its 3-clip curious section, it needs fewer excited
clips to feel substantial than excited-only does, which is carrying the
entire video on excited clips alone. Weight the split accordingly rather
than 50/50.

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

Manifest format — an object with an optional intro hook line plus the
segment list, in playback order:

```json
{
  "hook_text": "POV: guests didn't know what 'live portraits' actually meant",
  "segments": [
    {"file": "clip_04.mp4", "start": 12.4, "end": 15.1},
    {"file": "clip_11.mov", "start": 40.0, "end": 43.8, "zoom": 1.8, "crop_x": 0.3, "crop_y": 0.4,
     "caption": "You are so talented!"}
  ]
}
```

(`file` can be just the filename — it's resolved against `--raw-dir` — or an
absolute path. `zoom`/`crop_x`/`crop_y` are optional — see Step 4b.
`hook_text`/`caption` are optional — see Step 4c. A plain list with no
`hook_text` also still works, for a version with no text overlays at all.)

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

## Step 4c — Hook line and speech captions

Two kinds of on-screen text, both rendered by `build_reel.py` itself (via
PIL — this ffmpeg build has no `drawtext`/libfreetype, so text is rendered
to a transparent PNG and composited with the `overlay` filter instead):

**Hook line** (`hook_text`, one per manifest, shown over the first clip
only): a short, punchy line that frames what the viewer is about to watch,
Reels-style — e.g. "POV: guests didn't know what 'live portraits' actually
meant." Write one hook per version, not a shared one — curious-open and
excited-only are different videos with different opening beats, so the
hook should match what that specific version actually delivers (curious-
open can set up a "they had no idea..." payoff since it *has* the
confusion-to-payoff arc; excited-only has no build-up section, so its hook
should promise the reactions directly rather than a twist it doesn't have).
Pick the hook based on the actual clips chosen, optimizing for what would
make someone stop scrolling and watch to the end.

**Speech captions** (`caption`, per-segment, optional): the guest's actual
words, shown like the reference video's on-screen quotes. Only add one
where a clip has a clear, worth-quoting audible reaction — not every clip
needs one, and a caption on a clip with no real audible reaction (or a
garbled one) is worse than no caption at all.

**Whisper hallucinates fluently on this kind of audio — a single
transcription pass is not trustworthy enough to caption with.** On this
footage's distant venue-cam mic, outdoor noise, and overlapping voices, a
single pass with the "small" model produced confident-sounding but wrong
text on nearly every clip (e.g. "I hate people" over a clearly happy
couple). Bumping to "medium" alone doesn't fix it — it just produces
*different* wrong text. What actually works is treating agreement between
independent attempts as the accuracy signal, since hallucinations aren't
consistent across runs but real speech is:

```bash
python3 -c "import whisper" || pip install openai-whisper --break-system-packages
```

For each reaction clip, extract audio two ways — raw, and with a denoise/
highpass pass to cut venue rumble and wind — then transcribe both with the
`medium` model (small is noticeably worse on this kind of audio; the
accuracy is worth the extra size/time for the handful of clips this
applies to):

```bash
ffmpeg -ss <start> -i <file> -t <dur> -ar 16000 -ac 1 clip_raw.wav
ffmpeg -ss <start> -i <file> -t <dur> -ar 16000 -ac 1 \
  -af "highpass=f=120,afftdn=nf=-25,dynaudnorm" clip_clean.wav
```

```python
import whisper
model = whisper.load_model("medium")
for f in ["clip_raw.wav", "clip_clean.wav"]:
    print(f, "->", model.transcribe(f, fp16=False, language="en")["text"])
```

(Pad the extraction window by ~1s on each side of the visual segment for
context, independent of what `start`/`end` you use in the manifest. The
one-time model download, ~1.5GB for "medium", is the only part of this
skill that needs internet access.)

**Only caption a clip if the raw and clean transcripts tell the same
story** — not necessarily word-for-word, but clearly the same sentence.
When they agree (e.g. both land on "you both look very handsome... thank
you so much"), that's real signal. When they diverge into unrelated text,
that's exactly what happened on most clips in testing here, and it means
there's no reliable transcript — drop the caption for that clip rather
than picking whichever version sounds more plausible.

**Specifically distrust "Thank you for watching!" / "Thanks for
watching!"** — this is a well-known Whisper hallucination on silence or
noise-only input (it's copying training-data patterns from YouTube outros),
and it showed up repeatedly on clips with no real speech in this footage.
Treat it as a certain sign there's nothing to caption, not as content.

Before locking in a transcript as a caption, also check:
- Does it make sense in context (a guest reacting to seeing their
  portrait)? Discard anything generic, contradictory, or unrelated.
- Is it short enough to read in the clip's duration (roughly under ~8
  words for a 2-3s clip)? Trim filler if needed, but don't put words in
  someone's mouth they didn't say.

In practice, expect most candidate clips to fail this bar — on this
footage only 1 of ~10 candidates checked produced a transcript worth
using. That's expected, not a sign something's wrong with the process;
a wrong caption is worse than no caption.

## Step 5 — Build all three videos

```bash
python3 scripts/build_reel.py <workdir>/manifest_curious_open.json <output_dir>/highlight_reel_curious_open.mp4 --raw-dir <raw_dir>
python3 scripts/build_reel.py <workdir>/manifest_excited_only.json <output_dir>/highlight_reel_excited_only.mp4 --raw-dir <raw_dir>
python3 scripts/build_reel.py <workdir>/manifest_all_reactions_no_text.json <output_dir>/compiled_all_reactions_no_text.mp4 --raw-dir <raw_dir> --no-text --cache-dir <workdir>/reel_cache_all_no_text
```

Always pass `--no-text` for the all-selected compilation. It suppresses both
manifest-level hooks and per-segment captions even if text fields were copied
in accidentally. Use a dedicated cache directory for this output so a segment
previously rendered with an overlay cannot be reused.

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
shows as a black screen instead of an error. It also writes with
`-movflags +faststart`, which puts the `moov` atom (the file's index) at
the *start* of the file instead of the end — without it, ffmpeg's default
mp4 layout writes `moov` last, and plenty of players/previewers (browsers,
Finder/Quick Look, many upload pipelines) either stall partway through
playback waiting for it or need to download the whole file first, which
reads as the video randomly freezing a few seconds in even though the
encode itself is perfectly fine frame-by-frame. Defaults (`--width`,
`--height`, `--fps`, `--fade`) rarely need changing unless the user asks
for a different aspect ratio or a longer fade.

## Step 6 — Verify before handing off

Don't just assume it worked — check all outputs:

```bash
ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of default=noprint_wrappers=1 <output_dir>/highlight_reel_curious_open.mp4
ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of default=noprint_wrappers=1 <output_dir>/highlight_reel_excited_only.mp4
ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of default=noprint_wrappers=1 <output_dir>/compiled_all_reactions_no_text.mp4
```

For the all-selected compilation, verify that its segment count equals the
number of accepted standalone reaction clips, that each accepted interval
appears exactly once, and that sampled frames contain no text overlays.

Confirm resolution is 1080x1920 (or whatever was requested) and duration is
in the ~30-45s range for the curious-open version (the excited-only version
may legitimately be shorter — see "What this produces"). Also check
`ffprobe -show_entries stream=codec_type,duration` — video and audio stream
durations should match to within a fraction of a second; a growing gap
between them means the concat re-encode didn't happen correctly. Also
check `ffprobe -show_entries stream=pix_fmt,profile` — this should read
`yuv420p` / `High` (or `Main`/`Baseline`), never `yuv420p10le` / `High 10`
(see the 10-bit note in Step 5); the latter plays as a black screen in most
players. Also confirm the `moov` atom is first: `ffprobe -v trace out.mp4
2>&1 | grep -E "type:'(moov|mdat)'"` should list `moov` *before* `mdat` —
if `mdat` comes first, playback will stall/freeze partway through in a lot
of players even though the encode itself is fine (see the faststart note
in Step 5). Pull a couple of frames from each result (e.g. via `ffmpeg -ss
<t> -i out.mp4 -frames:v 1 preview.jpg`) and look at them with the Read
tool as a final sanity check that the tighter crop from Step 4b didn't cut
off faces.

If the duration is way off target, adjust the manifest (add/remove/trim
segments) and rerun Step 5 rather than re-running the whole detection pass.

## Notes

- Source clips can be any common format ffmpeg reads (mp4, mov, mkv, etc.).
- This is tuned for the "one main subject reacting to camera" framing typical
  of live-illustrator guest reaction footage. If a clip has multiple people
  in frame, the detector scores based on the largest face, which is usually
  the intended subject.
- Nothing here uploads footage anywhere. Detection (Haar cascades) needs no
  internet access; speech captioning (Step 4c) does, once, to download the
  whisper model — everything still runs and transcribes locally after that,
  audio never leaves the machine.
