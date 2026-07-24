#!/usr/bin/env python3
"""
build_reel.py — Assemble a curated list of clip segments into a single
vertical short-form video (Reels/TikTok ready) using ffmpeg.

Input is a manifest JSON: a plain list of segments IN THE ORDER you want
them to appear in the final video:

    [
      {"file": "guest1.mp4", "start": 12.4, "end": 15.1},
      {"file": "guest3.mov", "start": 40.0, "end": 43.8,
       "zoom": 2.0, "crop_x": 0.3, "crop_y": 0.4},
      ...
    ]

`file` may be a filename inside --raw-dir, or an absolute path.

`zoom`, `crop_x`, `crop_y` are optional per-segment framing controls (all
default to 1.0/0.5/0.5, i.e. the old plain centered fill-crop). `zoom` > 1.0
scales the source in before cropping, so the output frame shows a tighter
sub-region than a plain fill-crop would — use this to push in on a face
instead of showing the whole wide shot. `crop_x`/`crop_y` (0..1) then pick
where that crop window sits within the zoomed frame (0.5/0.5 = centered,
lower `crop_x` = further left, lower `crop_y` = further up).

Each segment is trimmed, scaled/cropped to fill the target frame (default
1080x1920, i.e. 9:16), and re-encoded to a consistent format so the
segments can be concatenated cleanly. A short audio fade in/out is applied
to the whole final video for polish.

Re-encoding a handful of seconds of 4K source per segment is the slow part
— on modest hardware it can take longer than a single short command
session allows if done for every segment back to back. So this script can
run incrementally: render one segment at a time with --only-index, caching
each rendered segment to disk, then do a final (fast, just concatenation)
pass with --finalize once every segment is rendered. If your environment
can comfortably run the whole thing in one go, just omit both flags and it
renders everything then finalizes automatically.

Usage:
    # all at once (fine for a handful of short/low-res segments):
    python3 build_reel.py manifest.json output.mp4 --raw-dir /path/to/raw

    # incrementally (safer for many/long/4K segments):
    python3 build_reel.py manifest.json output.mp4 --raw-dir /path/to/raw --only-index 0
    python3 build_reel.py manifest.json output.mp4 --raw-dir /path/to/raw --only-index 1
    ... (one call per segment index) ...
    python3 build_reel.py manifest.json output.mp4 --raw-dir /path/to/raw --finalize
"""
import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)


def resolve_path(file_ref, raw_dir):
    if os.path.isabs(file_ref) and os.path.exists(file_ref):
        return file_ref
    candidate = os.path.join(raw_dir, file_ref) if raw_dir else file_ref
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Could not resolve segment file: {file_ref}")


def render_segment(seg, i, cache_dir, raw_dir, width, height, fps):
    seg_out = os.path.join(cache_dir, f"seg_{i:03d}.mp4")
    if os.path.exists(seg_out):
        print(f"[{i}] already rendered, reusing {seg_out}", file=sys.stderr)
        return seg_out
    src = resolve_path(seg["file"], raw_dir)
    start, end = float(seg["start"]), float(seg["end"])
    dur = max(0.1, end - start)

    # zoom > 1.0 scales the source larger than the target frame before
    # cropping, so the crop can push in tighter than a plain fill-crop
    # would (e.g. to frame a face instead of the whole wide shot). crop_x/
    # crop_y (0..1) pick where within that zoomed-in range the crop window
    # sits — 0.5/0.5 reproduces the old centered behavior.
    zoom = float(seg.get("zoom", 1.0))
    crop_x = min(max(float(seg.get("crop_x", 0.5)), 0.0), 1.0)
    crop_y = min(max(float(seg.get("crop_y", 0.5)), 0.0), 1.0)
    scale_w, scale_h = int(round(width * zoom)), int(round(height * zoom))
    vf = (f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height}:x='(in_w-{width})*{crop_x}':y='(in_h-{height})*{crop_y}',"
          f"setsar=1,fps={fps}")
    print(f"[{i}] rendering {seg['file']} {start:.2f}-{end:.2f}s "
          f"(zoom={zoom}, crop_x={crop_x}, crop_y={crop_y}) -> {seg_out}", file=sys.stderr)
    run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(start), "-i", src, "-t", str(dur),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        seg_out,
    ])
    return seg_out


def finalize(manifest_len, cache_dir, output, fade):
    seg_paths = [os.path.join(cache_dir, f"seg_{i:03d}.mp4") for i in range(manifest_len)]
    missing = [p for p in seg_paths if not os.path.exists(p)]
    if missing:
        print(f"Cannot finalize — {len(missing)} segment(s) not yet rendered: {missing}",
              file=sys.stderr)
        sys.exit(1)

    concat_list = os.path.join(cache_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")

    # Each segment was encoded independently, so each has its own AAC encoder
    # priming/padding samples. Stream-copying them together (-c copy) at concat
    # time just concatenates those independently-timed streams as-is, which
    # accumulates into audible A/V drift and stutter at every cut. Decoding
    # and re-encoding through the concat demuxer instead produces one
    # continuous, correctly-timed stream.
    total_dur = 0.0
    for p in seg_paths:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", p],
            capture_output=True, text=True
        )
        try:
            total_dur += float(probe.stdout.strip())
        except ValueError:
            pass
    fade_out_start = max(0.0, total_dur - fade)

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-map", "0:v:0", "-map", "0:a:0", "-map_metadata", "-1",
        "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={fade_out_start}:d={fade}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        output,
    ])
    print(f"Wrote {output} ({total_dur:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="JSON file: list of {file, start, end}")
    ap.add_argument("output", help="output mp4 path")
    ap.add_argument("--raw-dir", default=None, help="base dir to resolve relative file names")
    ap.add_argument("--cache-dir", default=None,
                     help="where rendered segments are cached between incremental calls "
                          "(default: <output_dir>/.reel_cache)")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade", type=float, default=0.4, help="audio fade in/out seconds")
    ap.add_argument("--only-index", type=int, default=None,
                     help="render just this one segment index (0-based) and exit")
    ap.add_argument("--finalize", action="store_true",
                     help="skip rendering, just concatenate already-cached segments + fade")
    args = ap.parse_args()

    with open(args.manifest) as f:
        segments = json.load(f)
    if not segments:
        print("Manifest is empty — nothing to build.", file=sys.stderr)
        sys.exit(1)

    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(
        os.path.dirname(os.path.abspath(args.output)) or ".", ".reel_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if args.finalize:
        finalize(len(segments), cache_dir, args.output, args.fade)
        return

    if args.only_index is not None:
        if not (0 <= args.only_index < len(segments)):
            print(f"--only-index {args.only_index} out of range (0..{len(segments)-1})",
                  file=sys.stderr)
            sys.exit(1)
        render_segment(segments[args.only_index], args.only_index, cache_dir,
                        args.raw_dir, args.width, args.height, args.fps)
        return

    # no incremental flags given — render everything then finalize in one go
    for i, seg in enumerate(segments):
        render_segment(seg, i, cache_dir, args.raw_dir, args.width, args.height, args.fps)
    finalize(len(segments), cache_dir, args.output, args.fade)


if __name__ == "__main__":
    main()
