#!/usr/bin/env python3
"""
export_clips.py — Render a curated list of clip segments as SEPARATE output
files, one per segment, instead of stitching them into a single video.

Input is a manifest JSON: a plain list of segments, in any order:

    [
      {"file": "guest1.mp4", "start": 12.4, "end": 15.1, "category": "excited"},
      {"file": "guest3.mov", "start": 40.0, "end": 43.8, "category": "curious",
       "zoom": 1.8, "crop_x": 0.3, "crop_y": 0.4},
      ...
    ]

`file` may be a filename inside --raw-dir, or an absolute path.

`category` (optional, defaults to "clip") groups outputs into subfolders of
--output-dir, e.g. output/excited/, output/curious/ — handy for scrolling
through one mood at a time.

`zoom`, `crop_x`, `crop_y` are optional per-segment framing controls (all
default to 1.0/0.5/0.5, i.e. a plain centered fill-crop). `zoom` > 1.0 scales
the source in before cropping so the output frame shows a tighter sub-region
— use this to push in on a reacting face instead of showing the whole wide
shot. `crop_x`/`crop_y` (0..1) then pick where that crop window sits within
the zoomed frame (0.5/0.5 = centered).

Each segment is trimmed, scaled/cropped to fill the target frame (default
1080x1920, i.e. 9:16), and re-encoded with a short audio fade in/out. Unlike
build_reel.py, there's no concatenation step — each clip is a standalone,
final output file, so there's no cross-segment audio-drift concern and no
separate --finalize pass needed.

Rendering many/long/4K segments back to back can be slow, so this can also
run incrementally: render one segment at a time with --only-index. If your
environment can comfortably run the whole batch in one go, just omit that
flag and it renders every segment in the manifest.

Usage:
    # all at once:
    python3 export_clips.py manifest.json <output_dir> --raw-dir /path/to/raw

    # incrementally (safer for many/long/4K segments):
    python3 export_clips.py manifest.json <output_dir> --raw-dir /path/to/raw --only-index 0
    python3 export_clips.py manifest.json <output_dir> --raw-dir /path/to/raw --only-index 1
    ...
"""
import argparse
import json
import os
import re
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


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def output_path(seg, i, output_dir):
    category = sanitize(seg.get("category", "clip")) or "clip"
    cat_dir = os.path.join(output_dir, category)
    base = sanitize(os.path.splitext(os.path.basename(seg["file"]))[0])
    start, end = float(seg["start"]), float(seg["end"])
    fname = f"{i:03d}_{base}_{start:.1f}s-{end:.1f}s.mp4"
    return cat_dir, os.path.join(cat_dir, fname)


def render_clip(seg, i, output_dir, raw_dir, width, height, fps, fade):
    cat_dir, out_path = output_path(seg, i, output_dir)
    os.makedirs(cat_dir, exist_ok=True)
    if os.path.exists(out_path):
        print(f"[{i}] already rendered, skipping {out_path}", file=sys.stderr)
        return out_path

    src = resolve_path(seg["file"], raw_dir)
    start, end = float(seg["start"]), float(seg["end"])
    dur = max(0.1, end - start)
    clip_fade = min(fade, dur / 2 - 0.01) if dur > 0.1 else 0.0
    clip_fade = max(0.0, clip_fade)

    zoom = float(seg.get("zoom", 1.0))
    crop_x = min(max(float(seg.get("crop_x", 0.5)), 0.0), 1.0)
    crop_y = min(max(float(seg.get("crop_y", 0.5)), 0.0), 1.0)
    scale_w, scale_h = int(round(width * zoom)), int(round(height * zoom))
    vf = (f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height}:x='(in_w-{width})*{crop_x}':y='(in_h-{height})*{crop_y}',"
          f"setsar=1,fps={fps}")
    fade_out_start = max(0.0, dur - clip_fade)
    af = f"afade=t=in:st=0:d={clip_fade},afade=t=out:st={fade_out_start}:d={clip_fade}" if clip_fade > 0 else "anull"

    print(f"[{i}] rendering {seg['file']} {start:.2f}-{end:.2f}s "
          f"(category={seg.get('category', 'clip')}, zoom={zoom}, crop_x={crop_x}, crop_y={crop_y}) "
          f"-> {out_path}", file=sys.stderr)
    run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(start), "-i", src, "-t", str(dur),
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="JSON file: list of {file, start, end, category?}")
    ap.add_argument("output_dir", help="directory to write individual clip files into "
                                        "(one subfolder per category)")
    ap.add_argument("--raw-dir", default=None, help="base dir to resolve relative file names")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade", type=float, default=0.15,
                     help="audio fade in/out seconds per clip (kept short since these are "
                          "short standalone clips, not a single long video)")
    ap.add_argument("--only-index", type=int, default=None,
                     help="render just this one segment index (0-based) and exit")
    args = ap.parse_args()

    with open(args.manifest) as f:
        segments = json.load(f)
    if not segments:
        print("Manifest is empty — nothing to export.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.only_index is not None:
        if not (0 <= args.only_index < len(segments)):
            print(f"--only-index {args.only_index} out of range (0..{len(segments)-1})",
                  file=sys.stderr)
            sys.exit(1)
        render_clip(segments[args.only_index], args.only_index, args.output_dir,
                    args.raw_dir, args.width, args.height, args.fps, args.fade)
        return

    written = []
    for i, seg in enumerate(segments):
        written.append(render_clip(seg, i, args.output_dir, args.raw_dir,
                                    args.width, args.height, args.fps, args.fade))
    print(f"\nWrote {len(written)} clip(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
