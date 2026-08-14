#!/usr/bin/env python3
"""
scan_pipeline.py — One-command wrapper around the mechanical (non-judgment)
steps of finding reaction moments in a raw/ folder, so a single CLI
invocation replaces dozens of individual Claude tool calls:

  1. audio+smile candidate detection (calls analyze_clips.py)
  2. a dense, fixed-interval frame sweep across every file (catches quiet
     gestures like pointing that never trigger an audio spike)
  3. a motion-ranked scan: decode each file at a low fixed fps, score
     frame-to-frame pixel motion, and surface only the top spikes as
     individually-legible frames. This exists because the fixed-interval
     sweep has two blind spots the motion scan is specifically built to
     cover: (a) gestures shorter than the sweep interval can fall entirely
     between samples (e.g. a ~0.5s thumbs up), and (b) a real reaction
     packed into a 280px grid cell among dozens of others reads as
     indistinguishable from generic crowd milling. Ranking by motion lets
     sampling be much denser (cheap) while keeping the reviewed set small
     (the expensive part), and surfaces exactly the "someone's arm just
     moved fast" moments — pointing, raising a phone/portrait, a thumbs
     up — that both audio detection and eyeballing tiny grids missed.
  4. tiling frames into grid montages sized for a single Read/view each,
     with a manifest so any interesting grid cell can be traced back to
     its exact file + timestamp

Nothing in this script makes a judgment call about what's a "good"
reaction — that still needs a human or Claude to actually look at the
thumbnails/grids this produces. The point is to do all the ffmpeg/IO
work in one background-able process instead of one-tool-call-at-a-time,
which is what was burning excess turns/credits previously.

Usage:
    python3 scan_pipeline.py <raw_dir> <out_dir> [options]

Output layout under <out_dir>:
    analysis/analysis.json         — ranked audio+smile candidates
    analysis/thumbnails/*.jpg      — one thumb per candidate
    dense/manifest.json            — [{idx, file, t}, ...] for every dense frame
    dense/frames/fNNNN.jpg         — raw extracted frames (kept, in case a
                                      grid cell needs a closer look)
    dense/grids/gridNNN.jpg        — frames tiled cols x rows per image
    motion/manifest.json           — [{idx, file, t, score}, ...] for the
                                      top motion-spike candidates, ranked
    motion/frames/fNNNN.jpg        — individually-rendered candidate frames
                                      (larger than dense thumbs — few enough
                                      to afford it)
    motion/grids/gridNNN.jpg       — motion candidates tiled few-per-image
                                      so each one stays legible
    pipeline.log                   — combined stdout/stderr of every step

After this finishes, review motion/grids first (small set, high hit rate,
each cell big enough to actually see a gesture), then analysis/thumbnails
and dense/grids for anything with no motion signature (e.g. a static held
pose with no arm movement to trigger on). Note promising (grid_index, row,
col) -> compute the flat index `grid_index * cols*rows + row*cols + col`,
look it up in the corresponding manifest.json for file/timestamp, then zoom
in with ffmpeg to confirm before adding to a manifest for export_clips.py /
build_reel.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def log(msg, logf):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def run_audio_smile_detection(raw_dir, out_dir, args, logf):
    analysis_dir = os.path.join(out_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_clips.py")
    cmd = [
        sys.executable, script, raw_dir,
        "--out-dir", analysis_dir,
        "--percentile", str(args.percentile),
        "--top-n", str(args.top_n),
        "--max-per-file", str(args.max_per_file),
        "--audio-weight", str(args.audio_weight),
        "--smile-weight", str(args.smile_weight),
    ]
    log(f"Running audio+smile detection: {' '.join(cmd)}", logf)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logf.write(proc.stdout)
    logf.write(proc.stderr)
    logf.flush()
    if proc.returncode != 0:
        log("analyze_clips.py FAILED — see pipeline.log for details", logf)
    else:
        log("Audio+smile detection done.", logf)
    return analysis_dir


def collect_files(raw_dir, skip_files, skip_substrings):
    files = []
    for f in sorted(os.listdir(raw_dir)):
        if os.path.splitext(f)[1].lower() not in VIDEO_EXTS:
            continue
        if f in skip_files:
            continue
        if any(sub in f for sub in skip_substrings):
            continue
        dur = ffprobe_duration(os.path.join(raw_dir, f))
        if dur <= 0:
            continue
        files.append((f, dur))
    return files


def run_dense_sweep(raw_dir, out_dir, args, logf):
    dense_dir = os.path.join(out_dir, "dense")
    frames_dir = os.path.join(dense_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    skip_files = set(f.strip() for f in args.skip_files.split(",") if f.strip())
    skip_substrings = [s.strip() for s in args.skip_substrings.split(",") if s.strip()]
    files = collect_files(raw_dir, skip_files, skip_substrings)

    manifest = []
    idx = 0
    for fname, dur in files:
        t = 0.0
        while t < dur:
            manifest.append({"idx": idx, "file": fname, "t": round(t, 2)})
            idx += 1
            t += args.dense_interval

    manifest_path = os.path.join(dense_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    log(f"Dense sweep: {len(manifest)} frames across {len(files)} files "
        f"(interval={args.dense_interval}s)", logf)

    durations = dict(files)
    for m in manifest:
        out = os.path.join(frames_dir, f"f{m['idx']:04d}.jpg")
        if os.path.exists(out):
            continue
        # clamp away from the exact end of the file — seeking within ~0.15s
        # of true duration can land past the last decodable frame and make
        # ffmpeg silently write nothing, leaving a gap in the sweep
        seek_t = min(m["t"], max(0.0, durations[m["file"]] - 0.15))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", str(seek_t), "-i",
             os.path.join(raw_dir, m["file"]), "-frames:v", "1",
             "-vf", f"scale={args.thumb_size}:{args.thumb_size}", out],
            capture_output=True
        )
        if not os.path.exists(out):
            log(f"  WARNING: frame {m['idx']:04d} ({m['file']} @ {m['t']}s) "
                f"failed to extract — grid tiling will show a gap here", logf)
        if m["idx"] % 100 == 0:
            log(f"  extraction progress: {m['idx']}/{len(manifest)}", logf)

    log("Dense frame extraction done.", logf)
    return dense_dir, manifest


def build_grids(base_dir, manifest, cols, rows, logf):
    frames_dir = os.path.join(base_dir, "frames")
    grids_dir = os.path.join(base_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)

    per_grid = cols * rows
    n = len(manifest)
    for gi in range(0, n, per_grid):
        grid_idx = gi // per_grid
        grid_out = os.path.join(grids_dir, f"grid{grid_idx:03d}.jpg")
        if os.path.exists(grid_out):
            continue
        chunk = manifest[gi:gi + per_grid]
        while len(chunk) < per_grid and chunk:
            chunk.append(chunk[-1])  # pad last row for a clean tile
        row_imgs = []
        for r in range(rows):
            row_chunk = chunk[r * cols:(r + 1) * cols]
            if not row_chunk:
                break
            row_paths = []
            last_good = None
            for m in row_chunk:
                p = os.path.join(frames_dir, f"f{m['idx']:04d}.jpg")
                if not os.path.exists(p):
                    # extraction failed for this one frame (e.g. seek landed
                    # past the last decodable frame) — reuse a neighbor so
                    # the grid still tiles cleanly instead of crashing
                    p = last_good or p
                else:
                    last_good = p
                row_paths.append(p)
            if len(row_paths) < 2:
                row_imgs.append(row_paths[0])
                continue
            row_out = os.path.join(grids_dir, f"_row_{grid_idx}_{r}.jpg")
            cmd = ["ffmpeg", "-y", "-v", "error"]
            for p in row_paths:
                cmd += ["-i", p]
            cmd += ["-filter_complex", f"hstack=inputs={len(row_paths)}", row_out]
            subprocess.run(cmd, capture_output=True)
            row_imgs.append(row_out)
        if len(row_imgs) < 2:
            os.replace(row_imgs[0], grid_out)
            continue
        cmd = ["ffmpeg", "-y", "-v", "error"]
        for p in row_imgs:
            cmd += ["-i", p]
        cmd += ["-filter_complex", f"vstack=inputs={len(row_imgs)}", grid_out]
        subprocess.run(cmd, capture_output=True)
        for p in row_imgs:
            if os.path.exists(p) and p != grid_out:
                os.remove(p)
        if grid_idx % 5 == 0:
            log(f"  grid {grid_idx} done ({gi}-{gi + len(chunk)})", logf)

    log(f"Grid building done: {(n + per_grid - 1)//per_grid} grids "
        f"({cols}x{rows} each) in {grids_dir}", logf)


def _block_max_diff(gray_a, gray_b, blocks):
    """Max mean abs-diff over an NxN grid of blocks, not the whole-frame
    mean. A hand/arm gesture only changes a small patch of a crowded wide
    shot; averaging over the whole frame buries it under ambient crowd
    motion. Taking the hottest block instead makes a small localized
    gesture register even when the rest of the frame is calm-ish."""
    import cv2
    diff = cv2.absdiff(gray_a, gray_b)
    h, w = diff.shape
    bh, bw = max(1, h // blocks), max(1, w // blocks)
    best = 0.0
    for by in range(0, h, bh):
        for bx in range(0, w, bw):
            block = diff[by:by + bh, bx:bx + bw]
            if block.size == 0:
                continue
            m = float(block.mean())
            if m > best:
                best = m
    return best


def run_motion_scan(raw_dir, out_dir, args, logf):
    import cv2
    import statistics

    motion_dir = os.path.join(out_dir, "motion")
    frames_dir = os.path.join(motion_dir, "frames")
    scratch_dir = os.path.join(motion_dir, "_scratch")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    scores_cache = os.path.join(motion_dir, "all_scores.json")
    if args.motion_reuse_scores and os.path.exists(scores_cache):
        with open(scores_cache) as f:
            per_file_scores = {k: [tuple(p) for p in v] for k, v in json.load(f).items()}
        log(f"Loaded cached per-frame scores from {scores_cache} "
            f"(--motion-reuse-scores) — skipping decode/scoring.", logf)
    else:
        skip_files = set(f.strip() for f in args.skip_files.split(",") if f.strip())
        skip_substrings = [s.strip() for s in args.skip_substrings.split(",") if s.strip()]
        files = collect_files(raw_dir, skip_files, skip_substrings)

        # a single continuous decode per file at a fixed fps — no seeking, no
        # gaps. This used to mean decoding every native 4K frame just to keep
        # a fraction of them, which was the actual bottleneck (~2s of CPU decode
        # per second of source footage). Hardware decode (VideoToolbox on this
        # macOS box) cuts that ~7x, and since decode cost is dominated by native
        # frame count rather than the target fps, there's little reason not to
        # sample fairly densely (motion_fps default 3.0 => ~0.33s resolution).
        hwaccel_args = ["-hwaccel", args.motion_hwaccel] if args.motion_hwaccel != "none" else []
        per_file_scores = {}
        for fi, (fname, dur) in enumerate(files):
            for f in os.listdir(scratch_dir):
                os.remove(os.path.join(scratch_dir, f))
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error"] + hwaccel_args +
                ["-i", os.path.join(raw_dir, fname),
                 "-vf", f"fps={args.motion_fps},scale={args.motion_scale}:-1",
                 os.path.join(scratch_dir, "s%05d.jpg")],
                capture_output=True
            )
            scratch_files = sorted(os.listdir(scratch_dir))
            scores = []
            prev_gray = None
            for i, sf in enumerate(scratch_files):
                img = cv2.imread(os.path.join(scratch_dir, sf))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                t = round(i / args.motion_fps, 2)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    scores.append((t, _block_max_diff(gray, prev_gray, args.motion_blocks)))
                prev_gray = gray
            per_file_scores[fname] = scores
            if fi % 10 == 0:
                log(f"  motion-scored {fi}/{len(files)} files "
                    f"({fname}: {len(scores)} samples)", logf)

        for f in os.listdir(scratch_dir):
            os.remove(os.path.join(scratch_dir, f))
        os.rmdir(scratch_dir)

        # cache raw scores so re-tuning the picking logic (top-n, min-gap,
        # per-file cap) later doesn't require redoing the expensive decode —
        # rerun with --motion-reuse-scores to reuse this
        with open(scores_cache, "w") as f:
            json.dump(per_file_scores, f)

    total = sum(len(v) for v in per_file_scores.values())
    log(f"Motion scoring done: {total} scored samples across "
        f"{len(per_file_scores)} files.", logf)

    # rank by how far a sample stands out from ITS OWN file's baseline
    # (robust z-score via median/MAD), not by raw score pooled globally —
    # otherwise a file with a lot of ambient camera panning dominates the
    # top-N and quieter files with one real, localized spike get crowded out
    normalized = []
    for fname, scores in per_file_scores.items():
        vals = [s for _, s in scores]
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals]) or 1.0
        for t, s in scores:
            normalized.append({"file": fname, "t": t, "score": round(s, 2),
                                "z": round((s - med) / mad, 2)})

    normalized.sort(key=lambda c: -c["z"])
    picked = []
    picked_t_by_file = {}
    count_by_file = {}
    for c in normalized:
        existing = picked_t_by_file.setdefault(c["file"], [])
        if any(abs(c["t"] - e) < args.motion_min_gap for e in existing):
            continue
        # cap per-file so one long/busy file (lots of legitimate local peaks
        # just by having more timeline) can't crowd out the one real spike
        # in an otherwise-quiet short file
        if count_by_file.get(c["file"], 0) >= args.motion_max_per_file:
            continue
        existing.append(c["t"])
        count_by_file[c["file"]] = count_by_file.get(c["file"], 0) + 1
        picked.append(c)
        if len(picked) >= args.motion_top_n:
            break
    picked.sort(key=lambda c: (c["file"], c["t"]))

    manifest = []
    for i, c in enumerate(picked):
        out_path = os.path.join(frames_dir, f"f{i:04d}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", str(c["t"]), "-i",
             os.path.join(raw_dir, c["file"]), "-frames:v", "1",
             "-vf", f"scale={args.motion_out_size}:-1", out_path],
            capture_output=True
        )
        manifest.append({"idx": i, "file": c["file"], "t": c["t"],
                          "score": c["score"], "z": c["z"]})

    manifest_path = os.path.join(motion_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"Motion scan done: top {len(manifest)} motion-spike candidates "
        f"(min_gap={args.motion_min_gap}s, ranked by per-file z-score) "
        f"in {frames_dir}", logf)
    return motion_dir, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--skip-audio-detection", action="store_true",
                     help="skip the analyze_clips.py pass (dense sweep only)")
    ap.add_argument("--skip-dense-sweep", action="store_true",
                     help="skip the dense frame sweep (audio detection only)")
    # audio+smile detection knobs (passed through to analyze_clips.py)
    ap.add_argument("--percentile", type=float, default=45)
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--max-per-file", type=int, default=15)
    ap.add_argument("--audio-weight", type=float, default=0.6)
    ap.add_argument("--smile-weight", type=float, default=0.4)
    # dense sweep knobs
    ap.add_argument("--dense-interval", type=float, default=3.0,
                     help="seconds between frames in the dense sweep")
    ap.add_argument("--thumb-size", type=int, default=280,
                     help="square thumbnail size for dense-sweep frames")
    ap.add_argument("--skip-files", default="",
                     help="comma-separated exact filenames to exclude from the "
                          "dense sweep (e.g. files already deep-scanned before)")
    ap.add_argument("--skip-substrings", default="video 2.mp4",
                     help="comma-separated substrings; a file is excluded from the "
                          "dense sweep if any substring matches (default filters out "
                          "'video 2.mp4' duplicate-copy files)")
    ap.add_argument("--grid-cols", type=int, default=8)
    ap.add_argument("--grid-rows", type=int, default=6)
    # motion scan knobs
    ap.add_argument("--skip-motion-scan", action="store_true",
                     help="skip the motion-spike scan (catches brief/quiet "
                          "gestures — thumbs up, pointing, raising a phone "
                          "or portrait — that fall between fixed-interval "
                          "samples or get lost in small grid thumbnails)")
    ap.add_argument("--motion-fps", type=float, default=3.0,
                     help="continuous decode+sample rate for motion scoring "
                          "(frames/sec, gapless). Cost is dominated by native "
                          "frame decode, not this value, so it's cheap to keep "
                          "fairly dense")
    ap.add_argument("--motion-hwaccel", default="videotoolbox",
                     help="ffmpeg -hwaccel method for decode (default "
                          "videotoolbox, macOS). ~7x faster than software decode "
                          "on 4K source; set to 'none' to disable if unsupported")
    ap.add_argument("--motion-blocks", type=int, default=16,
                     help="NxN block grid for scoring; score is the single "
                          "hottest block's mean diff, not the whole-frame mean — "
                          "a small localized gesture would otherwise get diluted "
                          "away by ambient crowd motion elsewhere in frame")
    ap.add_argument("--motion-scale", type=int, default=640,
                     help="downscale width used only for the diff computation "
                          "(speed knob, not output quality — finer than the old "
                          "320 default so a small gesture isn't blurred away "
                          "before block-scoring even sees it)")
    ap.add_argument("--motion-top-n", type=int, default=150,
                     help="max number of motion-spike candidates to keep, "
                          "across the whole raw_dir")
    ap.add_argument("--motion-min-gap", type=float, default=4.0,
                     help="min seconds between two kept candidates in the same "
                          "file, so one sustained gesture doesn't crowd out others")
    ap.add_argument("--motion-max-per-file", type=int, default=3,
                     help="max candidates kept from any single file, so a long "
                          "or generally busy file can't crowd out the top-n list")
    ap.add_argument("--motion-reuse-scores", action="store_true",
                     help="if motion/all_scores.json already exists in out_dir, "
                          "load it instead of re-decoding/scoring — use this to "
                          "cheaply re-tune --motion-top-n/--motion-min-gap/"
                          "--motion-max-per-file without paying the decode cost again")
    ap.add_argument("--motion-out-size", type=int, default=560,
                     help="width of each rendered motion-candidate frame — "
                          "bigger than dense thumbs since there are far fewer")
    ap.add_argument("--motion-grid-cols", type=int, default=3)
    ap.add_argument("--motion-grid-rows", type=int, default=3)
    args = ap.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "pipeline.log"), "a") as logf:
        log(f"=== scan_pipeline start: raw_dir={raw_dir} out_dir={out_dir} ===", logf)

        if not args.skip_audio_detection:
            run_audio_smile_detection(raw_dir, out_dir, args, logf)
        else:
            log("Skipping audio+smile detection (--skip-audio-detection).", logf)

        if not args.skip_dense_sweep:
            dense_dir, manifest = run_dense_sweep(raw_dir, out_dir, args, logf)
            build_grids(dense_dir, manifest, args.grid_cols, args.grid_rows, logf)
        else:
            log("Skipping dense sweep (--skip-dense-sweep).", logf)

        if not args.skip_motion_scan:
            motion_dir, motion_manifest = run_motion_scan(raw_dir, out_dir, args, logf)
            build_grids(motion_dir, motion_manifest, args.motion_grid_cols,
                        args.motion_grid_rows, logf)
        else:
            log("Skipping motion scan (--skip-motion-scan).", logf)

        log("=== scan_pipeline done ===", logf)

    print(f"\nDone. Review:\n"
          f"  {out_dir}/motion/grids/*.jpg — start here (small set, ranked) "
          f"(cols={args.motion_grid_cols}, rows={args.motion_grid_rows} per grid; "
          f"flat index = grid*{args.motion_grid_cols*args.motion_grid_rows} + "
          f"row*{args.motion_grid_cols} + col; look up in motion/manifest.json)\n"
          f"  {out_dir}/analysis/analysis.json + thumbnails/\n"
          f"  {out_dir}/dense/grids/*.jpg "
          f"(cols={args.grid_cols}, rows={args.grid_rows} per grid; "
          f"flat index = grid*{args.grid_cols*args.grid_rows} + row*{args.grid_cols} + col; "
          f"look up that index in dense/manifest.json for file/timestamp)")


if __name__ == "__main__":
    main()
