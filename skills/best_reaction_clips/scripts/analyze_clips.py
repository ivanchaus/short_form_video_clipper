#!/usr/bin/env python3
"""
analyze_clips.py — Scan raw video clips and score short time windows for
"high energy" guest reactions, using two signals:

  1. Audio energy: RMS loudness, z-scored within each clip, to catch
     laughs, gasps, cheering, exclamations. This is extracted for the
     WHOLE clip cheaply (audio-only demux, no video decode).
  2. Face/smile evidence: for each audio-energy candidate window only,
     a handful of frames are grabbed with ffmpeg fast-seek (-ss before -i,
     which jumps to the nearest keyframe rather than decoding from the
     start) and checked with OpenCV Haar cascades for a smiling face.

This two-stage design matters for real footage: continuously decoding
every frame of long/high-resolution source video (e.g. 4K 50fps action-cam
footage) to run face detection is drastically slower than realtime on
modest hardware. Audio is cheap to scan in full, many strong reactions are
audible, and fast-seek frame grabs only cost something for candidate
windows that already look promising. Comprehensive searches must pair this
fast pass with scan_pipeline.py's full-timeline visual sweep.

This script does NOT decide the final cut. It surfaces candidates with a
multi-frame review sheet that includes pre/post context, so a human (or
Claude) sees facial beats that lead or lag the detected audio window. Face/
smile detection is supporting evidence, not a gate: open-mouth screams and
profile/wide-shot faces frequently score zero even when the reaction is
excellent.

Usage:
    python3 analyze_clips.py <raw_dir> [options]

Output:
    <out_dir>/analysis.json
    <out_dir>/thumbnails/seg_XXX.jpg
"""
import argparse
import io
import json
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


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


def extract_audio_rms(path, window, hop):
    """Return (times, rms) sampled every `hop` seconds using a `window`-second
    analysis window. Audio-only extraction — fast regardless of video
    resolution/length since it doesn't touch the video decoder."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", "-"],
        capture_output=True
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.array([]), np.array([])

    wf = wave.open(io.BytesIO(proc.stdout), "rb")
    sr = wf.getframerate()
    n = wf.getnframes()
    raw = wf.readframes(n)
    wf.close()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return np.array([]), np.array([])

    win_n = max(1, int(window * sr))
    hop_n = max(1, int(hop * sr))
    times, rms = [], []
    for start in range(0, max(1, samples.size - win_n), hop_n):
        chunk = samples[start:start + win_n]
        if chunk.size == 0:
            continue
        rms.append(float(np.sqrt(np.mean(chunk ** 2)) + 1e-9))
        times.append(start / sr)
    return np.array(times), np.array(rms)


def zscore(arr):
    if arr.size == 0:
        return arr
    mu, sd = arr.mean(), arr.std()
    if sd < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sd


_face_cascade = None
_smile_cascade = None


def get_cascades():
    global _face_cascade, _smile_cascade
    if _face_cascade is None:
        base = cv2.data.haarcascades
        _face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        _smile_cascade = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
    return _face_cascade, _smile_cascade


def grab_frame(path, t, max_dim=640):
    """Fast-seek a single frame near timestamp t via ffmpeg (-ss before -i
    jumps to the nearest keyframe, so this stays fast no matter how long
    the source file is) and return it as a BGR numpy array, downscaled so
    Haar cascade detection stays cheap."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(max(0, t)), "-i", path,
         "-frames:v", "1", "-vf", f"scale='min({max_dim},iw)':'min({max_dim},ih)':force_original_aspect_ratio=decrease",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def smile_score_for_frame(frame):
    """Return (score in [0,1], annotated usable bool) based on whether the
    largest detected face shows a smile."""
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_c, smile_c = get_cascades()
    faces = face_c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        return 0.0
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    face_roi = gray[fy:fy + fh, fx:fx + fw]
    smiles = smile_c.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=18,
                                       minSize=(int(fw * 0.3), int(fh * 0.15)))
    if len(smiles) == 0:
        return 0.1  # face present, no smile locked in on this frame
    sw = max(s[2] for s in smiles)
    return min(1.0, 0.6 + 0.4 * (sw / fw))


def make_review_sheet(frames, times, cols=3, cell_w=360, cell_h=640):
    """Tile frames from across a candidate into one legible review image.

    A single midpoint/start thumbnail routinely hides the useful beat in a
    multi-second candidate. Fixed-size letterboxed cells also keep portrait
    and landscape sources comparable without distorting faces.
    """
    cells = []
    for frame, t in zip(frames, times):
        if frame is None:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        else:
            h, w = frame.shape[:2]
            scale = min(cell_w / max(1, w), cell_h / max(1, h))
            rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            x, y = (cell_w - rw) // 2, (cell_h - rh) // 2
            cell[y:y + rh, x:x + rw] = resized
        label = f"t={float(t):.1f}s"
        cv2.putText(cell, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(cell, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(cell)
    if not cells:
        return None
    while len(cells) % cols:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def merge_windows(grid, scores, hop, threshold, min_len, pad, clip_duration):
    """Merge contiguous grid points above threshold into segments."""
    segments = []
    in_seg = False
    seg_start = None
    seg_scores = []
    for t, s in zip(grid, scores):
        if s >= threshold:
            if not in_seg:
                in_seg = True
                seg_start = t
                seg_scores = []
            seg_scores.append(s)
        else:
            if in_seg:
                seg_end = t
                if seg_end - seg_start >= min_len:
                    segments.append((max(0, seg_start - pad),
                                      min(clip_duration, seg_end + pad),
                                      float(np.mean(seg_scores))))
                in_seg = False
    if in_seg:
        seg_end = grid[-1] + hop if len(grid) else seg_start
        if seg_end - seg_start >= min_len:
            segments.append((max(0, seg_start - pad),
                              min(clip_duration, seg_end + pad),
                              float(np.mean(seg_scores))))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--out-dir", default=None, help="defaults to <raw_dir>/../analysis")
    ap.add_argument("--window", type=float, default=0.6, help="audio analysis window seconds")
    ap.add_argument("--hop", type=float, default=0.3, help="hop between audio windows, seconds")
    ap.add_argument("--audio-weight", type=float, default=0.55)
    ap.add_argument("--smile-weight", type=float, default=0.45)
    ap.add_argument("--percentile", type=float, default=85,
                     help="audio score percentile (per-clip) used as the candidate threshold")
    ap.add_argument("--min-segment", type=float, default=1.2, help="min segment length, seconds")
    ap.add_argument("--pad", type=float, default=0.35, help="context padding around each segment")
    ap.add_argument("--frames-per-candidate", type=int, default=6,
                     help="how many fast-seek frames to sample per audio candidate for smile check")
    ap.add_argument("--face-sample-size", type=int, default=960,
                    help="max frame dimension for face/smile checks; wide shots need more than 640")
    ap.add_argument("--review-frames", type=int, default=9,
                    help="frames spread across each candidate plus context in its review sheet")
    ap.add_argument("--review-preroll", type=float, default=4.0,
                    help="seconds before each detected window included in its review sheet")
    ap.add_argument("--review-postroll", type=float, default=4.0,
                    help="seconds after each detected window included in its review sheet")
    ap.add_argument("--max-per-file", type=int, default=8,
                     help="cap on audio candidate windows scored per clip (keeps runtime bounded "
                          "on long clips even with a loose --percentile)")
    ap.add_argument("--top-n", type=int, default=25, help="max candidate segments to keep")
    ap.add_argument("--only-file", default=None,
                     help="process just this one filename (basename inside raw_dir) — lets a big "
                          "raw folder be processed incrementally across several short runs instead "
                          "of one long one. Results accumulate in <out-dir>/state.json between runs.")
    args = ap.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = args.out_dir or os.path.join(os.path.dirname(raw_dir), "analysis")
    thumb_dir = os.path.join(out_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    state_path = os.path.join(out_dir, "state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state_segments = json.load(f)
    else:
        state_segments = []

    if args.only_file:
        files = [args.only_file]
        if not os.path.exists(os.path.join(raw_dir, args.only_file)):
            print(f"{args.only_file} not found in {raw_dir}", file=sys.stderr)
            sys.exit(1)
        # drop any stale entries for this file so reruns are idempotent
        state_segments = [s for s in state_segments if s["file"] != args.only_file]
    else:
        files = sorted(f for f in os.listdir(raw_dir)
                        if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
        if not files:
            print(f"No video files found in {raw_dir}", file=sys.stderr)
            sys.exit(1)
        state_segments = []  # full rescan replaces state entirely

    all_segments = []
    for fname in files:
        fpath = os.path.join(raw_dir, fname)
        duration = ffprobe_duration(fpath)
        if duration <= 0:
            print(f"Skipping {fname}: could not read duration", file=sys.stderr)
            continue
        print(f"[{fname}] duration {duration:.1f}s — scanning audio...", file=sys.stderr)

        a_times, a_rms = extract_audio_rms(fpath, args.window, args.hop)
        if a_times.size == 0:
            print(f"[{fname}] no audio track found, skipping", file=sys.stderr)
            continue
        a_z = zscore(a_rms)
        a_norm = 1 / (1 + np.exp(-a_z))  # squash to 0..1

        threshold = np.percentile(a_norm, args.percentile)
        audio_segments = merge_windows(a_times, a_norm, args.hop, threshold,
                                        args.min_segment, args.pad, duration)
        # bound the smile-check workload regardless of how loose --percentile
        # is or how long the clip runs — keep only the strongest audio peaks
        audio_segments.sort(key=lambda s: s[2], reverse=True)
        audio_segments = audio_segments[:args.max_per_file]
        print(f"[{fname}] {len(audio_segments)} audio candidate window(s) above p{args.percentile} "
              f"(capped at {args.max_per_file})", file=sys.stderr)

        for start, end, audio_score in audio_segments:
            sample_ts = np.linspace(start, end, args.frames_per_candidate + 2)[1:-1]
            best_smile = 0.0
            for t in sample_ts:
                frame = grab_frame(fpath, float(t), args.face_sample_size)
                s = smile_score_for_frame(frame)
                if s > best_smile:
                    best_smile = s

            combined = args.audio_weight * audio_score + args.smile_weight * best_smile
            # A failed face/smile cascade must never demote a strong audio
            # event. Open-mouth screams and small/profile faces are common
            # false negatives for Haar cascades.
            ranking_score = max(float(audio_score), float(combined))
            all_segments.append({
                "file": fname,
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "duration": round(float(end - start), 2),
                "source_duration": round(float(duration), 2),
                "audio_score": round(float(audio_score), 4),
                "smile_score": round(float(best_smile), 4),
                "combined_score": round(float(combined), 4),
                "score": round(ranking_score, 4),
            })

    # merge this run's results into the persisted state, then rank across
    # everything scanned so far (so incremental --only-file runs still
    # produce a correct global top-n)
    state_segments.extend(all_segments)
    with open(state_path, "w") as f:
        json.dump(state_segments, f, indent=2)

    ranked = sorted(state_segments, key=lambda s: s["score"], reverse=True)
    top_segments = ranked[:args.top_n]

    # Regrab a strip of frames across every finalized segment. A single
    # thumbnail can land before/after the actual reaction and caused known
    # misses even when audio detection found the correct window. Note: some
    # sandboxed filesystems
    # don't allow deleting/overwriting already-written files, so this
    # never deletes old thumbnails — names are content-deterministic
    # (same file+start -> same name), so an existing file is just reused
    # rather than rewritten. Stray thumbnails from segments that dropped
    # out of a previous top-n are harmless leftovers; analysis.json's
    # segment list (not a directory listing) is the source of truth.
    for i, seg in enumerate(top_segments):
        # name by content (file+start), not rank — rank shifts across
        # incremental runs, but the same segment reuses the same thumbnail
        thumb_name = (f"seg_{os.path.splitext(seg['file'])[0]}_"
                      f"{seg['start']:.2f}s_context_review.jpg")
        thumb_path = os.path.join(thumb_dir, thumb_name)
        fpath = os.path.join(raw_dir, seg["file"])
        source_duration = float(seg.get("source_duration") or
                                (ffprobe_duration(fpath) if os.path.exists(fpath) else
                                 seg["end"] + args.review_postroll))
        review_start = max(0.0, seg["start"] - args.review_preroll)
        review_end = min(source_duration, seg["end"] + args.review_postroll)
        review_span = max(0.01, review_end - review_start)
        review_ts = np.linspace(review_start + review_span * 0.05,
                                review_end - review_span * 0.05,
                                max(2, args.review_frames))
        seg["review_start"] = round(float(review_start), 2)
        seg["review_end"] = round(float(review_end), 2)
        seg["review_times"] = [round(float(t), 2) for t in review_ts]
        if os.path.exists(thumb_path):
            seg["thumbnail"] = os.path.join("thumbnails", thumb_name)
            continue
        frames = ([grab_frame(fpath, float(t), args.face_sample_size) for t in review_ts]
                  if os.path.exists(fpath) else [])
        sheet = make_review_sheet(frames, review_ts)
        if sheet is not None:
            cv2.imwrite(thumb_path, sheet)
            seg["thumbnail"] = os.path.join("thumbnails", thumb_name)
        else:
            seg.pop("thumbnail", None)

    result = {
        "raw_dir": raw_dir,
        "num_clips_scanned_this_run": len(files),
        "num_candidates_total": len(state_segments),
        "num_candidates": len(top_segments),
        "segments": top_segments,
    }
    analysis_path = os.path.join(out_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {len(top_segments)} candidate segments (of {len(state_segments)} scanned so far) "
          f"to {analysis_path}", file=sys.stderr)
    print(analysis_path)


if __name__ == "__main__":
    main()
