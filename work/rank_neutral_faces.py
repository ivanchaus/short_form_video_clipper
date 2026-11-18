#!/usr/bin/env python3
"""Rank non-smiling face frames from an existing dense reaction scan."""

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def content_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero((gray > 2).astype("uint8"))
    if points is None:
        return image
    x, y, w, h = cv2.boundingRect(points)
    return image[y:y + h, x:x + w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dense_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--per-file", type=int, default=5)
    parser.add_argument("--min-gap", type=float, default=5.0)
    parser.add_argument("--min-luma", type=float, default=0.0)
    parser.add_argument("--files", help="optional comma-separated source filenames")
    args = parser.parse_args()

    included_files = set(args.files.split(",")) if args.files else None

    manifest = json.loads((args.dense_dir / "manifest.json").read_text())
    frame_dir = args.dense_dir / "frames"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = args.out_dir / "grids"
    grid_dir.mkdir(exist_ok=True)

    cascade_root = Path(cv2.data.haarcascades)
    frontal = cv2.CascadeClassifier(str(cascade_root / "haarcascade_frontalface_default.xml"))
    profile = cv2.CascadeClassifier(str(cascade_root / "haarcascade_profileface.xml"))
    smile = cv2.CascadeClassifier(str(cascade_root / "haarcascade_smile.xml"))

    by_file = {}
    for item in manifest:
        if included_files is not None and item["file"] not in included_files:
            continue
        path = frame_dir / f"f{int(item['idx']):04d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            continue
        content = content_crop(image)
        raw_gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        visible_pixels = raw_gray[raw_gray > 3]
        luma = float(visible_pixels.mean()) if visible_pixels.size else 0.0
        if luma < args.min_luma:
            continue
        gray = cv2.equalizeHist(raw_gray)
        min_side = max(24, min(gray.shape[:2]) // 14)
        faces = list(frontal.detectMultiScale(gray, 1.08, 4, minSize=(min_side, min_side)))
        profiles = list(profile.detectMultiScale(gray, 1.08, 4, minSize=(min_side, min_side)))
        flipped = cv2.flip(gray, 1)
        for x, y, w, h in profile.detectMultiScale(flipped, 1.08, 4,
                                                    minSize=(min_side, min_side)):
            profiles.append((gray.shape[1] - x - w, y, w, h))
        candidates = [(tuple(face), "front") for face in faces]
        candidates.extend((tuple(face), "profile") for face in profiles)
        if not candidates:
            continue
        face, kind = max(candidates, key=lambda pair: pair[0][2] * pair[0][3])
        x, y, w, h = face
        has_smile = False
        if kind == "front":
            roi = gray[y:y + h, x:x + w]
            smiles = smile.detectMultiScale(
                roi, scaleFactor=1.7, minNeighbors=18,
                minSize=(max(8, int(w * 0.3)), max(5, int(h * 0.15))),
            )
            has_smile = len(smiles) > 0
        if has_smile:
            continue
        area = (w * h) / max(1, gray.shape[0] * gray.shape[1])
        score = area * (1.05 if kind == "profile" else 1.0)
        by_file.setdefault(item["file"], []).append({
            **item, "score": score, "face_kind": kind, "luma": luma,
        })

    selected = []
    for filename in sorted(by_file):
        kept = []
        for item in sorted(by_file[filename], key=lambda value: value["score"], reverse=True):
            if all(abs(float(item["t"]) - float(other["t"])) >= args.min_gap for other in kept):
                kept.append(item)
            if len(kept) >= args.per_file:
                break
        selected.extend(sorted(kept, key=lambda value: value["t"]))

    (args.out_dir / "manifest.json").write_text(json.dumps(selected, indent=2))

    cell_w, cell_h, label_h = 270, 480, 34
    cols, rows = 4, 3
    per_grid = cols * rows
    for start in range(0, len(selected), per_grid):
        chunk = selected[start:start + per_grid]
        canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "black")
        draw = ImageDraw.Draw(canvas)
        for offset, item in enumerate(chunk):
            image = cv2.imread(str(frame_dir / f"f{int(item['idx']):04d}.jpg"))
            content = content_crop(image)
            rgb = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(rgb)
            frame.thumbnail((cell_w, cell_h))
            tile = Image.new("RGB", (cell_w, cell_h), "black")
            tile.paste(frame, ((cell_w - frame.width) // 2, (cell_h - frame.height) // 2))
            row, col = divmod(offset, cols)
            x, y = col * cell_w, row * (cell_h + label_h)
            canvas.paste(tile, (x, y))
            label = (f"#{start + offset} {item['file']} {float(item['t']):.1f}s "
                     f"{item['face_kind']} L{float(item['luma']):.0f}")
            draw.text((x + 4, y + cell_h + 9), label, fill="white")
        grid_index = start // per_grid
        canvas.save(grid_dir / f"grid{grid_index:03d}.jpg", quality=91)

    print(f"Selected {len(selected)} neutral/profile face frames across {len(by_file)} files")
    print(f"Wrote {(len(selected) + per_grid - 1) // per_grid} grids to {grid_dir}")


if __name__ == "__main__":
    main()
