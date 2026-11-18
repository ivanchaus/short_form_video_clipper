#!/usr/bin/env python3
"""Build a labeled start/middle/end contact sheet for rendered MP4 files."""

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    cell_w, cell_h, label_h = 180, 320, 40
    columns = 3
    sheet = Image.new(
        "RGB", (cell_w * columns, (cell_h + label_h) * len(args.files)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for row, path in enumerate(args.files):
        clip_duration = duration(path)
        draw.text((6, row * (cell_h + label_h) + 8), path.stem, fill="white")
        for column, fraction in enumerate((0.05, 0.5, 0.95)):
            timestamp = clip_duration * fraction
            temp = args.out.parent / f"qa_{row:03d}_{column}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", str(timestamp), "-i", str(path),
                 "-vf", f"scale={cell_w}:{cell_h}", "-frames:v", "1", str(temp)],
                check=True,
            )
            frame = Image.open(temp).convert("RGB")
            ImageDraw.Draw(frame).text((5, 5), f"{timestamp:.1f}s", fill="white", stroke_width=2,
                                       stroke_fill="black")
            sheet.paste(frame, (column * cell_w, row * (cell_h + label_h) + label_h))
            temp.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)


if __name__ == "__main__":
    main()
