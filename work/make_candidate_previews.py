#!/usr/bin/env python3
"""Render four-point vertical crop previews for a candidate manifest."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    segments = json.loads(args.manifest.read_text())
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        zoom = float(segment.get("zoom", 1.0))
        crop_x = float(segment.get("crop_x", 0.5))
        crop_y = float(segment.get("crop_y", 0.5))
        width, height = 270, 480
        scale_w, scale_h = round(width * zoom), round(height * zoom)
        vf = (
            f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:x='(in_w-{width})*{crop_x}':"
            f"y='(in_h-{height})*{crop_y}',setsar=1,format=yuvj420p"
        )
        frames = []
        for frame_index, fraction in enumerate((0.02, 0.25, 0.5, 0.75, 0.98)):
            timestamp = start + (end - start) * fraction
            frame_path = args.out_dir / f"tmp_{index:03d}_{frame_index}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", str(timestamp), "-i",
                 str(args.raw_dir / segment["file"]), "-vf", vf,
                 "-frames:v", "1", str(frame_path)],
                check=True,
            )
            frame = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, width, 24), fill="black")
            draw.text((6, 5), f"{timestamp:.2f}s", fill="white")
            frames.append(frame.copy())
            frame_path.unlink()

        title_height = 34
        sheet = Image.new("RGB", (width * len(frames), height + title_height), "black")
        for frame_index, frame in enumerate(frames):
            sheet.paste(frame, (width * frame_index, title_height))
        ImageDraw.Draw(sheet).text(
            (6, 10), f"{index:03d} {segment['file']} {start:.1f}-{end:.1f}", fill="white"
        )
        sheet.save(args.out_dir / f"preview_{index:03d}.jpg", quality=91)


if __name__ == "__main__":
    main()
