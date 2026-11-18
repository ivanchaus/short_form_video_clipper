#!/usr/bin/env python3
"""Extract review audio for a reaction-candidate manifest."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for index, segment in enumerate(json.loads(args.manifest.read_text())):
        output = args.out_dir / f"{index:03d}_{Path(segment['file']).stem}.wav"
        duration = float(segment["end"]) - float(segment["start"])
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", str(segment["start"]),
                "-i", str(args.raw_dir / segment["file"]), "-t", str(duration),
                "-vn", "-af", "highpass=f=100,lowpass=f=10000",
                "-ar", "16000", "-ac", "1", str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
