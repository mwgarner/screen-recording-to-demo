#!/usr/bin/env python3
"""Generate WebVTT captions from a narration audio manifest.json.

Uses timeline placement from the manifest (matches full-recording-timeline.wav mux).

Usage:
  python3 generate_demo_captions.py \\
    output/demo-timeline/<cut>/latest/audio/manifest.json \\
    output/demo-timeline/<cut>/deliverables/captions.vtt \\
    --video-length 1:30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_clock(value: str) -> float:
    """Parse M:SS or H:MM:SS to seconds."""
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise argparse.ArgumentTypeError(f"Invalid clock: {value}")


def ms_to_vtt(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def display_text(narration: str) -> str:
    """Caption text: strip TTS audio tags like [short pause]."""
    text = re.sub(r"\s*\[[^\]]*\]\s*", " ", narration)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WebVTT from audio manifest")
    parser.add_argument("manifest", type=Path, help="audio/manifest.json from a demo run")
    parser.add_argument("output", type=Path, help="Output .vtt path")
    parser.add_argument(
        "--video-length",
        type=parse_clock,
        help="Optional video duration cap (M:SS or H:MM:SS)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        print("error: manifest has no segments[]", file=sys.stderr)
        sys.exit(1)

    video_ms = int(args.video_length * 1000) if args.video_length else None

    segment_times: list[tuple[int, int]] = []
    for index, segment in enumerate(segments):
        start_ms = int(segment["timeline_start_ms"])
        if index + 1 < len(segments):
            end_ms = int(segments[index + 1]["timeline_start_ms"])
        else:
            end_ms = start_ms + int(segment["duration_ms"])
        segment_times.append((start_ms, end_ms))

    lines = ["WEBVTT", ""]
    cue_count = 0
    for index, (segment, (start_ms, end_ms)) in enumerate(zip(segments, segment_times), start=1):
        if video_ms is not None:
            end_ms = min(end_ms, video_ms)
            if start_ms >= video_ms:
                break

        cue_text = display_text(str(segment.get("narration", "")))
        if not cue_text:
            continue

        lines.append(str(index))
        lines.append(f"{ms_to_vtt(start_ms)} --> {ms_to_vtt(end_ms)}")
        lines.append(cue_text)
        lines.append("")
        cue_count += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({cue_count} cues)")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
