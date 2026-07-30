#!/usr/bin/env python3
"""
Extract a visual action timeline from a demo screen recording using Gemini.

Usage:
  python3 extract_demo_timeline.py "/path/to/demo.mp4" --narration

Each invocation writes an immutable run directory (never overwrites a prior run):

  output/demo-timeline/<cut-slug>/
    runs/<YYYYMMDD-HHMMSS>/
      run.json
      visual_timeline.json|.md
      coverage_report.md
      narration_script.model.json   # model draft (immutable once written)
      narration_script.json|.md     # working copy — edit this before TTS
    latest -> runs/<YYYYMMDD-HHMMSS>

Pass the run dir (or the `latest` symlink) to generate_demo_narration_audio.py.

Default extraction is dense multi-pass (chapters → per-chapter events → gap fill).
Use --single-pass for faster but less complete capture.

Defaults: gemini-3.5-flash (video analysis), gemini-3.1-flash-lite (narration).

Requires GEMINI_API_KEY in the environment or a local .env / .env.local file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

# Models for analyzing an already-recorded screen demo (not for app generation).
DEFAULT_VISUAL_MODEL = "gemini-3.5-flash"
DEFAULT_NARRATION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OUTPUT_DIR = Path("output/demo-timeline")
DEFAULT_PRODUCT_CONTEXT = Path(__file__).resolve().parent / "examples" / "product-context.example.md"


def slugify_cut_name(stem: str) -> str:
    """Stable folder name from a video filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "demo"


def guess_video_mime(path: Path) -> str:
    """Best-effort MIME type for Gemini File API / Part references."""
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }.get(path.suffix.lower(), "video/mp4")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def allocate_run_dir(output_root: Path, cut_slug: str, run_id: str | None = None) -> Path:
    """Create output/demo-timeline/<cut>/runs/<run-id>/ and point latest at it."""
    rid = run_id or new_run_id()
    cut_dir = output_root / cut_slug
    runs_dir = cut_dir / "runs"
    run_dir = runs_dir / rid
    if run_dir.exists():
        raise SystemExit(f"ERROR: run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    latest = cut_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # Relative symlink so the tree stays portable when moved.
        latest.symlink_to(Path("runs") / rid)
    except OSError as exc:
        # Common on Windows without Developer Mode / symlink privilege.
        print(
            f"WARNING: could not create 'latest' symlink ({exc}). "
            f"Pass the run directory explicitly to the next script:\n  {run_dir}",
            file=sys.stderr,
        )
    return run_dir


def write_run_meta(run_dir: Path, meta: dict) -> None:
    path = run_dir / "run.json"
    existing: dict = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(meta)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

VISUAL_TIMELINE_PROMPT_TEMPLATE = """
Analyze this product demo screen recording and produce a detailed, chronological
description of all visual actions, scene changes, UI states, and key on-screen text.

Focus on what a first-time viewer would need explained in a voiceover:
- Page or screen transitions
- Clicks, form input, scrolling, modals, toasts, loading states, results shown
- Readable button labels, menu items, headings, and field names (verbatim when legible)
- Ignore idle cursor movement with no effect

Use the product context below so narration_hint fields explain WHY each action matters
to the target user, not just what changed on screen.

--- PRODUCT CONTEXT ---
{product_context}
--- END PRODUCT CONTEXT ---

Return JSON only with this shape:
{
  "title": "short inferred title",
  "duration_estimate": "MM:SS",
  "chapters": [
    {
      "start": "MM:SS",
      "end": "MM:SS",
      "screen": "screen or page name",
      "summary": "one sentence chapter summary"
    }
  ],
  "events": [
    {
      "timestamp": "MM:SS",
      "event_type": "navigation|click|type|scroll|modal|toast|loading|result|other",
      "subject": "what was interacted with or shown",
      "on_screen_text": "verbatim labels if readable, else empty string",
      "visual_change": "one clear sentence describing what changed",
      "narration_hint": "what a voiceover should explain here, factual not marketing"
    }
  ]
}

Include only significant events. Merge rapid typing into one event.
Use MM:SS timestamps (round to nearest second if uncertain).
"""

CHAPTERS_PROMPT_TEMPLATE = """
Analyze this full product demo screen recording and segment it into logical chapters.

--- PRODUCT CONTEXT ---
{product_context}
--- END PRODUCT CONTEXT ---

Return JSON only:
{
  "title": "short inferred title",
  "duration_estimate": "MM:SS",
  "chapters": [
    {
      "start": "MM:SS",
      "end": "MM:SS",
      "screen": "screen or page name",
      "summary": "one sentence chapter summary"
    }
  ]
}

Cover the entire video from first frame to last. Chapters must not overlap and must span the full duration.
"""

CHAPTER_EVENTS_PROMPT_TEMPLATE = """
You are analyzing segment {start} to {end} ONLY ({screen}) of a product demo.

Your job is EXHAUSTIVE capture — list every distinct user-visible action in this segment.
Do not summarize multiple actions into one event. When in doubt, include the event.

Capture ALL of:
- Navigation: sidebar clicks, tabs, links, back buttons, route changes
- Modals: open, close, confirm, cancel
- Forms: field focus, typing (summarize final text in on_screen_text), submit/save
- Buttons and controls: every click that changes UI state
- Scroll: when new content becomes visible (list what appeared)
- Loading states: spinners, skeletons, "thinking", "searching"
- Results: panels, summaries, lists, tables that appear or update
- Documents: viewer open, page changes if visible, related panels open/close
- Canvas / map / diagram interactions: zoom, pan, select, drag, tooltip

--- PRODUCT CONTEXT ---
{product_context}
--- END PRODUCT CONTEXT ---

Chapter summary for context: {summary}

Return JSON only:
{
  "events": [
    {
      "timestamp": "MM:SS",
      "event_type": "navigation|click|type|scroll|modal|toast|loading|result|other",
      "subject": "what was interacted with or shown",
      "on_screen_text": "verbatim labels if readable, else empty string",
      "visual_change": "one clear sentence describing what changed",
      "narration_hint": "what a voiceover should explain here, factual not marketing"
    }
  ]
}

Rules:
- Timestamps must fall between {start} and {end} (inclusive).
- Use whole-second timestamps only (MM:SS format, no decimals).
- Include sidebar navigation clicks even if the destination was shown earlier.
- Do not skip scrolling, loading, or intermediate screens.
"""

GAP_FILL_PROMPT_TEMPLATE = """
Re-analyze ONLY the segment {start} to {end} of this demo video.
A prior pass may have missed actions in this window. List EVERY visible action again, exhaustively.

--- PRODUCT CONTEXT ---
{product_context}
--- END PRODUCT CONTEXT ---

Return JSON only: {{ "events": [ ...same event shape as before... ] }}
Timestamps must be between {start} and {end}.
"""

NARRATION_PROMPT_TEMPLATE = """
You are writing a voiceover for a product demo. Given the visual timeline JSON below,
write a professional product demo voiceover script.

--- PRODUCT CONTEXT ---
{product_context}
--- END PRODUCT CONTEXT ---

Rules:
- Present tense, speak to the target user described in product context
- ~130 words per minute pacing; aim to cover most of the video duration with useful copy
- Each line must include its start timestamp aligned to on-screen actions
- Tie actions to outcomes using product context
- Use exact UI labels when they appear on screen
- Do not describe cursor movement unless it teaches something
- Do not invent features not present in the timeline or product context
- Do not invent stats, superlatives, or competitor comparisons
- Follow any terminology and claim guardrails in product context

Return JSON only:
{
  "script_lines": [
    {
      "timestamp": "MM:SS",
      "narration": "spoken line",
      "duration_estimate_sec": 3
    }
  ],
  "full_script": "continuous narration text for recording"
}

Visual timeline JSON:
"""

# Higher FPS improves capture of fast UI clicks (API default is ~1 FPS).
CHAPTER_ANALYSIS_FPS = 2.0
GAP_THRESHOLD_SEC = 6
GAP_FILL_MAX = 8


def timestamp_to_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Invalid timestamp: {value}")


def normalize_timestamp(value: str) -> str:
    total_seconds = round(timestamp_to_seconds(value))
    return seconds_to_timestamp(int(total_seconds))


def seconds_to_timestamp(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def offset_seconds(seconds: int) -> str:
    return f"{seconds}s"


def merge_events(chapters: list[dict], event_groups: list[list[dict]]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for events in event_groups:
        for event in events:
            raw_ts = event.get("timestamp", "00:00")
            try:
                event["timestamp"] = normalize_timestamp(raw_ts)
            except ValueError:
                continue
            ts = event["timestamp"]
            key = (
                ts,
                event.get("event_type", ""),
                event.get("subject", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)

    merged.sort(key=lambda e: timestamp_to_seconds(e.get("timestamp", "00:00")))
    return merged


def find_coverage_gaps(events: list[dict], duration_sec: float) -> list[dict]:
    if not events:
        return [{"start": "00:00", "end": seconds_to_timestamp(duration_sec), "gap_sec": duration_sec}]

    gaps: list[dict] = []
    sorted_events = sorted(events, key=lambda e: timestamp_to_seconds(e.get("timestamp", "00:00")))
    timestamps = [timestamp_to_seconds(e.get("timestamp", "00:00")) for e in sorted_events]

    for idx in range(len(timestamps) - 1):
        gap = timestamps[idx + 1] - timestamps[idx]
        if gap >= GAP_THRESHOLD_SEC:
            gaps.append(
                {
                    "start": seconds_to_timestamp(timestamps[idx]),
                    "end": seconds_to_timestamp(timestamps[idx + 1]),
                    "gap_sec": gap,
                }
            )

    trailing = duration_sec - timestamps[-1]
    if trailing >= GAP_THRESHOLD_SEC:
        gaps.append(
            {
                "start": seconds_to_timestamp(int(timestamps[-1])),
                "end": seconds_to_timestamp(int(duration_sec)),
                "gap_sec": int(trailing),
            }
        )

    return gaps


def write_coverage_report(
    timeline: dict,
    gaps: list[dict],
    output_path: Path,
    *,
    chapter_event_counts: dict[str, int],
) -> None:
    events = timeline.get("events", [])
    duration = timeline.get("duration_estimate", "?")
    lines = [
        "# Timeline coverage report",
        "",
        f"Duration: {duration}",
        f"Total events: {len(events)}",
        "",
        "## Events per chapter",
        "",
    ]

    for chapter in timeline.get("chapters", []):
        label = f"{chapter.get('start', '?')}–{chapter.get('end', '?')} ({chapter.get('screen', '')})"
        lines.append(f"- **{label}**: {chapter_event_counts.get(label, 0)} events")

    lines.extend(["", "## Gaps (no event for ≥6 seconds)", ""])
    if not gaps:
        lines.append("No significant gaps detected.")
    else:
        for gap in gaps:
            lines.append(
                f"- **{gap['start']}–{gap['end']}** ({gap['gap_sec']}s without an event)"
            )

    lines.extend(
        [
            "",
            "## Manual review checklist",
            "",
            "- Scrub each gap window in the video — add missing events to `visual_timeline.json`",
            "- Confirm every sidebar navigation click is captured",
            "- Confirm each search → results → document open chain is complete",
            "- Confirm Intelligence flow: profile → compare → project → pay app",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_product_context(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        print(f"WARNING: Product context file not found: {resolved}", file=sys.stderr)
        return ""
    return resolved.read_text(encoding="utf-8").strip()


def build_chapters_prompt(product_context: str) -> str:
    return CHAPTERS_PROMPT_TEMPLATE.replace("{product_context}", product_context or "(none)")


def build_chapter_events_prompt(product_context: str, chapter: dict) -> str:
    return (
        CHAPTER_EVENTS_PROMPT_TEMPLATE.replace("{product_context}", product_context or "(none)")
        .replace("{start}", chapter.get("start", "00:00"))
        .replace("{end}", chapter.get("end", "00:00"))
        .replace("{screen}", chapter.get("screen", "Unknown"))
        .replace("{summary}", chapter.get("summary", ""))
    )


def build_gap_fill_prompt(product_context: str, gap: dict) -> str:
    return (
        GAP_FILL_PROMPT_TEMPLATE.replace("{product_context}", product_context or "(none)")
        .replace("{start}", gap["start"])
        .replace("{end}", gap["end"])
    )


def build_visual_timeline_prompt(product_context: str) -> str:
    return VISUAL_TIMELINE_PROMPT_TEMPLATE.replace(
        "{product_context}", product_context or "(none)"
    )


def build_narration_prompt(product_context: str, timeline: dict) -> str:
    return (
        NARRATION_PROMPT_TEMPLATE.replace("{product_context}", product_context or "(none)")
        + json.dumps(timeline, indent=2)
    )


def write_narration_outputs(
    client: genai.Client,
    model: str,
    timeline: dict,
    product_context: str,
    output_dir: Path,
    *,
    keep_working_script: bool = False,
) -> None:
    """Write model draft + working narration scripts.

    - narration_script.model.json is the immutable model draft for this run.
    - narration_script.json is the working copy editors tweak before TTS.
    When keep_working_script is True (e.g. --narration-only after a human edit),
    only the .model.json is refreshed.
    """
    print(f"Generating voiceover script with {model}...")
    narration = generate_json(
        client,
        model,
        build_narration_prompt(product_context, timeline),
    )
    payload = json.dumps(narration, indent=2) + "\n"
    model_path = output_dir / "narration_script.model.json"
    model_path.write_text(payload, encoding="utf-8")
    print(f"Wrote {model_path}")

    working_json = output_dir / "narration_script.json"
    working_md = output_dir / "narration_script.md"
    if keep_working_script and working_json.is_file():
        print(f"Kept existing working script: {working_json}")
    else:
        working_json.write_text(payload, encoding="utf-8")
        write_markdown_narration(narration, working_md)
        print(f"Wrote {working_json}")
        print(f"Wrote {working_md}")

    write_run_meta(
        output_dir,
        {
            "narration_model": model,
            "narration_script_model": "narration_script.model.json",
            "narration_script_working": "narration_script.json",
            "narration_generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def load_api_key() -> str:
    """Load GEMINI_API_KEY from the environment or a local .env file.

    Never commit API keys. Supported sources (first match wins):
      1. GEMINI_API_KEY environment variable
      2. .env or .env.local in the current working directory
      3. .env or .env.local next to this script
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key.strip()

    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent / ".env.local",
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value

    print(
        "ERROR: GEMINI_API_KEY not set. Export it or put it in a local .env file "
        "(see .env.example). Do not commit API keys.",
        file=sys.stderr,
    )
    sys.exit(1)


def wait_for_file_active(client: genai.Client, uploaded_file: types.File) -> types.File:
    name = uploaded_file.name
    if not name:
        raise RuntimeError("Uploaded file has no name")

    for _ in range(60):
        current = client.files.get(name=name)
        state = getattr(current, "state", None)
        state_name = getattr(state, "name", state)
        if state_name == "ACTIVE":
            return current
        if state_name == "FAILED":
            raise RuntimeError(f"File processing failed: {current}")
        time.sleep(2)

    raise TimeoutError("Timed out waiting for uploaded video to become ACTIVE")


class UploadedVideo:
    """File API handle plus a guaranteed MIME for Part references."""

    def __init__(self, file: types.File, mime_type: str) -> None:
        self.file = file
        self.mime_type = mime_type or getattr(file, "mime_type", None) or "video/mp4"

    @property
    def name(self) -> str | None:
        return getattr(self.file, "name", None)

    @property
    def uri(self) -> str | None:
        return getattr(self.file, "uri", None)


def upload_video(client: genai.Client, video_path: Path) -> UploadedVideo:
    mime = guess_video_mime(video_path)
    size_mb = video_path.stat().st_size / 1024 / 1024
    print(f"Uploading video ({size_mb:.1f} MB, {mime})...")
    try:
        uploaded = client.files.upload(file=str(video_path), config={"mime_type": mime})
    except TypeError:
        # Older google-genai SDKs may not accept config=mime_type.
        uploaded = client.files.upload(file=str(video_path))
    active = wait_for_file_active(client, uploaded)
    resolved = getattr(active, "mime_type", None) or mime
    print(f"Upload ready: {active.uri} ({resolved})")
    return UploadedVideo(active, resolved)


def extract_json_from_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise


def generate_json(
    client: genai.Client,
    model: str,
    prompt: str,
    *,
    video_file: UploadedVideo | types.File | None = None,
    start_sec: int | None = None,
    end_sec: int | None = None,
    fps: float | None = None,
) -> dict:
    contents: list[types.Part | str] = []
    if video_file is not None:
        video_metadata = None
        if start_sec is not None or end_sec is not None or fps is not None:
            video_metadata = types.VideoMetadata(
                start_offset=offset_seconds(start_sec) if start_sec is not None else None,
                end_offset=offset_seconds(end_sec) if end_sec is not None else None,
                fps=fps,
            )
        uri = getattr(video_file, "uri", None)
        mime_type = getattr(video_file, "mime_type", None) or "video/mp4"
        contents.append(
            types.Part(
                file_data=types.FileData(file_uri=uri, mime_type=mime_type),
                video_metadata=video_metadata,
            )
        )
    contents.append(prompt)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")

    return extract_json_from_response(text)


def count_events_per_chapter(chapters: list[dict], events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chapter in chapters:
        start = timestamp_to_seconds(chapter.get("start", "00:00"))
        end = timestamp_to_seconds(chapter.get("end", "00:00"))
        label = f"{chapter.get('start', '?')}–{chapter.get('end', '?')} ({chapter.get('screen', '')})"
        counts[label] = sum(
            1
            for event in events
            if start <= timestamp_to_seconds(event.get("timestamp", "00:00")) <= end
        )
    return counts


def extract_timeline_dense(
    client: genai.Client,
    model: str,
    video_file: UploadedVideo | types.File,
    product_context: str,
    *,
    gap_fill: bool = True,
) -> tuple[dict, list[dict], dict[str, int]]:
    print("Pass 1: segmenting video into chapters...")
    chapter_result = generate_json(
        client,
        model,
        build_chapters_prompt(product_context),
        video_file=video_file,
    )
    chapters = chapter_result.get("chapters", [])
    if not chapters:
        raise RuntimeError("Chapter segmentation returned no chapters")

    event_groups: list[list[dict]] = []
    for index, chapter in enumerate(chapters, start=1):
        start_sec = timestamp_to_seconds(chapter.get("start", "00:00"))
        end_sec = timestamp_to_seconds(chapter.get("end", "00:00"))
        print(
            f"Pass 2 ({index}/{len(chapters)}): dense events "
            f"{chapter.get('start')}–{chapter.get('end')} — {chapter.get('screen', '')}..."
        )
        chapter_events: dict = {}
        for attempt in range(2):
            try:
                chapter_events = generate_json(
                    client,
                    model,
                    build_chapter_events_prompt(product_context, chapter),
                    video_file=video_file,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    fps=CHAPTER_ANALYSIS_FPS,
                )
                break
            except (json.JSONDecodeError, RuntimeError, ValueError) as error:
                if attempt == 1:
                    raise
                print(f"  retry after error: {error}")
        events = chapter_events.get("events", [])
        event_groups.append(events)
        print(f"  captured {len(events)} events")

    events = merge_events(chapters, event_groups)
    duration_estimate = chapter_result.get("duration_estimate", chapters[-1].get("end", "00:00"))
    duration_sec = timestamp_to_seconds(duration_estimate)
    gaps = find_coverage_gaps(events, duration_sec)

    if gap_fill and gaps:
        print(f"Pass 3: re-analyzing {min(len(gaps), GAP_FILL_MAX)} gap windows...")
        for gap in gaps[:GAP_FILL_MAX]:
            start_sec = timestamp_to_seconds(gap["start"])
            end_sec = timestamp_to_seconds(gap["end"])
            print(f"  gap fill {gap['start']}–{gap['end']} ({gap['gap_sec']}s)...")
            gap_events = generate_json(
                client,
                model,
                build_gap_fill_prompt(product_context, gap),
                video_file=video_file,
                start_sec=start_sec,
                end_sec=end_sec,
                fps=CHAPTER_ANALYSIS_FPS,
            )
            event_groups.append(gap_events.get("events", []))
        events = merge_events(chapters, event_groups)
        gaps = find_coverage_gaps(events, duration_sec)

    timeline = {
        "title": chapter_result.get("title", "Demo timeline"),
        "duration_estimate": duration_estimate,
        "chapters": chapters,
        "events": events,
    }
    chapter_counts = count_events_per_chapter(chapters, events)
    return timeline, gaps, chapter_counts


def write_markdown_timeline(timeline: dict, output_path: Path) -> None:
    lines = [
        f"# {timeline.get('title', 'Demo timeline')}",
        "",
        f"Estimated duration: {timeline.get('duration_estimate', 'unknown')}",
        "",
        "## Chapters",
        "",
    ]

    for chapter in timeline.get("chapters", []):
        lines.append(
            f"- **{chapter.get('start', '?')}–{chapter.get('end', '?')}** "
            f"— {chapter.get('screen', 'Unknown screen')}: {chapter.get('summary', '')}"
        )

    lines.extend(["", "## Visual events", ""])
    for event in timeline.get("events", []):
        lines.append(
            f"- **[{event.get('timestamp', '?')}]** ({event.get('event_type', 'other')}) "
            f"{event.get('visual_change', '')}"
        )
        if event.get("on_screen_text"):
            lines.append(f"  - On screen: \"{event['on_screen_text']}\"")
        if event.get("narration_hint"):
            lines.append(f"  - Narration hint: {event['narration_hint']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_narration(narration: dict, output_path: Path) -> None:
    lines = ["# Voiceover script", ""]
    for line in narration.get("script_lines", []):
        lines.append(f"**[{line.get('timestamp', '?')}]** {line.get('narration', '')}")
    lines.extend(["", "## Full script", "", narration.get("full_script", "")])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract demo video visual timeline with Gemini")
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Path to local MP4/MOV screen recording (omit with --narration-only)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VISUAL_MODEL,
        help=f"Model for video analysis (default: {DEFAULT_VISUAL_MODEL})",
    )
    parser.add_argument(
        "--narration-model",
        default=DEFAULT_NARRATION_MODEL,
        help=f"Model for voiceover script from timeline JSON (default: {DEFAULT_NARRATION_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root for cut/run trees (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cut-slug",
        type=str,
        default=None,
        help="Override cut folder name under --output-dir (default: slugified video stem)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run id (default: local YYYYMMDD-HHMMSS). Must be unique under the cut.",
    )
    parser.add_argument(
        "--product-context",
        type=Path,
        default=DEFAULT_PRODUCT_CONTEXT,
        help="Markdown file with product/value context for the narrator (default: examples/product-context.example.md)",
    )
    parser.add_argument(
        "--no-product-context",
        action="store_true",
        help="Skip loading product context",
    )
    parser.add_argument(
        "--narration",
        action="store_true",
        help="Also generate a voiceover script from the visual timeline",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Use one-pass extraction (faster, less complete). Default is dense multi-pass.",
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Skip gap-fill pass after dense chapter extraction",
    )
    parser.add_argument(
        "--narration-only",
        type=Path,
        metavar="RUN_DIR",
        help="Regenerate narration from existing visual_timeline.json in this run directory",
    )
    parser.add_argument(
        "--keep-working-script",
        action="store_true",
        help="With --narration-only: refresh narration_script.model.json but keep narration_script.json",
    )
    parser.add_argument(
        "--keep-uploaded-file",
        action="store_true",
        help="Do not delete the video from the Gemini File API after analysis "
        "(default: delete when possible)",
    )
    args = parser.parse_args()

    product_context = "" if args.no_product_context else load_product_context(args.product_context)
    client = genai.Client(api_key=load_api_key())

    if args.narration_only:
        output_dir = args.narration_only.expanduser().resolve()
        timeline_json_path = output_dir / "visual_timeline.json"
        if not timeline_json_path.is_file():
            print(f"ERROR: Timeline not found: {timeline_json_path}", file=sys.stderr)
            sys.exit(1)
        timeline = json.loads(timeline_json_path.read_text(encoding="utf-8"))
        write_narration_outputs(
            client,
            args.narration_model,
            timeline,
            product_context,
            output_dir,
            keep_working_script=args.keep_working_script,
        )
        print("Done.")
        return

    if args.video is None:
        print("ERROR: video path required unless using --narration-only", file=sys.stderr)
        sys.exit(1)

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cut_slug = args.cut_slug or slugify_cut_name(video_path.stem)
    output_root = args.output_dir.expanduser().resolve()
    output_dir = allocate_run_dir(output_root, cut_slug, args.run_id)
    print(f"Run directory: {output_dir}")
    print(f"Latest symlink: {output_root / cut_slug / 'latest'}")

    write_run_meta(
        output_dir,
        {
            "cut_slug": cut_slug,
            "run_id": output_dir.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_video": str(video_path),
            "source_video_bytes": video_path.stat().st_size,
            "visual_model": args.model,
            "product_context": None if args.no_product_context else str(args.product_context),
        },
    )

    video_file = upload_video(client, video_path)

    if args.single_pass:
        print(f"Single-pass analysis with {args.model}...")
        timeline = generate_json(
            client,
            args.model,
            build_visual_timeline_prompt(product_context),
            video_file=video_file,
        )
        duration_sec = timestamp_to_seconds(timeline.get("duration_estimate", "00:00"))
        gaps = find_coverage_gaps(timeline.get("events", []), duration_sec)
        chapter_counts = count_events_per_chapter(timeline.get("chapters", []), timeline.get("events", []))
    else:
        timeline, gaps, chapter_counts = extract_timeline_dense(
            client,
            args.model,
            video_file,
            product_context,
            gap_fill=not args.no_gap_fill,
        )

    timeline_json_path = output_dir / "visual_timeline.json"
    timeline_md_path = output_dir / "visual_timeline.md"
    coverage_path = output_dir / "coverage_report.md"
    timeline_json_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    write_markdown_timeline(timeline, timeline_md_path)
    write_coverage_report(timeline, gaps, coverage_path, chapter_event_counts=chapter_counts)

    print(f"Wrote {timeline_json_path} ({len(timeline.get('events', []))} events)")
    print(f"Wrote {timeline_md_path}")
    print(f"Wrote {coverage_path}")
    if gaps:
        print(f"WARNING: {len(gaps)} gap(s) ≥{GAP_THRESHOLD_SEC}s remain — review coverage_report.md")

    write_run_meta(
        output_dir,
        {
            "visual_timeline": "visual_timeline.json",
            "event_count": len(timeline.get("events", [])),
            "duration_estimate": timeline.get("duration_estimate"),
            "timeline_extracted_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    if args.narration:
        write_narration_outputs(client, args.narration_model, timeline, product_context, output_dir)

    if args.keep_uploaded_file:
        print(f"Keeping uploaded File API object: {video_file.name}")
    else:
        try:
            if video_file.name:
                client.files.delete(name=video_file.name)
                print(f"Deleted uploaded File API object: {video_file.name}")
        except Exception as exc:
            print(
                f"WARNING: could not delete uploaded File API object "
                f"{getattr(video_file, 'name', None)}: {exc}",
                file=sys.stderr,
            )
            print(
                "Delete it manually in AI Studio / the Files API if it contains sensitive footage.",
                file=sys.stderr,
            )

    print("Done.")
    print(f"Next: edit {output_dir / 'narration_script.json'} then run generate_demo_narration_audio.py on:")
    print(f"  {output_root / cut_slug / 'latest'}")


if __name__ == "__main__":
    main()
