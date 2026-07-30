#!/usr/bin/env python3
"""
Generate voiceover audio from narration_script.json using Gemini TTS.

Default mode (continuous): one TTS pass for the full script — same voice, pacing,
and tone throughout — then split on paragraph pauses into segment files.

Usage:
  python3 generate_demo_narration_audio.py output/demo-timeline/<cut>/latest
  python3 generate_demo_narration_audio.py ... --mode segmented  # per-line (less consistent)
  python3 generate_demo_narration_audio.py ... --voice Orus  # override default voice

Pass a run directory from extract_demo_timeline.py (prefer the cut's `latest` symlink).
Reads narration_script.json (the working/edited copy). Does not touch
narration_script.model.json.

Requires GEMINI_API_KEY in the environment or a local .env / .env.local file, and ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import array
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

# Per-line alignment: clamp the tempo nudge so no line ever sounds sped-up/slowed
# unnaturally. Lines whose natural read is close to their on-screen window get a
# near-invisible nudge; grossly mismatched copy is caught by the script rebalance.
ALIGN_MIN_FACTOR = 0.78
ALIGN_MAX_FACTOR = 1.22

# 3.1 Flash TTS is the current model — more expressive and controllable via
# Director's Notes. It occasionally 500s or returns text tokens (documented
# preview bug), so synthesize() retries and falls back to the 2.5 models.
DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"
FALLBACK_MODELS = (
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)
DEFAULT_VOICE = "Charon"
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
MAX_TTS_ATTEMPTS = 4

# Director's Notes prompt structure (per Gemini TTS docs). The preamble + labeled
# TRANSCRIPT prevents the classifier from reading the notes aloud, and the
# steady-pacing note keeps run-to-run tempo consistent for cleaner alignment.
VOICE_DIRECTION = """
Synthesize speech for the transcript below. Do not read these director's notes aloud — speak only the text under TRANSCRIPT.

# AUDIO PROFILE: The Product Narrator
A polished B2B software demo voiceover artist introducing the product to its target users.

## THE SCENE
A clean, modern studio voiceover for a screen-recorded product tour. Confident and credible — a knowledgeable insider showing peers something valuable, not a hype ad.

### DIRECTOR'S NOTES
Style: Warm, confident authority with a subtle vocal smile. Trustworthy and articulate. Land concrete nouns and numbers cleanly. No radio-announcer hype, no vocal fry.
Pacing: Steady and even throughout, about 130 words per minute. Hold the same energy from first line to last — do not speed up or trail off. A brief natural breath between paragraphs; no long gaps.
Accent: Neutral American English.

#### TRANSCRIPT
"""


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


def timestamp_to_ms(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp: {value}")
    minutes, seconds = parts
    return int(minutes) * 60_000 + int(float(seconds) * 1000)


def strip_audio_tags(text: str) -> str:
    """Remove inline audio tags like [short pause] or [warmly] from text.

    Tags steer the TTS delivery but must not count toward filenames or the
    word-weights used to split the continuous take.
    """
    return re.sub(r"\s*\[[^\]]*\]\s*", " ", text).strip()


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", strip_audio_tags(text).lower()).strip("-")
    return slug[:max_len].strip("-") or "segment"


def save_pcm_wav(path: Path, pcm: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    with wave.open(str(path), "rb") as wf:
        return int(wf.getnframes() / wf.getframerate() * 1000)


def extract_pcm(response) -> bytes:
    if not response.candidates:
        raise RuntimeError("No candidates in TTS response")
    candidate = response.candidates[0]
    finish = getattr(candidate, "finish_reason", None) or getattr(candidate, "finishReason", None)
    if finish and str(finish).endswith("OTHER"):
        raise RuntimeError(f"TTS finished with OTHER (no usable audio): {finish}")
    for part in candidate.content.parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data
    raise RuntimeError("No audio inline_data in TTS response")


def model_fallback_chain(preferred: str) -> tuple[str, ...]:
    ordered = [preferred, *FALLBACK_MODELS]
    seen: set[str] = set()
    chain: list[str] = []
    for model in ordered:
        if model not in seen:
            seen.add(model)
            chain.append(model)
    return tuple(chain)


def synthesize(
    client: genai.Client,
    model: str,
    voice: str,
    prompt: str,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for candidate_model in model_fallback_chain(model):
        for attempt in range(1, MAX_TTS_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=build_speech_config(voice),
                    ),
                )
                return extract_pcm(response), candidate_model
            except Exception as error:
                last_error = error
                if attempt < MAX_TTS_ATTEMPTS:
                    wait = 2 ** attempt
                    print(
                        f"  retry {attempt}/{MAX_TTS_ATTEMPTS} for {candidate_model} "
                        f"after {type(error).__name__}: {error}; sleeping {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                print(f"  giving up on {candidate_model}: {error}", file=sys.stderr)
                break
    raise RuntimeError(f"All TTS models failed. Last error: {last_error}")


def build_speech_config(voice: str) -> types.SpeechConfig:
    return types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
        )
    )


def build_continuous_prompt(paragraphs: list[str]) -> str:
    return VOICE_DIRECTION.strip() + "\n\n" + "\n\n".join(p.strip() for p in paragraphs if p.strip())


def find_silence_runs(
    samples: "array.array",
    *,
    sample_rate: int,
    frame_ms: int,
    threshold: int,
) -> list[tuple[int, int]]:
    """Return (start_sample, end_sample) for each contiguous below-threshold run."""
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for frame_start in range(0, len(samples), frame_size):
        frame = samples[frame_start : frame_start + frame_size]
        if not frame:
            break
        peak = max(abs(value) for value in frame)
        if peak < threshold:
            if run_start is None:
                run_start = frame_start
        else:
            if run_start is not None:
                runs.append((run_start, frame_start))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(samples)))
    return runs


def split_pcm_on_silence(
    pcm: bytes,
    *,
    expected_parts: int,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 20,
    silence_threshold: int = 500,
    min_silence_ms: int = 180,
) -> list[bytes]:
    """Split mono 16-bit PCM into exactly `expected_parts` by cutting at the
    longest internal pauses. Deterministic: always returns `expected_parts`
    segments (when the audio is long enough), cut at the most prominent silences.
    """
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples or expected_parts <= 1:
        return [pcm] if pcm else []

    min_silence_samples = int(sample_rate * min_silence_ms / 1000)
    interior_pad = int(sample_rate * 0.5)  # ignore pauses near absolute start/end

    runs = find_silence_runs(
        samples,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        threshold=silence_threshold,
    )

    # Keep interior pauses long enough to be paragraph boundaries.
    candidates = [
        (run_end - run_start, (run_start + run_end) // 2)
        for run_start, run_end in runs
        if (run_end - run_start) >= min_silence_samples
        and run_start > interior_pad
        and run_end < len(samples) - interior_pad
    ]

    if not candidates:
        return [pcm]

    # Take the N-1 longest pauses, then order them by position to cut in sequence.
    candidates.sort(key=lambda item: item[0], reverse=True)
    cut_samples = sorted(midpoint for _, midpoint in candidates[: expected_parts - 1])

    boundaries = [0, *cut_samples, len(samples)]
    segments: list[bytes] = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk = samples[start:end]
        if chunk:
            segments.append(array.array("h", chunk).tobytes())
    return segments


def split_pcm_guided(
    pcm: bytes,
    weights: list[float],
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 20,
    silence_threshold: int = 500,
    min_silence_ms: int = 120,
) -> list[bytes]:
    """Split a single continuous take into len(weights) segments.

    Unlike the longest-pause heuristic, this predicts where each line should
    end — its share of the take, by word weight — and snaps that boundary to
    the nearest real pause within a search window. This stays robust when a
    model (e.g. 3.1 TTS) inserts long intra-line pauses that would otherwise
    fool a "cut at the N-1 longest silences" approach.
    """
    samples = array.array("h")
    samples.frombytes(pcm)
    n_parts = len(weights)
    if not samples or n_parts <= 1:
        return [pcm] if pcm else []

    total_samples = len(samples)
    min_silence_samples = int(sample_rate * min_silence_ms / 1000)
    interior_pad = int(sample_rate * 0.3)

    runs = find_silence_runs(
        samples,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        threshold=silence_threshold,
    )
    candidates = [
        (run_start + run_end) // 2
        for run_start, run_end in runs
        if (run_end - run_start) >= min_silence_samples
        and run_start > interior_pad
        and run_end < total_samples - interior_pad
    ]
    candidates.sort()

    total_weight = float(sum(weights)) or 1.0
    mean_seg = total_samples / n_parts
    radius = int(max(0.5 * mean_seg, sample_rate * 0.6))
    min_gap = int(sample_rate * 0.2)

    cuts: list[int] = []
    prev = 0
    cumulative = 0.0
    for index in range(n_parts - 1):
        cumulative += weights[index]
        target = int((cumulative / total_weight) * total_samples)
        low = max(prev + min_gap, target - radius)
        high = target + radius
        best: int | None = None
        best_distance: int | None = None
        for candidate in candidates:
            if candidate <= prev + min_gap or candidate < low or candidate > high:
                continue
            distance = abs(candidate - target)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = candidate
        if best is None:
            best = max(target, prev + min_gap)
        cuts.append(best)
        prev = best

    boundaries = [0, *cuts, total_samples]
    segments: list[bytes] = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk = samples[start:end]
        if chunk:
            segments.append(array.array("h", chunk).tobytes())
    return segments


def atempo_pcm(pcm: bytes, factor: float) -> bytes:
    """Time-scale a PCM clip by `factor` via ffmpeg atempo (pitch-preserving).

    factor > 1 speeds up (shortens); factor < 1 slows down (lengthens). Used to
    fit each line to its on-screen window so a single natural take hits picture.
    """
    if not pcm or abs(factor - 1.0) < 1e-3:
        return pcm
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.wav")
        out_path = os.path.join(tmp, "out.wav")
        save_pcm_wav(Path(in_path), pcm)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", in_path,
                "-filter:a", f"atempo={factor:.5f}",
                "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                out_path,
            ],
            check=True,
        )
        with wave.open(out_path, "rb") as wf:
            return wf.readframes(wf.getnframes())


def align_segments_to_windows(
    segments: list[bytes],
    suggested_ms: list[int],
    *,
    video_length_ms: int | None,
) -> tuple[list[bytes], list[float]]:
    """Fit each segment to the gap between its timestamp and the next line's.

    Returns the fitted segments and the tempo factor applied to each (for logging).
    Factors are clamped so the nudge stays natural; residual mismatch is absorbed
    by the placement step (tiny pause or push).
    """
    count = len(segments)
    windows: list[int] = []
    for index in range(count):
        if index < count - 1:
            window = suggested_ms[index + 1] - suggested_ms[index]
        elif video_length_ms is not None:
            window = video_length_ms - suggested_ms[index]
        else:
            window = pcm_duration_ms(segments[index])
        windows.append(max(1, window))

    # Global normalization: absorb the whole-take over/undershoot as one uniform
    # tempo change (natural, measured pace) so the per-line fit only makes tiny
    # residual nudges instead of dumping the deficit into clamped lines/gaps.
    total_duration = sum(pcm_duration_ms(pcm) for pcm in segments)
    total_window = sum(windows)
    global_factor = total_duration / total_window if total_window and total_duration else 1.0

    fitted: list[bytes] = []
    factors: list[float] = []
    for index, pcm in enumerate(segments):
        duration_ms = pcm_duration_ms(pcm)
        if duration_ms <= 0:
            fitted.append(pcm)
            factors.append(1.0)
            continue
        residual = duration_ms / (global_factor * windows[index])
        residual = max(ALIGN_MIN_FACTOR, min(ALIGN_MAX_FACTOR, residual))
        net_factor = max(0.5, min(2.0, global_factor * residual))
        fitted.append(atempo_pcm(pcm, net_factor))
        factors.append(net_factor)
    return fitted, factors


def pcm_duration_ms(pcm: bytes) -> int:
    return int(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000)


def build_timeline_placed_pcm(
    placements: list[tuple[int, bytes]],
    *,
    pad_to_ms: int | None = None,
) -> tuple[bytes, list[int]]:
    """Place each (suggested_start_ms, pcm) on a silent bed at its timestamp.

    If a segment would overlap the previous one, it is pushed to start right
    after the previous ends (no speech is lost). Returns (pcm, actual_start_ms list).
    """
    bytes_per_ms = int(SAMPLE_RATE * SAMPLE_WIDTH / 1000)
    timeline = bytearray()
    actual_starts: list[int] = []
    cursor_byte = 0

    for suggested_ms, pcm in placements:
        target_byte = suggested_ms * bytes_per_ms
        start_byte = max(target_byte, cursor_byte)
        if start_byte > len(timeline):
            timeline.extend(b"\x00" * (start_byte - len(timeline)))
        end_byte = start_byte + len(pcm)
        if end_byte > len(timeline):
            timeline.extend(b"\x00" * (end_byte - len(timeline)))
        timeline[start_byte:end_byte] = pcm
        actual_starts.append(start_byte // bytes_per_ms)
        cursor_byte = end_byte

    if pad_to_ms is not None:
        target_len = pad_to_ms * bytes_per_ms
        if target_len > len(timeline):
            timeline.extend(b"\x00" * (target_len - len(timeline)))

    return bytes(timeline), actual_starts


def write_manifest(
    timeline_dir: Path,
    audio_dir: Path,
    *,
    model: str,
    voice: str,
    mode: str,
    manifest_segments: list[dict],
    timeline_reference: str | None,
    notes: str,
) -> Path:
    manifest = {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH,
        "model": model,
        "voice": voice,
        "mode": mode,
        "segments": manifest_segments,
        "editor_notes": notes,
    }
    if timeline_reference:
        manifest["timeline_reference"] = timeline_reference
        manifest["full_recording"] = timeline_reference

    manifest_path = audio_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def generate_continuous(
    client: genai.Client,
    model: str,
    voice: str,
    lines: list[dict],
    timeline_dir: Path,
    audio_dir: Path,
    segments_dir: Path,
    video_length_ms: int | None = None,
    align: bool = True,
) -> None:
    paragraphs = [line.get("narration", "").strip() for line in lines]
    prompt = build_continuous_prompt(paragraphs)

    print(f"Continuous take: {model} (voice: {voice}) — one API call for {len(lines)} paragraphs...")
    full_pcm, model_used = synthesize(client, model, voice, prompt)

    full_path = audio_dir / "full-recording.wav"
    full_duration_ms = save_pcm_wav(full_path, full_pcm)
    print(f"  model used: {model_used}")
    print(f"  full recording (talk track, no gaps): {full_duration_ms / 1000:.1f}s → {full_path}")

    weights = [max(1.0, float(len(strip_audio_tags(line.get("narration", "")).split()))) for line in lines]
    split_segments = split_pcm_guided(full_pcm, weights)
    if len(split_segments) != len(lines):
        print(
            f"  WARNING: guided split produced {len(split_segments)} parts, expected {len(lines)}. "
            f"Use full-recording.wav in your editor; segment files are best-effort.",
            file=sys.stderr,
        )

    suggested_all = [timestamp_to_ms(line.get("timestamp", "00:00")) for line in lines]

    if align and len(split_segments) == len(lines):
        print("  aligning each line to its on-screen window (per-line tempo fit)...")
        split_segments, factors = align_segments_to_windows(
            split_segments, suggested_all, video_length_ms=video_length_ms
        )
        for index, factor in enumerate(factors):
            if abs(factor - 1.0) >= 0.05:
                tag = "faster" if factor > 1 else "slower"
                print(f"    line {index:02d}: {factor:.2f}x ({tag})")

    manifest_segments: list[dict] = []
    placements: list[tuple[int, bytes]] = []

    for index, line in enumerate(lines):
        timestamp = line.get("timestamp", "00:00")
        text = line.get("narration", "").strip()
        ts_slug = timestamp.replace(":", "-")
        file_slug = slugify(text)
        filename = f"{ts_slug}-{index:02d}-{file_slug}.wav"
        wav_path = segments_dir / filename
        suggested_ms = suggested_all[index]

        if index < len(split_segments):
            seg_pcm = split_segments[index]
            duration_ms = save_pcm_wav(wav_path, seg_pcm)
            placements.append((suggested_ms, seg_pcm))
            seg_file = str(wav_path.relative_to(timeline_dir))
        else:
            duration_ms = 0
            seg_file = None

        manifest_segments.append(
            {
                "index": index,
                "timestamp": timestamp,
                "suggested_timeline_ms": suggested_ms,
                "duration_ms": duration_ms,
                "file": seg_file,
                "narration": text,
            }
        )

    # Timeline-placed master: segments at their timestamps on a silent bed,
    # padded to the video length so it drops straight onto the video track.
    timeline_pcm, actual_starts = build_timeline_placed_pcm(placements, pad_to_ms=video_length_ms)
    timeline_path = audio_dir / "full-recording-timeline.wav"
    timeline_duration_ms = save_pcm_wav(timeline_path, timeline_pcm)
    print(
        f"  timeline-placed master (video-aligned): {timeline_duration_ms / 1000:.1f}s → {timeline_path}"
    )

    for entry, actual_start in zip(manifest_segments, actual_starts):
        entry["timeline_start_ms"] = actual_start

    write_manifest(
        timeline_dir,
        audio_dir,
        model=model_used,
        voice=voice,
        mode="continuous",
        manifest_segments=manifest_segments,
        timeline_reference=str(timeline_path.relative_to(timeline_dir)),
        notes=(
            "Two masters: full-recording.wav (continuous talk track, no gaps) and "
            "full-recording-timeline.wav (segments placed at their video timestamps, "
            "padded to video length — drop this straight onto the video track). "
            "timeline_start_ms is the actual placement after overlap resolution; "
            "suggested_timeline_ms is the script's target."
        ),
    )


def generate_segmented(
    client: genai.Client,
    model: str,
    voice: str,
    lines: list[dict],
    timeline_dir: Path,
    audio_dir: Path,
    segments_dir: Path,
    video_length_ms: int | None = None,
) -> None:
    print(
        f"Segmented mode: {len(lines)} separate API calls — exact per-line durations "
        f"for reliable video sync (slight tone drift possible between clips).",
        file=sys.stderr,
    )

    manifest_segments: list[dict] = []
    placements: list[tuple[int, bytes]] = []

    segment_prompt_prefix = (
        "Synthesize speech. Do not read these instructions aloud.\n\n"
        "### PERFORMANCE\nProfessional product demo narrator. Steady ~130 wpm. Minimal vocal fry.\n\n"
        "#### TRANSCRIPT\n"
    )
    model_used = model

    for index, line in enumerate(lines):
        timestamp = line.get("timestamp", "00:00")
        text = line.get("narration", "").strip()
        if not text:
            continue

        ts_slug = timestamp.replace(":", "-")
        filename = f"{ts_slug}-{index:02d}-{slugify(text)}.wav"
        wav_path = segments_dir / filename
        suggested_ms = timestamp_to_ms(timestamp)

        print(f"  [{timestamp}] {text[:70]}{'...' if len(text) > 70 else ''}")
        pcm, model_used = synthesize(client, model, voice, segment_prompt_prefix + text)
        duration_ms = save_pcm_wav(wav_path, pcm)

        manifest_segments.append(
            {
                "index": index,
                "timestamp": timestamp,
                "suggested_timeline_ms": suggested_ms,
                "duration_ms": duration_ms,
                "file": str(wav_path.relative_to(timeline_dir)),
                "narration": text,
            }
        )
        placements.append((suggested_ms, pcm))

    timeline_pcm, actual_starts = build_timeline_placed_pcm(placements, pad_to_ms=video_length_ms)
    timeline_path = audio_dir / "full-recording-timeline.wav"
    timeline_duration_ms = save_pcm_wav(timeline_path, timeline_pcm)
    print(
        f"  timeline-placed master (video-aligned): {timeline_duration_ms / 1000:.1f}s → {timeline_path}"
    )

    for entry, actual_start in zip(manifest_segments, actual_starts):
        entry["timeline_start_ms"] = actual_start

    write_manifest(
        timeline_dir,
        audio_dir,
        model=model_used,
        voice=voice,
        mode="segmented",
        manifest_segments=manifest_segments,
        timeline_reference=str(timeline_path.relative_to(timeline_dir)),
        notes=(
            "Per-line generation for exact durations and reliable sync. "
            "full-recording-timeline.wav is the video-aligned master (padded to video length) — "
            "drop it straight onto the video track. timeline_start_ms is actual placement after "
            "overlap resolution; suggested_timeline_ms is the script target."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo narration audio")
    parser.add_argument("timeline_dir", type=Path, help="Directory with narration_script.json")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"TTS model (default: {DEFAULT_MODEL}). "
            "gemini-3.1-flash-tts-preview is preview-only and may 500; script auto-fallbacks."
        ),
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Voice name (default: {DEFAULT_VOICE})")
    parser.add_argument(
        "--mode",
        choices=("continuous", "segmented"),
        default="continuous",
        help="continuous = one take (consistent); segmented = per-line (default: continuous)",
    )
    parser.add_argument("--script", type=Path, help="Override narration_script.json path")
    parser.add_argument(
        "--video-length",
        default="1:30",
        help="Video length MM:SS to pad the timeline-placed master to (default: 1:30)",
    )
    parser.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        help="Disable per-line tempo fit to on-screen windows (continuous mode only)",
    )
    parser.set_defaults(align=True)
    args = parser.parse_args()

    video_length_ms = timestamp_to_ms(args.video_length) if args.video_length else None

    timeline_dir = args.timeline_dir.expanduser().resolve()
    script_path = (args.script or timeline_dir / "narration_script.json").resolve()
    if not script_path.is_file():
        print(f"ERROR: Narration script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    lines = json.loads(script_path.read_text(encoding="utf-8")).get("script_lines", [])
    if not lines:
        print("ERROR: narration_script.json has no script_lines", file=sys.stderr)
        sys.exit(1)

    audio_dir = timeline_dir / "audio"
    segments_dir = audio_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=load_api_key())

    if args.mode == "continuous":
        generate_continuous(
            client,
            args.model,
            args.voice,
            lines,
            timeline_dir,
            audio_dir,
            segments_dir,
            video_length_ms=video_length_ms,
            align=args.align,
        )
    else:
        generate_segmented(
            client,
            args.model,
            args.voice,
            lines,
            timeline_dir,
            audio_dir,
            segments_dir,
            video_length_ms=video_length_ms,
        )

    print(f"\nWrote {audio_dir / 'manifest.json'}")

    run_meta_path = timeline_dir / "run.json"
    if run_meta_path.is_file():
        try:
            meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            meta.update(
                {
                    "audio_generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "audio_model": args.model,
                    "audio_voice": args.voice,
                    "audio_mode": args.mode,
                    "audio_align": bool(args.align),
                    "audio_timeline_wav": "audio/full-recording-timeline.wav",
                    "audio_manifest": "audio/manifest.json",
                    "video_length": args.video_length,
                }
            )
            run_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"Updated {run_meta_path}")
        except Exception as exc:
            print(f"WARNING: could not update run.json: {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
