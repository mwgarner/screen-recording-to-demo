#!/usr/bin/env bash
# Mux an existing narration WAV onto a picture master for YouTube / web.
#
# Reuses audio from a prior extract/generate run — no TTS regeneration.
# Pads the picture with a cloned last frame if the WAV is longer than the video
# (avoids truncating the voiceover with ffmpeg -shortest alone).
#
# Usage:
#   ./mux_demo_deliverable.sh \
#     --cut my-product-demo \
#     --source path/to/picture-master.mov \
#     --resolution 1080p
#
#   ./mux_demo_deliverable.sh \
#     --cut my-product-demo \
#     --source output/demo-timeline/my-product-demo/masters/picture-4k.mp4 \
#     --run runs/20260725-214500 \
#     --resolution 4k
set -euo pipefail

CUT_SLUG=""
SOURCE=""
RUN_DIR=""
RESOLUTION="1080p"
CRF=16
PRESET=slow

usage() {
  sed -n '2,20p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cut) CUT_SLUG="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --run) RUN_DIR="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --crf) CRF="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$CUT_SLUG" && -n "$SOURCE" ]] || usage

ROOT="$(cd "$(dirname "$0")" && pwd)"
CUT_ROOT="$ROOT/output/demo-timeline/$CUT_SLUG"
SOURCE="${SOURCE/#\~/$HOME}"

resolve_path() {
  python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

if [[ -z "$RUN_DIR" ]]; then
  [[ -e "$CUT_ROOT/latest" ]] || { echo "Missing latest symlink: $CUT_ROOT/latest" >&2; exit 1; }
  RUN_DIR="$(resolve_path "$CUT_ROOT/latest")"
else
  if [[ "$RUN_DIR" = /* ]]; then
    RUN_DIR="$(resolve_path "$RUN_DIR")"
  else
    RUN_DIR="$(resolve_path "$CUT_ROOT/$RUN_DIR")"
  fi
fi

WAV="$RUN_DIR/audio/full-recording-timeline.wav"
[[ -f "$WAV" ]] || { echo "Missing WAV: $WAV" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "Missing source: $SOURCE" >&2; exit 1; }

command -v ffmpeg >/dev/null || { echo "ffmpeg not found on PATH" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found on PATH" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
DELIV="$CUT_ROOT/deliverables/${STAMP}"
mkdir -p "$DELIV"

VID_DUR="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$SOURCE")"
AUD_DUR="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$WAV")"
PAD="$(python3 -c "print(f'{max(0.0, float(\"$AUD_DUR\") - float(\"$VID_DUR\")):.3f}')")"

echo "source=$SOURCE"
echo "audio=$WAV"
echo "video=${VID_DUR}s audio=${AUD_DUR}s pad=${PAD}s resolution=$RESOLUTION"

PICTURE="$DELIV/picture-for-mux.mp4"
if python3 -c "exit(0 if float('$PAD') > 0.001 else 1)"; then
  ffmpeg -y -loglevel error -i "$SOURCE" \
    -vf "tpad=stop_mode=clone:stop_duration=$PAD" \
    -an -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p -movflags +faststart \
    "$PICTURE"
else
  ffmpeg -y -loglevel error -i "$SOURCE" -an \
    -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p -movflags +faststart \
    "$PICTURE"
fi

case "$RESOLUTION" in
  4k|2160p)
    SCALE='scale=3840:2160:flags=lanczos,format=yuv420p'
    OUT_NAME="${CUT_SLUG}-narrated-4k.mp4"
    ;;
  1080p)
    SCALE='scale=1920:1080:flags=lanczos,format=yuv420p'
    OUT_NAME="${CUT_SLUG}-narrated-1080p.mp4"
    ;;
  *)
    echo "Unknown resolution: $RESOLUTION (use 4k or 1080p)" >&2
    exit 1
    ;;
esac

OUT="$DELIV/$OUT_NAME"
ffmpeg -y -loglevel error -i "$PICTURE" -i "$WAV" \
  -map 0:v:0 -map 1:a:0 \
  -vf "$SCALE" \
  -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart -shortest \
  "$OUT"

# Copy captions from a prior deliverable for this run when present.
PRIOR_DELIV="$CUT_ROOT/deliverables/$(basename "$RUN_DIR")"
if [[ -d "$PRIOR_DELIV" ]]; then
  VTT="$(find "$PRIOR_DELIV" -maxdepth 1 -name '*.vtt' -print -quit 2>/dev/null || true)"
  if [[ -n "${VTT:-}" && -f "$VTT" ]]; then
    cp "$VTT" "$DELIV/"
  fi
fi

ffprobe -hide_banner -v error -select_streams v:0 \
  -show_entries stream=width,height,codec_name,bit_rate \
  -show_entries format=duration,size,bit_rate -of default=nw=1 "$OUT"
ls -lh "$OUT"
echo "Deliverable: $OUT"
