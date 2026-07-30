# screen-recording-to-demo

Turn a raw screen recording into a polished, AI-narrated product demo where the
voice is natural, consistent, and synced to the on-screen action.

Two Python scripts + [Google Gemini](https://ai.google.dev/). One watches the
video and writes a timestamped timeline; the other writes and speaks the
narration, then syncs it to the video. Optional helpers mux a deliverable and
emit WebVTT captions.

> Shared as-is, no warranty. Do whatever you want with it (MIT).

## The one idea worth stealing

Generate the whole voiceover as a **single TTS take** (so the voice never
drifts), then **time-stretch each line by a tiny, inaudible amount** so it lands
on the action it describes. Natural *and* synced.

Per-line TTS calls give easy timing but audible voice drift. Global stretch
drifts mid-video. Fit-to-picture (what VO editors do) keeps one performance and
fixes timing in post.

When a cut has long intentional film/wait gaps, skip per-line tempo fit with
`--no-align` so speech is not dragged across dead air.

## Requirements

- Python 3.10+
- [`google-genai`](https://pypi.org/project/google-genai/) SDK
- [`ffmpeg`](https://ffmpeg.org/) / `ffprobe` on `PATH`
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg   # or your platform's equivalent

cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
# or: export GEMINI_API_KEY=...
```

**Never commit `.env` or API keys.** The repo gitignores them.

## Quick start

```bash
# 0. Record a high-res screen demo first (OBS / screen recorder). Prefer 4K.
#    This pipeline does not capture video for you.

CUT=my-product-demo
mkdir -p "output/demo-timeline/$CUT/masters"

# 1. Encode a compressed proxy for Gemini analysis (once per cut).
#    Keep the high-res master separately for the final mux.
ffmpeg -y -i "path/to/your-master.mov" \
  -an -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -movflags +faststart \
  "output/demo-timeline/$CUT/masters/picture-1080p.mp4"

# 2. Extract timeline + draft narration (creates output/.../runs/<timestamp>/)
python3 extract_demo_timeline.py \
  "output/demo-timeline/$CUT/masters/picture-1080p.mp4" \
  --cut-slug "$CUT" \
  --narration \
  --product-context examples/product-context.example.md

# 3. Edit the WORKING script only (leave narration_script.model.json alone)
#    output/demo-timeline/$CUT/latest/narration_script.json

# 4. Generate voiceover. Pass the real video length.
#    Prefer --no-align when the cut has long film/wait gaps.
python3 generate_demo_narration_audio.py \
  "output/demo-timeline/$CUT/latest" \
  --video-length 1:30
# python3 generate_demo_narration_audio.py ... --video-length 1:30 --no-align

# 5. Mux onto the high-res master (pads picture if audio is longer)
./mux_demo_deliverable.sh \
  --cut "$CUT" \
  --source "path/to/your-master.mov" \
  --resolution 1080p

# 6. Optional captions for YouTube
python3 generate_demo_captions.py \
  "output/demo-timeline/$CUT/latest/audio/manifest.json" \
  "output/demo-timeline/$CUT/deliverables/captions.vtt" \
  --video-length 1:30
```

Optional loudness normalize (YouTube-ish target, −14 LUFS):

```bash
ffmpeg -i narrated.mp4 -af loudnorm=I=-14:TP=-1.5:LRA=11 \
  -c:v copy -c:a aac -b:a 192k narrated-normalized.mp4
```

## Run layout

```
output/demo-timeline/<cut-slug>/
  masters/
    picture-1080p.mp4          # H.264 proxy for Gemini (recommended)
  runs/<YYYYMMDD-HHMMSS>/
    run.json
    visual_timeline.json|.md
    coverage_report.md
    narration_script.model.json   # model draft — leave alone
    narration_script.json|.md     # WORKING copy — edit before TTS
    audio/                        # after generate_demo_narration_audio.py
  latest -> runs/<...>
  deliverables/                   # muxed mp4 + captions
```

## Architecture

```
Screen recording (prefer 4K master; analyze a proxy)
   │
   ▼
extract_demo_timeline.py   ── Gemini video understanding (multi-pass)
   │   • Pass 1: segment into chapters
   │   • Pass 2: exhaustive per-chapter events (higher FPS)
   │   • Pass 3: gap-fill re-analysis of any ≥6s window with no event
   ▼
visual_timeline.json  +  narration_script.json (draft)
   │
   │   (human edits the narration copy to fit the beats)
   ▼
generate_demo_narration_audio.py   ── Gemini TTS + alignment
   │   • ONE continuous TTS take (consistent voice)
   │   • split at natural pauses (word-weight guided)
   │   • fit each line to its on-screen window (tempo)  [or --no-align]
   ▼
audio/full-recording-timeline.wav
   │
   ├── mux_demo_deliverable.sh  → deliverables/*.mp4
   └── generate_demo_captions.py → .vtt
```

## How alignment works

1. Split the continuous take at natural pauses, guided by each line's word count
   (`split_pcm_guided`): predict where each line should end as its share of the
   take, then snap to the nearest real silence.
2. Globally normalize the whole take to video length with one gentle uniform
   tempo change.
3. Apply a small, clamped per-line tempo nudge (pitch-preserving via ffmpeg
   `atempo`, capped ~0.78–1.22×) so each line lands on its beat
   (`align_segments_to_windows`).

Result: one natural take, no jumpy silent gaps, every line within ~1s of its
mark. It also absorbs TTS run-to-run speed variance.

Verify sync in `audio/manifest.json`: each line's `timeline_start_ms` should sit
near its `suggested_timeline_ms`. A large printed tempo factor means the copy is
mis-sized for its window — rewrite the line rather than fighting the audio.

**`--no-align`:** place each line at its timestamp without per-line tempo fit.
Use this when long intentional pauses would otherwise stretch speech across waits.

**`--video-length`:** defaults to `1:30` if omitted. Always pass your real cut
length so padding and windows match the picture.

## Prompting tips that mattered

- Use the TTS "Director's Notes" format (Audio Profile / Scene / Style-Pacing-
  Accent). Explicit "steady ~130 wpm, hold energy" reduces pace drift.
- Add a preamble so the model does not read stage directions aloud.
- Inline tags like `[short pause]` steer delivery; they are stripped from
  word-weights and captions so they do not skew timing or on-screen text.
- Ground copy with a product-context file (see `examples/`). Review every line
  for hallucinated numbers and legal-risky wording before you ship.

## Cost and runtime (rough)

- **Video understanding dominates cost** — multi-pass sends the clip several
  times. Use `--single-pass` when you do not need exhaustive capture.
- **TTS is one call** for the whole script in continuous mode — cheap/fast.
- **ffmpeg is negligible**, especially for mostly-static screen recordings.
- Prefer a **compressed proxy** for Gemini analysis; always **encode the final
  deliverable from the high-res master** (encoding from a 720p proxy is the usual
  cause of a "fuzzy" final video).

## Limitations

- Alignment is tempo-based, not word-based. Fine for narration over a screen
  tour; not enough for tight action-by-action callouts or lip-sync.
- Preview TTS models vary run-to-run and occasionally fail; the script retries
  with exponential backoff and a model fallback chain.
- If a line's copy is badly mis-sized, the per-line factor hits the clamp and a
  short silence can remain. Fix the copy.
- Model IDs in the defaults are moving targets; override with `--model` /
  `--narration-model` when Google renames them.

## Alternatives considered

| Approach | Why not (for this use case) |
|---|---|
| Per-line TTS | Exact timing, but audible voice drift between calls |
| Whisper forced alignment | More precise cuts; heavier dependency for marginal gain at demo length |
| SSML hard timing | Gemini TTS does not expose reliable word timings / break clocks |
| Manual VO in a NLE | Higher ceiling, not reproducible when copy changes |

## Privacy and security

- **API keys:** put `GEMINI_API_KEY` only in the environment or a local `.env` /
  `.env.local` (gitignored). Never commit keys. Rotate any key that was ever
  pasted into a chat, ticket, or public gist.
- **Your recording, your key:** `extract_demo_timeline.py` uploads the video
  path *you* pass to Google's Gemini File API using *your* API key so the model
  can analyze it. That is the caller's footage and account — not anyone else's
  demo masters. The script deletes the uploaded File API object when finished
  (unless `--keep-uploaded-file`). If delete fails, remove it manually in AI
  Studio / the Files API.
- **Do not upload sensitive footage** you cannot risk leaving server-side even
  briefly (credentials on screen, PII, unreleased product UI you cannot share
  with the model provider under their terms).
- **Do not commit** `output/`, recordings, WAVs, or product-specific context with
  private copy. Those paths are gitignored.
- This repo contains no credentials and no proprietary product context. Abuse of
  Gemini still requires an attacker's own API key and billing.

## License

MIT — see [LICENSE](LICENSE).
