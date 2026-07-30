# screen-recording-to-demo

Turn a raw screen recording into a polished, AI-narrated product demo where the
voice is natural, consistent, and synced to the on-screen action.

Two Python scripts + [Google Gemini](https://ai.google.dev/). One watches the
video and writes a timestamped timeline; the other writes and speaks the
narration, then syncs it to the video.

> Shared as-is, no warranty. Do whatever you want with it (MIT).

## The one idea worth stealing

Generate the whole voiceover as a **single TTS take** (so the voice never
drifts), then **time-stretch each line by a tiny, inaudible amount** so it lands
on the action it describes. Natural *and* synced.

Per-line TTS calls give easy timing but audible voice drift. Global stretch
drifts mid-video. Fit-to-picture (what VO editors do) keeps one performance and
fixes timing in post.

## Requirements

- Python 3.10+
- [`google-genai`](https://pypi.org/project/google-genai/) SDK
- [`ffmpeg`](https://ffmpeg.org/) on `PATH`
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
# 1. Record a high-res screen demo first (OBS / screen recorder). Prefer 4K.
#    The pipeline does not capture video for you.

# 2. Extract timeline + draft narration
python3 extract_demo_timeline.py "demo.mov" --narration \
  --product-context examples/product-context.example.md

# 3. Edit the working script (timestamps + copy)
#    output/demo-timeline/<cut>/latest/narration_script.json

# 4. Generate a video-aligned voiceover master
python3 generate_demo_narration_audio.py output/demo-timeline/<cut>/latest \
  --video-length 2:38

# 5. Mux onto the master and encode (1080p from a 4K source shown here)
ffmpeg -y -i "demo.mov" \
  -i output/demo-timeline/<cut>/latest/audio/full-recording-timeline.wav \
  -map 0:v:0 -map 1:a:0 \
  -vf "scale=1920:1080:flags=lanczos" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart -shortest narrated-1080p.mp4
```

Optional loudness normalize (YouTube-ish target, −14 LUFS):

```bash
ffmpeg -i narrated-1080p.mp4 -af loudnorm=I=-14:TP=-1.5:LRA=11 \
  -c:v copy -c:a aac -b:a 192k narrated-1080p-normalized.mp4
```

## Architecture

```
Screen recording (prefer 4K master)
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
   │   • fit each line to its on-screen window (tempo)
   ▼
audio/full-recording-timeline.wav  (video-aligned master)
   │
   ▼
ffmpeg mux onto master → MP4 → YouTube / site embed
```

Each extract run writes an immutable directory under
`output/demo-timeline/<cut>/runs/<timestamp>/` and updates a `latest` symlink.

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

## Prompting tips that mattered

- Use the TTS "Director's Notes" format (Audio Profile / Scene / Style-Pacing-
  Accent). Explicit "steady ~130 wpm, hold energy" reduces pace drift.
- Add a preamble so the model does not read stage directions aloud.
- Inline tags like `[short pause]` steer delivery; they are stripped from
  word-weights so they do not skew the split.
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

## Alternatives considered

| Approach | Why not (for this use case) |
|---|---|
| Per-line TTS | Exact timing, but audible voice drift between calls |
| Whisper forced alignment | More precise cuts; heavier dependency for marginal gain at demo length |
| SSML hard timing | Gemini TTS does not expose reliable word timings / break clocks |
| Manual VO in a NLE | Higher ceiling, not reproducible when copy changes |

## Security

- Put keys only in the environment or a local `.env` / `.env.local` (gitignored).
- Uploaded videos are deleted from the Gemini File API at the end of extraction
  when possible — still avoid uploading sensitive footage you cannot risk
  leaving server-side.
- Do not commit recordings, WAVs, or product-specific context with private copy.

## License

MIT — see [LICENSE](LICENSE).
