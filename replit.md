# ClearTake

Cleans church recordings (sermons, services, Bible studies) for public sharing: removes fan, air-con, hum and room noise, lifts low or uneven volume, and exports MP3/WAV.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/api-server/audio_service/main.py` — Python/FastAPI audio engine (all DSP lives here)
- `artifacts/api-server/src/` — Express server; spawns the Python service and proxies `/api/audio/*` to it
- `artifacts/audio-cleaner/src/pages/audio-cleaner.tsx` — the single-page UI
- `lib/api-spec/openapi.yaml` — API contract; run codegen after editing

## Architecture decisions

- Pipeline: FFmpeg decode (+ ebur128 input meter, 80 Hz high-pass, optional 50/60 Hz notch) → DeepFilterNet denoise → optional leveling (acompressor / dynaudnorm) → two-pass EBU R128 `loudnorm` (TP −1.5 dBTP) → encode.
- Everything streams from disk (RF64 intermediates) so 3-hour recordings don't need to fit in RAM.
- Processing is mono: speech-focused, halves DeepFilterNet CPU time, avoids stereo phase artifacts.
- Near-silent input (< −60 LUFS) is not boosted, to avoid amplifying noise.
- Jobs live in memory + temp dir; a restart loses them. Single worker thread, so jobs run one at a time.

## Product

- Upload MP3/WAV/M4A/AAC/FLAC/OGG/OPUS/WMA/AIFF/MP4/MOV/WEBM, up to 1 GB / 3 h (env: `AUDIO_MAX_UPLOAD_MB`, `AUDIO_MAX_DURATION_SECONDS`)
- Settings: noise removal (light/standard/strong), volume boost (off/even out/strong), loudness (−19/−16/−14 LUFS), hum removal (off/50/60 Hz), rumble filter
- Live progress, A/B playback, before/after loudness readout, download

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Requires `ffmpeg`/`ffprobe` on PATH (in `.replit` nix packages). The engine reports "unavailable" without them.
- `dfnstream-py` only ships Linux/macOS binaries; DeepFilterNet cannot run on Windows.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
