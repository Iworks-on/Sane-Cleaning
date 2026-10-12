# ClearTake — AI Audio Cleaner

ClearTake is a single-recording audio cleanup app. It accepts WAV and MP3, applies real DeepFilterNet speech denoising, measures integrated loudness with `pyloudnorm`, normalizes toward -16 LUFS, and uses FFmpeg peak protection before exporting WAV or MP3.

## Use it

1. Choose or drop a WAV or MP3 voice recording.
2. Select WAV for a lossless output or MP3 for a smaller file.
3. Start cleaning, then compare the original and processed players.
4. Download the processed recording or start a new session.

The current defaults allow files up to 100 MiB and recordings up to 10 minutes. Processing runs one job at a time, with a short queue. Limits can be adjusted with `AUDIO_MAX_UPLOAD_MB`, `AUDIO_MAX_DURATION_SECONDS`, and `AUDIO_MAX_ACTIVE_JOBS`.

## Runtime

The web UI lives in this artifact. The existing API server starts the FastAPI audio service as a local sidecar on `127.0.0.1:8099`; the browser reaches it through the same-origin `/api/audio` routes. The sidecar uses temporary files under `/tmp/audio-cleaner-jobs` by default. Completed and failed jobs expire after 30 minutes; resetting a session removes its files immediately. `AUDIO_JOB_RETENTION_SECONDS` and `AUDIO_TEMP_DIR` can change those settings.

The Python packages and compatible x86-64 model runtime are listed in `../api-server/audio_service/requirements.txt`. DeepFilterNet is provided by the MIT-licensed `dfnstream-py` package. FFmpeg and `ffprobe` must be available on the server. If the model cannot load, the app reports the engine as unavailable and does not return a pretend-cleaned file.

For local development, start the API server and web artifact using their configured Replit workflows. The API workflow builds and runs the Node API plus its FastAPI sidecar; the web workflow runs Vite. Public publishing requires a runtime that includes the Python packages, the model wheel, FFmpeg, and enough memory for the configured recording limit.

## Audio data and limits

Audio is not saved to a database or object store. The original upload is kept separately from the cleaned output so it remains available for comparison. The server deletes job files on reset and automatically expires completed or failed jobs. Job records are held in memory, so an API restart ends active sessions and removes stale temporary directories.
