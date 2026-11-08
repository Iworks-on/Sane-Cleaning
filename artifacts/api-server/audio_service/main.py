"""Audio cleaning service.

Pipeline (all steps stream from disk, so multi-hour sermon recordings do not need
to fit in memory):

1. decode     FFmpeg reads any common audio/video file, measures the original
              loudness, removes low rumble (fans, HVAC, stage thumps) and
              optional mains hum, then writes 48 kHz mono float audio.
2. denoise    DeepFilterNet removes steady and changing background noise
              (fans, air conditioning, room hiss, crowd murmur).
3. analyze    Optional volume leveling evens out quiet and loud speakers, then
              FFmpeg's EBU R128 loudnorm measures the result.
4. encode     A second loudnorm pass raises (or lowers) the volume to the chosen
              target while keeping true peaks below the ceiling, then encodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse


DATA_DIR = Path(os.environ.get("AUDIO_TEMP_DIR", "/tmp/audio-cleaner-jobs"))
MAX_UPLOAD_BYTES = max(1, int(os.environ.get("AUDIO_MAX_UPLOAD_MB", "1024"))) * 1024 * 1024
MAX_DURATION_SECONDS = max(30, int(os.environ.get("AUDIO_MAX_DURATION_SECONDS", str(3 * 60 * 60))))
MAX_ACTIVE_JOBS = max(1, int(os.environ.get("AUDIO_MAX_ACTIVE_JOBS", "4")))
RETENTION_SECONDS = max(60, int(os.environ.get("AUDIO_JOB_RETENTION_SECONDS", str(2 * 60 * 60))))
JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SAMPLE_RATE = 48_000
TRUE_PEAK_CEILING = -1.5
LOUDNESS_RANGE = 11.0
SILENCE_LUFS = -60.0

# Formats people actually bring from phones, Zoom/Tascam recorders, mixing desks and livestreams.
ACCEPTED_EXTENSIONS = [
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "aiff", "aif", "mp4", "mov", "webm",
]
MEDIA_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "aac": "audio/aac",
    "flac": "audio/flac", "ogg": "audio/ogg", "opus": "audio/ogg", "wma": "audio/x-ms-wma",
    "aiff": "audio/aiff", "aif": "audio/aiff", "mp4": "video/mp4", "mov": "video/quicktime",
    "webm": "audio/webm",
}

# DeepFilterNet attenuation limit in dB. Higher removes more noise; 100 means no limit.
NOISE_REDUCTION_LEVELS = {"light": 12.0, "standard": 24.0, "strong": 100.0}
LOUDNESS_TARGETS = {"standard": -16.0, "louder": -14.0, "softer": -19.0}
LEVELING_MODES = {"off", "gentle", "strong"}
HUM_MODES = {"off", "50", "60"}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
PROCESSING_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-cleaner")
ENGINE_ERROR: str | None = None
ENGINE_NAME = "DeepFilterNet speech enhancement with rumble and hum filters"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("audio-cleaner")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_engine_check() -> tuple[bool, str]:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            return False, f"{binary} is not installed on the server, so recordings cannot be decoded."
    try:
        from dfnstream_py import DeepFilterNetStreaming

        processor = DeepFilterNetStreaming(log_level="error", atten_lim=15.0)
        processor.close()
        return True, ENGINE_NAME
    except Exception:
        return False, "The DeepFilterNet model could not be loaded. Check the server logs and model installation."


def _cleanup_expired() -> None:
    cutoff = time.time() - RETENTION_SECONDS
    with JOBS_LOCK:
        expired = [
            job_id
            for job_id, job in JOBS.items()
            if job["status"] in {"completed", "failed"} and job["_created_epoch"] < cutoff
        ]
        for job_id in expired:
            shutil.rmtree(job_path(job_id), ignore_errors=True)
            JOBS.pop(job_id, None)


def job_path(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="This audio job was not found or has expired.")
    return DATA_DIR / job_id


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def update_job(job_id: str, **changes: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(changes)


def ffmpeg_timeout(duration: float) -> int:
    return int(max(600, duration * 3))


def run_command(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio conversion took too long.") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or "").strip().splitlines()
        detail = message[-1] if message else "Audio conversion failed."
        raise RuntimeError(detail[:240]) from exc


def run_ffmpeg(
    args: list[str],
    *,
    duration: float,
    on_progress: Callable[[float], None] | None = None,
) -> str:
    """Run FFmpeg, report fractional progress, and return its stderr log (info level)."""
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-y", "-progress", "pipe:1", *args]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace"
    )
    stderr_lines: list[str] = []
    reader = threading.Thread(target=lambda: stderr_lines.extend(process.stderr), daemon=True)  # type: ignore[arg-type]
    reader.start()
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(ffmpeg_timeout(duration), kill)
    watchdog.start()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if on_progress and line.startswith("out_time_us=") and duration > 0:
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                    on_progress(min(1.0, max(0.0, seconds / duration)))
                except ValueError:
                    pass
        process.wait()
    finally:
        watchdog.cancel()
        reader.join(timeout=5)
    log = "".join(stderr_lines)
    if timed_out.is_set():
        raise RuntimeError("Audio conversion took too long.")
    if process.returncode != 0:
        tail = [line for line in log.strip().splitlines() if line.strip()][-3:]
        raise RuntimeError(("FFmpeg failed: " + " | ".join(tail))[:400])
    return log


def inspect_audio(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=duration",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    try:
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams") or []
        if not streams:
            raise ValueError("This file does not contain an audio track.")
        stream = streams[0]
        raw_duration = metadata.get("format", {}).get("duration") or stream.get("duration")
        duration = float(raw_duration)
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except ValueError as exc:
        if "audio track" in str(exc):
            raise
        raise ValueError("The recording could not be read. The file may be damaged or in an unusual format.") from exc
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The recording could not be read. The file may be damaged or in an unusual format.") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The recording length could not be read. Try exporting it again as WAV or MP3.")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"This recording is {int(duration // 60)} minutes long. The current limit is "
            f"{MAX_DURATION_SECONDS // 60} minutes."
        )
    if sample_rate < 8_000 or channels < 1:
        raise ValueError("This recording's audio format is not supported.")
    return {"duration": duration, "sample_rate": sample_rate, "channels": channels}


def prefilter_chain(settings: dict[str, Any]) -> str:
    """Filters applied before noise reduction. The ebur128 meter measures the untouched original."""
    filters = ["ebur128=framelog=quiet"]
    if settings["rumbleFilter"]:
        # 80 Hz sits below the lowest speaking voices but removes fan/HVAC rumble and mic handling thumps.
        filters.append("highpass=f=80:poles=2")
    if settings["humRemoval"] != "off":
        base = int(settings["humRemoval"])
        for harmonic in range(1, 5):
            # Two cascaded notches cut deeper. Wider at the low hum frequencies, narrower where voices live.
            q = 4 if harmonic <= 2 else 10
            notch = f"bandreject=f={base * harmonic}:width_type=q:w={q}"
            filters.extend([notch, notch])
    return ",".join(filters)


def leveling_chain(mode: str) -> str:
    """Volume leveling so a quiet speaker, a distant reader and a loud preacher end up closer together."""
    if mode == "gentle":
        return "acompressor=threshold=0.1:ratio=2.5:attack=10:release=250:knee=4:makeup=1"
    if mode == "strong":
        return (
            "dynaudnorm=f=250:g=21:p=0.9:m=10,"
            "acompressor=threshold=0.125:ratio=3:attack=8:release=200:knee=4:makeup=1"
        )
    return ""


def decode(
    source: Path, decoded: Path, settings: dict[str, Any], duration: float,
    on_progress: Callable[[float], None],
) -> float | None:
    log = run_ffmpeg(
        [
            "-i", str(source), "-map", "0:a:0", "-vn",
            "-af", prefilter_chain(settings),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-c:a", "pcm_f32le", "-rf64", "auto", str(decoded),
        ],
        duration=duration,
        on_progress=on_progress,
    )
    matches = re.findall(r"Integrated loudness:\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", log)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    return value if math.isfinite(value) and value > -70 else None


def denoise(decoded: Path, enhanced: Path, atten_lim: float, on_progress: Callable[[float], None]) -> None:
    from dfnstream_py import DeepFilterNetStreaming

    block_size = 4_800
    with sf.SoundFile(str(decoded)) as reader, sf.SoundFile(
        str(enhanced), mode="w", samplerate=SAMPLE_RATE, channels=1, format="RF64", subtype="FLOAT"
    ) as writer:
        expected_frames = reader.frames
        written_frames = 0
        last_reported = -1.0
        processor = DeepFilterNetStreaming(log_level="error", atten_lim=atten_lim)
        try:
            while True:
                block = reader.read(block_size, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                cleaned = np.asarray(processor.process_chunk(block[:, 0]), dtype=np.float32).reshape(-1)
                remaining = expected_frames - written_frames
                if remaining > 0 and len(cleaned):
                    chunk = cleaned[:remaining]
                    writer.write(chunk)
                    written_frames += len(chunk)
                fraction = reader.tell() / max(1, expected_frames)
                if fraction - last_reported >= 0.01:
                    on_progress(fraction)
                    last_reported = fraction

            tail = np.asarray(processor.finalize(), dtype=np.float32).reshape(-1)
            remaining = expected_frames - written_frames
            if remaining > 0 and len(tail):
                chunk = tail[:remaining]
                writer.write(chunk)
                written_frames += len(chunk)
            if written_frames != expected_frames:
                raise RuntimeError("The speech model did not return a complete audio stream.")
        finally:
            processor.close()


def parse_loudnorm(log: str) -> dict[str, str]:
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", log, flags=re.DOTALL)
    if not blocks:
        raise RuntimeError("Loudness measurement did not return a result.")
    return json.loads(blocks[-1])


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_and_encode(
    enhanced: Path,
    output: Path,
    settings: dict[str, Any],
    duration: float,
    on_analyze: Callable[[float], None],
    on_encode: Callable[[float], None],
) -> tuple[float | None, float | None, str]:
    target = LOUDNESS_TARGETS[settings["loudnessTarget"]]
    leveler = leveling_chain(settings["leveling"])
    prefix = f"{leveler}," if leveler else ""
    base_loudnorm = f"loudnorm=I={target}:TP={TRUE_PEAK_CEILING}:LRA={LOUDNESS_RANGE}"

    measure_log = run_ffmpeg(
        ["-i", str(enhanced), "-af", f"{prefix}{base_loudnorm}:print_format=json", "-f", "null", "-"],
        duration=duration,
        on_progress=on_analyze,
    )
    measured = parse_loudnorm(measure_log)
    measured_i = as_float(measured.get("input_i"))

    output_rate = 44_100 if settings["outputFormat"] == "mp3" else SAMPLE_RATE
    codec_args = (
        ["-c:a", "libmp3lame", "-b:a", "128k"]
        if settings["outputFormat"] == "mp3"
        else ["-c:a", "pcm_s24le", "-rf64", "auto"]
    )

    if measured_i is None or measured_i < SILENCE_LUFS:
        # Boosting near-silence would only make the leftover noise loud.
        limiter = f"alimiter=limit={10 ** (TRUE_PEAK_CEILING / 20):.6f}:level=false"
        run_ffmpeg(
            ["-i", str(enhanced), "-af", f"{prefix}{limiter}", "-ar", str(output_rate), "-ac", "1", *codec_args, str(output)],
            duration=duration,
            on_progress=on_encode,
        )
        return None, None, "The recording is nearly silent, so the volume was not raised to avoid amplifying noise."

    second_pass = (
        f"{prefix}{base_loudnorm}"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true:print_format=json"
    )
    encode_log = run_ffmpeg(
        ["-i", str(enhanced), "-af", second_pass, "-ar", str(output_rate), "-ac", "1", *codec_args, str(output)],
        duration=duration,
        on_progress=on_encode,
    )
    result = parse_loudnorm(encode_log)
    output_i = as_float(result.get("output_i"))
    output_tp = as_float(result.get("output_tp"))
    gain = target - measured_i
    direction = "raised" if gain >= 0 else "lowered"
    if str(result.get("normalization_type", "")).lower() == "dynamic":
        note = (
            f"Volume {direction} to about {target:.0f} LUFS. Loud moments were gently held back "
            f"so nothing clips."
        )
    else:
        note = f"Volume {direction} by {abs(gain):.1f} dB to reach {target:.0f} LUFS with peaks kept safe."
    return output_i, output_tp, note


def process_job(job_id: str, source: Path, metadata: dict[str, Any], settings: dict[str, Any]) -> None:
    directory = job_path(job_id)
    decoded = directory / "decoded_48k.wav"
    enhanced = directory / "enhanced_48k.wav"
    output = directory / f"cleaned.{settings['outputFormat']}"
    duration = metadata["duration"]

    def stage(start: float, end: float) -> Callable[[float], None]:
        return lambda fraction: update_job(job_id, progress=round(start + (end - start) * fraction))

    try:
        update_job(job_id, status="decoding", progress=0, message="Reading the recording and removing rumble and hum.")
        input_lufs = decode(source, decoded, settings, duration, stage(0, 10))

        update_job(job_id, status="denoising", progress=10, message="Removing fan, air-conditioning and room noise.")
        denoise(decoded, enhanced, NOISE_REDUCTION_LEVELS[settings["noiseReduction"]], stage(10, 75))
        decoded.unlink(missing_ok=True)

        update_job(job_id, status="normalizing", progress=75, message="Evening out the volume and measuring loudness.")
        analyze = stage(75, 87)

        def on_encode(fraction: float) -> None:
            update_job(
                job_id, status="encoding", progress=round(87 + 13 * fraction),
                message=f"Raising the volume and preparing the {settings['outputFormat'].upper()} file.",
            )

        output_lufs, peak_dbtp, note = normalize_and_encode(enhanced, output, settings, duration, analyze, on_encode)
        update_job(
            job_id,
            status="completed",
            progress=100,
            message=note,
            inputLufs=round(input_lufs, 1) if input_lufs is not None else None,
            outputLufs=round(output_lufs, 1) if output_lufs is not None else None,
            outputPeakDbfs=round(peak_dbtp, 1) if peak_dbtp is not None else None,
        )
    except Exception:
        logger.exception("Audio processing failed for job %s", job_id)
        update_job(
            job_id,
            status="failed",
            message="Audio cleaning did not finish.",
            error="The audio engine could not process this file. Try exporting it again as WAV or MP3, or try a shorter section.",
        )
        output.unlink(missing_ok=True)
    finally:
        for temporary in (decoded, enhanced):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def engine_status() -> dict[str, Any]:
    available = ENGINE_ERROR is None
    return {
        "status": "ready" if available else "unavailable",
        "message": "The audio engine is ready." if available else ENGINE_ERROR,
        "noiseReduction": ENGINE_NAME if available else "Unavailable",
        "loudnessNormalization": "Volume leveling and EBU R128 loudness targeting with true-peak protection",
        "maxUploadBytes": MAX_UPLOAD_BYTES,
        "maxDurationSeconds": MAX_DURATION_SECONDS,
        "acceptedExtensions": ACCEPTED_EXTENSIONS,
    }


async def cleanup_loop() -> None:
    while True:
        _cleanup_expired()
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ENGINE_ERROR
    DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    DATA_DIR.chmod(0o700)
    for stale_dir in DATA_DIR.iterdir():
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir, ignore_errors=True)
    ready, result = await asyncio.to_thread(_safe_engine_check)
    ENGINE_ERROR = None if ready else result
    if not ready:
        logger.warning("Audio engine unavailable: %s", result)
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        PROCESSING_POOL.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Audio Cleaner Processing API", version="2.0.0", lifespan=lifespan)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "audioEngine": "ready" if ENGINE_ERROR is None else "unavailable"}


@app.get("/audio/engine")
async def get_engine() -> dict[str, Any]:
    _cleanup_expired()
    return engine_status()


def _choice(value: str, allowed: set[str] | dict[str, Any], label: str) -> str:
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"Choose a valid {label} setting.")
    return value


@app.post("/audio/jobs", status_code=202)
async def create_audio_job(
    file: UploadFile = File(...),
    output_format: str = Form(..., alias="outputFormat"),
    noise_reduction: str = Form("standard", alias="noiseReduction"),
    leveling: str = Form("gentle"),
    loudness_target: str = Form("standard", alias="loudnessTarget"),
    hum_removal: str = Form("off", alias="humRemoval"),
    rumble_filter: str = Form("true", alias="rumbleFilter"),
) -> JSONResponse:
    _cleanup_expired()
    if ENGINE_ERROR:
        raise HTTPException(status_code=503, detail=ENGINE_ERROR)
    settings = {
        "outputFormat": _choice(output_format, {"wav", "mp3"}, "output format"),
        "noiseReduction": _choice(noise_reduction, NOISE_REDUCTION_LEVELS, "noise reduction"),
        "leveling": _choice(leveling, LEVELING_MODES, "volume leveling"),
        "loudnessTarget": _choice(loudness_target, LOUDNESS_TARGETS, "loudness target"),
        "humRemoval": _choice(hum_removal, HUM_MODES, "hum removal"),
        "rumbleFilter": rumble_filter.strip().lower() not in {"false", "0", "off", "no"},
    }
    with JOBS_LOCK:
        active_jobs = sum(job["status"] not in {"completed", "failed"} for job in JOBS.values())
    if active_jobs >= MAX_ACTIVE_JOBS:
        raise HTTPException(status_code=429, detail="The audio queue is full. Wait for a job to finish and try again.")

    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="This file type is not supported. Use " + ", ".join(e.upper() for e in ACCEPTED_EXTENSIONS) + ".",
        )

    job_id = uuid.uuid4().hex
    directory = job_path(job_id)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    original = directory / f"original.{suffix}"
    size = 0
    try:
        with original.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"This file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                target.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The selected file is empty.")
        metadata = await asyncio.to_thread(inspect_audio, original)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        detail = str(exc) if isinstance(exc, ValueError) else "This audio file is unreadable or damaged."
        raise HTTPException(status_code=400, detail=detail) from exc
    finally:
        await file.close()

    display_name = Path(file.filename or f"recording.{suffix}").name[:180]
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Your recording is queued for cleaning.",
        "originalName": display_name,
        "originalFormat": suffix,
        "outputFormat": settings["outputFormat"],
        "settings": {**settings, "targetLufs": LOUDNESS_TARGETS[settings["loudnessTarget"]]},
        "durationSeconds": round(metadata["duration"], 2),
        "inputLufs": None,
        "outputLufs": None,
        "outputPeakDbfs": None,
        "error": None,
        "createdAt": _utc_now(),
        "_created_epoch": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    PROCESSING_POOL.submit(process_job, job_id, original, metadata, settings)
    return JSONResponse(status_code=202, content=public_job(job))


@app.get("/audio/jobs/{job_id}")
async def get_audio_job(job_id: str) -> dict[str, Any]:
    _cleanup_expired()
    job_path(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="This audio job was not found or has expired.")
        return public_job(job)


@app.delete("/audio/jobs/{job_id}", status_code=204)
async def delete_audio_job(job_id: str) -> None:
    job_directory = job_path(job_id)
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="This audio job was not found or has expired.")
        JOBS.pop(job_id, None)
    shutil.rmtree(job_directory, ignore_errors=True)


def media_response(job_id: str, filename: str, *, attachment: bool = False) -> FileResponse:
    job_directory = job_path(job_id)
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="This audio job was not found or has expired.")
        job = JOBS[job_id]
        if filename == "cleaned":
            if job["status"] != "completed":
                raise HTTPException(status_code=409, detail="The cleaned audio is not ready yet.")
            extension = job["outputFormat"]
            path = job_directory / f"cleaned.{extension}"
            name = f"{Path(job['originalName']).stem}-cleaned.{extension}"
        else:
            extension = job["originalFormat"]
            path = job_directory / f"original.{extension}"
            name = job["originalName"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This audio file has expired.")
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(extension, "application/octet-stream"),
        filename=name if attachment else None,
        content_disposition_type="attachment" if attachment else "inline",
    )


@app.get("/audio/jobs/{job_id}/original")
async def get_original(job_id: str) -> FileResponse:
    return media_response(job_id, "original")


@app.get("/audio/jobs/{job_id}/cleaned")
async def get_cleaned(job_id: str) -> FileResponse:
    return media_response(job_id, "cleaned")


@app.get("/audio/jobs/{job_id}/cleaned/download")
async def download_cleaned(job_id: str) -> FileResponse:
    return media_response(job_id, "cleaned", attachment=True)
