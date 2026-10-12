from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from scipy.signal import resample_poly


SERVICE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("AUDIO_TEMP_DIR", "/tmp/audio-cleaner-jobs"))
MAX_UPLOAD_BYTES = max(1, int(os.environ.get("AUDIO_MAX_UPLOAD_MB", "100"))) * 1024 * 1024
MAX_DURATION_SECONDS = max(30, int(os.environ.get("AUDIO_MAX_DURATION_SECONDS", "600")))
MAX_ACTIVE_JOBS = max(1, int(os.environ.get("AUDIO_MAX_ACTIVE_JOBS", "4")))
RETENTION_SECONDS = max(60, int(os.environ.get("AUDIO_JOB_RETENTION_SECONDS", "1800")))
JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
TARGET_LUFS = -16.0
TRUE_PEAK_CEILING = -1.0
SAMPLE_RATE = 48_000

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
PROCESSING_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-cleaner")
ENGINE_ERROR: str | None = None
ENGINE_NAME = "DeepFilterNet streaming speech enhancement"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("audio-cleaner")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_engine_check() -> tuple[bool, str]:
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


def run_command(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio conversion took too long. Try a shorter recording.") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or "").strip().splitlines()
        detail = message[-1] if message else "Audio conversion failed."
        raise RuntimeError(detail[:240]) from exc


def inspect_audio(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        timeout=20,
    )
    try:
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        audio_format = metadata.get("format", {})
        duration = float(audio_format["duration"])
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec = str(stream["codec_name"]).lower()
        format_names = str(audio_format.get("format_name", "")).lower()
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("The file does not contain readable WAV or MP3 audio.") from exc

    is_wav = codec.startswith("pcm_") and "wav" in format_names
    is_mp3 = codec == "mp3" and ("mp3" in format_names or "mp2" in format_names)
    if not (is_wav or is_mp3):
        raise ValueError("Only WAV and MP3 recordings are supported.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The recording duration could not be read. The audio file may be damaged.")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"This recording is {int(duration // 60)} minutes long. The current limit is "
            f"{MAX_DURATION_SECONDS // 60} minutes."
        )
    if sample_rate < 8_000 or sample_rate > 192_000 or channels < 1 or channels > 2:
        raise ValueError("This recording's sample rate or channel count is not supported. Use mono or stereo audio.")
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "codec": codec,
        "format": "mp3" if is_mp3 else "wav",
    }


def decode_for_model(source: Path, decoded: Path, channels: int) -> None:
    run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            str(decoded),
        ],
        timeout=MAX_DURATION_SECONDS + 120,
    )


def measure_lufs(path: Path) -> float | None:
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-7:
            return None
        measurement = float(pyln.Meter(sample_rate).integrated_loudness(audio))
        return measurement if math.isfinite(measurement) else None
    except (ValueError, RuntimeError, sf.SoundFileError):
        return None


def true_peak_linear(audio: np.ndarray, block_frames: int = 240_000) -> float:
    peak = 0.0
    frame_count = len(audio)
    overlap = 32
    for start in range(0, frame_count, block_frames):
        end = min(frame_count, start + block_frames)
        left = max(0, start - overlap)
        right = min(frame_count, end + overlap)
        oversampled = resample_poly(
            audio[left:right].astype(np.float64),
            4,
            1,
            axis=0,
        )
        core_start = (start - left) * 4
        core_end = (end - left) * 4
        if core_end > core_start:
            peak = max(peak, float(np.max(np.abs(oversampled[core_start:core_end]))))
    return peak


def denoise(decoded: Path, enhanced: Path) -> None:
    from dfnstream_py import DeepFilterNetStreaming

    block_size = 4_800
    with sf.SoundFile(decoded) as reader, sf.SoundFile(
        str(enhanced), mode="w", samplerate=SAMPLE_RATE, channels=reader.channels, format="WAV", subtype="FLOAT"
    ) as writer:
        expected_frames = reader.frames
        written_frames = 0
        processors = [
            DeepFilterNetStreaming(log_level="error", atten_lim=15.0)
            for _ in range(reader.channels)
        ]
        try:
            while True:
                block = reader.read(block_size, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                cleaned_channels = [
                    np.asarray(processors[channel].process_chunk(block[:, channel]), dtype=np.float32).reshape(-1)
                    for channel in range(reader.channels)
                ]
                lengths = {len(channel) for channel in cleaned_channels}
                if len(lengths) != 1:
                    raise RuntimeError("The speech model returned inconsistent channel lengths.")
                remaining = expected_frames - written_frames
                if remaining > 0:
                    output_block = np.stack(cleaned_channels, axis=1)[:remaining]
                    writer.write(output_block)
                    written_frames += len(output_block)

            tails = [
                np.asarray(processor.finalize(), dtype=np.float32).reshape(-1)
                for processor in processors
            ]
            if tails and any(len(tail) for tail in tails):
                if len({len(tail) for tail in tails}) != 1:
                    raise RuntimeError("The speech model returned inconsistent final audio frames.")
                remaining = expected_frames - written_frames
                if remaining > 0:
                    output_tail = np.stack(tails, axis=1)[:remaining]
                    writer.write(output_tail)
                    written_frames += len(output_tail)
            if written_frames != expected_frames:
                raise RuntimeError("The speech model did not return a complete audio stream.")
        finally:
            for processor in processors:
                processor.close()


def normalize_and_encode(
    enhanced: Path,
    output: Path,
    analysis: Path,
    output_format: str,
    source_sample_rate: int,
    channels: int,
    on_encoding: Callable[[], None],
) -> tuple[float | None, float | None, str]:
    audio, sample_rate = sf.read(str(enhanced), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise RuntimeError("The speech model produced an empty recording.")

    current_lufs: float | None = None
    current_peak = float(np.max(np.abs(audio)))
    current_true_peak = true_peak_linear(audio)
    if current_peak >= 1e-7:
        try:
            measured = float(pyln.Meter(sample_rate).integrated_loudness(audio))
            current_lufs = measured if math.isfinite(measured) else None
        except (ValueError, RuntimeError):
            current_lufs = None

    explanation = "Normalized toward -16 LUFS with a -1 dBFS peak ceiling."
    gain_db = 0.0
    if current_lufs is None or current_peak < 1e-7:
        explanation = "The recording is nearly silent, so it was left unamplified to avoid raising background noise."
    else:
        requested_gain = TARGET_LUFS - current_lufs
        true_peak_dbtp = 20.0 * math.log10(max(current_true_peak, 1e-12))
        safe_peak_gain = TRUE_PEAK_CEILING - true_peak_dbtp
        gain_db = min(requested_gain, safe_peak_gain, 24.0)
        if gain_db < requested_gain - 0.05:
            explanation = "Peak protection or the 24 dB gain limit prevented reaching -16 LUFS safely."

    del audio
    on_encoding()
    filter_chain = (
        f"volume={gain_db:.4f}dB,"
        f"alimiter=limit={10 ** (TRUE_PEAK_CEILING / 20):.6f}:level=false:latency=true"
    )
    codec_args = (
        ["-c:a", "libmp3lame", "-q:a", "2"]
        if output_format == "mp3"
        else ["-c:a", "pcm_s24le"]
    )
    encode_args = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(enhanced),
            "-af",
            filter_chain,
            "-ar",
            str(source_sample_rate),
            "-ac",
            str(channels),
            *codec_args,
            str(output),
        ]
    peak_is_safe = False
    for _ in range(5):
        run_command(encode_args, timeout=MAX_DURATION_SECONDS + 120)
        run_command(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(output),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                str(channels),
                "-c:a",
                "pcm_f32le",
                str(analysis),
            ],
            timeout=MAX_DURATION_SECONDS + 120,
        )
        final_audio, _ = sf.read(str(analysis), dtype="float32", always_2d=True)
        measured_true_peak = true_peak_linear(final_audio)
        true_peak_dbtp = 20.0 * math.log10(max(measured_true_peak, 1e-12))
        if true_peak_dbtp <= TRUE_PEAK_CEILING:
            peak_is_safe = True
            break
        gain_db += TRUE_PEAK_CEILING - true_peak_dbtp - 0.1
        filter_chain = (
            f"volume={gain_db:.4f}dB,"
            f"alimiter=limit={10 ** (TRUE_PEAK_CEILING / 20):.6f}:level=false:latency=true"
        )
        encode_args[encode_args.index("-af") + 1] = filter_chain
    if not peak_is_safe:
        raise RuntimeError("The encoded audio could not be kept below the safe peak ceiling.")
    run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(output),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            str(analysis),
        ],
        timeout=MAX_DURATION_SECONDS + 120,
    )

    output_lufs = measure_lufs(analysis)
    final_audio, _ = sf.read(str(analysis), dtype="float32", always_2d=True)
    final_peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
    peak_dbfs = 20.0 * math.log10(max(final_peak, 1e-12)) if final_peak else None
    return output_lufs, peak_dbfs, explanation


def process_job(job_id: str, source: Path, metadata: dict[str, Any], output_format: str) -> None:
    directory = job_path(job_id)
    decoded = directory / "decoded_48k.wav"
    enhanced = directory / "enhanced_48k.wav"
    analysis = directory / "output_analysis.wav"
    output = directory / f"cleaned.{output_format}"
    try:
        update_job(job_id, status="decoding", message="Checking and decoding the recording.")
        decode_for_model(source, decoded, metadata["channels"])
        input_lufs = measure_lufs(decoded)
        update_job(job_id, status="denoising", message="Reducing steady background noise with DeepFilterNet.")
        denoise(decoded, enhanced)
        update_job(job_id, status="normalizing", message="Measuring loudness and protecting peaks.")
        update_job(job_id, status="encoding", message=f"Preparing the {output_format.upper()} download.")
        output_lufs, peak_dbfs, note = normalize_and_encode(
            enhanced,
            output,
            analysis,
            output_format,
            metadata["sample_rate"],
            metadata["channels"],
            lambda: update_job(
                job_id,
                status="encoding",
                message=f"Preparing the {output_format.upper()} download.",
            ),
        )
        update_job(
            job_id,
            status="completed",
            message=note,
            inputLufs=round(input_lufs, 1) if input_lufs is not None else None,
            outputLufs=round(output_lufs, 1) if output_lufs is not None else None,
            outputPeakDbfs=round(peak_dbfs, 1) if peak_dbfs is not None else None,
        )
    except Exception:
        logger.exception("Audio processing failed for job %s", job_id)
        update_job(
            job_id,
            status="failed",
            message="Audio cleaning did not finish.",
            error="The audio engine could not process this file. Try a shorter WAV or MP3 recording, or check that DeepFilterNet is available.",
        )
        for partial in (decoded, enhanced, analysis, output):
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        for temporary in (decoded, enhanced, analysis):
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
        "loudnessNormalization": "pyloudnorm, target -16 LUFS, with FFmpeg peak limiting",
        "maxUploadBytes": MAX_UPLOAD_BYTES,
        "maxDurationSeconds": MAX_DURATION_SECONDS,
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
        logger.warning("DeepFilterNet unavailable: %s", result)
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        PROCESSING_POOL.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Audio Cleaner Processing API", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "audioEngine": "ready" if ENGINE_ERROR is None else "unavailable"}


@app.get("/audio/engine")
async def get_engine() -> dict[str, Any]:
    _cleanup_expired()
    return engine_status()


@app.post("/audio/jobs", status_code=202)
async def create_audio_job(
    file: UploadFile = File(...),
    output_format: str = Form(..., alias="outputFormat"),
) -> JSONResponse:
    _cleanup_expired()
    if ENGINE_ERROR:
        raise HTTPException(status_code=503, detail=ENGINE_ERROR)
    if output_format not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="Choose WAV or MP3 for the cleaned audio.")
    with JOBS_LOCK:
        active_jobs = sum(job["status"] not in {"completed", "failed"} for job in JOBS.values())
    if active_jobs >= MAX_ACTIVE_JOBS:
        raise HTTPException(status_code=429, detail="The audio queue is full. Wait for a job to finish and try again.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise HTTPException(status_code=400, detail="Choose a WAV or MP3 audio file.")

    job_id = uuid.uuid4().hex
    directory = job_path(job_id)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    original = directory / f"original{suffix}"
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

    actual_original = directory / f"original.{metadata['format']}"
    if actual_original != original:
        original.rename(actual_original)
        original = actual_original

    display_name = Path(file.filename or f"recording{suffix}").name[:180]
    job = {
        "id": job_id,
        "status": "queued",
        "message": "Your recording is queued for cleaning.",
        "originalName": display_name,
        "originalFormat": metadata["format"],
        "outputFormat": output_format,
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
    PROCESSING_POOL.submit(process_job, job_id, original, metadata, output_format)
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
            path = job_directory / f"cleaned.{job['outputFormat']}"
            name = f"{Path(job['originalName']).stem}-cleaned.{job['outputFormat']}"
        else:
            path = job_directory / f"original.{job['originalFormat']}"
            name = job["originalName"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This audio file has expired.")
    media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(
        path,
        media_type=media_type,
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
