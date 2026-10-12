import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, RefObject } from 'react';
import {
  AudioJobInputOutputFormat,
  AudioJobStatus,
  getGetAudioJobQueryKey,
  useCreateAudioJob,
  useDeleteAudioJob,
  useGetAudioEngine,
  useGetAudioJob,
} from '@workspace/api-client-react';
import type { AudioJob } from '@workspace/api-client-react';
import {
  AudioLines,
  Check,
  ChevronDown,
  CircleAlert,
  Download,
  FileAudio2,
  Headphones,
  LoaderCircle,
  LockKeyhole,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';

type OutputFormat = 'wav' | 'mp3';
type WorkFile = { file: File; duration: number };

const terminalStatuses: AudioJobStatus[] = [AudioJobStatus.completed, AudioJobStatus.failed];
const acceptedExtensions = ['wav', 'mp3'];

function readableBytes(bytes: number) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

function timeLabel(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

function errorMessage(error: unknown, fallback: string) {
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object') {
    const record = error as Record<string, unknown>;
    const response = record.response as { data?: unknown } | undefined;
    const data = response?.data;
    if (typeof data === 'string' && data.trim()) return data;
    if (data && typeof data === 'object') {
      const body = data as Record<string, unknown>;
      for (const key of ['message', 'error', 'detail']) {
        if (typeof body[key] === 'string' && body[key]) return body[key] as string;
      }
    }
    if (typeof record.message === 'string' && record.message) return record.message;
  }
  return fallback;
}

function inspectDuration(file: File) {
  return new Promise<number>((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => {
      const duration = audio.duration;
      URL.revokeObjectURL(url);
      if (Number.isFinite(duration) && duration > 0) resolve(duration);
      else reject(new Error('The recording duration could not be read. Try exporting it as WAV or MP3.'));
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('This audio file could not be opened in your browser. Try WAV or MP3.'));
    };
    audio.src = url;
  });
}

function prettyStatus(status: string) {
  const labels: Record<string, string> = {
    queued: 'In the queue',
    decoding: 'Reading your recording',
    denoising: 'Removing background noise',
    normalizing: 'Balancing the listening level',
    encoding: 'Preparing your clean file',
    completed: 'Your recording is ready',
    failed: 'Cleaning could not finish',
  };
  return labels[status] ?? 'Working on your recording';
}

export default function AudioCleaner() {
  const queryClient = useQueryClient();
  const { data: engine, isLoading: engineLoading, isError: engineError, error: engineFailure, refetch: retryEngine } = useGetAudioEngine();
  const [file, setFile] = useState<WorkFile | null>(null);
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('wav');
  const [jobId, setJobId] = useState('');
  const [localError, setLocalError] = useState('');
  const [checkingFile, setCheckingFile] = useState(false);
  const [activeCompare, setActiveCompare] = useState<'original' | 'cleaned'>('cleaned');
  const [dragging, setDragging] = useState(false);
  const originalAudioRef = useRef<HTMLAudioElement>(null);
  const cleanedAudioRef = useRef<HTMLAudioElement>(null);
  const pickerRef = useRef<HTMLInputElement>(null);
  const createJob = useCreateAudioJob();
  const deleteJob = useDeleteAudioJob();
  const jobQuery = useGetAudioJob(jobId, {
    query: {
      queryKey: getGetAudioJobQueryKey(jobId),
      enabled: Boolean(jobId),
      refetchInterval: (query) => {
        const current = query.state.data as AudioJob | undefined;
        return current && terminalStatuses.includes(current.status) ? false : 1800;
      },
    },
  });
  const job = jobQuery.data;
  const localUrl = useMemo(() => (file ? URL.createObjectURL(file.file) : ''), [file]);

  useEffect(() => () => {
    if (localUrl) URL.revokeObjectURL(localUrl);
  }, [localUrl]);

  useEffect(() => {
    if (!job || job.status !== AudioJobStatus.completed) return;
    const original = originalAudioRef.current;
    const cleaned = cleanedAudioRef.current;
    if (original) original.pause();
    if (cleaned) cleaned.pause();
  }, [job?.status]);

  const engineReady = engine?.status === 'ready';
  const engineUnavailable = !engineLoading && (engineError || engine?.status === 'unavailable');
  const busy = createJob.isPending || checkingFile;
  const processing = Boolean(job && !terminalStatuses.includes(job.status));
  const canDownload = job?.status === AudioJobStatus.completed;
  const originalUrl = jobId ? `/api/audio/jobs/${jobId}/original` : localUrl;
  const cleanedUrl = jobId ? `/api/audio/jobs/${jobId}/cleaned` : '';
  const cleanedDownloadUrl = jobId ? `${cleanedUrl}/download` : '';
  const maxSize = engine?.maxUploadBytes ?? 0;
  const maxDuration = engine?.maxDurationSeconds ?? 0;

  async function setSelectedFile(candidate: File) {
    setLocalError('');
    setFile(null);
    if (!candidate.size) {
      setLocalError('That file is empty. Choose an audio recording with sound.');
      return;
    }
    const extension = candidate.name.split('.').pop()?.toLowerCase() ?? '';
    if (!acceptedExtensions.includes(extension)) {
      setLocalError('Choose a WAV or MP3 recording. Other audio formats are not supported yet.');
      return;
    }
    if (maxSize > 0 && candidate.size > maxSize) {
      setLocalError(`This file is too large. Choose a recording under ${readableBytes(maxSize)}.`);
      return;
    }
    setCheckingFile(true);
    try {
      const duration = await inspectDuration(candidate);
      if (maxDuration > 0 && duration > maxDuration) {
        setLocalError(`This recording is ${timeLabel(duration)} long. The maximum is ${timeLabel(maxDuration)}.`);
        return;
      }
      setFile({ file: candidate, duration });
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Unable to read this audio file.');
    } finally {
      setCheckingFile(false);
    }
  }

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const candidate = event.target.files?.[0];
    if (candidate) await setSelectedFile(candidate);
    event.target.value = '';
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const candidate = event.dataTransfer.files[0];
    if (candidate) void setSelectedFile(candidate);
  }

  function submitJob() {
    if (!file || !engineReady || busy || jobId) return;
    if (engine.maxUploadBytes > 0 && file.file.size > engine.maxUploadBytes) {
      setLocalError(`This file is too large. Choose a recording under ${readableBytes(engine.maxUploadBytes)}.`);
      return;
    }
    if (engine.maxDurationSeconds > 0 && file.duration > engine.maxDurationSeconds) {
      setLocalError(`This recording is ${timeLabel(file.duration)} long. The maximum is ${timeLabel(engine.maxDurationSeconds)}.`);
      return;
    }
    setLocalError('');
    createJob.mutate(
      { data: { file: file.file, outputFormat: outputFormat === 'wav' ? AudioJobInputOutputFormat.wav : AudioJobInputOutputFormat.mp3 } },
      {
        onSuccess: (created) => {
          setJobId(created.id);
          queryClient.setQueryData(getGetAudioJobQueryKey(created.id), created);
        },
        onError: (error) => setLocalError(errorMessage(error, 'The upload could not be started. Check your connection and try again.')),
      },
    );
  }

  function playSide(side: 'original' | 'cleaned') {
    const target = side === 'original' ? originalAudioRef.current : cleanedAudioRef.current;
    const other = side === 'original' ? cleanedAudioRef.current : originalAudioRef.current;
    if (!target || !target.src) return;
    if (other && !other.paused) other.pause();
    setActiveCompare(side);
    void target.play().catch(() => setLocalError('Playback was blocked by your browser. Press play on the audio player to start.'));
  }

  async function resetSession() {
    if (jobId) {
      try {
        await deleteJob.mutateAsync({ jobId });
        queryClient.removeQueries({ queryKey: getGetAudioJobQueryKey(jobId) });
      } catch (error) {
        setLocalError(errorMessage(error, 'The temporary recording could not be removed. Please try again.'));
        return;
      }
    }
    originalAudioRef.current?.pause();
    cleanedAudioRef.current?.pause();
    setJobId('');
    setFile(null);
    setLocalError('');
    setActiveCompare('cleaned');
  }

  const statusText = job ? (job.status === AudioJobStatus.failed ? (job.error || job.message || prettyStatus(job.status)) : job.message || prettyStatus(job.status)) : '';
  const downloadName = job ? `${job.originalName.replace(/\.[^.]+$/, '')}-cleaned.${job.outputFormat}` : 'cleaned-recording';

  return (
    <main className="studio-shell min-h-[100dvh]">
      <header className="topbar">
        <a className="brand" href="/" aria-label="ClearTake home">
          <span className="brand-mark"><AudioLines size={19} strokeWidth={2.2} /></span>
          <span>cleartake<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-right">
          <span className="header-caption">A quieter way to sound clear</span>
          <span className={`engine-chip ${engineLoading ? 'is-loading' : engineReady ? 'is-ready' : 'is-offline'}`} data-testid="status-engine">
            <span className="chip-dot" />
            {engineLoading ? 'Checking studio' : engineReady ? 'Studio ready' : 'Studio unavailable'}
          </span>
        </div>
      </header>

      <div className="workspace">
        <section className="intro">
          <div className="eyebrow"><span className="eyebrow-line" /> VOICE RECORDING CLEANUP</div>
          <h1>Make every<br className="mobile-break" /> word <em>clear.</em></h1>
          <p className="intro-copy">A little room noise happens. Drop in your recording and we’ll take care of the rest.</p>
        </section>

        <div className="workbench">
          <section className="main-panel" aria-label="Audio cleanup workspace">
            {!jobId ? (
              <>
                <div className="panel-heading">
                  <div>
                    <p className="step-label">01 <span>·</span> ADD YOUR RECORDING</p>
                    <h2>Start with a voice file</h2>
                  </div>
                  <span className="step-complete"><span>1</span> of 1</span>
                </div>

                <div
                  className={`dropzone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''} ${engineUnavailable ? 'is-disabled' : ''}`}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={onDrop}
                  onClick={() => !engineUnavailable && !busy && pickerRef.current?.click()}
                  role="group"
                  aria-label="Choose an audio recording or drop one here"
                  data-testid="dropzone-audio"
                >
                  <input
                    ref={pickerRef}
                    className="visually-hidden"
                    type="file"
                    accept="audio/wav,audio/x-wav,audio/mpeg,.wav,.mp3"
                    onChange={onFileChange}
                    disabled={engineUnavailable || busy}
                    data-testid="input-audio-file"
                    aria-label="Choose audio file"
                  />
                  {checkingFile ? (
                    <div className="dropzone-inner">
                      <span className="upload-icon checking"><LoaderCircle size={23} /></span>
                      <strong>Checking your recording…</strong>
                      <span className="drop-hint">Reading file details in your browser</span>
                    </div>
                  ) : file ? (
                    <div className="selected-file">
                      <span className="upload-icon"><FileAudio2 size={23} /></span>
                      <div className="selected-details">
                        <strong title={file.file.name}>{file.file.name}</strong>
                        <span>{file.file.size > 0 ? readableBytes(file.file.size) : 'Audio'} <i /> {timeLabel(file.duration)}</span>
                      </div>
                      <button
                        type="button"
                        className="icon-button remove-file"
                        aria-label="Remove selected recording"
                        data-testid="button-remove-file"
                        onClick={(event) => { event.stopPropagation(); setFile(null); setLocalError(''); }}
                      ><X size={17} /></button>
                    </div>
                  ) : (
                    <div className="dropzone-inner">
                      <span className="upload-icon"><Upload size={22} /></span>
                      <strong>Drop your recording here</strong>
                      <span className="drop-hint">or <button className="browse-trigger" type="button" onClick={(event) => { event.stopPropagation(); pickerRef.current?.click(); }}>browse files</button> on your device</span>
                      <span className="supported-note">WAV or MP3 · up to {readableBytes(maxSize)} · {timeLabel(maxDuration)} maximum</span>
                    </div>
                  )}
                </div>

                <div className="form-row">
                  <div className="format-label">
                    <span className="format-icon"><SlidersHorizontal size={15} /></span>
                    <div><strong>Output file</strong><small>Choose the format for your clean recording</small></div>
                  </div>
                  <label className="select-wrap">
                    <span className="sr-only">Output format</span>
                    <select value={outputFormat} onChange={(event) => setOutputFormat(event.target.value as OutputFormat)} disabled={Boolean(jobId)}>
                      <option value="wav">WAV · best quality</option>
                      <option value="mp3">MP3 · smaller file</option>
                    </select>
                    <ChevronDown className="select-chevron" size={15} />
                  </label>
                </div>

                {localError && <div className="notice notice-error" role="alert" data-testid="text-error"><CircleAlert size={17} /><span>{localError}</span><button aria-label="Dismiss message" data-testid="button-dismiss-error" onClick={() => setLocalError('')}><X size={14} /></button></div>}
                {engineUnavailable && (
                  <div className="notice notice-warning" role="alert" data-testid="text-engine-unavailable">
                    <CircleAlert size={18} />
                    <span><strong>Cleaning is temporarily unavailable.</strong> {engine?.message || errorMessage(engineFailure, 'The processing studio could not be reached. Try again shortly.')}</span>
                    <button className="retry-inline" onClick={() => void retryEngine()} aria-label="Retry engine connection"><RefreshCw size={15} /></button>
                  </div>
                )}
                {engineLoading && (
                  <div className="engine-loading" role="status"><span className="skeleton-dot" /> Checking that the studio is ready…</div>
                )}

                <button
                  type="button"
                  className="primary-button"
                  onClick={submitJob}
                  disabled={!file || !engineReady || busy || createJob.isPending}
                  data-testid="button-clean-audio"
                >
                  {createJob.isPending ? <><LoaderCircle className="spin" size={17} /> Sending your recording…</> : <><Sparkles size={17} /> Clean my recording <span className="button-arrow">→</span></>}
                </button>
                <div className="privacy-note"><LockKeyhole size={13} /> Your recording is only kept while you work. Reset to remove it.</div>
              </>
            ) : (
              <div className="result-view">
                <div className="panel-heading">
                  <div>
                    <p className="step-label">01 <span>·</span> YOUR RECORDING</p>
                    <h2>{canDownload ? 'Hear the difference' : job?.status === AudioJobStatus.failed ? 'Let’s get you back on track' : 'Cleaning in progress'}</h2>
                  </div>
                  {canDownload ? <span className="result-badge"><Check size={13} /> READY</span> : <span className={`status-orb ${job?.status === AudioJobStatus.failed ? 'failed' : ''}`}>{job?.status === AudioJobStatus.failed ? <CircleAlert size={14} /> : <LoaderCircle className="spin" size={14} />}</span>}
                </div>

                {jobQuery.isLoading && <div className="status-loading" role="status"><span className="skeleton-bar" /><span className="skeleton-bar short" /><span className="sr-only">Loading recording status</span></div>}
                {jobQuery.isError && (
                  <div className="notice notice-error" role="alert">
                    <CircleAlert size={17} /><span>{errorMessage(jobQuery.error, 'Could not refresh the processing status.')}</span>
                    <button className="retry-inline" aria-label="Retry status" onClick={() => void jobQuery.refetch()}><RefreshCw size={15} /></button>
                  </div>
                )}

                {job && (
                  <>
                    <div className={`progress-message ${job.status === AudioJobStatus.failed ? 'is-failed' : canDownload ? 'is-done' : ''}`} role="status" aria-live="polite" data-testid="status-job">
                      <span className="progress-mark">{canDownload ? <Check size={15} /> : job.status === AudioJobStatus.failed ? <CircleAlert size={15} /> : <LoaderCircle className="spin" size={15} />}</span>
                      <span><strong>{prettyStatus(job.status)}</strong><small>{statusText}</small></span>
                    </div>

                    {job.status === AudioJobStatus.failed ? (
                      <div className="failed-help">
                        <p>The original is still here. You can try a different format or choose another recording.</p>
                        <button className="secondary-button" onClick={() => void resetSession()} disabled={deleteJob.isPending}>
                          {deleteJob.isPending ? <LoaderCircle className="spin" size={15} /> : <RotateCcw size={15} />} Start over
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="audio-compare" aria-label="Original and cleaned audio players">
                          <AudioCard
                            title="Original"
                            subtitle="Before cleanup"
                            audioRef={originalAudioRef}
                            src={originalUrl}
                            duration={job.durationSeconds ?? file?.duration ?? null}
                            format={job.originalFormat}
                            active={activeCompare === 'original'}
                            onChoose={() => playSide('original')}
                            disabled={!jobId}
                          />
                          <div className="compare-divider"><span>A / B</span></div>
                          <AudioCard
                            title="Cleaned"
                            subtitle={canDownload ? 'Noise reduced · normalized' : 'Ready when processing finishes'}
                            audioRef={cleanedAudioRef}
                            src={canDownload ? cleanedUrl : ''}
                            duration={job.durationSeconds ?? file?.duration ?? null}
                            format={job.outputFormat}
                            active={activeCompare === 'cleaned'}
                            onChoose={() => playSide('cleaned')}
                            disabled={!canDownload}
                          />
                        </div>
                        {canDownload && (
                          <div className="result-stats" aria-label="Audio levels">
                            <div><span>BEFORE</span><strong>{job.inputLufs == null ? '—' : `${job.inputLufs.toFixed(1)} LUFS`}</strong></div>
                            <div><span>AFTER</span><strong>{job.outputLufs == null ? '—' : `${job.outputLufs.toFixed(1)} LUFS`} <small>/ −16 LUFS target</small></strong></div>
                            <div><span>PEAK LEVEL</span><strong>{job.outputPeakDbfs == null ? '—' : `${job.outputPeakDbfs.toFixed(1)} dBFS`}</strong></div>
                          </div>
                        )}
                        <div className="result-actions">
                          <a
                            className={`primary-button download-button ${canDownload ? '' : 'is-disabled'}`}
                            href={canDownload ? cleanedDownloadUrl : undefined}
                            download={canDownload ? downloadName : undefined}
                            aria-disabled={!canDownload}
                            onClick={(event) => { if (!canDownload) event.preventDefault(); }}
                            data-testid="link-download-cleaned"
                          ><Download size={17} /> Download clean {job.outputFormat.toUpperCase()}</a>
                          <button type="button" className="text-button" onClick={() => void resetSession()} disabled={deleteJob.isPending}>
                            {deleteJob.isPending ? <LoaderCircle className="spin" size={14} /> : <RotateCcw size={14} />} New recording
                          </button>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            )}
          </section>

          <aside className="side-panel">
            <div className="side-top">
              <div className="side-icon"><Headphones size={19} /></div>
              <span className="side-kicker">MADE FOR THE HUMAN VOICE</span>
              <h3>Less noise.<br />More you.</h3>
              <p>No knobs to learn. ClearTake gently cleans your recording and brings it to a comfortable listening level.</p>
            </div>
            <div className="side-divider" />
            <div className="process-list">
              <div className="process-item">
                <span className="process-symbol"><AudioLines size={16} /></span>
                <div><strong>Noise reduction</strong><span>{engine?.noiseReduction || 'Background noise, softened'}</span></div>
              </div>
              <div className="process-item">
                <span className="process-symbol"><SlidersHorizontal size={16} /></span>
                <div><strong>Even listening level</strong><span>{engine?.loudnessNormalization || 'Balanced to −16 LUFS'}</span></div>
              </div>
            </div>
            <div className="side-note">
              <ShieldCheck size={15} />
              <span>Private by design<span>Files are temporary and removed when you start over.</span></span>
            </div>
            {engineReady && <p className="limits-note">UP TO {readableBytes(engine.maxUploadBytes)} <i /> {timeLabel(engine.maxDurationSeconds)} MAX</p>}
          </aside>
        </div>

        <footer className="footer">
          <span>ClearTake <span className="footer-dot">·</span> A small studio for clearer words</span>
          <span>Made for recordings that sound like you.</span>
        </footer>
      </div>
    </main>
  );
}

function AudioCard({
  title,
  subtitle,
  audioRef,
  src,
  duration,
  format,
  active,
  onChoose,
  disabled,
}: {
  title: string;
  subtitle: string;
  audioRef: RefObject<HTMLAudioElement | null>;
  src: string;
  duration: number | null;
  format: string;
  active: boolean;
  onChoose: () => void;
  disabled: boolean;
}) {
  return (
    <div className={`audio-card ${active ? 'is-active' : ''} ${disabled ? 'is-pending' : ''}`} data-testid={`card-audio-${title.toLowerCase()}`}>
      <div className="audio-card-top">
        <span className="audio-card-icon"><AudioLines size={16} /></span>
        <div className="audio-card-copy"><strong>{title}</strong><span>{subtitle}</span></div>
        <span className="audio-format">{format.toUpperCase()}</span>
      </div>
      <div className="waveform" aria-hidden="true">
        {Array.from({ length: 40 }, (_, index) => {
          const height = [14, 25, 19, 34, 22, 40, 29, 17, 32, 23, 38, 19, 28, 40, 23, 31, 15, 26, 38, 21, 33, 18, 29, 40, 25, 16, 34, 22, 30, 40, 19, 28, 15, 35, 23, 31, 20, 39, 25, 17][index];
          return <span key={index} style={{ height }} className={active ? 'wave-active' : ''} />;
        })}
      </div>
      <audio ref={audioRef} src={src || undefined} controls preload="metadata" aria-label={`${title} recording`} />
      <div className="audio-card-bottom">
        <button type="button" className="listen-button" onClick={onChoose} disabled={disabled} data-testid={`button-play-${title.toLowerCase()}`}>
          {disabled ? <><span className="listen-dot muted" /> Not ready yet</> : <><span className={`listen-dot ${active ? 'playing' : ''}`} /> Listen to {title.toLowerCase()}</>}
        </button>
        <span>{timeLabel(duration)}</span>
      </div>
    </div>
  );
}
