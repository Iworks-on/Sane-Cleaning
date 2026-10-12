import { spawn, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { logger } from "./logger";

const AUDIO_SERVICE_PORT = Number(process.env.AUDIO_SERVICE_PORT ?? 8099);
const AUDIO_SERVICE_DIR = fileURLToPath(new URL("../audio_service/", import.meta.url));

export type AudioServiceProcess = {
  child: ChildProcess;
  ready: Promise<boolean>;
};

export function startAudioService(): AudioServiceProcess {
  const child = spawn(
    process.env.AUDIO_PYTHON_BIN ?? "python3",
    [
      "-m",
      "uvicorn",
      "main:app",
      "--app-dir",
      AUDIO_SERVICE_DIR,
      "--host",
      "127.0.0.1",
      "--port",
      String(AUDIO_SERVICE_PORT),
      "--log-level",
      "warning",
    ],
    {
      cwd: AUDIO_SERVICE_DIR,
      env: process.env,
      stdio: ["ignore", "inherit", "inherit"],
    },
  );

  child.on("error", (error) => {
    logger.error({ err: error }, "Could not start the audio processing service");
  });
  child.on("exit", (code, signal) => {
    if (code !== 0 && code !== null) {
      logger.error({ code, signal }, "Audio processing service exited unexpectedly");
    }
  });

  const ready = waitForAudioService(child);
  process.once("exit", () => child.kill());
  process.once("SIGTERM", () => child.kill("SIGTERM"));
  process.once("SIGINT", () => child.kill("SIGINT"));

  return { child, ready };
}

async function waitForAudioService(child: ChildProcess): Promise<boolean> {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline && child.exitCode === null) {
    try {
      const response = await fetch(`http://127.0.0.1:${AUDIO_SERVICE_PORT}/healthz`, {
        signal: AbortSignal.timeout(1_000),
      });
      if (response.ok) return true;
    } catch {
      // The Python app can take a few seconds to load its bundled speech model.
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  logger.warn("Audio processing service did not become ready before the startup wait ended");
  return false;
}
