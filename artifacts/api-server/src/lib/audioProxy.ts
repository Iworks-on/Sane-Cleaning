import { request as httpRequest } from "node:http";
import type { RequestHandler } from "express";
import { logger } from "./logger";

const AUDIO_SERVICE_PORT = Number(process.env.AUDIO_SERVICE_PORT ?? 8099);

export const audioProxy: RequestHandler = (req, res) => {
  const suffix = req.url || "/";
  const upstream = httpRequest(
    {
      hostname: "127.0.0.1",
      port: AUDIO_SERVICE_PORT,
      method: req.method,
      path: `/audio${suffix}`,
      headers: {
        ...req.headers,
        host: `127.0.0.1:${AUDIO_SERVICE_PORT}`,
      },
    },
    (response) => {
      res.status(response.statusCode ?? 502);
      for (const [name, value] of Object.entries(response.headers)) {
        if (value !== undefined && name !== "transfer-encoding" && name !== "connection") {
          res.setHeader(name, value);
        }
      }
      response.pipe(res);
    },
  );

  upstream.on("error", () => {
    if (!res.headersSent) {
      res.status(503).json({
        detail: "The audio processing service is not available yet. Please wait a moment and try again.",
      });
    } else {
      res.destroy();
    }
    logger.warn({ path: req.path }, "Audio service proxy request failed");
  });

  req.on("aborted", () => upstream.destroy());
  req.pipe(upstream);
};
