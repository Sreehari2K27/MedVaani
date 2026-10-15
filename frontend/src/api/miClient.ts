// Typed client for the dedicated Medical Information chat endpoints.
// Same-origin in development via the Vite proxy (see vite.config.ts, "/mi").
//
// Answers are text only: this client never requests or returns audio, and
// there is no TTS playback path for MI responses.

export interface MiChatResponse {
  answer: string;
  status: string;
  category: string;
  grounded: boolean;
  supported: boolean;
  source_page_numbers: number[];
  retrieved_chunk_ids: string[];
  urgent: boolean;
  suggest_ae_report: boolean;
}

export interface MiTranscribeResponse {
  transcript: string;
}

export class MiApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "MiApiError";
    this.status = status;
  }
}

const CHAT_ERRORS: Record<number, string> = {
  400: "MedVaani couldn't process that question. Please try again.",
  429: "Too many requests. Please wait a moment and try again.",
  500: "We're having trouble with MedVaani right now. Please try again in a moment.",
  502: "The medical information service is unavailable right now. Please try again shortly.",
};

const TRANSCRIBE_ERRORS: Record<number, string> = {
  400: "MedVaani couldn't read that recording. Please try again.",
  429: "Too many requests. Please wait a moment and try again.",
  500: "We're having trouble with the speech service right now. Please try again in a moment.",
  502: "The speech service is unavailable right now. Please try again shortly.",
};

function messageFor(
  status: number,
  detail: string | null,
  errors: Record<number, string>,
  label: string,
): string {
  const base = errors[status] ?? `${label} (${status}). Please try again.`;
  // Append the backend's safe, human-readable detail (never secrets) so
  // distinct failures stay distinguishable instead of one generic message.
  if (detail) {
    return `${base} (${detail})`;
  }
  return base;
}

async function requestJson<T>(
  path: string,
  init: RequestInit,
  errors: Record<number, string>,
  label: string,
): Promise<T> {
  const response = await fetch(path, init);

  if (!response.ok) {
    let detail: string | null = null;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error body; fall back to the generic message.
    }
    throw new MiApiError(
      response.status,
      messageFor(response.status, detail, errors, label),
    );
  }

  return (await response.json()) as T;
}

// Submit one MI question. The backend answers with grounded text only -
// never audio, an AE session, or provider internals.
export async function postMiChat(question: string): Promise<MiChatResponse> {
  return requestJson<MiChatResponse>(
    "/mi/chat",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
    CHAT_ERRORS,
    "Medical information request failed",
  );
}

// Transcribe one recorded question via the existing Prisma STT integration.
// The transcript is returned for review in the composer; it is never
// auto-submitted.
export async function postMiTranscribe(
  blob: Blob,
  language = "en-IN",
): Promise<MiTranscribeResponse> {
  const form = new FormData();
  form.append("audio_file", blob, "recording.wav");
  form.append("language", language);
  return requestJson<MiTranscribeResponse>(
    "/mi/transcribe",
    { method: "POST", body: form },
    TRANSCRIBE_ERRORS,
    "Speech transcription failed",
  );
}
