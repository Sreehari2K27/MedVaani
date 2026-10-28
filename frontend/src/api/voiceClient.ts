// Typed client for the MedVaani voice conversation endpoint.
// Same-origin in development via the Vite proxy (see vite.config.ts).

export interface AeTurnInformation {
  extracted_information: Record<string, unknown>;
  collected_information: Record<string, unknown>;
}

export interface VoiceTurnResponse {
  session_id: string;
  transcript: string;
  intent: string;
  workflow: string;
  reason: string;
  response_text: string;
  audio_path: string | null;
  current_state: string;
  case_id: string;
  priority: string;
  human_review_required: boolean;
  completed: boolean;
  missing_information: string[] | null;
  next_question: [string, string] | null;
  safety_status: string;
  mi: unknown;
  ae: AeTurnInformation | null;
  // Additive language-mirroring metadata. The English `response_text` above is
  // unchanged; these describe the spoken form (all optional for compatibility).
  // `spoken_response_text` is null on text turns, which produce no audio.
  conversation_language?: string;
  spoken_response_text?: string | null;
  voice?: string | null;
  language_translated?: boolean;
  language_source?: string;
}

export class VoiceApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "VoiceApiError";
    this.status = status;
  }
}

const FRIENDLY_ERRORS: Record<number, string> = {
  400: "MedVaani couldn't read that recording. Please try again.",
  429: "Too many requests. Please wait a moment and try again.",
  500: "We're having trouble with the speech service right now. Please try again in a moment.",
  502: "The speech service is unavailable right now. Please try again shortly.",
};

const TEXT_ERRORS: Record<number, string> = {
  400: "MedVaani couldn't process that message. Please try again.",
  429: FRIENDLY_ERRORS[429],
  500: FRIENDLY_ERRORS[500],
  502: FRIENDLY_ERRORS[502],
};

function messageFor(
  status: number,
  detail: string | null,
  errors: Record<number, string>,
  label: string,
): string {
  const base = errors[status] ?? `${label} (${status}). Please try again.`;
  // Append the backend's safe, human-readable detail (never secrets) so
  // distinct failures (auth, rate limit, empty transcript, timeout) are
  // distinguishable instead of collapsing into one generic message.
  if (detail) {
    return `${base} (${detail})`;
  }
  return base;
}

async function postTurn<T>(
  path: string,
  form: FormData,
  errors: Record<number, string>,
  label: string,
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    let detail: string | null = null;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error body; fall back to the generic message.
    }
    throw new VoiceApiError(
      response.status,
      messageFor(response.status, detail, errors, label),
    );
  }

  return (await response.json()) as T;
}

// Map a browser locale (e.g. "hi-IN", "ml") to the BCP-47 language hint the
// backend accepts for STT/TTS. Only languages the existing integration
// supports are mapped; anything else (including English and missing/unknown
// locales) falls back to en-IN. This is an initial hint only: the backend
// switches the session language automatically once a transcript identifies it.
const LOCALE_TO_LANGUAGE: Record<string, string> = {
  hi: "hi-IN",
  bn: "bn-IN",
  gu: "gu-IN",
  kn: "kn-IN",
  ml: "ml-IN",
  mr: "mr-IN",
  pa: "pa-IN",
  ta: "ta-IN",
  te: "te-IN",
};

export function resolveRequestLanguage(locale?: string): string {
  const base = (locale ?? "").trim().toLowerCase().split(/[-_]/)[0];
  return LOCALE_TO_LANGUAGE[base] ?? "en-IN";
}

// The ten locales the backend supports, for the explicit language selector.
// Labels pair the readable native name with the BCP-47 code the backend
// accepts; `code` is exactly the value posted as the initial `language` hint.
export interface LanguageOption {
  code: string;
  label: string;
}

export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { code: "en-IN", label: "English (en-IN)" },
  { code: "hi-IN", label: "हिन्दी (hi-IN)" },
  { code: "bn-IN", label: "বাংলা (bn-IN)" },
  { code: "gu-IN", label: "ગુજરાતી (gu-IN)" },
  { code: "kn-IN", label: "ಕನ್ನಡ (kn-IN)" },
  { code: "ml-IN", label: "മലയാളം (ml-IN)" },
  { code: "mr-IN", label: "मराठी (mr-IN)" },
  { code: "pa-IN", label: "ਪੰਜਾਬੀ (pa-IN)" },
  { code: "ta-IN", label: "தமிழ் (ta-IN)" },
  { code: "te-IN", label: "తెలుగు (te-IN)" },
];

export async function postVoiceTurn(
  sessionId: string,
  blob: Blob,
  language = "en-IN",
): Promise<VoiceTurnResponse> {
  const form = new FormData();
  form.append("audio_file", blob, "recording.wav");
  form.append("session_id", sessionId);
  form.append("language", language);
  return postTurn<VoiceTurnResponse>(
    "/voice/conversation",
    form,
    FRIENDLY_ERRORS,
    "Voice request failed",
  );
}

// Typed AE/MI turn through the shared backend workflow (no STT/TTS).
export async function postTextTurn(
  sessionId: string,
  message: string,
  language = "en-IN",
): Promise<VoiceTurnResponse> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("message", message);
  form.append("language", language);
  return postTurn<VoiceTurnResponse>(
    "/text/conversation",
    form,
    TEXT_ERRORS,
    "Text request failed",
  );
}

// "output/conv_ae_<session>.wav" -> "/audio/conv_ae_<session>.wav?_=<timestamp>"
// The backend overwrites the same file every turn, so a cache-busting query
// parameter is required to avoid replaying stale audio.
export function audioUrlFor(audioPath: string): string {
  const normalized = audioPath.replace(/\\/g, "/");
  const name = normalized.substring(normalized.lastIndexOf("/") + 1);
  return `/audio/${name}?_=${Date.now()}`;
}

// One-shot delivery of the approved AE opening for a fresh voice session.
// The endpoint also marks the session's workflow introduction as shown, so
// the first utterance is never answered with a repeated introduction. The
// text below is the same fixed wording the chat already seeds as its single
// greeting; it is returned for verification only and must NOT be pushed as
// another chat message.
export interface GreetingResponse {
  session_id: string;
  response_text: string;
  audio_path: string | null;
  // Same additive language-mirroring metadata as a turn response.
  conversation_language?: string;
  spoken_response_text?: string | null;
  voice?: string | null;
  language_translated?: boolean;
  language_source?: string;
}

export async function postGreeting(
  sessionId: string,
  language = "en-IN",
): Promise<GreetingResponse> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("language", language);
  return postTurn<GreetingResponse>(
    "/voice/introduction",
    form,
    TEXT_ERRORS,
    "Greeting request failed",
  );
}

// Visible text must always match what is spoken: when the backend
// translated a response, the spoken form is authoritative; otherwise the
// English text is what was spoken too (translation is all-or-nothing).
export function assistantDisplayText(result: VoiceTurnResponse): string {
  if (result.language_translated && result.spoken_response_text) {
    return result.spoken_response_text;
  }
  return result.response_text;
}
