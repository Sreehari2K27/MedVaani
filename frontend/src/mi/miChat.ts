// Pure helpers for the Medical Information chat composer and recorder.
// Kept free of React/DOM so they can be checked with Node's native
// TypeScript support (see miChat.test.ts).

// The MI microphone is capped well below the speech service's documented
// 60-second limit; exceeding it stops the recording with a clear notice.
export const MI_RECORDING_MAX_SECONDS = 30;

export const WELCOME_EXAMPLES: readonly string[] = [
  "What is Lantus SoloStar used for?",
  "How should Lantus SoloStar be stored?",
  "What are the common adverse reactions of Lantus SoloStar?",
  "How does Lantus SoloStar work?",
];

export type ComposerKeyAction = "submit" | "newline" | "none";

// Enter submits the draft; Shift+Enter inserts a newline; keys are ignored
// entirely while a request or recording is in flight.
export function composerKeyAction(
  key: string,
  shiftKey: boolean,
  busy: boolean,
): ComposerKeyAction {
  if (key !== "Enter") return "none";
  if (shiftKey) return "newline";
  if (busy) return "none";
  return "submit";
}

// A draft may be sent only when it has non-whitespace content and no other
// request is being processed (prevents duplicate submissions).
export function canSubmit(text: string, busy: boolean): boolean {
  return !busy && text.trim().length > 0;
}

// A recognized transcript may fill the composer only when it carries real
// content; empty/failed transcriptions are never submitted.
export function usableTranscript(text: string | null | undefined): string {
  return (text ?? "").trim();
}

export interface AnswerSegment {
  text: string;
  citation: boolean;
}

const CITATION_PATTERN = /(\(\s*Source:[^)]*\)\.?)/g;

// Split an assistant answer into prose and source-citation segments so the
// citation can be styled (and identified) separately from the answer text.
export function splitCitationSegments(text: string): AnswerSegment[] {
  return text
    .split(CITATION_PATTERN)
    .filter((part) => part.length > 0)
    .map((part) => ({
      text: part,
      citation: /^\(\s*source:/i.test(part),
    }));
}
