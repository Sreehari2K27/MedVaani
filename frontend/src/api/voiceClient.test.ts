/**
 * Focused regression tests for the voice client's language hint.
 *
 * No test framework exists in this frontend (no vitest/jest dependency), so
 * these are plain assertion checks runnable directly with Node's native
 * TypeScript support (same convention as aeDemoConversation.test.ts):
 *
 *   node src/api/voiceClient.test.ts
 */

import {
  postTextTurn,
  postVoiceTurn,
  resolveRequestLanguage,
} from "./voiceClient.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

// ---- Locale mapping ---------------------------------------------------------

check(resolveRequestLanguage("hi") === "hi-IN", "hi maps to hi-IN");
check(resolveRequestLanguage("hi-IN") === "hi-IN", "hi-IN maps to hi-IN");
check(resolveRequestLanguage("en-US") === "en-IN", "en-US falls back to en-IN");
check(resolveRequestLanguage("en") === "en-IN", "en falls back to en-IN");
check(resolveRequestLanguage("ml-IN") === "ml-IN", "ml-IN maps to ml-IN");
check(resolveRequestLanguage("ml") === "ml-IN", "ml maps to ml-IN");
check(resolveRequestLanguage(undefined) === "en-IN", "missing locale falls back to en-IN");
check(resolveRequestLanguage("") === "en-IN", "empty locale falls back to en-IN");
check(resolveRequestLanguage("xx-YY") === "en-IN", "unknown locale falls back to en-IN");

// ---- Requests carry the explicit language hint ------------------------------

interface CapturedCall {
  url: string;
  language: string | null;
}

const captured: CapturedCall[] = [];
const originalFetch = globalThis.fetch;

globalThis.fetch = (async (url: RequestInfo | URL, init?: RequestInit) => {
  const body = init?.body;
  const language =
    body instanceof FormData ? (body.get("language") as string | null) : null;
  captured.push({ url: String(url), language });
  return new Response(JSON.stringify({ session_id: "test" }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

try {
  await postVoiceTurn("sess-1", new Blob(["fake-audio"]), "hi-IN");
  await postTextTurn("sess-1", "नमस्ते", "hi-IN");
  await postVoiceTurn("sess-2", new Blob(["fake-audio"]));
} finally {
  globalThis.fetch = originalFetch;
}

check(
  captured[0]?.url === "/voice/conversation" &&
    captured[0]?.language === "hi-IN",
  "voice request includes the explicit language in its FormData",
);
check(
  captured[1]?.url === "/text/conversation" &&
    captured[1]?.language === "hi-IN",
  "text request includes the explicit language in its FormData",
);
check(
  captured[2]?.language === "en-IN",
  "voice request without an explicit language still defaults to en-IN",
);

// ---- Result -----------------------------------------------------------------

if (failures > 0) {
  throw new Error(`${failures} voice client test(s) failed`);
}
console.log("All voice client tests passed");
