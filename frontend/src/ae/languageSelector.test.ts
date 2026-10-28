/**
 * Focused tests for the explicit report-language selector.
 *
 * No test framework exists in this frontend (no vitest/jest dependency), so
 * these are plain assertion checks runnable directly with Node's native
 * TypeScript support (same convention as voiceClient.test.ts):
 *
 *   node src/ae/languageSelector.test.ts
 */

import {
  postTextTurn,
  postVoiceTurn,
  resolveRequestLanguage,
  SUPPORTED_LANGUAGES,
} from "../api/voiceClient.ts";
import { languageSelectDisabled } from "./aeVoiceStartup.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

// ---- The selector offers exactly the ten supported locales -------------------

const EXPECTED_CODES = [
  "en-IN",
  "hi-IN",
  "bn-IN",
  "gu-IN",
  "kn-IN",
  "ml-IN",
  "mr-IN",
  "pa-IN",
  "ta-IN",
  "te-IN",
];

check(SUPPORTED_LANGUAGES.length === 10, "the selector offers exactly ten languages");
check(
  JSON.stringify(SUPPORTED_LANGUAGES.map((option) => option.code)) ===
    JSON.stringify(EXPECTED_CODES),
  "the selector options are exactly the ten supported locale codes",
);
check(
  SUPPORTED_LANGUAGES.every(
    (option) =>
      typeof option.label === "string" &&
      option.label.trim().length > 0 &&
      option.label.includes(option.code),
  ),
  "each option shows a readable name alongside its locale code",
);

// ---- Default selection mirrors the browser-language behaviour ----------------

check(
  resolveRequestLanguage("ta-IN") === "ta-IN" &&
    resolveRequestLanguage("bn-IN") === "bn-IN" &&
    resolveRequestLanguage("mr-IN") === "mr-IN",
  "a browser locale that maps to a supported language becomes the default",
);
check(
  resolveRequestLanguage("en-US") === "en-IN" && resolveRequestLanguage("en") === "en-IN",
  "an English browser locale defaults to en-IN",
);
check(
  resolveRequestLanguage("fr-FR") === "en-IN" && resolveRequestLanguage(undefined) === "en-IN",
  "an unrelated or missing browser locale defaults to en-IN",
);

// ---- The selector locks instead of resetting the session ---------------------

check(
  languageSelectDisabled({ liveStarted: false, busy: false }) === false,
  "the selector is editable before the conversation starts",
);
check(
  languageSelectDisabled({ liveStarted: true, busy: false }) === true,
  "the selector locks once the conversation has started",
);
check(
  languageSelectDisabled({ liveStarted: false, busy: true }) === true,
  "the selector locks while a turn is in flight",
);

// ---- Requests carry the explicitly selected locale ---------------------------

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
  await postVoiceTurn("sess-lang", new Blob(["fake-audio"]), "ta-IN");
  await postTextTurn("sess-lang", "எனக்கு தலைவலி உள்ளது", "ta-IN");
} finally {
  globalThis.fetch = originalFetch;
}

check(
  captured[0]?.url === "/voice/conversation" && captured[0]?.language === "ta-IN",
  "the voice request carries the explicitly selected locale",
);
check(
  captured[1]?.url === "/text/conversation" && captured[1]?.language === "ta-IN",
  "the text request carries the explicitly selected locale",
);

// ---- Result ------------------------------------------------------------------

if (failures > 0) {
  throw new Error(`${failures} language selector test(s) failed`);
}
console.log("All language selector tests passed");
