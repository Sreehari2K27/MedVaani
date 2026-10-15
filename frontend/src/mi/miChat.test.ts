/**
 * Focused checks for the MI chat helpers and text-only invariants.
 * No test framework exists in this frontend; these are plain assertion
 * checks runnable with Node's native TypeScript support:
 *
 *   node --experimental-strip-types src/mi/miChat.test.ts
 */

import { readFileSync } from "node:fs";
import {
  MI_RECORDING_MAX_SECONDS,
  WELCOME_EXAMPLES,
  canSubmit,
  composerKeyAction,
  splitCitationSegments,
  usableTranscript,
} from "./miChat.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

// ---- composer behaviour -----------------------------------------------------

check(composerKeyAction("Enter", false, false) === "submit", "Enter submits an idle draft");
check(composerKeyAction("Enter", true, false) === "newline", "Shift+Enter inserts a newline");
check(composerKeyAction("Enter", false, true) === "none", "Enter never submits while busy");
check(composerKeyAction("a", false, false) === "none", "other keys leave the composer alone");

check(canSubmit("What is Lantus?", false) === true, "a real draft can be submitted");
check(canSubmit("   ", false) === false, "a blank draft is never submitted");
check(canSubmit("question", true) === false, "busy state blocks duplicate submissions");

// ---- transcription review ---------------------------------------------------

check(usableTranscript("  How does it work?  ") === "How does it work?", "transcripts are trimmed for review");
check(usableTranscript("") === "", "an empty transcript is unusable");
check(usableTranscript(null) === "", "a failed transcription is unusable");

// ---- citation rendering -----------------------------------------------------

const cited = splitCitationSegments(
  'Hypoglycemia is the most frequent adverse reaction. (Source: Lantus package insert, page 6, section "Adverse Reactions").',
);
check(cited.length === 2, "an answer splits into prose and one citation");
check(cited[0].citation === false && cited[1].citation === true, "the source segment is flagged as a citation");
check(
  splitCitationSegments("No citation here.").every((s) => !s.citation),
  "answers without citations stay unflagged",
);

// ---- welcome state and recording limit --------------------------------------

check(MI_RECORDING_MAX_SECONDS === 30, "MI recording is capped at 30 seconds");
check(
  WELCOME_EXAMPLES.length > 0 && WELCOME_EXAMPLES.every((q) => q.includes("Lantus")),
  "welcome examples all target Lantus",
);

// ---- text-only and navigation invariants ------------------------------------

const miPage = readFileSync(
  new URL("../components/pages/MiPage.tsx", import.meta.url),
  "utf-8",
);
check(
  !/audioUrlFor|<audio|new Audio\(/.test(miPage),
  "MiPage never plays or generates audio (text-only answers)",
);

const styles = readFileSync(
  new URL("../mi-styles.css", import.meta.url),
  "utf-8",
);
check(
  /white-space:\s*pre-wrap/.test(styles),
  "assistant messages preserve paragraphs and citations (pre-wrap)",
);

const app = readFileSync(new URL("../App.tsx", import.meta.url), "utf-8");
check(
  /onGetProductInfo/.test(app) && /navigate\("mi"\)/.test(app),
  "the homepage MI card stays wired to the MI page",
);
check(
  />Medical Information</.test(miPage),
  "the page heading is 'Medical Information'",
);
check(
  !/onHome|mi-home-btn/.test(miPage) && !/onHome/.test(app),
  "no per-page Home button: main navigation is preserved instead",
);

if (failures > 0) {
  throw new Error(`${failures} MI chat test(s) failed`);
}
console.log("All MI chat helper tests passed");
