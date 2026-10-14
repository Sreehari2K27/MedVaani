/**
 * Focused regression tests for the scripted text-only AE demo journey.
 *
 * No test framework exists in this frontend (no vitest/jest dependency), so
 * these are plain assertion checks runnable directly with Node's native
 * TypeScript support:
 *
 *   node src/ae/aeDemoConversation.test.ts
 */

import {
  AE_STEPS,
  createConversation,
  handleUserInput,
  type AeChatMessage,
  type AeConversationState,
} from "./aeDemoConversation.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

const lastAssistantText = (state: AeConversationState): string => {
  const messages = [...state.messages]
    .reverse()
    .filter((m: AeChatMessage) => m.role === "assistant");
  return messages[0]?.text ?? "";
};

const summaryOf = (state: AeConversationState): AeChatMessage => {
  const summary = state.messages.find((m: AeChatMessage) => m.role === "summary");
  if (!summary || !summary.summary) {
    throw new Error("Expected a summary message at the end of the journey");
  }
  return summary;
};

const rowValue = (summary: AeChatMessage, label: string): string => {
  const row = summary.summary?.find((r) => r.label === label);
  if (!row) {
    throw new Error(`Expected a summary row labelled "${label}"`);
  }
  return row.value;
};

const stepQuestion = (id: string): string => {
  const step = AE_STEPS.find((s) => s.id === id);
  if (!step) {
    throw new Error(`Unknown step "${id}"`);
  }
  return step.question;
};

// ---- 1. Duplicate reporter question ----------------------------------------

const eventText = "I took Lantus yesterday and today I have a bad stomach pain.";
let state = createConversation();
state = handleUserInput(state, eventText);

const reporterQuestion = stepQuestion("reporter");
const reporterAsked = state.messages.filter((m) => m.text === reporterQuestion);
check(
  reporterAsked.length === 1,
  "reporter-identification question is asked exactly once after the event turn",
);
check(state.stepIndex === 0, "step progression lands on the reporter step");
check(
  state.answers["event"] === eventText,
  "event description is preserved in demo state",
);

const confirmation = state.messages[state.messages.length - 2];
check(
  !((confirmation.text ?? "").includes("person who experienced")),
  "event confirmation does not embed a duplicate reporter question",
);
check(
  (confirmation.text ?? "").includes("For this demo"),
  "generic event confirmation identifies the demo",
);
check(
  !((confirmation.text ?? "").includes("headache")),
  "generic event confirmation does not claim uncaptured symptoms",
);

// ---- 2. Consent branching ---------------------------------------------------

state = handleUserInput(state, "Myself, I am the patient"); // reporter
state = handleUserInput(state, "45"); // age
state = handleUserInput(state, "No, please do not contact me"); // consent declined

check(
  lastAssistantText(state) === stepQuestion("onset"),
  "declining contact consent skips method/contact and asks onset next",
);
check(
  !("method" in state.answers) && !("contact" in state.answers),
  "no contact-method or contact-detail answers are recorded after decline",
);
check(
  state.stepIndex === AE_STEPS.findIndex((s) => s.id === "onset"),
  "step index advanced past the skipped contact steps",
);

// Consent-accept path still asks the contact-method question.
let acceptState = createConversation();
acceptState = handleUserInput(acceptState, eventText);
acceptState = handleUserInput(acceptState, "Myself");
acceptState = handleUserInput(acceptState, "45");
acceptState = handleUserInput(acceptState, "Yes, you can contact me");
check(
  lastAssistantText(acceptState) === stepQuestion("method"),
  "accepting contact consent still asks the contact-method question",
);

// ---- 3 + 4. Summary: real event description, validated onset ----------------

state = handleUserInput(state, "yesterday morning"); // onset (valid)
state = handleUserInput(state, "No"); // seriousness
state = handleUserInput(state, "No"); // medical care -> report prepared (no outcome question)

check(state.finished === true, "journey finishes after the final step");
check(
  AE_STEPS.every((s) => s.id !== "outcome") &&
    state.messages.every(
      (m) => !(m.text ?? "").includes("How are you feeling now"),
    ),
  "no post-documentation outcome question is asked after the report",
);
check(
  lastAssistantText(state).includes("qualified human reviewer"),
  "conversation ends with the existing closing message after the report",
);
check(
  summaryOf(state).role === "summary",
  "the report summary is displayed before the closing message",
);

const summary = summaryOf(state);
check(
  rowValue(summary, "Reported events") === eventText,
  "summary reports the user's actual event description",
);
const serialized = JSON.stringify(summary.summary);
check(
  !serialized.includes("Severe headache") && !serialized.includes("Vomiting"),
  "summary never substitutes fabricated symptoms",
);
check(
  rowValue(summary, "Onset") === "Yesterday morning",
  "valid onset answer is presented as an onset date",
);
check(
  (summary.text ?? "").includes("demo information"),
  "summary remains explicitly identified as demo information",
);
check(
  !serialized.toLowerCase().includes("submitted") &&
    !serialized.toLowerCase().includes("regulatory"),
  "summary makes no submission/regulatory claims",
);

// Invalid onset ("No") must be labelled, not echoed.
let invalidOnsetState = createConversation();
invalidOnsetState = handleUserInput(invalidOnsetState, "stomach cramps after injection");
invalidOnsetState = handleUserInput(invalidOnsetState, "reporting for my son");
invalidOnsetState = handleUserInput(invalidOnsetState, "8");
invalidOnsetState = handleUserInput(invalidOnsetState, "Yes");
invalidOnsetState = handleUserInput(invalidOnsetState, "Phone");
invalidOnsetState = handleUserInput(invalidOnsetState, "9876543210");
invalidOnsetState = handleUserInput(invalidOnsetState, "No"); // invalid onset
invalidOnsetState = handleUserInput(invalidOnsetState, "Yes");
invalidOnsetState = handleUserInput(invalidOnsetState, "Yes"); // medical care -> finish

const invalidSummary = summaryOf(invalidOnsetState);
check(
  rowValue(invalidSummary, "Onset") === "Not provided",
  "invalid onset answer is labelled Not provided instead of echoed",
);
check(
  rowValue(invalidSummary, "Reported events") ===
    "stomach cramps after injection",
  "summary keeps the reported event even on the consent-accept path",
);
check(
  invalidSummary.summary?.some((r) => r.label === "Consent" && r.value === "Yes") ===
    true,
  "consent-accept path records consent in the summary",
);

// ---- Result -----------------------------------------------------------------

if (failures > 0) {
  throw new Error(`${failures} demo conversation test(s) failed`);
}
console.log("All demo conversation tests passed");
