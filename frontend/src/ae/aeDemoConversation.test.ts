/**
 * Focused regression checks for the AE conversation sidebar/opening module.
 *
 * The client-side demo journey was removed: typed ("demo") and spoken turns
 * now share the authoritative backend AE workflow, so the approved question
 * order can no longer diverge in the frontend. No test framework exists in
 * this frontend, so these are plain assertion checks runnable directly with
 * Node's native TypeScript support:
 *
 *   node --experimental-strip-types src/ae/aeDemoConversation.test.ts
 */

import {
  AE_STAGES,
  OPENING_MESSAGE,
  stageStatusText,
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

// ---- approved opening -------------------------------------------------------

check(
  OPENING_MESSAGE ===
    "Thank you for taking the time to speak with us. I'm here to help record the event for review. I'm not a substitute for a healthcare professional, and I can't provide medical advice. Please tell me what happened, in your own words.",
  "opening matches the approved introduction and invitation exactly",
);

// ---- sidebar stages follow the approved flow --------------------------------

check(
  AE_STAGES.map((stage) => stage.id).join(",") ===
    "what-happened,reporter-details,patient-details,contact-information,consent,seriousness,event-timing,review",
  "sidebar stages follow the approved order (contact before consent, severity before event timing)",
);
const removedStage = "outcome";
check(
  AE_STAGES.every((stage) => (stage.id as string) !== removedStage),
  "no post-documentation outcome stage exists",
);

check(stageStatusText("review", "done") === "Ready for human review", "review stage has a done label");
check(stageStatusText("consent", "todo") === "Not started", "todo stages are labelled");

// ---- the divergent client-side journey is gone ------------------------------

import * as demoModule from "./aeDemoConversation.ts";

check(
  !("AE_STEPS" in demoModule) &&
    !("handleUserInput" in demoModule) &&
    !("createConversation" in demoModule),
  "no client-side question sequence remains in the frontend module",
);
check(
  !JSON.stringify(Object.keys(demoModule)).includes("medical"),
  "no medical-care question is exported",
);

if (failures > 0) {
  throw new Error(`${failures} demo conversation test(s) failed`);
}
console.log("All demo conversation tests passed");
