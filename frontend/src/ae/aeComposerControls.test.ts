/**
 * Focused tests for the AE composer control visibility by mode.
 *
 * Contract under test:
 *   - voice-only (live) AE mode: the text input is absent entirely and the
 *     microphone is present (with no send button),
 *   - normal typed chat (demo) mode: the text input and send button are
 *     present and the microphone button is absent (unchanged behaviour),
 *   - switching modes never leaves stale controls: the decision is a pure
 *     function of the mode, and the visible mic button always lines up with
 *     micClickAction's decision.
 *
 * No browser and no test framework: plain fakes driven with the repo's
 * Node-native TypeScript convention:
 *
 *   node src/ae/aeComposerControls.test.ts
 */

import { composerControls } from "./aeComposerControls.ts";
import { micClickAction } from "./aeVoiceStartup.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

// ---- voice-only AE mode: no text field, microphone present -------------------

{
  const controls = composerControls({ mode: "live" });
  check(
    controls.textInput === false,
    "voice-only AE mode hides the text input entirely",
  );
  check(
    controls.micButton === true,
    "voice-only AE mode keeps the microphone control",
  );
  check(
    controls.sendButton === false,
    "voice-only AE mode has no send button (nothing typed to send)",
  );
}

// ---- normal typed chat mode: text input present -------------------------------

{
  const controls = composerControls({ mode: "demo" });
  check(
    controls.textInput === true,
    "normal chat mode keeps the text input",
  );
  check(
    controls.sendButton === true,
    "normal chat mode keeps the send button",
  );
  check(
    controls.micButton === false,
    "normal chat mode has no microphone button (unchanged behaviour)",
  );
}

// ---- switching modes never leaves stale controls -------------------------------

{
  // AeChat's mode is fixed for the lifetime of a mount (useState
  // initializer; the entry screen unmounts it), so a mode switch is always
  // a fresh mount. Purity is therefore the stale-state guarantee: every
  // call reflects exactly the mode it was given, never a previous one.
  let controls = composerControls({ mode: "demo" });
  for (let i = 0; i < 4; i += 1) {
    const mode = i % 2 === 0 ? "live" : "demo";
    controls = composerControls({ mode });
    if (mode === "live") {
      check(
        controls.textInput === false && controls.micButton === true,
        `switch ${i + 1} to voice-only shows only the microphone`,
      );
    } else {
      check(
        controls.textInput === true &&
          controls.sendButton === true &&
          controls.micButton === false,
        `switch ${i + 1} back to typed chat restores the text composer`,
      );
    }
  }
}

// ---- the visible mic button matches the mic click decision ---------------------

{
  for (const mode of ["live", "demo"] as const) {
    const controls = composerControls({ mode });
    const action = micClickAction({
      demoMode: mode === "demo",
      busy: false,
      recording: false,
    });
    check(
      (controls.micButton && action === "start") ||
        (!controls.micButton && action === "demo-notice"),
      `in ${mode} mode the mic button visibility matches micClickAction`,
    );
  }
}

if (failures > 0) {
  throw new Error(`${failures} composer controls test(s) failed`);
}
console.log("All composer controls tests passed");
