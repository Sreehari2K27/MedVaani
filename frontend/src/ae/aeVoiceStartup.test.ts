/**
 * Focused tests for the voice-first AE greeting startup behavior.
 *
 * Contract under test:
 *   - the introduction plays automatically once per fresh session,
 *   - playback NEVER starts the microphone (the startup module has no
 *     microphone port at all; `ready` is the end state),
 *   - the fallback buttons only play/retry the introduction,
 *   - the mic button decision (micClickAction) is what starts recording,
 *   - rerenders/StrictMode never duplicate the greeting.
 *
 * No browser and no test framework: plain fakes driven with the repo's
 * Node-native TypeScript convention:
 *
 *   node src/ae/aeVoiceStartup.test.ts
 */

import {
  createVoiceStartup,
  micClickAction,
  type StartupAudio,
  type StartupDeps,
  type StartupGreeting,
  type VoiceStartupState,
} from "./aeVoiceStartup.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

// Capture the module's failure logging: it must exist, be concise, and
// carry nothing but its fixed prefix and an error name.
const warnings: string[] = [];
const originalWarn = console.warn;
console.warn = (...args: unknown[]): void => {
  warnings.push(args.map((part) => String(part)).join(" "));
};

interface HarnessOptions {
  fetchError?: boolean;
  noAudioPath?: boolean;
  playBlockedFirst?: boolean;
  /** Every play() rejects with an Error carrying this name. */
  playFailName?: string;
}

interface Harness {
  deps: StartupDeps;
  states: VoiceStartupState[];
  calls: { fetch: number; audioCreated: number; play: number; pause: number };
  fireEnded(): void;
  fireLoadError(): void;
}

function namedError(name: string): Error {
  const error = new Error(`${name} at playback`);
  error.name = name;
  return error;
}

function makeHarness(options: HarnessOptions = {}): Harness {
  const states: VoiceStartupState[] = [];
  const calls = { fetch: 0, audioCreated: 0, play: 0, pause: 0 };
  let playCount = 0;
  let ended: (() => void) | null = null;
  let loadError: ((reason: string) => void) | null = null;

  const audio: StartupAudio = {
    play: () => {
      calls.play += 1;
      playCount += 1;
      if (options.playFailName) {
        return Promise.reject(namedError(options.playFailName));
      }
      if (options.playBlockedFirst && playCount === 1) {
        return Promise.reject(namedError("NotAllowedError"));
      }
      return Promise.resolve();
    },
    pause: () => {
      calls.pause += 1;
    },
    onEnded: (listener) => {
      ended = listener;
    },
    onError: (listener) => {
      loadError = listener;
    },
  };

  const deps: StartupDeps = {
    fetchGreeting: () => {
      calls.fetch += 1;
      if (options.fetchError) {
        return Promise.reject(new Error("network down"));
      }
      const greeting: StartupGreeting = {
        audio_path: options.noAudioPath
          ? null
          : "/audio/conv_intro_sess.wav?_=1",
        response_text: "Thank you for taking the time to speak with us.",
      };
      return Promise.resolve(greeting);
    },
    createAudio: () => {
      calls.audioCreated += 1;
      return audio;
    },
    onState: (state) => {
      states.push(state);
    },
  };

  return {
    deps,
    states,
    calls,
    fireEnded: () => {
      ended?.();
    },
    fireLoadError: (reason = "load-error") => {
      loadError?.(reason);
    },
  };
}

const last = (states: VoiceStartupState[]): VoiceStartupState =>
  states[states.length - 1];

// ---- automatic greeting, microphone untouched --------------------------------

{
  const h = makeHarness();
  check(
    !("startMicrophone" in h.deps),
    "the startup module has no microphone capability at all",
  );
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(state === "playing", "the introduction plays automatically on a fresh session");
  check(
    h.calls.fetch === 1 && h.calls.audioCreated === 1 && h.calls.play === 1,
    "the introduction is fetched and played exactly once",
  );
  h.fireEnded();
  check(
    last(h.states) === "ready",
    "playback finishing settles on ready — recording never starts",
  );
}

// ---- rerenders/retries never duplicate the introduction ----------------------

{
  const h = makeHarness();
  const startup = createVoiceStartup(h.deps);
  await startup.begin();
  await startup.begin();
  await startup.begin();
  check(h.calls.fetch === 1, "repeated begin() calls never refetch the introduction");
  check(
    h.calls.audioCreated === 1 && h.calls.play === 1,
    "repeated begin() calls (rerenders/StrictMode) never replay the introduction",
  );
  check(
    h.states.filter((state) => state === "loading").length === 1,
    "exactly one loading transition",
  );
}

// ---- resume during an in-flight greeting is a no-op --------------------------

{
  const h = makeHarness();
  const startup = createVoiceStartup(h.deps);
  const pending = startup.begin();
  const early = await startup.resume();
  check(early === "loading", "resume during loading does not interrupt the greeting");
  await pending;
}

// ---- autoplay blocked: manual fallback plays the introduction only -----------

{
  const h = makeHarness({ playBlockedFirst: true });
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(state === "blocked", "autoplay rejection surfaces the blocked state");
  const resumed = await startup.resume();
  check(resumed === "playing", "the fallback button plays the introduction");
  check(h.calls.fetch === 1, "fallback playback never refetches the introduction");
  h.fireEnded();
  check(
    last(h.states) === "ready",
    "fallback playback finishing also settles on ready, not recording",
  );
}

{
  const h = makeHarness({ playFailName: "NotAllowedError" });
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(state === "blocked", "persistent autoplay rejection stays blocked");
  const resumed = await startup.resume();
  check(
    resumed === "blocked" && h.calls.play === 2,
    "a blocked retry that fails again never leaves the blocked state (no mic path)",
  );
}

// ---- greeting audio failures stay playback-only ------------------------------

{
  const h = makeHarness({ playFailName: "NotSupportedError" });
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(
    state === "audio-error",
    "an unloadable greeting surfaces audio-error, not a fake autoplay block",
  );
  const resumed = await startup.resume();
  check(
    resumed === "audio-error" && h.calls.play === 2,
    "the retry button only retries playback — it cannot record",
  );
}

{
  const h = makeHarness();
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(state === "playing", "greeting begins playing before the load error");
  h.fireLoadError();
  check(
    last(h.states) === "audio-error",
    "a late load failure surfaces audio-error instead of hanging silently",
  );
}

{
  const h = makeHarness({ fetchError: true });
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(state === "unavailable", "a failed greeting request surfaces unavailable");
  const retried = await startup.resume();
  check(retried === "unavailable" && h.calls.fetch === 2, "retry refetches the introduction");
  await startup.resume();
  check(h.calls.fetch === 3, "each explicit retry may refetch, never more automatically");
}

{
  const h = makeHarness({ noAudioPath: true });
  const startup = createVoiceStartup(h.deps);
  const state = await startup.begin();
  check(state === "unavailable", "a greeting without audio surfaces unavailable");
  const resumed = await startup.resume();
  check(
    resumed === "unavailable" && h.calls.fetch === 1,
    "with nothing to retry, resume is a harmless no-op (mic stays with the button)",
  );
}

// ---- finished introduction stays finished ------------------------------------

{
  const h = makeHarness();
  const startup = createVoiceStartup(h.deps);
  await startup.begin();
  h.fireEnded();
  const resumed = await startup.resume();
  check(
    resumed === "ready" && h.calls.play === 1,
    "resume after the introduction finished never replays or records",
  );
}

// ---- manual takeover cancels any pending greeting playback -------------------

{
  const h = makeHarness();
  const startup = createVoiceStartup(h.deps);
  const pending = startup.begin();
  startup.cancel();
  await pending;
  check(
    h.calls.audioCreated === 0 && h.calls.play === 0,
    "cancel during an in-flight greeting prevents any playback",
  );
  check(
    h.states.every((state) => state !== "playing" && state !== "ready"),
    "a cancelled greeting never plays and never settles ready",
  );
}

{
  const h = makeHarness();
  const startup = createVoiceStartup(h.deps);
  await startup.begin();
  startup.cancel();
  check(h.calls.pause === 1, "cancel pauses greeting playback");
  h.fireEnded();
  check(
    h.states.every((state) => state !== "ready"),
    "no completion state after the greeting was cancelled",
  );
}

// ---- the mic button is the only recording entry point -------------------------

check(
  micClickAction({ demoMode: true, busy: false, recording: false }) === "demo-notice",
  "mic click in typed mode shows the notice (typed mode unchanged)",
);
check(
  micClickAction({ demoMode: false, busy: true, recording: false }) === "ignore",
  "mic click while a turn is in flight is ignored",
);
check(
  micClickAction({ demoMode: false, busy: true, recording: true }) === "ignore",
  "busy takes precedence over an active recording",
);
check(
  micClickAction({ demoMode: false, busy: false, recording: true }) === "stop",
  "mic click while recording stops and sends",
);
check(
  micClickAction({ demoMode: false, busy: false, recording: false }) === "start",
  "clicking the mic on an idle voice session starts recording",
);

// ---- failure logging ----------------------------------------------------------

console.warn = originalWarn;
check(
  warnings.length > 0 &&
    warnings.every((warning) => warning.startsWith("[ae-greeting] ")),
  "playback failures log only concise [ae-greeting] prefixed warnings",
);
check(
  warnings.some((warning) => warning.includes("NotSupportedError")),
  "the failing error name is logged at the exact failure point",
);

if (failures > 0) {
  throw new Error(`${failures} voice startup test(s) failed`);
}
console.log("All voice startup tests passed");
