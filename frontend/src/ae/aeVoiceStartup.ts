// Voice-first AE startup sequencing for a fresh live (spoken) session.
//
// Pure orchestration over injected browser ports so it is testable with
// plain Node scripts: fetch the approved greeting and play it once. When
// playback finishes the state settles on "ready" — this module has NO
// microphone capability at all (there is no startMicrophone port): the mic
// button is the only action that starts recording, via micClickAction().
// Playback failure is never silently swallowed: it is classified, logged
// concisely (error name only) and surfaced as a state the UI can explain.
// This module NEVER emits chat messages: the single seeded greeting message
// in AeChat is the only display of the introduction, so rerenders and
// retries can never duplicate it.

export type VoiceStartupState =
  | "idle"
  | "loading"
  | "playing"
  | "ready"
  | "blocked"
  | "audio-error"
  | "unavailable";

export interface StartupGreeting {
  audio_path: string | null;
  response_text: string;
  // Language-mirroring metadata: when the backend localized the spoken
  // introduction, the display text must switch to the spoken form so the
  // screen always matches the audio (same rule as conversation turns).
  spoken_response_text?: string | null;
  language_translated?: boolean;
}

export interface StartupAudio {
  play(): Promise<void>;
  pause(): void;
  onEnded(listener: () => void): void;
  onError(listener: (reason: string) => void): void;
}

export interface StartupDeps {
  fetchGreeting(): Promise<StartupGreeting>;
  createAudio(url: string): StartupAudio;
  onState(state: VoiceStartupState): void;
}

export interface VoiceStartup {
  begin(): Promise<VoiceStartupState>;
  resume(): Promise<VoiceStartupState>;
  /** User took over (manual mic tap or typed turn): never play the greeting. */
  cancel(): void;
}

export function createVoiceStartup(deps: StartupDeps): VoiceStartup {
  let begun = false;
  let cancelled = false;
  let state: VoiceStartupState = "idle";
  let audio: StartupAudio | null = null;
  let audioEnded = false;
  let audioFailed = false;
  let fetchFailed = false;

  const setState = (next: VoiceStartupState): VoiceStartupState => {
    state = next;
    deps.onState(next);
    return next;
  };

  // One concise, non-sensitive line at the exact failure point (error name
  // only — never URLs, prompts or session data), then a state the UI can
  // explain: genuine autoplay blocking vs. audio that cannot play at all.
  const classify = (error: unknown): VoiceStartupState => {
    const name = error instanceof Error ? error.name : typeof error;
    console.warn(`[ae-greeting] playback failed (${name})`);
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "blocked";
    }
    return "audio-error";
  };

  const fetchAndPlay = async (): Promise<VoiceStartupState> => {
    let greeting: StartupGreeting;
    try {
      greeting = await deps.fetchGreeting();
      fetchFailed = false;
    } catch {
      if (cancelled) return state;
      fetchFailed = true;
      return setState("unavailable");
    }
    if (cancelled) return state;
    if (!greeting.audio_path) {
      // The text greeting exists; only the spoken form is missing.
      return setState("unavailable");
    }
    audio = deps.createAudio(greeting.audio_path);
    audio.onEnded(() => {
      // The introduction finished: settle on "ready" and wait. The
      // microphone is NEVER started here — only the mic button does that.
      if (cancelled) return;
      audioEnded = true;
      setState("ready");
    });
    audio.onError(() => {
      // A load failure (bad URL, missing file, undecodable bytes) after
      // playback started would otherwise hang silently: no `ended`, no
      // sound. Surface it instead.
      if (cancelled || audioEnded) return;
      console.warn("[ae-greeting] greeting audio failed to load");
      audioFailed = true;
      setState("audio-error");
    });
    try {
      await audio.play();
      if (cancelled || audioFailed) return state;
      return setState("playing");
    } catch (error) {
      return setState(classify(error));
    }
  };

  const begin = async (): Promise<VoiceStartupState> => {
    if (begun) {
      // Rerenders (including React StrictMode's double effect) and repeat
      // calls never fetch or play the greeting twice.
      return state;
    }
    begun = true;
    setState("loading");
    return fetchAndPlay();
  };

  const cancel = (): void => {
    cancelled = true;
    audio?.pause();
  };

  // Manual playback only (the blocked/failed-audio fallback buttons).
  // Success returns to playing and later "ready"; it never touches a
  // microphone — there is no microphone port in these deps.
  const resume = async (): Promise<VoiceStartupState> => {
    if (cancelled || state === "loading") {
      return state;
    }
    if (audio) {
      if (audioEnded) {
        return state;
      }
      audioFailed = false;
      try {
        await audio.play();
        if (cancelled || audioFailed) return state;
        return setState("playing");
      } catch (error) {
        return setState(classify(error));
      }
    }
    if (fetchFailed) {
      // The greeting request itself failed earlier: retry it (the endpoint
      // is idempotent and its text is never re-displayed).
      setState("loading");
      return fetchAndPlay();
    }
    return state;
  };

  return { begin, resume, cancel };
}

// ---- mic button decision (the only path that starts recording) ---------------

export type MicClickAction = "demo-notice" | "ignore" | "stop" | "start";

// Mirrors AeChat's mic-button precedence exactly: demo mode shows a notice,
// busy turns are ignored, an active recording is stopped/sent, and every
// other click starts recording — the single entry point into the recorder.
export function micClickAction(input: {
  demoMode: boolean;
  busy: boolean;
  recording: boolean;
}): MicClickAction {
  if (input.demoMode) return "demo-notice";
  if (input.busy) return "ignore";
  if (input.recording) return "stop";
  return "start";
}

// ---- language selector state (pure decision, mirrors micClickAction) --------

// Changing the language hint mid-conversation is not safely supported: the
// backend establishes the session language from the first transcript, so the
// selector locks once the conversation has started (or while a turn is in
// flight) instead of ever resetting the session.
export function languageSelectDisabled(input: {
  liveStarted: boolean;
  busy: boolean;
}): boolean {
  return input.liveStarted || input.busy;
}
