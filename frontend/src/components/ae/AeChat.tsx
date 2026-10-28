import { useCallback, useEffect, useRef, useState } from "react";
import lantusPen from "../../assets/lantus-solostar.png.png";
import {
  AE_STAGES,
  initialLiveMessages,
  stageStatusText,
  type AeChatMessage,
} from "../../ae/aeDemoConversation";
import {
  buildLiveSummary,
  liveStageStates,
  missingLabels,
  type AeCollectedInfo,
} from "../../ae/aeLiveConversation";
import {
  createVoiceStartup,
  micClickAction,
  type VoiceStartup,
  type VoiceStartupState,
} from "../../ae/aeVoiceStartup";
import {
  assistantDisplayText,
  audioUrlFor,
  postGreeting,
  postTextTurn,
  postVoiceTurn,
  resolveRequestLanguage,
  SUPPORTED_LANGUAGES,
  type VoiceTurnResponse,
} from "../../api/voiceClient";
import { useWavRecorder } from "../../hooks/useWavRecorder";
import { composerControls } from "../../ae/aeComposerControls";
import { languageSelectDisabled } from "../../ae/aeVoiceStartup";

function CheckIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  );
}

function MicIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
      <line x1="12" y1="19" x2="12" y2="23"></line>
      <line x1="8" y1="23" x2="16" y2="23"></line>
    </svg>
  );
}

function SendIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m22 2-7 20-4-9-9-4z"></path>
      <path d="M22 2 11 13"></path>
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
      <line x1="12" y1="9" x2="12" y2="13"></line>
      <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
  );
}

function renderMessage(message: AeChatMessage) {
  switch (message.role) {
    case "safety":
      return (
        <div className="ae-safety-alert" key={message.id} role="alert">
          <span className="ae-safety-alert-icon" aria-hidden="true">
            <WarningIcon />
          </span>
          <div className="ae-safety-alert-text">
            <strong>{message.lines?.[0]}</strong>
            {message.lines?.[1] ? <span>{message.lines[1]}</span> : null}
          </div>
        </div>
      );
    case "notice":
      return (
        <div className="ae-info-notice" key={message.id}>
          {message.text}
        </div>
      );
    case "summary":
      return (
        <div className="ae-summary-card" key={message.id}>
          <h4 className="ae-summary-title">{message.text}</h4>
          <dl className="ae-summary-rows">
            {message.summary?.map((row) => (
              <div className="ae-summary-row" key={row.label}>
                <dt>{row.label}</dt>
                <dd>{row.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      );
    case "user":
      return (
        <div className="ae-turn ae-turn--user" key={message.id}>
          <div className="ae-msg-block ae-msg-block--user">
            <span className="ae-msg-name">You</span>
            <div className="ae-msg-body ae-msg-body--user">
              {message.text}
            </div>
          </div>
        </div>
      );
    default:
      return (
        <div className="ae-turn ae-turn--assistant" key={message.id}>
          <span className="ae-msg-avatar" aria-hidden="true">
            M
          </span>
          <div className="ae-msg-block">
            <span className="ae-msg-name">MedVaani</span>
            <div className="ae-msg-body ae-msg-body--assistant">
              {message.text}
            </div>
          </div>
        </div>
      );
  }
}

function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `ae-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// Initial language hint from the browser locale (safe in non-DOM contexts).
// The backend mirrors the session language after the first transcript, so
// this only affects the very first turn of a conversation.
function currentRequestLanguage(): string {
  return resolveRequestLanguage(
    typeof navigator !== "undefined" ? navigator.language : undefined,
  );
}

interface LiveBackendState {
  currentState: string;
  completed: boolean;
  humanReviewRequired: boolean;
  missing: string[];
  collected: AeCollectedInfo | null;
}

interface AeChatProps {
  initialMode?: "demo" | "live";
}

function AeChat({ initialMode = "demo" }: AeChatProps) {
  const [mode] = useState<"demo" | "live">(initialMode === "live" ? "live" : "demo");

  // Both the typed ("demo") and spoken (live) journeys run the same
  // authoritative backend AE workflow: typed turns go through
  // POST /text/conversation and recorded turns through /voice/conversation,
  // sharing one session so the approved question order, confirmations and
  // branches are identical. The mic stays disabled in typed mode.
  // The composer itself is mode-specific (composerControls): typed mode owns
  // the text field and the send button; voice-only mode hides the text input
  // entirely and offers only the microphone.
  const [draft, setDraft] = useState("");
  const [micNotice, setMicNotice] = useState(false);

  const [sessionId] = useState(newSessionId);
  const [liveMessages, setLiveMessages] =
    useState<AeChatMessage[]>(initialLiveMessages);
  const [liveStarted, setLiveStarted] = useState(false);
  const [liveBackend, setLiveBackend] = useState<LiveBackendState | null>(null);
  const [liveFinished, setLiveFinished] = useState(false);
  const [busy, setBusy] = useState(false);
  const [voiceStart, setVoiceStart] = useState<VoiceStartupState | null>(null);
  // Explicit language hint: defaults to the browser-language mapping and is
  // locked once the backend has established the conversation language (the
  // selector disables; the session itself is never reset).
  const [requestLanguage, setRequestLanguage] = useState(currentRequestLanguage);

  const scrollRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const liveIdRef = useRef(1);
  const startupRef = useRef<VoiceStartup | null>(null);
  const startupBegunRef = useRef(false);
  const greetingAudioRef = useRef<{ pause(): void } | null>(null);
  // Latest selection for the greeting request: read through a ref so the
  // one-shot startup effect never re-runs (its cleanup would pause a playing
  // introduction) while still picking up a change made during "loading".
  const requestLanguageRef = useRef(requestLanguage);
  requestLanguageRef.current = requestLanguage;

  const pushLive = useCallback((message: Omit<AeChatMessage, "id">) => {
    liveIdRef.current += 1;
    const entry: AeChatMessage = { ...message, id: `ae-live-${liveIdRef.current}` };
    setLiveMessages((prev) => [...prev, entry]);
  }, []);

  // Failed attempts show one clear error bubble; an identical consecutive
  // error never stacks a duplicate on top of itself.
  const pushErrorNotice = useCallback((text: string) => {
    setLiveMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "notice" && last.text === text) {
        return prev;
      }
      liveIdRef.current += 1;
      return [...prev, { id: `ae-live-${liveIdRef.current}`, role: "notice", text }];
    });
  }, []);

  const playAudio = useCallback((url: string) => {
    audioRef.current?.pause();
    const element = new Audio(url);
    audioRef.current = element;
    element.play().catch(() => undefined);
  }, []);

  // A manual mic tap or typed message supersedes the greeting: pause it and
  // cancel the startup so an in-flight greeting request can never start
  // speaking over the user's turn (nor auto-start a second recording).
  const takeOverGreeting = useCallback(() => {
    greetingAudioRef.current?.pause();
    startupRef.current?.cancel();
    setVoiceStart(null);
  }, []);

  // Shared envelope handling for both voice and text turns: the backend
  // response is identical apart from audio, so the UI renders them the same.
  const applyTurnResult = useCallback(
    (result: VoiceTurnResponse) => {
      setLiveStarted(true);
      setLiveBackend({
        currentState: result.current_state,
        completed: result.completed,
        humanReviewRequired: result.human_review_required,
        missing: result.missing_information ?? [],
        collected: result.ae?.collected_information ?? null,
      });

      pushLive({
        role: "user",
        text: result.transcript.trim() || "(No speech detected)",
      });

      // Step 11: the report is displayed with the existing report-summary
      // mechanism, immediately before the assistant's closing turn, for both
      // completion branches (case completed and urgent/human review).
      const reportReady = (result.completed || result.human_review_required);
      if (reportReady && !liveFinished) {
        setLiveFinished(true);
        pushLive({
          role: "summary",
          text: "Report summary",
          summary: buildLiveSummary(result.ae?.collected_information),
        });
      }

      // Visible text always matches the spoken form: the backend returns
      // English `response_text` plus the translated `spoken_response_text`
      // actually synthesized (all-or-nothing), so the screen and the audio
      // can never disagree after a language switch.
      if (result.safety_status === "urgent_life_threatening") {
        pushLive({
          role: "safety",
          lines: [
            assistantDisplayText(result),
            "Your report information can be preserved for human review.",
          ],
        });
      } else if (result.safety_status === "treatment_request_blocked") {
        pushLive({ role: "notice", text: assistantDisplayText(result) });
      } else {
        pushLive({ role: "assistant", text: assistantDisplayText(result) });
      }

      if (result.audio_path) {
        playAudio(audioUrlFor(result.audio_path));
      }
    },
    [liveFinished, playAudio, pushLive],
  );

  const sendRecording = useCallback(
    async (blob: Blob) => {
      setBusy(true);
      try {
        const result = await postVoiceTurn(sessionId, blob, requestLanguage);
        applyTurnResult(result);
      } catch (error) {
        const text =
          error instanceof Error
            ? error.message
            : "Something went wrong while sending your recording. Please try again.";
        pushErrorNotice(text);
      } finally {
        setBusy(false);
      }
    },
    [applyTurnResult, pushErrorNotice, requestLanguage, sessionId],
  );

  // Typed live turn through POST /text/conversation (same backend workflow).
  const sendTextTurn = useCallback(
    async (text: string) => {
      takeOverGreeting();
      setBusy(true);
      try {
        const result = await postTextTurn(sessionId, text, requestLanguage);
        applyTurnResult(result);
        setDraft("");
      } catch (error) {
        const text =
          error instanceof Error
            ? error.message
            : "Something went wrong while sending your message. Please try again.";
        pushErrorNotice(text);
      } finally {
        setBusy(false);
      }
    },
    [applyTurnResult, pushErrorNotice, requestLanguage, sessionId, takeOverGreeting],
  );

  const recorder = useWavRecorder({
    onStop: (blob) => {
      void sendRecording(blob);
    },
  });

  // Voice-first startup (live sessions only): fetch the approved greeting
  // and play it once. `startupBegunRef` guards React StrictMode's double
  // effect so the greeting is fetched and played exactly once per mount;
  // the seeded chat message above is the only display of the introduction
  // (the startup module never pushes messages), so rerenders and retries
  // cannot duplicate it. Playback NEVER starts the microphone: after the
  // introduction finishes the state settles on "ready" and recording waits
  // for an explicit mic-button click (handleMicClick below).
  useEffect(() => {
    if (mode !== "live" || startupBegunRef.current) {
      return;
    }
    startupBegunRef.current = true;
    const startup = createVoiceStartup({
      fetchGreeting: async () => {
        const greeting = await postGreeting(sessionId, requestLanguageRef.current);
        // Visible text must match what is spoken: when the backend localized
        // the introduction, swap the seeded English greeting for the spoken
        // form (all-or-nothing translation; English fallback stays English).
        if (greeting.language_translated && greeting.spoken_response_text) {
          const spoken = greeting.spoken_response_text;
          setLiveMessages((prev) =>
            prev.map((message) =>
              message.id === "ae-live-open" ? { ...message, text: spoken } : message,
            ),
          );
        }
        return greeting;
      },
      createAudio: (url) => {
        // `audio_path` is a filesystem path (output\conv_intro_*.wav); it
        // must go through audioUrlFor() to become a served /audio/ URL,
        // exactly like turn audio — a raw path would 404 and stay silent.
        const element = new Audio(audioUrlFor(url));
        greetingAudioRef.current = element;
        return {
          play: () => element.play(),
          pause: () => element.pause(),
          onEnded: (listener) =>
            element.addEventListener("ended", listener, { once: true }),
          onError: (listener) =>
            element.addEventListener("error", () => listener("load-error"), {
              once: true,
            }),
        };
      },
      onState: (next) => setVoiceStart(next),
    });
    startupRef.current = startup;
    void startup.begin();
    return () => {
      greetingAudioRef.current?.pause();
      greetingAudioRef.current = null;
    };
  }, [mode, sessionId]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [liveMessages]);

  useEffect(() => () => audioRef.current?.pause(), []);

  const handleSend = () => {
    const text = draft.trim();
    if (!text) return;
    // Both modes share the backend text workflow (POST /text/conversation);
    // typed turns never consume speech credits and follow the same approved
    // question order as recorded turns.
    if (busy) return;
    void sendTextTurn(text);
  };

  const handleMicClick = () => {
    // The mic button is the ONLY action that starts recording (the startup
    // module has no microphone port); the shared decision helper keeps the
    // precedence testable: demo notice, busy ignore, stop/sent, else start.
    const action = micClickAction({
      demoMode: mode === "demo",
      busy,
      recording: recorder.recording,
    });
    if (action === "demo-notice") {
      setMicNotice(true);
      return;
    }
    if (action === "ignore") return;
    if (action === "stop") {
      recorder.stop();
      return;
    }
    takeOverGreeting();
    void recorder.start();
  };

  const demoMode = mode === "demo";
  const composer = composerControls({ mode });
  const liveStates = liveStageStates(
    liveBackend?.collected,
    liveBackend?.completed ?? false,
    liveBackend?.humanReviewRequired ?? false,
    liveStarted,
  );

  const stageState = (stageId: (typeof AE_STAGES)[number]["id"]) =>
    liveStates[stageId];

  const captured = [
    ...(liveStates["what-happened"] === "done"
      ? ["Suspected product", "Reported symptoms"]
      : []),
    ...AE_STAGES.slice(1)
      .filter((stage) => liveStates[stage.id] === "done")
      .map((stage) => stage.label),
  ];

  const stillNeeded = missingLabels(liveBackend?.missing ?? []);

  const hint = demoMode ? (
    micNotice ? (
      <span className="ae-composer-hint--notice">
        Voice capture isn&apos;t enabled yet — you can type for now.
      </span>
    ) : (
      <span>Type or use your voice</span>
    )
  ) : recorder.error ? (
    <span className="ae-composer-hint--notice">{recorder.error}</span>
  ) : voiceStart === "playing" ? (
    <span>The introduction is playing — tap the mic when you&apos;re ready</span>
  ) : recorder.recording ? (
    <span className="ae-composer-hint--recording">
      Recording… tap the mic again to send (auto-stops at 60s)
    </span>
  ) : busy ? (
    <span>Please wait…</span>
  ) : (
    <span>Tap the mic to record (max 60 seconds)</span>
  );

  // Visible states for the greeting: loading, autoplay blocked (manual
  // play fallback), and greeting-audio failure. Every fallback button here
  // only plays/retries the introduction — none of them start the
  // microphone; recording begins exclusively at the mic button below.
  const resumeStartup = () => {
    void startupRef.current?.resume();
  };

  const renderVoiceStart = () => {
    if (demoMode || !voiceStart) return null;
    if (voiceStart === "loading") {
      return (
        <div className="ae-voice-start ae-voice-start--loading" role="status">
          Preparing the introduction…
        </div>
      );
    }
    if (voiceStart === "blocked") {
      return (
        <div className="ae-voice-start" role="alert">
          <span className="ae-voice-start-text">
            Your browser blocked automatic playback. Hear the introduction
            when you&apos;re ready — the mic stays off until you tap it.
          </span>
          <button
            type="button"
            className="ae-voice-start-btn"
            onClick={resumeStartup}
          >
            Hear the introduction
          </button>
        </div>
      );
    }
    if (voiceStart === "audio-error") {
      return (
        <div className="ae-voice-start" role="alert">
          <span className="ae-voice-start-text">
            The introduction audio couldn&apos;t be played. You can retry, or
            use the mic button when you&apos;re ready.
          </span>
          <button
            type="button"
            className="ae-voice-start-btn"
            onClick={resumeStartup}
          >
            Try again
          </button>
        </div>
      );
    }
    if (voiceStart === "unavailable") {
      return (
        <div className="ae-voice-start" role="alert">
          <span className="ae-voice-start-text">
            The spoken introduction isn&apos;t available right now — it&apos;s
            shown above.
          </span>
          <button
            type="button"
            className="ae-voice-start-btn"
            onClick={resumeStartup}
          >
            Try again
          </button>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="container ae-convo">
      <section className="ae-convo-main" aria-label="Conversation">
        <div className="ae-convo-head">
          <div className="ae-convo-head-text">
            <h2 className="ae-convo-title">Report a Side Effect</h2>
            <p className="ae-convo-sub">
              MedVaani will guide you through a few questions to document what
              happened.
            </p>
          </div>
          <span className="ae-product-pill">Lantus® SoloStar®</span>
        </div>

        <div
          className="ae-convo-messages"
          ref={scrollRef}
          aria-live="polite"
        >
          {liveMessages.map(renderMessage)}
          {busy ? (
            <div className="ae-info-notice">
              MedVaani is processing your message…
            </div>
          ) : null}
        </div>

        <div className="ae-composer-zone">
          {renderVoiceStart()}
          <div className="ae-composer-card">
            {composer.textInput ? (
              <input
                type="text"
                value={draft}
                placeholder="Describe what happened…"
                aria-label="Describe what happened"
                autoComplete="off"
                disabled={busy}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") handleSend();
                }}
              />
            ) : null}
            {composer.micButton ? (
              <button
                type="button"
                className={
                  recorder.recording
                    ? "ae-mic-btn ae-mic-btn--recording"
                    : "ae-mic-btn"
                }
                aria-label={
                  recorder.recording
                    ? "Stop recording and send"
                    : "Start recording"
                }
                title={
                  recorder.recording
                    ? "Tap to send your recording"
                    : "Tap to record (max 60 seconds)"
                }
                disabled={busy}
                onClick={handleMicClick}
              >
                <MicIcon />
              </button>
            ) : null}
            {composer.sendButton ? (
              <button
                type="button"
                className="ae-send-btn"
                aria-label="Send message"
                onClick={handleSend}
                disabled={draft.trim().length === 0 || busy}
              >
                <SendIcon />
              </button>
            ) : null}
          </div>
          <div className="ae-lang-picker">
            <label className="ae-lang-picker-label" htmlFor="ae-language-select">
              Report language
            </label>
            <select
              id="ae-language-select"
              className="ae-lang-picker-select"
              value={requestLanguage}
              disabled={languageSelectDisabled({ liveStarted, busy })}
              onChange={(event) => setRequestLanguage(event.target.value)}
            >
              {SUPPORTED_LANGUAGES.map((option) => (
                <option key={option.code} value={option.code}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <p className="ae-composer-lang-note">
            For your comfort, you may describe your experience in the language
            you are most comfortable speaking.
          </p>
          <div className="ae-composer-hint">{hint}</div>
        </div>

        <div className="ae-convo-safety" role="note">
          <WarningIcon />
          <span>
            If you are experiencing a serious or life-threatening reaction,
            seek immediate medical attention.
          </span>
        </div>

        <p className="ae-convo-disclaimer">
          MedVaani helps document safety reports and provides product
          information from referenced product documents. It does not diagnose
          conditions or provide treatment advice.
        </p>
      </section>

      <aside className="ae-report-card" aria-label="Report progress">
        <div className="ae-report-pen-block">
          <span className="ae-report-pen-name">Lantus® SoloStar®</span>
          <img
            src={lantusPen}
            alt="Lantus SoloStar pen"
            className="ae-report-pen"
          />
        </div>

        <div className="ae-report-head">
          <h3 className="ae-report-title">Report progress</h3>
          <p className="ae-report-sub">Building your safety report</p>
        </div>

        <ol className="ae-stages">
          {AE_STAGES.map((stage) => {
            const state = stageState(stage.id);
            return (
              <li
                key={stage.id}
                className={`ae-stage ae-stage--${state}`}
              >
                <span className="ae-stage-node" aria-hidden="true">
                  {state === "done" ? (
                    <CheckIcon />
                  ) : state === "active" ? (
                    <span className="ae-stage-dot" />
                  ) : null}
                </span>
                <div className="ae-stage-text">
                  <div className="ae-stage-label">{stage.label}</div>
                  <div className="ae-stage-status">
                    {stageStatusText(stage.id, state)}
                  </div>
                </div>
              </li>
            );
          })}
        </ol>

        <div className="ae-info-block ae-info-block--captured">
          <h4 className="ae-info-title">Information captured</h4>
          <ul className="ae-info-list">
            {captured.map((item) => (
              <li key={item}>
                <span className="ae-info-check" aria-hidden="true">
                  <CheckIcon size={15} />
                </span>
                {item}
              </li>
            ))}
          </ul>
        </div>

        <div className="ae-info-block ae-info-block--needed">
          <h4 className="ae-info-title">Still needed</h4>
          <ul className="ae-info-list">
            {stillNeeded.map((item) => (
              <li key={item}>
                <span className="ae-info-bullet" aria-hidden="true" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}

export default AeChat;
