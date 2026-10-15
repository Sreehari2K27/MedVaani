import { useCallback, useEffect, useRef, useState } from "react";
import lantusPen from "../../assets/lantus-solostar.png.png";
import {
  AE_STAGES,
  OPENING_MESSAGE,
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
  audioUrlFor,
  postTextTurn,
  postVoiceTurn,
  resolveRequestLanguage,
  type VoiceTurnResponse,
} from "../../api/voiceClient";
import { useWavRecorder } from "../../hooks/useWavRecorder";

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
  const [draft, setDraft] = useState("");
  const [micNotice, setMicNotice] = useState(false);

  const [sessionId] = useState(newSessionId);
  const [liveMessages, setLiveMessages] = useState<AeChatMessage[]>([
    { id: "ae-live-open", role: "assistant", text: OPENING_MESSAGE },
  ]);
  const [liveStarted, setLiveStarted] = useState(false);
  const [liveBackend, setLiveBackend] = useState<LiveBackendState | null>(null);
  const [liveFinished, setLiveFinished] = useState(false);
  const [busy, setBusy] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const liveIdRef = useRef(1);

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

      if (result.safety_status === "urgent_life_threatening") {
        pushLive({
          role: "safety",
          lines: [
            result.response_text,
            "Your report information can be preserved for human review.",
          ],
        });
      } else if (result.safety_status === "treatment_request_blocked") {
        pushLive({ role: "notice", text: result.response_text });
      } else {
        pushLive({ role: "assistant", text: result.response_text });
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
        const result = await postVoiceTurn(sessionId, blob, currentRequestLanguage());
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
    [applyTurnResult, pushErrorNotice, sessionId],
  );

  // Typed live turn through POST /text/conversation (same backend workflow).
  const sendTextTurn = useCallback(
    async (text: string) => {
      setBusy(true);
      try {
        const result = await postTextTurn(sessionId, text, currentRequestLanguage());
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
    [applyTurnResult, pushErrorNotice, sessionId],
  );

  const recorder = useWavRecorder({
    onStop: (blob) => {
      void sendRecording(blob);
    },
  });

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
    if (mode === "demo") {
      setMicNotice(true);
      return;
    }
    if (busy) return;
    if (recorder.recording) {
      recorder.stop();
      return;
    }
    void recorder.start();
  };

  const demoMode = mode === "demo";
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
  ) : recorder.recording ? (
    <span className="ae-composer-hint--recording">
      Recording… tap the mic again to send (auto-stops at 60s)
    </span>
  ) : busy ? (
    <span>Please wait…</span>
  ) : (
    <span>Type your message, or tap the mic to record (max 60s)</span>
  );

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
          <div className="ae-composer-card">
            <input
              type="text"
              value={draft}
              placeholder={
                demoMode
                  ? "Describe what happened…"
                  : "Type your message…"
              }
              aria-label="Describe what happened"
              autoComplete="off"
              disabled={busy}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") handleSend();
              }}
            />
            {!demoMode ? (
              <button
                type="button"
                className={
                  !demoMode && recorder.recording
                    ? "ae-mic-btn ae-mic-btn--recording"
                    : "ae-mic-btn"
                }
                aria-label={
                  demoMode
                    ? "Voice input"
                    : recorder.recording
                      ? "Stop recording and send"
                      : "Start recording"
                }
                title={
                  demoMode
                    ? "Voice capture is not enabled in this prototype yet"
                    : recorder.recording
                      ? "Tap to send your recording"
                      : "Tap to record (max 60 seconds)"
                }
                disabled={!demoMode && busy}
                onClick={handleMicClick}
              >
                <MicIcon />
              </button>
            ) : null}
            {demoMode ? (
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
