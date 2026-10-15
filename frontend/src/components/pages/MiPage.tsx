import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { resolveRequestLanguage } from "../../api/voiceClient";
import {
  MiApiError,
  postMiChat,
  postMiTranscribe,
  type MiChatResponse,
} from "../../api/miClient";
import {
  MI_RECORDING_MAX_SECONDS,
  WELCOME_EXAMPLES,
  canSubmit,
  composerKeyAction,
  splitCitationSegments,
  usableTranscript,
} from "../../mi/miChat";
import { useWavRecorder } from "../../hooks/useWavRecorder";
import lantusPen from "../../assets/lantus-solostar.png.png";
import "../../mi-styles.css";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  urgent?: boolean;
  suggestAe?: boolean;
}

function MicIcon({ active }: { active: boolean }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke={active ? "#b91c1c" : "#475569"}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
      <path d="M19 10v1a7 7 0 0 1-14 0v-1"></path>
      <line x1="12" y1="19" x2="12" y2="22"></line>
    </svg>
  );
}

function MiPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);

  const nextIdRef = useRef(1);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const pushMessage = useCallback((message: Omit<ChatMessage, "id">) => {
    setMessages((prev) => [...prev, { ...message, id: nextIdRef.current++ }]);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy, error, notice]);

  const sendQuestion = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || busy) return;
      setBusy(true);
      setError(null);
      setNotice(null);
      setPendingQuestion(trimmed);
      try {
        const result: MiChatResponse = await postMiChat(trimmed);
        pushMessage({ role: "user", text: trimmed });
        pushMessage({
          role: "assistant",
          text: result.answer,
          urgent: result.urgent,
          suggestAe: result.suggest_ae_report,
        });
        setDraft("");
        setPendingQuestion(null);
      } catch (err) {
        const text =
          err instanceof MiApiError
            ? err.message
            : "Something went wrong while fetching the answer. Please try again.";
        setError(text);
      } finally {
        setBusy(false);
      }
    },
    [busy, pushMessage],
  );

  const retryLastQuestion = useCallback(() => {
    if (pendingQuestion) void sendQuestion(pendingQuestion);
  }, [pendingQuestion, sendQuestion]);

  const submitDraft = useCallback(() => {
    if (canSubmit(draft, busy || transcribing)) void sendQuestion(draft);
  }, [busy, draft, sendQuestion, transcribing]);

  const handleComposerKeyDown = (
    event: ReactKeyboardEvent<HTMLTextAreaElement>,
  ) => {
    const action = composerKeyAction(event.key, event.shiftKey, busy || transcribing);
    if (action === "submit") {
      event.preventDefault();
      submitDraft();
    }
    // "newline": let the browser insert the newline; "none": no action.
  };

  const onRecordingStop = useCallback(
    (blob: Blob) => {
      void (async () => {
        setTranscribing(true);
        setError(null);
        setNotice(null);
        try {
          const result = await postMiTranscribe(
            blob,
            resolveRequestLanguage(),
          );
          const transcript = usableTranscript(result.transcript);
          if (!transcript) {
            setNotice(
              "No speech was recognized in that recording. Please try again, or type your question.",
            );
            return;
          }
          // The transcript is placed in the composer for review - it is
          // never submitted automatically.
          setDraft((prev) => (prev.trim() ? `${prev.trim()} ${transcript}` : transcript));
          setNotice("Recognized speech is in the text box. Review it, then press Send.");
        } catch (err) {
          const text =
            err instanceof MiApiError
              ? err.message
              : "Couldn't transcribe the recording. Please try again.";
          setError(text);
        } finally {
          setTranscribing(false);
        }
      })();
    },
    [],
  );

  const recorder = useWavRecorder({
    maxSeconds: MI_RECORDING_MAX_SECONDS,
    onStop: onRecordingStop,
    onLimitReached: () =>
      setNotice(
        `Recording stopped — the maximum is ${MI_RECORDING_MAX_SECONDS} seconds.`,
      ),
  });

  const toggleRecording = useCallback(() => {
    if (recorder.recording) {
      recorder.stop();
      return;
    }
    setNotice(null);
    setError(null);
    void recorder.start();
  }, [recorder]);

  const fillExample = useCallback((example: string) => {
    setDraft(example);
    setError(null);
    setNotice(null);
  }, []);

  const micBusy = busy || transcribing;

  return (
    <section className="mi-page">
      <div className="mi-head">
        <div className="mi-head-text">
          <h1>Medical Information</h1>
          <p className="mi-sub">
            Grounded answers about Lantus® SoloStar® from the approved product
            information.
          </p>
        </div>
        <div className="mi-head-right">
          <img
            className="mi-product-img"
            src={lantusPen}
            alt="Lantus SoloStar insulin pen"
          />
        </div>
      </div>

      <div className="mi-thread" ref={scrollRef} aria-live="polite">
        {messages.length === 0 ? (
          <div className="mi-welcome">
            <p>
              Ask anything about Lantus® SoloStar® — uses, storage, how it
              works, or its adverse reactions. Answers are shown as text and
              cite the package insert.
            </p>
            <p className="mi-welcome-label">Try one of these:</p>
            <div className="mi-examples">
              {WELCOME_EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="mi-example"
                  onClick={() => fillExample(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`mi-msg mi-msg--${message.role}${
              message.urgent ? " mi-msg--safety" : ""
            }`}
          >
            <div className="mi-msg-text">
              {splitCitationSegments(message.text).map((segment, index) =>
                segment.citation ? (
                  <span key={index} className="mi-citation">
                    {segment.text}
                  </span>
                ) : (
                  <span key={index}>{segment.text}</span>
                ),
              )}
            </div>
            {message.suggestAe ? (
              <div className="mi-ae-hint">
                Looking to report a side effect? Return Home and choose
                “Report a Side Effect” to start a report.
              </div>
            ) : null}
          </div>
        ))}

        {busy ? (
          <div className="mi-msg mi-msg--assistant mi-msg--loading">
            Preparing an answer from the product information…
          </div>
        ) : null}
      </div>

      {error ? (
        <div className="mi-error" role="alert">
          <span>{error}</span>
          {pendingQuestion ? (
            <button type="button" className="mi-retry" onClick={retryLastQuestion}>
              Retry
            </button>
          ) : null}
        </div>
      ) : null}
      {notice ? <div className="mi-notice">{notice}</div> : null}
      {recorder.error ? (
        <div className="mi-error" role="alert">
          <span>{recorder.error}</span>
          <button
            type="button"
            className="mi-retry"
            onClick={() => recorder.dismissError()}
          >
            Dismiss
          </button>
        </div>
      ) : null}

      <div className="mi-composer">
        <textarea
          className="mi-input"
          rows={2}
          value={draft}
          placeholder="Ask about Lantus SoloStar… (Enter to send, Shift+Enter for a new line)"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleComposerKeyDown}
          disabled={micBusy}
          aria-label="Your question"
        />
        <div className="mi-composer-actions">
          <button
            type="button"
            className={`mi-mic${recorder.recording ? " mi-mic--active" : ""}`}
            onClick={toggleRecording}
            disabled={micBusy && !recorder.recording}
            aria-pressed={recorder.recording}
            aria-label={recorder.recording ? "Stop recording" : "Speak your question"}
          >
            <MicIcon active={recorder.recording} />
            <span>
              {recorder.recording
                ? "Listening… tap to stop"
                : transcribing
                  ? "Transcribing…"
                  : "Speak"}
            </span>
          </button>
          <button
            type="button"
            className="mi-send"
            onClick={submitDraft}
            disabled={!canSubmit(draft, micBusy)}
          >
            Send
          </button>
        </div>
      </div>

      <p className="mi-disclaimer">
        MedVaani provides general product information about Lantus® SoloStar®
        from the approved package insert. It is not personalized medical
        advice, a diagnosis, or treatment guidance, and it is not a substitute
        for the approved prescribing information. For decisions about your
        care, consult a qualified healthcare professional.
      </p>
    </section>
  );
}

export default MiPage;
