// Stage model for the AE conversation sidebar. The conversation itself is
// driven by the shared backend AE workflow (POST /voice/conversation and
// POST /text/conversation) for both typed and spoken turns, so there is no
// separate client-side question sequence: the approved question order,
// confirmations and branches all live in the backend workflow.

export type AeStageId =
  | "what-happened"
  | "patient-details"
  | "reporter-details"
  | "contact-information"
  | "consent"
  | "seriousness"
  | "event-timing"
  | "review";

export type AeStageState = "done" | "active" | "todo";

export interface AeStage {
  id: AeStageId;
  label: string;
}

// The approved flow collects identity and contact details, then consent,
// then screens severity before the incident date, and finally prepares the
// report for human review.
export const AE_STAGES: AeStage[] = [
  { id: "what-happened", label: "What happened" },
  { id: "reporter-details", label: "Reporter details" },
  { id: "patient-details", label: "Patient details" },
  { id: "contact-information", label: "Contact information" },
  { id: "consent", label: "Consent" },
  { id: "seriousness", label: "Seriousness" },
  { id: "event-timing", label: "Event timing" },
  { id: "review", label: "Review" },
];

export type AeChatRole =
  | "assistant"
  | "user"
  | "safety"
  | "notice"
  | "summary";

export interface AeSummaryRow {
  label: string;
  value: string;
}

export interface AeChatMessage {
  id: string;
  role: AeChatRole;
  text?: string;
  lines?: string[];
  summary?: AeSummaryRow[];
}

export function stageStatusText(
  stageId: AeStageId,
  state: AeStageState,
): string {
  if (state === "done") {
    if (stageId === "what-happened") return "Side effect described";
    if (stageId === "review") return "Ready for human review";
    return "Completed";
  }
  if (state === "active") return "In progress";
  return "Not started";
}

// The fixed opening (Step 1 introduction + Step 2 invitation): shown once
// per session, and never repeated after the person has described the event.
export const OPENING_MESSAGE =
  "Thank you for taking the time to speak with us. I'm here to help record the event for review. I'm not a substitute for a healthcare professional, and I can't provide medical advice. Please tell me what happened, in your own words.";

// The single greeting message seeded into a fresh session's chat. AeChat
// passes this to its state initializer (runs once per mount), and the
// voice-startup module never pushes another copy — the introduction is
// displayed exactly once however the greeting request or playback behaves.
export function initialLiveMessages(): AeChatMessage[] {
  return [{ id: "ae-live-open", role: "assistant", text: OPENING_MESSAGE }];
}
