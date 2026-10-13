export type AeStageId =
  | "what-happened"
  | "patient-details"
  | "reporter-details"
  | "contact-information"
  | "consent"
  | "event-timing"
  | "seriousness"
  | "outcome"
  | "review";

export type AeStageState = "done" | "active" | "todo";

export interface AeStage {
  id: AeStageId;
  label: string;
}

export const AE_STAGES: AeStage[] = [
  { id: "what-happened", label: "What happened" },
  { id: "reporter-details", label: "Reporter details" },
  { id: "patient-details", label: "Patient details" },
  { id: "consent", label: "Consent" },
  { id: "contact-information", label: "Contact information" },
  { id: "event-timing", label: "Event timing" },
  { id: "seriousness", label: "Seriousness" },
  { id: "outcome", label: "Outcome" },
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

export interface AeStep {
  id: string;
  stage: AeStageId;
  question: string;
}

export const AE_STEPS: AeStep[] = [
  {
    id: "reporter",
    stage: "reporter-details",
    question:
      "Are you the person who experienced this reaction, or are you reporting it for someone else?",
  },
  {
    id: "age",
    stage: "patient-details",
    question: "How old is the person this happened to?",
  },
  {
    id: "consent",
    stage: "consent",
    question:
      "If we need a little more information later, is it alright if we contact you?",
  },
  {
    id: "method",
    stage: "contact-information",
    question: "Would you prefer to be contacted by phone or by email?",
  },
  {
    id: "contact",
    stage: "contact-information",
    question: "What is the best phone number or email address to reach you?",
  },
  {
    id: "onset",
    stage: "event-timing",
    question: "When did the side effect first start?",
  },
  {
    id: "seriousness",
    stage: "seriousness",
    question: "Would you consider this problem serious?",
  },
  {
    id: "medical_care",
    stage: "seriousness",
    question:
      "Have you spoken to a doctor or received medical care for it?",
  },
  {
    id: "outcome",
    stage: "outcome",
    question:
      "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?",
  },
];

export const OPENING_MESSAGE =
  "I'm here to help you report a side effect. Tell me what happened in your own words.";

export const DEMO_EVENT_INPUT =
  "I took Lantus yesterday and since this morning I have a severe headache and vomiting.";

export const DEMO_EVENT_CONFIRMATION =
  "Thank you for sharing that. For this demo, I've noted your report about Lantus SoloStar with severe headache and vomiting.";

export const GENERIC_EVENT_CONFIRMATION =
  "Thank you for sharing that. For this demo, I've noted your description exactly as you gave it.";

export const URGENT_SAFETY_TITLE =
  "Please seek immediate medical attention or emergency care for a serious or potentially life-threatening reaction.";

export const URGENT_SAFETY_SUB =
  "Your report information can be preserved for human review.";

export const DOSING_SAFE_RESPONSE =
  "MedVaani can't recommend a personal insulin dose or medication change. Please consult a qualified healthcare professional for personal medical guidance.";

export const READY_FOR_REVIEW =
  "Your safety report is ready for review.";

export const FINAL_MESSAGE =
  "MedVaani helps document safety reports. A qualified human reviewer is responsible for reviewing the report and determining any required follow-up.";

export const FINISHED_REPLY =
  "Your safety report is already complete and ready for human review.";

export interface AeConversationState {
  stepIndex: number;
  started: boolean;
  finished: boolean;
  urgent: boolean;
  answers: Record<string, string>;
  messages: AeChatMessage[];
  idCounter: number;
}

export function createConversation(): AeConversationState {
  return {
    stepIndex: -1,
    started: false,
    finished: false,
    urgent: false,
    answers: {},
    messages: [
      { id: "ae-open", role: "assistant", text: OPENING_MESSAGE },
    ],
    idCounter: 1,
  };
}

const URGENT_PATTERNS: string[] = [
  "difficulty breathing",
  "trouble breathing",
  "trouble catching my breath",
  "can't breathe",
  "can't catch my breath",
  "cannot breathe",
  "shortness of breath",
  "hard time breathing",
  "severe allergic",
  "anaphyla",
  "throat closing",
  "swelling of my face",
  "swelling of my lips",
  "swelling of my tongue",
  "swelling of my throat",
  "lip swelling",
  "tongue swelling",
  "facial swelling",
  "chest pain",
  "chest tightness",
  "fainting",
  "fainted",
  "unconscious",
  "unresponsive",
  "seizure",
  "convuls",
  "heart attack",
  "cardiac arrest",
  "stroke",
  "overdose",
];

const DOSING_PATTERNS: string[] = [
  "increase my lantus",
  "decrease my lantus",
  "increase my dose",
  "decrease my dose",
  "lower my dose",
  "raise my dose",
  "adjust my dose",
  "adjust my insulin",
  "change my dose",
  "change my insulin",
  "change my medication",
  "change my lantus",
  "more units",
  "how many units",
  "should i take more",
  "should i take less",
  "dose",
  "dosage",
  "medication change",
];

const meansYes = (value: string | undefined): boolean => {
  const t = (value ?? "").trim().toLowerCase();
  return /^(yes|yeah|yep|yup|i do|i consent|sure|of course|absolutely|correct|right|ok|okay)\b/.test(t);
};

const meansNo = (value: string | undefined): boolean => {
  const t = (value ?? "").trim().toLowerCase();
  return /^(no|nope|not really|i don't|i do not|no i)/.test(t);
};

const lastStepIndexFor = (stageId: AeStageId): number =>
  AE_STEPS.reduce(
    (acc, step, index) => (step.stage === stageId ? index : acc),
    -1,
  );

const firstStepIndexFor = (stageId: AeStageId): number => {
  const index = AE_STEPS.findIndex((step) => step.stage === stageId);
  return index;
};

export function stageStateFor(
  state: AeConversationState,
  stageId: AeStageId,
): AeStageState {
  if (stageId === "what-happened") {
    return state.started ? "done" : "active";
  }
  if (state.finished) return "done";
  const first = firstStepIndexFor(stageId);
  const last = lastStepIndexFor(stageId);
  const current = Math.max(state.stepIndex, 0);
  if (state.started && last !== -1 && current > last) return "done";
  if (state.started && first !== -1 && current >= first && current <= last) {
    return "active";
  }
  return "todo";
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

const push = (
  state: AeConversationState,
  message: Omit<AeChatMessage, "id">,
): AeConversationState => {
  const idCounter = state.idCounter + 1;
  return {
    ...state,
    idCounter,
    messages: [
      ...state.messages,
      { ...message, id: `ae-msg-${idCounter}` },
    ],
  };
};

const currentQuestion = (state: AeConversationState): string => {
  if (!state.started) return OPENING_MESSAGE;
  if (state.finished) return FINISHED_REPLY;
  return AE_STEPS[state.stepIndex].question;
};

const isDemoEventInput = (text: string): boolean => {
  const t = text.toLowerCase();
  return (
    t.includes("lantus") &&
    t.includes("headache") &&
    t.includes("vomiting")
  );
};

// Answers that cannot be an onset date/time ("No", "n/a", "don't know"...).
const NON_ANSWER_ONSET =
  /^(no|nope|yes|yeah|yup|n\/?a|none|nothing|unknown|unsure|not\s+sure|don'?t\s+know|do\s+not\s+know|no\s+idea|na)\b/i;

// Signals that an answer actually carries temporal information.
const ONSET_TEMPORAL =
  /(today|yesterday|tomorrow|morning|afternoon|evening|night|noon|o'?clock|\d|hour|hr\b|day|week|month|year|ago|since|after|before|started|began|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)/i;

// Present an onset answer only when it plausibly is a date/time; anything
// missing, refused or non-temporal is shown as "Not provided" instead of
// echoing a raw, misleading answer (e.g. "No").
const formatOnsetAnswer = (answer: string | undefined): string => {
  const raw = (answer ?? "").trim();
  if (!raw || NON_ANSWER_ONSET.test(raw) || !ONSET_TEMPORAL.test(raw)) {
    return "Not provided";
  }
  return raw.charAt(0).toUpperCase() + raw.slice(1);
};

const buildSummary = (state: AeConversationState): AeSummaryRow[] => {
  const self = meansYes(state.answers["reporter"]);
  const consentYes = meansYes(state.answers["consent"]);
  const noConsent = meansNo(state.answers["consent"]);
  const reportedEvents = (state.answers["event"] ?? "").trim() || "Not provided";
  return [
    { label: "Product", value: "Lantus® SoloStar®" },
    { label: "Reported events", value: reportedEvents },
    { label: "Onset", value: formatOnsetAnswer(state.answers["onset"]) },
    { label: "Reporter", value: self ? "Patient" : "Reporter" },
    {
      label: "Consent",
      value: consentYes ? "Yes" : noConsent ? "No" : "Yes",
    },
    { label: "Status", value: "Ready for human review (demo information)" },
  ];
};

const finishJourney = (state: AeConversationState): AeConversationState => {
  const finished: AeConversationState = {
    ...state,
    stepIndex: AE_STEPS.length,
    finished: true,
  };
  return push(
    push(
      push(
        finished,
        { role: "assistant", text: READY_FOR_REVIEW },
      ),
      {
        role: "summary",
        text: "Report summary (demo information)",
        summary: buildSummary(finished),
      },
    ),
    { role: "assistant", text: FINAL_MESSAGE },
  );
};

export function handleUserInput(
  state: AeConversationState,
  input: string,
): AeConversationState {
  const text = input.trim();
  let next = push(state, { role: "user", text });

  if (state.finished) {
    return push(next, {
      role: "assistant",
      text: FINISHED_REPLY,
    });
  }

  const t = text.toLowerCase();
  const urgent = URGENT_PATTERNS.some((pattern) => t.includes(pattern));
  if (urgent) {
    return push(
      {
        ...next,
        urgent: true,
      },
      {
        role: "safety",
        lines: [URGENT_SAFETY_TITLE, URGENT_SAFETY_SUB],
      },
    );
  }

  const dosing = DOSING_PATTERNS.some((pattern) => t.includes(pattern));
  if (dosing) {
    return push(
      push(next, {
        role: "notice",
        text: DOSING_SAFE_RESPONSE,
      }),
      {
        role: "assistant",
        text: currentQuestion(next),
      },
    );
  }

  if (!state.started) {
    const confirmation = isDemoEventInput(text)
      ? DEMO_EVENT_CONFIRMATION
      : GENERIC_EVENT_CONFIRMATION;
    next = {
      ...next,
      started: true,
      stepIndex: 0,
      // Preserve the reporter's own event description for the final summary.
      answers: { ...state.answers, event: text },
    } as AeConversationState;
    // The confirmation acknowledges the event only; the reporter question
    // is asked exactly once, by AE_STEPS[0] right below.
    return push(
      push(next, {
        role: "assistant",
        text: confirmation,
      }),
      {
        role: "assistant",
        text: AE_STEPS[0].question,
      },
    );
  }

  const currentStep = AE_STEPS[state.stepIndex];
  next = {
    ...next,
    answers: { ...state.answers, [currentStep.id]: text },
  } as AeConversationState;

  let nextIndex = state.stepIndex + 1;
  if (currentStep.id === "consent" && meansNo(text)) {
    // Contact consent declined: skip the contact-method and contact-detail
    // questions entirely instead of asking for details the person refused.
    while (
      nextIndex < AE_STEPS.length &&
      (AE_STEPS[nextIndex].id === "method" ||
        AE_STEPS[nextIndex].id === "contact")
    ) {
      nextIndex += 1;
    }
  }
  if (nextIndex >= AE_STEPS.length) {
    return finishJourney(next);
  }

  next = {
    ...next,
    stepIndex: nextIndex,
    urgent: false,
  } as AeConversationState;

  return push(next, {
    role: "assistant",
    text: AE_STEPS[nextIndex].question,
  });
}