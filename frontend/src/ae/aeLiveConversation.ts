// Mapping between the backend /voice/conversation response and the AE chat
// UI (stage sidebar, "still needed" list, completion summary). Pure helpers;
// the demo journey in aeDemoConversation.ts is untouched.

import {
  AE_STAGES,
  type AeStageId,
  type AeStageState,
  type AeSummaryRow,
} from "./aeDemoConversation";

export type AeCollectedInfo = Record<string, unknown>;

const asString = (value: unknown): string | null =>
  typeof value === "string" && value.trim() ? value.trim() : null;

const STAGE_DONE: Record<AeStageId, (info: AeCollectedInfo, completed: boolean) => boolean> = {
  "what-happened": (info) => asString(info.adverse_event) !== null || asString(info.product) !== null,
  "patient-details": (info) =>
    asString(info.patient_identifier) !== null && asString(info.patient_age) !== null,
  "reporter-details": (info) =>
    asString(info.reporter_role) !== null || asString(info.reporter_name) !== null,
  "contact-information": (info) =>
    asString(info.contact_phone) !== null || asString(info.contact_email) !== null,
  consent: (info) => info.consent_to_contact === true,
  "event-timing": (info) => asString(info.onset_date) !== null,
  seriousness: (info) => info.seriousness === "yes" || info.seriousness === "no",
  review: (_info, completed) => completed,
};

export function liveStageStates(
  collected: AeCollectedInfo | null | undefined,
  completed: boolean,
  humanReviewRequired: boolean,
  started: boolean,
): Record<AeStageId, AeStageState> {
  const info = collected ?? {};
  const states = {} as Record<AeStageId, AeStageState>;
  let activeAssigned = false;
  for (const stage of AE_STAGES) {
    if (started && STAGE_DONE[stage.id](info, completed)) {
      states[stage.id] = "done";
      continue;
    }
    if (!activeAssigned || (stage.id === "review" && humanReviewRequired && started)) {
      states[stage.id] = "active";
      activeAssigned = true;
    } else {
      states[stage.id] = "todo";
    }
  }
  return states;
}

const MISSING_LABELS: Record<string, string> = {
  product: "Suspected product",
  adverse_event: "Reported symptoms",
  patient_identifier: "Patient identifier",
  patient_age: "Patient age",
  reporter_role: "Reporter details",
  reporter_name: "Reporter name",
  contact_method: "Contact preference",
  contact_phone: "Contact information",
  contact_email: "Contact information",
  "contact_phone or contact_email": "Contact information",
  consent_to_contact: "Consent",
  onset_date: "Event timing",
  medical_care: "Medical care",
  seriousness: "Seriousness",
};

export function missingLabels(keys: string[]): string[] {
  return keys.map((key) => MISSING_LABELS[key] ?? key);
}

export function buildLiveSummary(collected: AeCollectedInfo | null | undefined): AeSummaryRow[] {
  const info = collected ?? {};
  const rows: AeSummaryRow[] = [];
  const product = asString(info.product);
  const events = asString(info.adverse_event);
  const identifier = asString(info.patient_identifier);
  const age = asString(info.patient_age);
  const onset = asString(info.onset_date);
  const onsetDisplay = asString(info.onset_date_display);
  const onsetPeriod = asString(info.onset_period);
  const role = asString(info.reporter_role);
  const phone = asString(info.contact_phone);
  const email = asString(info.contact_email);
  const seriousness = asString(info.seriousness);
  const outcome = asString(info.outcome);

  if (product) rows.push({ label: "Product", value: product });
  if (events) rows.push({ label: "Reported events", value: events });
  // Patient identity: the name and the date of birth are separate, correctly
  // labelled fields. A stored date of birth is never displayed as an age;
  // only a stored numeric age is shown as an age (no age is calculated).
  if (identifier) rows.push({ label: "Patient", value: identifier });
  if (age) {
    if (/^\d{1,3}$/.test(age)) {
      rows.push({ label: "Age", value: `${age} years` });
    } else {
      rows.push({ label: "Date of birth", value: age });
    }
  }
  if (onset) {
    // Readable calendar date (DD-MMM-YYYY) with the reported wording retained;
    // unknown date and unknown time are called out distinctly rather than merged.
    const datePart = onsetDisplay ?? "date unknown";
    const timePart = onsetPeriod ?? "time unknown";
    rows.push({
      label: "Onset",
      value: `${datePart} · ${timePart} · reported "${onset}"`,
    });
  }
  if (role) {
    rows.push({ label: "Reporter", value: role === "self" ? "Patient" : "Reporter" });
  }
  const contact = [phone, email].filter(Boolean).join(" · ");
  if (contact) rows.push({ label: "Contact", value: contact });
  // Consent is only "No" when it was explicitly declined: an unanswered
  // consent field is reported as not provided and never silently converted.
  rows.push({
    label: "Consent",
    value:
      info.consent_to_contact === true
        ? "Yes"
        : info.consent_to_contact === false
          ? "No"
          : "Not provided",
  });
  if (seriousness && seriousness !== "unknown") {
    rows.push({ label: "Seriousness", value: seriousness });
  }
  if (outcome) rows.push({ label: "Outcome", value: outcome });

  // Optional closing detail. Kept separate from the structured fields above;
  // an explicit decline is shown as "None provided" rather than omitted.
  const additionalInfo = asString(info.additional_information);
  const additionalStatus = asString(info.additional_information_status);
  if (additionalStatus === "provided" && additionalInfo) {
    rows.push({
      label: "Additional information provided by the patient",
      value: additionalInfo,
    });
  } else if (additionalStatus === "declined") {
    rows.push({
      label: "Additional information provided by the patient",
      value: "None provided",
    });
  }

  rows.push({
    label: "Status",
    value: "Structured report prepared for human PV review.",
  });
  return rows;
}
