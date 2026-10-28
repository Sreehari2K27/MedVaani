// Mapping between the backend /voice/conversation response and the AE chat
// UI (stage sidebar, "still needed" list, completion summary). Pure helpers;
// the demo journey in aeDemoConversation.ts is untouched.

import {
  AE_STAGES,
  type AeStageId,
  type AeStageState,
  type AeSummaryRow,
} from "./aeDemoConversation.ts";
import {
  capitalizeFirst,
  englishCategory,
  englishName,
  englishReportText,
} from "./reportEnglish.ts";

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
  // The reported event is NEVER translated, transliterated, or rewritten:
  // it is rendered from the stored verbatim value — the only display
  // formatting is capitalizing the first character of the sentence
  // ("abdominal pain…" -> "Abdominal pain…"), which is a no-op for
  // non-Latin scripts. Every other character stays byte-for-byte.
  if (events) rows.push({ label: "Reported event", value: capitalizeFirst(events) });
  // Patient identity: the name and the date of birth are separate, correctly
  // labelled fields. A stored date of birth is never displayed as an age;
  // only a stored numeric age is shown as an age (no age is calculated).
  // Final-report rendering only: a Devanagari name is transliterated to
  // Latin script ("यश गुप्ता" -> "Yash Gupta"); stored values are unchanged.
  if (identifier) rows.push({ label: "Patient", value: englishName(identifier) });
  if (age) {
    if (/^\d{1,3}$/.test(age)) {
      rows.push({ label: "Age", value: `${age} years` });
    } else {
      rows.push({ label: "Date of birth", value: age });
    }
  }
  // Unconfirmed held candidate (review-time only): shown exactly as
  // captured, never as a committed or verified value.
  const identifierHeld = asString(info.patient_identifier_held);
  if (identifierHeld) {
    rows.push({
      label: "Patient (unconfirmed)",
      value: `${englishName(identifierHeld)} — held for confirmation`,
    });
  }
  if (onset) {
    // Readable calendar date (DD-MMM-YYYY) with the reported wording retained;
    // unknown date and unknown time are called out distinctly rather than merged.
    // The reported wording itself is presented in English (the stored verbatim
    // value is untouched); the composed row starts capitalised
    // ("Date unknown · time unknown · reported \"…\"").
    const datePart = onsetDisplay ? englishReportText(onsetDisplay) : "date unknown";
    const timePart = onsetPeriod ? englishReportText(onsetPeriod) : "time unknown";
    rows.push({
      label: "Onset",
      value: capitalizeFirst(
        `${datePart} · ${timePart} · reported "${englishReportText(onset)}"`,
      ),
    });
  }
  if (role) {
    // The confirmed reporter identity: in the other-person branch the
    // reporter's own confirmed name is shown (transliterated like the
    // patient row above); the self branch IS the patient and stays
    // "Patient". Without a confirmed name the role label is kept.
    const reporterName = asString(info.reporter_name);
    rows.push({
      label: "Reporter",
      value:
        role === "self"
          ? "Patient"
          : reporterName
            ? englishName(reporterName)
            : "Reporter",
    });
  }
  // The reporter's own confirmed contact number: kept strictly separate
  // from the patient Contact row below (patient phone · email) — it is
  // never substituted by the email or the patient's number.
  const reporterPhone = asString(info.reporter_contact_phone);
  if (reporterPhone) {
    rows.push({ label: "Reporter contact", value: reporterPhone });
  }
  const contact = [phone, email].filter(Boolean).join(" · ");
  if (contact) rows.push({ label: "Contact", value: contact });
  // Unconfirmed held candidate (review-time only), like the held patient
  // name above: never presented as committed or verified.
  const phoneHeld = asString(info.contact_phone_held);
  if (phoneHeld) {
    rows.push({
      label: "Contact (unconfirmed)",
      value: `${phoneHeld} — held for confirmation`,
    });
  }
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
    rows.push({ label: "Seriousness", value: englishCategory(seriousness) });
  }
  if (outcome) rows.push({ label: "Outcome", value: outcome });

  // Reviewer-facing note (Option B): a short event description that matched
  // no recognized symptom/report cue may be unclear speech. The captured
  // description is still rendered above byte-for-byte untouched; this row
  // only tells the human reviewer the wording may not be a complete
  // description and must be judged by a person.
  if (info.event_description_unclear === true) {
    rows.push({
      label: "Review note",
      value:
        "The reported event description was captured verbatim and may be unclear; it needs qualified human review.",
    });
  }

  // Optional closing detail. Kept separate from the structured fields above;
  // an explicit decline is shown as "None provided" rather than omitted.
  const additionalInfo = asString(info.additional_information);
  const additionalStatus = asString(info.additional_information_status);
  if (additionalStatus === "provided" && additionalInfo) {
    rows.push({
      label: "Additional information provided by the patient",
      value: englishReportText(additionalInfo),
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
