/**
 * Focused regression tests for the final AE report's English rendering.
 *
 * Presentation layer only (reportEnglish + buildLiveSummary): Hindi names are
 * transliterated, yes/no and onset phrases translated, all other rows kept in
 * English — and the reported event description must remain byte-for-byte
 * identical to what was captured. No browser, no framework: plain assertion
 * checks runnable with Node's native TypeScript support:
 *
 *   node src/ae/reportEnglish.test.ts
 */

import {
  capitalizeFirst,
  englishCategory,
  englishName,
  englishReportText,
} from "./reportEnglish.ts";
import { buildLiveSummary, type AeCollectedInfo } from "./aeLiveConversation.ts";

let failures = 0;

const check = (condition: unknown, message: string): void => {
  if (condition) {
    console.log(`ok   - ${message}`);
  } else {
    failures += 1;
    console.error(`FAIL - ${message}`);
  }
};

// ---- Hindi name transliteration ---------------------------------------------

check(englishName("यश गुप्ता") === "Yash Gupta", "यश गुप्ता transliterates to Yash Gupta");
check(englishName("राजेश रस्तोगी") === "Rajesh Rastogi", "राजेश रस्तोगी transliterates to Rajesh Rastogi");
check(englishName("अंकित शर्मा") === "Ankit Sharma", "अंकित शर्मा transliterates to Ankit Sharma");
check(englishName("Beta Jones") === "Beta Jones", "a Latin name is returned unchanged");
check(
  englishName("  यश गुप्ता  ") === "Yash Gupta",
  "surrounding whitespace is trimmed without touching the name",
);

// ---- Malayalam name transliteration -----------------------------------------

check(
  englishName("സുരേഷ് ഗോപി") === "Suresh Gopi",
  "സുരേഷ് ഗോപി transliterates to Suresh Gopi",
);
check(englishName("രാജു") === "Raju", "രാജു transliterates to Raju");
check(englishName("രാമൻ") === "Raman", "a chillu-final name keeps its nasal without an inherent a");
check(
  englishName("രാജു പി കെ") === "Raju P K",
  "the observed name keeps spoken initials as their English letters",
);
check(
  englishName("Raju P K") === "Raju P K",
  "a name that already carries Latin initials is unchanged",
);
check(
  englishName("  സുരേഷ് ഗോപി  ") === "Suresh Gopi",
  "Malayalam surrounding whitespace is trimmed without touching the name",
);
check(
  buildLiveSummary({
    product: "Lantus SoloStar",
    adverse_event: "തലവേദന ഉണ്ടായി",
    patient_identifier: "സുരേഷ് ഗോപി",
    patient_age: "20-May-2006",
  }).find((row) => row.label === "Patient")?.value === "Suresh Gopi",
  "the report summary renders a Malayalam patient name in Roman script",
);

// ---- reviewer note for unclear event descriptions ----------------------------

// Unconfirmed held candidates (review-time only): rendered clearly as
// unconfirmed, never as committed or verified values.
{
  const rows = buildLiveSummary({
    product: "Lantus SoloStar",
    adverse_event: "rash",
    patient_identifier_held: "സുരേഷ് ഗോപി",
    contact_phone_held: "9224419103",
  });
  const nameRow = rows.find((row) => row.label === "Patient (unconfirmed)");
  check(
    nameRow !== undefined &&
      nameRow.value.includes("Suresh Gopi") &&
      nameRow.value.includes("held for confirmation"),
    "a held unconfirmed patient name is rendered as held for confirmation",
  );
  const phoneRow = rows.find((row) => row.label === "Contact (unconfirmed)");
  check(
    phoneRow !== undefined &&
      phoneRow.value.includes("9224419103") &&
      phoneRow.value.includes("held for confirmation"),
    "a held unconfirmed phone candidate is rendered as held for confirmation",
  );
  check(
    !rows.some((row) => row.label === "Patient"),
    "a held name is never also shown as a committed patient row",
  );
  check(
    !rows.some((row) => row.label === "Contact" && row.value === "9224419103"),
    "a held phone is never also shown as a committed contact row",
  );
}
check(
  !buildLiveSummary({ product: "Lantus SoloStar", adverse_event: "rash" }).some(
    (row) => row.label === "Patient (unconfirmed)" || row.label === "Contact (unconfirmed)",
  ),
  "no held rows appear when nothing is held",
);

const unclearEvent = "ഈ മേഖലയില് പാര്ട്ടി നല്കി";
const unclearRows = buildLiveSummary({
  product: "Lantus SoloStar",
  adverse_event: unclearEvent,
  event_description_unclear: true,
});
check(
  unclearRows.find((row) => row.label === "Reported event")?.value === unclearEvent,
  "an unclear event description is preserved verbatim",
);
check(
  unclearRows.some(
    (row) =>
      row.label === "Review note" &&
      row.value.includes("captured verbatim and may be unclear"),
  ),
  "the reviewer can see the description may be unclear",
);
check(
  !buildLiveSummary({
    product: "Lantus SoloStar",
    adverse_event: "I have a headache",
    event_description_unclear: false,
  }).some((row) => row.label === "Review note"),
  "a clear description adds no review note row",
);

// ---- yes/no and categorical values -----------------------------------------

check(englishReportText("नहीं") === "No", "नहीं renders as No");
check(englishReportText("नहीं।") === "No", "नहीं with a danda renders as No");
check(englishReportText("हाँ") === "Yes", "हाँ renders as Yes");
check(englishReportText("जी हाँ") === "Yes", "जी हाँ renders as Yes");
check(englishCategory("no") === "No", "categorical no renders as No");
check(englishCategory("yes") === "Yes", "categorical yes renders as Yes");
check(
  englishCategory("recovered") === "recovered",
  "already-English categorical values are unchanged",
);
check(capitalizeFirst("date unknown") === "Date unknown", "capitalise helper works");

// ---- onset / free-text translation ------------------------------------------

check(
  englishReportText("आज सुबह यह घटना हुई") === "The event occurred this morning",
  "the reported onset sentence translates exactly",
);
check(
  englishReportText("30-Jan-2001") === "30-Jan-2001",
  "English dates pass through byte-for-byte",
);
check(
  englishReportText("morning") === "morning",
  "English period words pass through unchanged",
);
const unknownWord = englishReportText("अजीब तकलीफ");
check(!/[\u0900-\u097F]/.test(unknownWord), "unknown Devanagari words fall back to Latin script");
check(
  unknownWord.charAt(0) === unknownWord.charAt(0).toUpperCase(),
  "translated free text is sentence-cased",
);

// ---- the full final report (expected rendering) -----------------------------

const EVENT = "maine kal ye davai le thi aur aaj mera sir dard kar raha hai";

const collected: AeCollectedInfo = {
  product: "Lantus SoloStar",
  adverse_event: EVENT,
  patient_identifier: "यश गुप्ता",
  patient_age: "30-Jan-2001",
  onset_date: "आज सुबह यह घटना हुई",
  onset_date_display: null,
  onset_period: null,
  reporter_role: "self",
  contact_phone: "922440000",
  consent_to_contact: false,
  seriousness: "no",
  additional_information: "नहीं",
  additional_information_status: "provided",
};

const rows = buildLiveSummary(collected);
const rowValue = (label: string): string | undefined =>
  rows.find((row) => row.label === label)?.value;

check(rowValue("Product") === "Lantus SoloStar", "product preserved exactly");
check(rowValue("Patient") === "Yash Gupta", "patient name rendered in English");
check(rowValue("Date of birth") === "30-Jan-2001", "date of birth kept in English format");
check(
  rowValue("Onset") ===
    'Date unknown · time unknown · reported "The event occurred this morning"',
  "onset row translated with unknown date/time called out",
);
check(rowValue("Reporter") === "Patient", "reporter role rendered in English");
check(rowValue("Contact") === "922440000", "numeric contact preserved untouched");
check(rowValue("Consent") === "No", "consent rendered in English");
check(rowValue("Seriousness") === "No", "seriousness rendered in English");
check(
  rowValue("Additional information provided by the patient") === "No",
  "additional information rendered in English",
);
check(
  rowValue("Status") === "Structured report prepared for human PV review.",
  "report status preserved",
);

const expectedLabels = [
  "Product",
  "Reported event",
  "Patient",
  "Date of birth",
  "Onset",
  "Reporter",
  "Contact",
  "Consent",
  "Seriousness",
  "Additional information provided by the patient",
  "Status",
];
check(
  JSON.stringify(rows.map((row) => row.label)) === JSON.stringify(expectedLabels),
  "final report rows and labels match the expected rendering exactly",
);

// ---- the reported event is never modified -----------------------------------

check(
  rowValue("Reported event") === "Maine kal ye davai le thi aur aaj mera sir dard kar raha hai",
  "the event display is sentence-capitalized and otherwise byte-for-byte",
);

const devanagariEvent = "आज सुबह मुझे बुखार आया और सिर दर्द हुआ";
const devanagariRows = buildLiveSummary({
  adverse_event: devanagariEvent,
  product: "Lantus SoloStar",
});
const devanagariEventRow = devanagariRows.find((row) => row.label === "Reported event");
check(
  devanagariEventRow?.value === devanagariEvent,  // non-Latin: untouched
  "a Devanagari event description is never transliterated or re-cased",
);
const oddCasedEvent = "miXed Case, punctUation! with Hindi दर्द inside";
check(
  buildLiveSummary({ adverse_event: oddCasedEvent }).some(
    (row) =>
      row.label === "Reported event" &&
      row.value === "MiXed Case, punctUation! with Hindi \u0926\u0930\u094d\u0926 inside",
  ),
  "event casing, punctuation and embedded script survive beyond the leading capital",
);

// ---- other non-event rows keep existing behaviour ---------------------------

const withEnglish = buildLiveSummary({
  product: "Lantus SoloStar",
  adverse_event: EVENT,
  patient_identifier: "Yash Gupta",
  patient_age: "30-Jan-2001",
  onset_date: "headache started two days ago",
  onset_date_display: "05-Feb-2026",
  onset_period: "morning",
  reporter_role: "someone_else",
  contact_email: "yash@example.com",
  consent_to_contact: true,
  seriousness: "yes",
  outcome: "recovered",
  additional_information: "none",
  additional_information_status: "declined",
});
const englishRow = (label: string): string | undefined =>
  withEnglish.find((row) => row.label === label)?.value;
check(englishRow("Patient") === "Yash Gupta", "Latin patient name untouched");
check(
  englishRow("Onset") === '05-Feb-2026 · morning · reported "headache started two days ago"',
  "English onset fields pass through unchanged",
);
check(englishRow("Reporter") === "Reporter", "someone-else reporter label unchanged");
check(englishRow("Contact") === "yash@example.com", "email contact unchanged");
check(englishRow("Consent") === "Yes", "consent yes unchanged");
check(englishRow("Seriousness") === "Yes", "seriousness yes rendered as Yes");
check(englishRow("Outcome") === "recovered", "outcome enum unchanged");
check(
  englishRow("Additional information provided by the patient") === "None provided",
  "declined additional information keeps the existing wording",
);
check(
  englishRow("Reported event") === "Maine kal ye davai le thi aur aaj mera sir dard kar raha hai",
  "event row stays verbatim apart from the leading capital",
);

// ---- confirmed reporter identity in the final summary -------------------------

// The reported other-person session: the reporter's confirmed name and own
// number must appear in the summary as separate reporter rows, distinct
// from the patient's rows and never substituted by the email.
const otherPersonRows = buildLiveSummary({
  product: "Lantus SoloStar",
  adverse_event: "rash",
  reporter_role: "someone_else",
  reporter_name: "രാജു കെ",
  reporter_contact_phone: "9886094321",
  patient_identifier: "Priya Nair",
  patient_age: "15-Mar-1995",
  contact_phone: "9224419103",
  contact_email: "sreh@gmail.com",
  consent_to_contact: true,
});
const otherRow = (label: string): string | undefined =>
  otherPersonRows.find((row) => row.label === label)?.value;
check(
  otherRow("Reporter") === "Raju K",
  "the confirmed reporter name is rendered in the summary",
);
check(
  otherRow("Reporter contact") === "9886094321",
  "the confirmed reporter phone has its own summary row",
);
check(
  otherRow("Contact") === "9224419103 · sreh@gmail.com",
  "the patient Contact row keeps the patient phone and email",
);
check(
  otherRow("Patient") === "Priya Nair",
  "the patient name stays distinct from the reporter name",
);
check(
  otherPersonRows.find((row) => row.label === "Reporter contact")?.value !==
    "sreh@gmail.com",
  "the email can never populate the reporter contact row",
);
check(
  otherPersonRows.some((row) => row.label === "Patient") &&
    otherPersonRows.some((row) => row.label === "Reporter contact"),
  "reporter and patient rows coexist once patient details are collected",
);

// The self branch has no separate reporter identity or number: the rows are
// unchanged and no empty reporter-contact row appears.
const selfRows = buildLiveSummary({
  product: "Lantus SoloStar",
  adverse_event: "rash",
  reporter_role: "self",
  contact_phone: "9224419103",
});
check(
  selfRows.find((row) => row.label === "Reporter")?.value === "Patient",
  "self branch keeps the Patient reporter row",
);
check(
  !selfRows.some((row) => row.label === "Reporter contact"),
  "no reporter-contact row appears without a confirmed reporter phone",
);
check(
  !buildLiveSummary({ product: "Lantus SoloStar", reporter_role: "someone_else" }).some(
    (row) => row.label === "Reporter contact",
  ),
  "no reporter-contact row appears without a confirmed reporter number",
);

if (failures > 0) {
  throw new Error(`${failures} report English test(s) failed`);
}
console.log("All report English tests passed");


// ---- Latin name capitalization + event display casing (reported issues) ------

check(englishName("raju") === "Raju", "a lower-case Latin first name is capitalized");
check(englishName("raju sharma") === "Raju Sharma", "each Latin name word is capitalized");
check(
  englishName("Eric McDonald") === "Eric McDonald",
  "meaningful internal capitalization is never rewritten",
);
check(
  englishName("raju McDonald") === "Raju McDonald",
  "only the first letter of a lower-case word changes",
);
check(
  englishName("\u0930\u093e\u091c\u0942 \u0936\u0930\u094d\u092e\u093e") !== undefined,
  "non-Latin names keep their transliteration path",
);

// The reported event DISPLAY copy: the first character of the sentence is
// capitalized; every other character stays byte-for-byte identical to the
// stored verbatim value.
{
  const rows = buildLiveSummary({
    product: "Lantus SoloStar",
    adverse_event: "abdominal pain after taking medicine this morning",
  });
  check(
    rows.find((row) => row.label === "Reported event")?.value ===
      "Abdominal pain after taking medicine this morning",
    "the event display copy is sentence-capitalized",
  );
  const mixedRows = buildLiveSummary({
    product: "Lantus SoloStar",
    adverse_event: "miXed Case, punctUation! with Hindi \u0926\u0930\u094d\u0926 inside",
  });
  check(
    mixedRows.find((row) => row.label === "Reported event")?.value ===
      "MiXed Case, punctUation! with Hindi \u0926\u0930\u094d\u0926 inside",
    "only the first character is capitalized; the rest stays byte-for-byte",
  );
  check(
    !buildLiveSummary({
      product: "Lantus SoloStar",
      adverse_event: "\u0926\u0930\u094d\u0926 \u0939\u0941\u0906",
    }).some(
      (row) =>
        row.label === "Reported event" &&
        row.value !== "\u0926\u0930\u094d\u0926 \u0939\u0941\u0906",
    ),
    "non-Latin event text keeps the byte-for-byte display",
  );
}
