"""Pytest configuration for MedVaani.

Contains ONLY the optional ``--locale-report`` multilingual regression
summary: an opt-in reporting plugin that groups executed test outcomes by
supported locale and scenario. It is inactive unless ``--locale-report`` is
explicitly passed, never contacts the network, adds no dependencies, and does
not alter ordinary pytest behaviour.

The report derives every cell from ACTUAL executed node ids and their
outcomes — a cell is never marked PASS because a related test exists or
another locale passed. Ids whose locale is ambiguous or absent are classified
SHARED (language-independent logic), never guessed into a locale.
"""
from __future__ import annotations

import re

LOCALES = ("bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN")

# Full BCP-47 codes anywhere in the id (the parametrized fixtures embed them).
_FULL_LOCALE_RE = re.compile(r"(?<![A-Za-z])(?:bn|en|gu|hi|kn|ml|mr|pa|ta|te)-IN(?![A-Za-z])")
# Short session/case prefixes ("em-ml-reject-replace", "keep-hi-1"): a locale
# fragment delimited by non-letters and a following dash or bracket.
_SHORT_LOCALE_RE = re.compile(r"(?<![A-Za-z])(bn|gu|hi|kn|ml|mr|pa|ta|te)(?=[-\]])")

# Scenario -> node-id regex. Matched against the function-name part of the
# id; the ids' parametrized segments carry the locale evidence.
SCENARIOS: tuple[tuple[str, str], ...] = (
    ("role-classification-per-locale", r"test_(?:self|someone_else)_reporter_answers_in_every_locale"),
    ("read-back-affirmatives-per-locale", r"test_natural_affirmatives_in_every_locale|test_clear_affirmatives_confirm_a_pending"),
    ("negatives-per-locale", r"test_negative_reply_in_every_locale|test_ambiguous_and_negative_replies"),
    ("field-choice-per-locale", r"test_field_selection_in_every_locale"),
    ("phone-digit-words-per-locale", r"test_spoken_digit_words_extract_phone_in_every_locale|test_locale_spoken_phone_tolerates"),
    ("phone-mixed-digits", r"test_mixed_digits_and_digit_words|test_devanagari_asr_spelling|test_incomplete_transliterated"),
    ("phone-length-validation", r"test_phone_length_plausibility|test_implausible_pending_phone|test_leading_zero_phone|test_marathi_eleven_digit"),
    ("email-spelling", r"test_malayalam_dictated_email|test_devanagari_|test_latin_letter_name|test_double_e|test_compact_|test_marathi_script_yes"),
    ("email-reject-safeguards", r"test_rejected_|test_truncated_|test_garbled_|test_incorrect_email"),
    ("email-optional-skip", r"test_empty_transcript_during_email|test_incorrect_email_phrase"),
    ("dob-spoken-per-locale", r"test_spoken_dob_parses_to_20_may_2000_across_locales|test_mixed_digit_and_spoken_dob|test_every_locale_month_token"),
    ("name-dob-combined", r"test_clear_hindi_name_and_dob|test_mixed_marathi_name|test_name_and_date_of_birth_are_extracted|test_combined_name|test_failed_hindi_date_attempt|test_workflow_keeps_name_and_reasks|test_marathi_name_lead_variants"),
    ("dob-clarification-recovery", r"test_ambiguous_dob_loop|test_dob_clarification|test_failed_hindi_date_attempt|test_workflow_keeps_name_and_reasks"),
    ("reporter-journey", r"test_other_person_flow|test_marathi_other_person|test_marathi_browser_regression|test_identity_question|test_malayalam_first_person|test_english_happened|test_marathi_self_experience"),
    ("session-language", r"test_marathi_session_survives|test_short_replies_do_not_revert|test_language_switches|test_explicit_language_hint|test_latin_marathi_replies|test_markerless_devanagari"),
    ("empty-transcript", r"test_empty_transcript"),
    ("onset", r"test_hindi_today_morning|test_malayalam_relative_today|test_onset"),
    ("consent-report-integrity", r"test_consent|test_unanswered_consent|test_report_includes|test_final_report|test_structured_report"),
    ("catalog-parity", r"test_new_language_catalog_covers|test_new_language_patterns_mirror"),
    ("voice-mapping", r"test_voice_mapping|test_normalize_and_voice|test_voice_mapping_covers"),
)

_RANK = {"passed": 0, "skipped": 1, "failed": 2}
_CELL_STATE = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}

_STATE: dict | None = None


def pytest_addoption(parser):
    parser.addoption(
        "--locale-report",
        action="store_true",
        default=False,
        help="Print the per-locale x per-scenario multilingual regression summary.",
    )


def pytest_configure(config):
    global _STATE
    _STATE = None
    if config.getoption("--locale-report"):
        _STATE = {"collected": [], "executed": {}}


def pytest_collection_finish(session):
    if _STATE is not None:
        _STATE["collected"] = [item.nodeid for item in session.items]


def pytest_runtest_logreport(report):
    if _STATE is None:
        return
    prior = _STATE["executed"].get(report.nodeid)
    outcome = report.outcome
    if prior is None or _RANK.get(outcome, 1) > _RANK.get(prior, 1):
        _STATE["executed"][report.nodeid] = outcome


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if getattr(config, "_medvaani_locale_report_off", None) or _STATE is None:
        return
    executed = _STATE["executed"]
    collected = _STATE["collected"]

    # cell[(scenario, locale)] = worst outcome across matching executed ids
    cells: dict[tuple[str, str], str] = {}
    failing_by_cell: dict[tuple[str, str], list[str]] = {}
    for nodeid, outcome in sorted(executed.items()):
        name = _test_name_of(nodeid)
        locale = _locale_of(nodeid)
        for scenario, pattern in SCENARIOS:
            if not re.search(pattern, name):
                continue
            key = (scenario, locale)
            prior = cells.get(key)
            if prior is None or _RANK.get(outcome, 1) > _RANK.get(prior, 1):
                cells[key] = outcome
            if outcome == "failed":
                failing_by_cell.setdefault(key, []).append(nodeid)

    # Pattern validation against the COLLECTED ids: a pattern matching
    # nothing collected is a mapping gap, never silently UNTESTED.
    unmatched = [
        scenario
        for scenario, pattern in SCENARIOS
        if not any(re.search(pattern, _test_name_of(cid)) for cid in collected)
    ]

    term = terminalreporter
    term.ensure_newline()
    term.write_line("=" * 30 + " MULTILINGUAL REGRESSION REPORT " + "=" * 30)
    header = f"{'scenario':<34}" + "".join(f"{loc:<7}" for loc in LOCALES) + "shared"
    term.write_line(header)
    term.write_line("-" * len(header))
    untested_cells: list[tuple[str, str]] = []
    for scenario, _pattern in SCENARIOS:
        if scenario in unmatched:
            term.write_line(
                f"{scenario:<34}PATTERN MATCHED NO COLLECTED TESTS (mapping gap)"
            )
            continue
        row = f"{scenario:<34}"
        for loc in LOCALES + ("shared",):
            outcome = cells.get((scenario, loc))
            if outcome is None:
                row += f"{'-':<7}"
                untested_cells.append((scenario, loc))
            else:
                row += f"{_CELL_STATE[outcome]:<7}"
        term.write_line(row)
    term.write_line("-" * len(header))
    term.write_line(
        "P=passed F=failed S=skipped -=untested (no executed test id for that cell)"
    )

    if failing_by_cell:
        term.write_line("\nFAILURES BY LOCALE/SCENARIO:")
        for (scenario, loc), ids in sorted(failing_by_cell.items()):
            term.write_line(f"  [{loc}] {scenario}:")
            for nodeid in ids:
                term.write_line(f"    FAIL {nodeid}")
    else:
        term.write_line("\nNo failures in the executed multilingual set.")

    if untested_cells:
        compact: dict[str, list[str]] = {}
        for scenario, loc in untested_cells:
            compact.setdefault(scenario, []).append(loc)
        term.write_line("\nUNTESTED CELLS (no executed id for that locale):")
        for scenario, locs in compact.items():
            term.write_line(f"  {scenario}: {', '.join(locs)}")
    else:
        term.write_line("\nEvery scenario/locale cell had at least one executed test.")

    if unmatched:
        term.write_line(
            "\nUNMATCHED PATTERNS (matched no collected test - fix the mapping):"
        )
        for scenario in unmatched:
            term.write_line(f"  {scenario}")
    else:
        term.write_line("\nAll scenario patterns matched at least one collected test.")


def _test_name_of(nodeid: str) -> str:
    return nodeid.split("::")[-1].split("[")[0]


def _locale_of(nodeid: str) -> str:
    full = set(_FULL_LOCALE_RE.findall(nodeid))
    if len(full) == 1:
        return next(iter(full))
    if len(full) > 1:
        return "shared"
    shorts = set(_SHORT_LOCALE_RE.findall(nodeid))
    if len(shorts) == 1:
        return next(iter(shorts)) + "-IN"
    return "shared"
