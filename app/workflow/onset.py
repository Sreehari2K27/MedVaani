"""Onset-date normalization: relative wording -> calendar dates.

Deterministic, offline, standard-library only. Converts a reporter's original
onset statement ("today morning", "yesterday evening", "two days ago", or an
explicit date) into an ISO calendar date using the conversation timestamp and
its timezone, while always preserving the original wording. No clock time is
ever invented from a period word such as "morning"; only the period itself is
recorded.

Reference handling (never silently assume UTC):
    - A timestamp carrying an explicit non-UTC offset frames the conversation's
      local calendar directly (``2026-10-09T08:30:00+05:30`` -> 2026-10-09).
    - A UTC (``Z``) or naive timestamp has no local frame on its own; it is only
      usable when the caller supplies ``conversation_timezone``.
    - Without a reliable reference no date is invented: the original wording is
      kept and the normalized date is marked ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re

_MONTH_ABBRS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})[\s-]([A-Za-z]{3,9})\.?[\s,-]+(\d{4})\b"
)
_MONTH_DAY_YEAR = re.compile(
    r"\b([A-Za-z]{3,9})\.?[\s-](\d{1,2}),?[\s,-]+(\d{4})\b"
)
_SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

# ---- spoken-out dates ("fourteenth of May twenty twenty six") --------------
# Filler words removed before matching a spoken date.
_SPOKEN_FILLER = re.compile(r"\b(?:the|of|on)\b", re.IGNORECASE)
# Ordinal suffixes on a numeric day ("14th" -> "14").
_ORDINAL_SUFFIX = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

_DAY_WORDS: dict[str, int] = {
    "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10,
    "eleven": 11, "eleventh": 11,
    "twelve": 12, "twelfth": 12,
    "thirteen": 13, "thirteenth": 13,
    "fourteen": 14, "fourteenth": 14,
    "fifteen": 15, "fifteenth": 15,
    "sixteen": 16, "sixteenth": 16,
    "seventeen": 17, "seventeenth": 17,
    "eighteen": 18, "eighteenth": 18,
    "nineteen": 19, "nineteenth": 19,
    "twenty": 20, "twentieth": 20,
    "thirty": 30, "thirtieth": 30,
}

_YEAR_ONES: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_YEAR_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_DAYS_AGO = re.compile(
    # "two days ago" / "2 days back" - "back" is a common spoken variant.
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+(?:ago|back)\b",
    re.IGNORECASE,
)
_DAY_BEFORE_YESTERDAY = re.compile(r"\bday\s+before\s+yesterday\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY = re.compile(
    r"\b(today|tonight|this\s+(?:morning|afternoon|evening))\b"
    # Malayalam relative today (bare Indic tokens; a trailing \b never
    # matches after a matra): "ഇന്നു രാവിലെ" = this morning.
    r"|ഇന്നു|ഇന്ന്"
    # Hindi relative today (the observed "आज सुबह" live wording), following the Malayalam bare-Indic-token precedent; no clock time is ever inferred.
    r"|आज",
    re.IGNORECASE,
)
_PERIOD = re.compile(
    r"\b(?:this\s+)?(morning|afternoon|evening|night|midnight|noon|midday)\b"
    r"|\btonight\b"
    # Malayalam morning ("രാവിലെ"), bare like the other Indic tokens.
    r"|രാവിലെ"
    # Hindi morning ("subah"), bare like the Malayalam token: "आज सुबह" = today morning.
    r"|सुबह",
    re.IGNORECASE,
)

_FIXED_OFFSET = re.compile(r"^([+-])(\d{1,2}):?(\d{2})$")


@dataclass(frozen=True)
class OnsetNormalization:
    """Result of interpreting one onset statement.

    ``normalized_iso`` is a calendar date only (never a time); ``source`` is
    ``"explicit"`` (a stated date), ``"normalized"`` (resolved from relative
    wording) or ``"unknown"`` (no reliable reference / ambiguous).
    """

    original: str
    normalized_iso: str | None
    period: str | None
    source: str

    @property
    def display(self) -> str | None:
        return format_iso_date(self.normalized_iso)


def format_report_date(value: date) -> str:
    """Render a date as ``DD-MMM-YYYY`` (e.g. ``09-Oct-2026``)."""
    return f"{value.day:02d}-{_MONTH_ABBRS[value.month - 1]}-{value.year:04d}"


def format_iso_date(value: str | None) -> str | None:
    """Render an ISO ``YYYY-MM-DD`` string as ``DD-MMM-YYYY`` if parseable."""
    if not value:
        return None
    try:
        return format_report_date(date.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def resolve_reference(
    timestamp: str | datetime | None,
    timezone_hint: str | None = None,
) -> datetime | None:
    """Resolve a conversation timestamp to an aware local datetime, or ``None``.

    Returns ``None`` when the timestamp is missing/unparseable, when it is naive
    and no timezone is supplied, or when it is UTC (``Z``/zero offset) and no
    local timezone is supplied. Local wall time is never assumed from a server
    default.
    """
    instant = _parse_instant(timestamp)
    if instant is None:
        return None
    tz = _parse_timezone(timezone_hint)
    if instant.tzinfo is None:
        if tz is None:
            return None
        return instant.replace(tzinfo=tz)
    if tz is not None:
        return instant.astimezone(tz)
    offset = instant.utcoffset()
    if offset is None or offset == timedelta(0):
        return None
    return instant


def is_recognizable_onset(statement: str | None) -> bool:
    """Whether the statement contains a recognizable date or time expression:
    an explicit supported date, a relative-day expression (today/yesterday/
    N days ago, including the supported native-script tokens), or a period
    word (morning/afternoon/… ). Used to gate onset capture so
    acknowledgements and non-date answers are never stored as onset dates.
    Recognition only - the actual value still resolves against the
    conversation's own reference, and no clock time is ever invented."""
    text = (statement or "").strip()
    if not text:
        return False
    if _parse_explicit_date(text) is not None:
        return True
    # The relative SHAPE is recognizable without a real reference date; the
    # stored value still resolves from the conversation's own reference.
    if (
        _DAY_BEFORE_YESTERDAY.search(text)
        or _YESTERDAY.search(text)
        or _DAYS_AGO.search(text)
        or _TODAY.search(text)
    ):
        return True
    return _PERIOD.search(text) is not None


def normalize_onset_statement(
    statement: str | None,
    reference: datetime | None = None,
) -> OnsetNormalization:
    """Interpret an onset statement, preserving the original wording."""
    original = (statement or "").strip()
    period = _extract_period(original)

    explicit = _parse_explicit_date(original)
    if explicit is not None:
        return OnsetNormalization(original, explicit.isoformat(), period, "explicit")

    if reference is not None:
        relative = _resolve_relative(original, reference.date())
        if relative is not None:
            return OnsetNormalization(original, relative.isoformat(), period, "normalized")

    return OnsetNormalization(original, None, period, "unknown")


def _parse_instant(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_timezone(value: str | None):
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    fixed = _FIXED_OFFSET.match(candidate)
    if fixed:
        sign = 1 if fixed.group(1) == "+" else -1
        hours, minutes = int(fixed.group(2)), int(fixed.group(3))
        if hours > 23 or minutes > 59:
            return None
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(candidate)
    except Exception:
        return None


_PERIOD_TRANSLATIONS: dict[str, str] = {
    # Malayalam morning (the established catalog token) and the Hindi
    # subah spoken by the code-switching reporters in the observed
    # live transcripts; both normalize to the English value the
    # workflow stores.
    "രാവിലെ": "morning",
    "सुबह": "morning",
}


def _extract_period(text: str) -> str | None:
    match = _PERIOD.search(text)
    if match is None:
        return None
    if match.group(1):
        return match.group(1).lower()
    # Bare Indic matches have no captured group: translate the matched word
    # when known; "tonight" and unknown words keep the existing day-part.
    matched = match.group(0).strip().lower()
    if matched in _PERIOD_TRANSLATIONS:
        return _PERIOD_TRANSLATIONS[matched]
    return "night"


def _resolve_relative(text: str, reference_date: date) -> date | None:
    if _DAY_BEFORE_YESTERDAY.search(text):
        return reference_date - timedelta(days=2)
    if _YESTERDAY.search(text):
        return reference_date - timedelta(days=1)
    days_ago = _DAYS_AGO.search(text)
    if days_ago is not None:
        token = days_ago.group(1).lower()
        days = int(token) if token.isdigit() else _WORD_NUMBERS.get(token)
        if days is not None and 0 < days <= 365:
            return reference_date - timedelta(days=days)
    if _TODAY.search(text):
        return reference_date
    return None


def _parse_explicit_date(text: str) -> date | None:
    iso = _ISO_DATE.search(text)
    if iso is not None:
        candidate = _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        if candidate is not None:
            return candidate

    day_month = _DAY_MONTH_YEAR.search(text)
    if day_month is not None:
        month = _MONTHS.get(day_month.group(2).lower())
        if month is not None:
            candidate = _safe_date(
                int(day_month.group(3)), month, int(day_month.group(1))
            )
            if candidate is not None:
                return candidate

    month_day = _MONTH_DAY_YEAR.search(text)
    if month_day is not None:
        month = _MONTHS.get(month_day.group(1).lower())
        if month is not None:
            candidate = _safe_date(
                int(month_day.group(3)), month, int(month_day.group(2))
            )
            if candidate is not None:
                return candidate

    slash = _SLASH_DATE.search(text)
    if slash is not None:
        first, second, year = int(slash.group(1)), int(slash.group(2)), int(slash.group(3))
        if first > 12 and second <= 12:
            return _safe_date(year, second, first)
        if second > 12 and first <= 12:
            return _safe_date(year, first, second)
        # Both components could be a month: ambiguous, do not guess.
        return None

    # Spoken-out dates ("fourteenth of May twenty twenty six"). Only a full
    # day + month + year is accepted; partial wording stays unresolved.
    return _parse_spoken_date(text)


def _word_day(text: str) -> int | None:
    """A spoken day ("fourteen", "fourteenth", "twenty seventh", "27")."""
    tokens = text.split()
    if not tokens:
        return None
    if len(tokens) == 1:
        token = tokens[0]
        if token.isdigit():
            return int(token) if 1 <= int(token) <= 31 else None
        return _DAY_WORDS.get(token)
    if len(tokens) == 2:
        tens = _DAY_WORDS.get(tokens[0])
        ones = _DAY_WORDS.get(tokens[1])
        if tens is not None and ones is not None and tens in (20, 30) and 1 <= ones <= 9:
            return tens + ones
    return None


def _word_year(text: str) -> int | None:
    """A spoken year ("two thousand", "twenty twenty six", "nineteen ninety eight")."""
    tokens = [t for t in re.sub(r"\band\b", " ", text.lower()).split() if t]
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].isdigit():
        return int(tokens[0]) if len(tokens[0]) == 4 else None

    def _tail(rest: list[str]) -> int | None:
        if not rest:
            return 0
        if len(rest) != 1:
            return None
        token = rest[0]
        if token.isdigit() and len(token) <= 2:
            return int(token)
        return _YEAR_ONES.get(token)

    if tokens[0] == "two" and len(tokens) >= 2 and tokens[1] == "thousand":
        add = _tail(tokens[2:])
        year = None if add is None else 2000 + add
    elif (
        tokens[0] in ("nineteen", "twenty")
        and len(tokens) >= 2
        and tokens[1] in _YEAR_TENS
    ):
        add = _tail(tokens[2:])
        if add is None:
            return None
        year = (1900 if tokens[0] == "nineteen" else 2000) + _YEAR_TENS[tokens[1]] + add
    else:
        return None
    if year is None:
        return None
    return year if 1900 <= year <= date.today().year else None


_MONTH_ANCHOR = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in sorted(_MONTHS, key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)


def _parse_spoken_date(text: str) -> date | None:
    """A full spoken date ("the 14th of May 2026"), or None when incomplete."""
    cleaned = _ORDINAL_SUFFIX.sub(r"\1", _SPOKEN_FILLER.sub(" ", text.lower()))
    anchor = _MONTH_ANCHOR.search(cleaned)
    if anchor is None:
        return None
    before = cleaned[: anchor.start()].strip(" .,;:-")
    after = cleaned[anchor.end() :].strip(" .,;:-")
    day = _word_day(before)
    if day is None or not 1 <= day <= 31:
        return None
    year = _word_year(after)
    if year is None:
        return None
    return _safe_date(year, _MONTHS[anchor.group(1).lower()], day)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
