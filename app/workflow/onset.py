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

_DAYS_AGO = re.compile(
    # "two days ago" / "2 days back" - "back" is a common spoken variant.
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+(?:ago|back)\b",
    re.IGNORECASE,
)
_DAY_BEFORE_YESTERDAY = re.compile(r"\bday\s+before\s+yesterday\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY = re.compile(
    r"\b(today|tonight|this\s+(?:morning|afternoon|evening))\b", re.IGNORECASE
)
_PERIOD = re.compile(
    r"\b(?:this\s+)?(morning|afternoon|evening|night|midnight|noon|midday)\b"
    r"|\btonight\b",
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


def _extract_period(text: str) -> str | None:
    match = _PERIOD.search(text)
    if match is None:
        return None
    if match.group(1):
        return match.group(1).lower()
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

    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
