"""Grounded MI answer generation.

Answering is extractive by default: when the LLM provider does not supply a
grounded answer, MIAnswerService quotes the most relevant passage from the top
retrieved chunk and cites the package insert page. This keeps answers
deterministic and source-grounded without adding an LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import re
import time

from app.mi.retrieval import LantusRetriever, RetrievedChunk
from app.mi.retrieval.retriever import expand_query, normalize_token, tokenize_text
from app.providers.base import LLMProvider

# Diagnostics only: safe metadata (outcome, provider class, page numbers,
# elapsed time) - never the question, the answer text, or any key.
logger = logging.getLogger(__name__)


@dataclass
class MIAnswerResult:
    answer: str
    intent: str = "product_info"
    source_chunks: List[Dict[str, Any]] = None
    source_page_numbers: List[int] = None
    grounded: bool = True
    unsupported: bool = False
    retrieved_chunk_ids: List[str] = None

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []
        if self.source_page_numbers is None:
            self.source_page_numbers = []
        if self.retrieved_chunk_ids is None:
            self.retrieved_chunk_ids = []


# Sentence boundaries: protects common abbreviations ("e.g.", "i.e.") so
# "kept (e.g. in a refrigerator)" is never split mid-abbreviation, and keeps
# STANDALONE single-letter enumeration markers ("A. ", "B. ", "C. ") attached
# to their instruction ("A. Take off the pen cap." stays one sentence). The
# \b guard matters: without it a normal word ending ("frozen.") would also
# look like a letter + period.
_SENTENCE_SPLIT = re.compile(
    r"(?<!e\.g\.)(?<!i\.e\.)(?<!\b[A-Za-z]\.)(?<=[.!?])\s+"
)
# Inline citation segments a provider may attach; checked separately from the
# answer content so a citation never carries an ungrounded claim past the gate
# and never causes a fully grounded answer to be rejected.
_SOURCE_PAREN = re.compile(r"\(\s*Source:[^)]*\)", re.IGNORECASE)
_MAX_PASSAGE_CHARS = 400
_GROUNDED_THRESHOLD = 0.9

_INSUFFICIENT_MESSAGE = (
    "Sufficient information is not available in the provided product reference."
)
_INSUFFICIENT_PHRASE = "sufficient information is not available"

# Flattened storage table in the approved insert (page 11): recognized by its
# intro line so its rows can be rendered as faithful sentences instead of a
# raw cell dump.
_TABLE_MARKER = re.compile(r"summarized in the following table", re.IGNORECASE)
_TABLE_END = re.compile(r"\n(?:Preparation and Handling|Date of permission)\b")

# Structural banner lines: ALL-CAPS section banners (PHARMACODYNAMICS,
# MODE OF ACTION/...) that are removed from the SCORING text so their words
# cannot inflate a sentence's score. Colon labels ("Indication:", "Hypoglycemia:")
# are deliberately KEPT while scoring (they carry the topic) and trimmed only
# when the chosen passage is displayed.
_LABEL_ALLCAPS = re.compile(r"^[^a-z]*[A-Z][^a-z]*$")
_LABEL_COLON = re.compile(r"^.{1,45}:$")
_GLYPHS = "\uf0b7\u2022\u00b7"


def _strip_banner_lines(text: str) -> str:
    """Drop ALL-CAPS section banners (scoring text only; raw lines kept)."""
    kept = []
    for line in text.split("\n"):
        s = line.strip().lstrip(_GLYPHS).strip()
        if s and _LABEL_ALLCAPS.match(s):
            continue
        kept.append(line)
    return "\n".join(kept)


def _cut_leading_labels(passage: str) -> str:
    """Keep the content after the last colon-label line of a quote.

    E.g. "Therapeutic … Indication:\\nFor the treatment of diabetes …" becomes
    the indication sentence itself; "\\uf0b7 Hypoglycemia:\\nHypoglycemia, in
    general …" loses its duplicate heading. Quotes without such a label are
    returned unchanged.
    """
    lines = passage.split("\n")
    last_label = -1
    for idx, line in enumerate(lines[:8]):
        s = line.strip().lstrip(_GLYPHS).strip()
        if idx < len(lines) - 1 and _LABEL_COLON.match(s):
            last_label = idx
    if last_label < 0:
        return passage
    return "\n".join(lines[last_label + 1 :]).lstrip(_GLYPHS).lstrip()


def _collapse_wraps(text: str) -> str:
    """Turn PDF line-wraps into clean prose (no invented content).

    Joins wrapped hyphenation ("28-\\nday") and single line breaks; blank
    paragraph breaks are preserved as a newline for readability.
    """
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)
    parts = re.split(r"\n[ \t]*\n", text)
    parts = [re.sub(r"[ \t]*\n[ \t]*", " ", p) for p in parts]
    joined = "\n".join(parts)
    return re.sub(r" {2,}", " ", joined).strip()


def _storage_table_summary(chunk_text: str) -> Optional[str]:
    """Render the flattened storage table as faithful, grammatical sentences.

    Rows are reassembled from the source's raw line layout (a row may wrap
    across lines, and the pen row's product cell interleaves with its values),
    so column relationships from the table headers are preserved exactly:
    values are never mixed across rows or columns. Returns None when the
    table shape is not recognized (never guesses).
    """
    marker = _TABLE_MARKER.search(chunk_text)
    if marker is None:
        return None
    block = chunk_text[marker.end() :]
    stop = _TABLE_END.search(block)
    if stop is not None:
        block = block[: stop.start()]
    lines = [
        line.strip().lstrip(_GLYPHS).strip()
        for line in block.splitlines()
        if line.strip()
    ]
    if "Not in use" not in " ".join(lines) or "In use" not in " ".join(lines):
        return None

    # Reassemble rows: a row starts at a product line; following lines belong
    # to it until the next product line.
    rows: List[str] = []
    current: Optional[str] = None
    for line in lines:
        if re.match(r"^(10mL vial|3mL cartridge)\b", line):
            if current is not None:
                rows.append(current)
            current = line
        elif current is not None:
            current = f"{current} {line}"
    if current is not None:
        rows.append(current)
    if not rows:
        return None

    sentences = []
    for row in rows:
        if " Until expiration date " not in row:
            return None  # never guess a row we cannot parse
        head, _, rest = row.partition(" Until expiration date ")
        # Trailing note and product-cell continuation may follow the values
        # in the flattened layout ("… Room temperature into pen (SoloStar®)
        # only (Do not refrigerate)"): pull them back onto the right fields.
        note = ""
        note_match = re.search(r"\s+only \([^)]*\)$", rest)
        if note_match is not None:
            note = rest[note_match.start() :].strip()
            rest = rest[: note_match.start()].strip()
        cont_match = re.search(r"\s+into pen \([^)]*\)", rest)
        if cont_match is not None:
            head = f"{head}{cont_match.group(0)}"
            rest = (rest[: cont_match.start()] + rest[cont_match.end() :]).strip()
        values = re.match(r"(?P<unopened_room>\d+\s*days?)\s+(?P<in_use>.+)$", rest)
        if values is None:
            return None  # never guess a row we cannot parse
        in_use = re.match(
            r"(?P<duration>\d+\s*days?)(?:\.\s*(?P<condition>.+))?$",
            values.group("in_use").strip(),
        )
        if in_use is None:
            return None
        condition = (in_use.group("condition") or "").strip()
        if note:
            condition = f"{condition} {note}".strip()
        condition_clause = f", {condition.lower()}" if condition else ""
        sentences.append(
            f"\u2022 {head.strip()}: not in use (unopened), until the expiration "
            f"date when refrigerated and {values.group('unopened_room')} at room "
            f"temperature; in use (opened), {in_use.group('duration')}"
            f"{condition_clause}."
        )
    return "These storage conditions are summarized below.\n" + "\n".join(sentences)


def _source_citation(chunks: List[RetrievedChunk]) -> str:
    """Package-insert citation over the retrieved pages (deduplicated)."""
    pages = sorted({r.page_number for r in chunks})
    label = "page" if len(pages) == 1 else "pages"
    return (
        "(Source: Lantus package insert, "
        f"{label} {', '.join(str(p) for p in pages)})."
    )

_PROVIDER_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "has", "have", "had", "who", "whom", "which", "their", "them",
        "they", "his", "her", "its", "not", "may", "can", "any", "all", "also",
        "into", "than", "then", "when", "what", "your", "you", "will", "been",
        "being", "there", "these", "those", "does", "would", "should", "such",
        "other", "some", "each", "both", "many", "much", "more", "only", "one",
        "two", "how", "about",
    }
)


def _provider_words(answer: str) -> set[str]:
    """Normalized content words of ``answer`` (suffix-stripped for matching)."""
    return {
        normalize_token(w)
        for w in re.findall(r"[a-z]{4,}", answer.lower())
        if w not in _PROVIDER_STOPWORDS
    }


def _is_grounded_in(answer: str, chunks: List[RetrievedChunk]) -> bool:
    """Provider answers are trusted only when fully supported by the context.

    Two gates, both required:
    - the answer's content words overall meet the overlap threshold, AND
    - EVERY sentence of the answer is individually supported (each of its
      content words appears in the retrieved context), so one additional
      unsupported medical claim cannot ride along with a grounded sentence.
    """
    words = _provider_words(answer)
    if not words:
        return False
    context = " ".join(r.text.lower() for r in chunks)
    context_words = _provider_words(context)
    if not context_words:
        return False
    present = sum(1 for w in words if w in context_words)
    if present / len(words) < _GROUNDED_THRESHOLD:
        return False
    for sentence in _SENTENCE_SPLIT.split(answer):
        sentence_words = _provider_words(sentence)
        if sentence_words and not sentence_words <= context_words:
            return False
    return True


def _best_passage(chunk: RetrievedChunk, query_terms: list[str]) -> str:
    """Pick the sentence in a chunk that best matches the query terms.

    The section heading and structural label lines are removed BEFORE
    splitting and scoring: heading/label words used to sit inside the first
    sentence and inflate its score (e.g. "Adverse Reactions" plus the CIOMS
    legend beat the sentence naming the most frequent adverse reaction;
    "PHARMACODYNAMICS MODE OF ACTION" beat the actual mode-of-action
    sentence). Selected passages are then cleaned of PDF line-wraps, and the
    storage table (when selected) is rendered as faithful sentences.
    """
    raw = chunk.text
    text = raw
    heading = (chunk.section_heading or "").strip()
    if heading and text.lstrip()[: len(heading)].lower() == heading.lower():
        text = text.lstrip()[len(heading) :]
    # ALL-CAPS banners are removed before scoring so section-banner words
    # cannot inflate a sentence's score; colon labels stay (they carry the
    # topic) and are trimmed only when the chosen passage is displayed.
    text = _strip_banner_lines(text)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return _collapse_wraps(raw)[:_MAX_PASSAGE_CHARS]

    def score(sentence: str) -> int:
        lowered = sentence.lower()
        return sum(1 for term in query_terms if term in lowered)

    best = max(sentences, key=score)
    if score(best) == 0 and len(sentences) > 1:
        best = sentences[0]

    # A selected storage-table fragment is expanded into structured sentences
    # parsed from the raw (unmodified) chunk text, preserving column values.
    if _TABLE_MARKER.search(best) or (
        "Until expiration date" in best and _TABLE_MARKER.search(raw)
    ):
        summary = _storage_table_summary(raw)
        if summary:
            return summary

    # Don't start a quote with the section heading itself.
    if chunk.section_heading and chunk.section_heading.strip():
        heading = chunk.section_heading.strip()
        if best.lstrip().lower().startswith(heading.lower()):
            best = best.lstrip()[len(heading):].lstrip()
    # ...nor with a leading list glyph / duplicate heading label.
    best = best.lstrip(_GLYPHS).lstrip()
    best = _cut_leading_labels(best)
    best = _collapse_wraps(best)
    if not best:
        best = _collapse_wraps(sentences[0])
    if len(best) > _MAX_PASSAGE_CHARS:
        best = best[:_MAX_PASSAGE_CHARS].rstrip()
    if not best.endswith((".", "!", "?")):
        best = f"{best}."
    return best


def _query_terms(question: str) -> list[str]:
    # Expand with the same canonical terms the retriever matches on, so the
    # best-passage pick inside the right chunk favours the sentence carrying
    # the question's actual subject (e.g. the "Indication:" sentence over the
    # product-name header).
    return tokenize_text(expand_query(question))


def _followup_score(passage: str, terms: List[str]) -> int:
    """Distinct passage words touched by the query terms (>= 2 = relevant)."""
    if not terms:
        return 0
    words = set(re.findall(r"[a-z]{4,}", passage.lower()))
    return sum(1 for word in words if any(term in word for term in terms))


# Adverse-effect questions are answered from the document's own "Adverse
# Reactions" section (its labeled groups), never from unrelated chunks whose
# cross-references happen to mention adverse reactions.
_ADVERSE_QUESTION = re.compile(
    r"(?i)side\s+effects?|adverse\s+reactions?|adverse\s+effects?"
)
# Sentences kept beyond the first of each labeled group: frequency facts and
# important (severe / life-threatening) risk statements - the source's own
# distinctions are preserved, groups are never merged.
_ADVERSE_KEEP = re.compile(
    r"\d+\s*%|\b(?:very\s+common|common|uncommon|very\s+rare|rare|frequent|"
    r"severe|life[\s-]?threatening)\b",
    re.IGNORECASE,
)
_ADVERSE_SENTENCE_CAP = 3
_ADVERSE_MAX_CHARS = 2400


def _adverse_section_chunk(chunks: List[RetrievedChunk]) -> Optional[RetrievedChunk]:
    """The retrieved chunk that IS the Adverse Reactions section, if any."""
    for chunk in chunks:
        heading = (chunk.section_heading or "").lower()
        if "adverse reaction" in heading:
            return chunk
        if chunk.text.lstrip().startswith("Adverse Reactions"):
            return chunk
    return None


def _adverse_section_answer(chunk: RetrievedChunk) -> Optional[str]:
    """One bullet per labeled group of the Adverse Reactions section.

    The CIOMS frequency legend (it defines the scale, it is not a reaction)
    and anything before the first group label are dropped. Each group keeps
    its first sentence plus frequency/severity sentences (max 3), so every
    group stays a distinct, complete-sentence bullet in source order - the
    key hypoglycemia risk first, exactly as the source presents it. Returns
    None when the section has no recognizable groups (never guesses).
    """
    text = chunk.text
    heading = (chunk.section_heading or "").strip()
    if heading and text.lstrip()[: len(heading)].lower() == heading.lower():
        text = text.lstrip()[len(heading) :]

    blocks: List[tuple] = []
    label: Optional[str] = None
    buffer: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip().lstrip(_GLYPHS).strip()
        if stripped.endswith(":") and len(stripped) <= 45:
            if label is not None:
                blocks.append((label, buffer))
            label = stripped.rstrip(":").strip()
            buffer = []
        elif label is not None:
            # Content lines after the first label (pre-label legend dropped).
            buffer.append(line)
    if label is not None:
        blocks.append((label, buffer))
    if not blocks:
        return None

    bullets = []
    total = 0
    for group_label, lines in blocks:
        body = _collapse_wraps("\n".join(lines))
        if not body:
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip()]
        kept = []
        for index, sentence in enumerate(sentences):
            if index == 0 or _ADVERSE_KEEP.search(sentence):
                kept.append(sentence)
            if len(kept) >= _ADVERSE_SENTENCE_CAP:
                break
        if not kept:
            continue
        joined = " ".join(kept)
        if not joined.endswith((".", "!", "?")):
            joined = f"{joined}."
        bullet = f"\u2022 {group_label}: {joined}"
        if bullets and total + len(bullet) > _ADVERSE_MAX_CHARS:
            break
        bullets.append(bullet)
        total += len(bullet) + 1
    if not bullets:
        return None
    return "\n".join(bullets)


# ---- SoloStar Instruction Leaflet (pages 12-15) -----------------------------
# Pen-use questions are answered from the leaflet's own numbered steps,
# reconstructed across page boundaries in source order. The questions that
# qualify are matched narrowly so mechanism/storage/adverse questions are
# never hijacked.

_PEN_QUESTION = re.compile(
    r"(?i)(?:"
    r"\bhow\b[^.?!]{0,120}\b(?:use|us(?:e|ed|ing)|inject\w*|dose|safety\s+test|"
    r"attach\w*|needle|selector|discard\w*|dispose\w*|remove\w*|check\w*|cap|units?)\b"
    r"|\bsafety\s+test\b"
    r"|attach\w*\s+(?:a\s+|new\s+)?needle"
    r"|(?:remove|removing|dispose|disposing|discard|discarding|unscrew)\w*"
    r"\s+[^.?!]{0,30}needle"
    r"|check\w*\s+[^.?!]{0,40}insulin"
    r")"
)
_STEP_MARKER = re.compile(r"Step (\d)\. ")
_LEAFLET_MIN_PAGE = 12
_LEAFLET_MAX_PAGE = 15


def _pages_citation(pages: List[int]) -> str:
    """Package-insert citation for a list of pages (same format as
    ``_source_citation``)."""
    unique = sorted(set(pages))
    label = "page" if len(unique) == 1 else "pages"
    return (
        "(Source: Lantus package insert, "
        f"{label} {', '.join(str(p) for p in unique)})."
    )


def _leaflet_steps(document_chunks: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Reconstruct the SoloStar Instruction Leaflet steps from local chunks.

    Pages 12-15 are concatenated in page order and split on the source's own
    "Step N. " markers, so headings, lettered substeps, warnings and the
    original sequence survive page-boundary chunk splits. Accepts both raw
    document dicts and ``RetrievedChunk`` objects. Returns None when the
    structure cannot be found (caller falls back to the normal path).
    """

    def page_of(chunk: Any) -> int:
        return chunk["page_number"] if isinstance(chunk, dict) else chunk.page_number

    def text_of(chunk: Any) -> str:
        return chunk["text"] if isinstance(chunk, dict) else chunk.text

    leaflet = sorted(
        (
            c
            for c in document_chunks
            if _LEAFLET_MIN_PAGE <= page_of(c) <= _LEAFLET_MAX_PAGE
            and _STEP_MARKER.search(text_of(c))
        ),
        key=page_of,
    )
    if not leaflet:
        return None

    pieces: List[str] = []
    bounds: List[tuple] = []  # (start, end, page)
    offset = 0
    for chunk in leaflet:
        text = text_of(chunk)
        pieces.append(text)
        bounds.append((offset, offset + len(text), page_of(chunk)))
        offset += len(text) + 1  # "\n" join
    combined = "\n".join(pieces)

    matches = list(_STEP_MARKER.finditer(combined))
    if not matches:
        return None
    steps: List[Dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(combined)
        raw = combined[start:end]
        pages = sorted(
            {page for a, b, page in bounds if a < end and b > start}
        )
        # The Step 6 part runs to the end of the leaflet; cut the trailing
        # Storage Instructions (answered by the normal storage path).
        cut = raw.find("\nStorage Instructions")
        if cut != -1:
            raw = raw[:cut]
        heading_line, _, body = raw.partition("\n")
        steps.append(
            {
                "number": int(match.group(1)),
                "heading": heading_line.strip(),
                "body": body.strip("\n"),
                "pages": pages,
            }
        )
    numbers = [s["number"] for s in steps]
    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        return None  # never answer from a sequence we cannot trust
    return steps


def _clean_leaflet_body(text: str) -> str:
    """Tidy one step's body without altering its wording.

    PDF line-wraps are joined back onto their sentence (continuation lines),
    while lettered substeps ("A. …"), list items and separate sentences keep
    their own lines. Hyphen wraps are repaired. Wording is never changed.
    """
    lines: List[str] = []
    for line in text.split("\n"):
        raw = line.strip()
        if not raw:
            continue
        had_glyph = raw[0] in _GLYPHS
        stripped = raw.lstrip(_GLYPHS).strip()
        if re.match(r"^o\s+", stripped):
            stripped = re.sub(r"^o\s+", "", stripped)
        if not stripped:
            continue
        is_substep = bool(re.match(r"^[A-Z]\.\s", stripped))
        is_continuation = (
            lines
            and not is_substep
            and not had_glyph
            and not re.search(r"[.!?:]$", lines[-1])
        )
        if is_continuation:
            if lines[-1].endswith("-"):
                lines[-1] = lines[-1][:-1] + stripped
            else:
                lines[-1] = f"{lines[-1]} {stripped}"
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _pen_steps_for_question(question: str) -> List[int]:
    """Map a pen-use question to its source step numbers (ascending)."""
    q = (question or "").lower()
    wanted = set()
    if re.search(r"safety\s+test|prim(?:e|ing)", q):
        wanted.add(3)
    if re.search(r"attach\w*\s+[^.?!]{0,20}needle|needle\s+(?:attach|screw)", q):
        wanted.add(2)
    if re.search(
        r"check\w*\s+[^.?!]{0,40}(?:insulin|label)|take\s+off\s+[^.?!]{0,10}cap"
        r"|pen\s+cap|appearance\s+of\s+(?:the\s+)?insulin",
        q,
    ):
        wanted.add(1)
    if re.search(
        r"select\w*\s+[^.?!]{0,30}dose|set(?:ting)?\s+[^.?!]{0,20}dose"
        r"|dose\s+(?:selector|window)|dial\w*\s+[^.?!]{0,20}dose",
        q,
    ):
        wanted.add(4)
    # Step 5 only when injection itself is the question: temporal mentions
    # ("... before injecting", "... after injection") are context, not the act.
    if re.search(r"inject", q) and not re.search(
        r"(?:before|after|prior\s+to)\s+inject", q
    ):
        wanted.add(5)
    if re.search(
        r"(?:remove|removing|dispose|disposing|discard|discarding|unscrew)\w*"
        r"\s+[^.?!]{0,40}needle|needle\s+(?:removal|disposal)",
        q,
    ):
        wanted.add(6)
    return sorted(wanted)


def _pen_step_answer(question: str, document_chunks: List[Any]) -> Optional[str]:
    """Source-grounded pen-use answer from the Instruction Leaflet steps.

    One step -> that step's complete instructions in source order; several
    steps -> the steps in their original sequence; a general "how do I use"
    question -> an ordered Steps 1-6 overview (each step's own heading plus
    its first instruction). Every fact is quoted from the approved leaflet;
    no dose is ever selected, calculated, or recommended here. Returns None
    when the question has no matching step and no overview applies.
    """
    steps = _leaflet_steps(document_chunks)
    if not steps:
        return None

    wanted = _pen_steps_for_question(question)
    if wanted:
        selected = [s for s in steps if s["number"] in wanted]
        if len(selected) != len(wanted):
            return None  # a mapped step is missing from the source: fall back
        blocks = []
        pages: List[int] = []
        for step in selected:
            body = _clean_leaflet_body(step["body"])
            if not body:
                return None
            blocks.append(f"{step['heading']}\n{body}")
            pages.extend(step["pages"])
        return "\n\n".join(blocks) + "\n" + _pages_citation(pages)

    # General how-to-use question: ordered overview of every reconstructed step.
    lines = []
    for step in steps:
        body = _clean_leaflet_body(step["body"])
        if not body:
            return None
        first_line = body.split("\n")[0]
        lines.append(f"{step['heading']} - {first_line}")
    return "\n".join(lines) + "\n" + _pages_citation(
        [page for step in steps for page in step["pages"]]
    )


def _build_grounded_answer(
    chunks: List[RetrievedChunk],
    question: str,
    document_chunks: Optional[List[Any]] = None,
) -> str:
    terms = _query_terms(question)

    # Adverse-effect questions: answer from the Adverse Reactions section
    # itself (prioritized even when other chunks ranked higher), with no
    # unrelated passages appended.
    if _ADVERSE_QUESTION.search(question):
        adverse = _adverse_section_chunk(chunks)
        if adverse is not None:
            body = _adverse_section_answer(adverse)
            if body:
                location = f"page {adverse.page_number}"
                if adverse.section_heading and adverse.section_heading.strip():
                    location += f", section \"{adverse.section_heading.strip()}\""
                return f"{body}\n(Source: Lantus package insert, {location})."

    # Pen-use questions: answer from the Instruction Leaflet's own steps.
    if document_chunks and _PEN_QUESTION.search(question):
        pen_answer = _pen_step_answer(question, document_chunks)
        if pen_answer:
            return pen_answer

    passages = []
    for index, chunk in enumerate(chunks[:2]):
        passage = _best_passage(chunk, terms)
        # Follow-up passages must actually relate to the question (at least
        # two touched words, or all terms for very short queries), so an
        # unrelated second chunk never rides along with the answer.
        if index > 0 and _followup_score(passage, terms) < min(2, len(terms)):
            continue
        location = f"page {chunk.page_number}"
        if chunk.section_heading and chunk.section_heading.strip():
            location += f", section \"{chunk.section_heading.strip()}\""
        # The citation sits on its own line, clearly separated from the answer.
        passages.append(f"{passage}\n(Source: Lantus package insert, {location}).")
        if len("\n".join(passages)) >= 240:
            break
    return "\n".join(passages)


class MIAnswerService:
    def __init__(self, retriever: LantusRetriever = None, llm_provider: LLMProvider = None) -> None:
        from app.providers import MockLLMProvider

        self.retriever = retriever or LantusRetriever()
        self.llm_provider = llm_provider or MockLLMProvider()

    def answer(self, question: str, top_k: int = 3) -> MIAnswerResult:
        started = time.perf_counter()
        if not question or not question.strip():
            return MIAnswerResult(
                answer="No question provided.",
                intent="product_info",
                grounded=False,
                unsupported=True,
            )

        question = question.strip()
        retrieved = self.retriever.retrieve(question, k=top_k)

        # Build context for LLM
        context_parts = []
        for r in retrieved:
            context_parts.append(f"=== Page {r.page_number} ===\n{r.text}")

        context_text = "\n\n".join(context_parts) if context_parts else ""

        grounding_instruction = (
            "Answer ONLY using the supplied reference content. "
            "Do NOT invent facts. Do NOT use outside medical knowledge. "
            "If the supplied reference does not contain enough information, "
            "explicitly say that sufficient information is not available in the provided product reference."
        )

        llm_context = {
            "question": question,
            "context": context_text,
            "grounding_instruction": grounding_instruction,
            "retrieved_chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "text": r.text,
                }
                for r in retrieved
            ],
        }

        # A provider failure (key, timeout, network, parse) degrades to the
        # deterministic extractive answer below; it never fails the request.
        provider_error_type: Optional[str] = None
        provider_accepted = False
        provider_refused = False
        try:
            llm_result = self.llm_provider.interpret(
                f"grounding_instruction: {grounding_instruction}\n\nQuestion: {question}\n\nContext: {context_text}",
                llm_context,
            )
            provider_answer = (llm_result.extracted_fields or {}).get("answer")
            provider_intent = getattr(llm_result, "intent", None)
        except Exception as exc:
            # Only the exception type is logged/propagated - never its payload.
            provider_answer = None
            provider_intent = None
            provider_error_type = type(exc).__name__

        answer = None
        if provider_answer and retrieved and provider_intent == "product_info":
            provider_answer = str(provider_answer)
            if _INSUFFICIENT_PHRASE in provider_answer.lower():
                # A provider refusal while retrieval DID return evidence: the
                # grounded extractive quote below answers from the actual
                # source instead of surfacing a generic refusal (the refusal
                # phrase remains the answer only when nothing was retrieved).
                provider_refused = True
            else:
                # Citations are excluded from the grounding check (they are
                # metadata, not claims) and re-attached below if missing.
                core = _SOURCE_PAREN.sub("", provider_answer).strip() or provider_answer
                if _is_grounded_in(core, retrieved):
                    answer = provider_answer
                    provider_accepted = True
            if answer is not None and not _SOURCE_PAREN.search(answer):
                answer = f"{answer}\n{_source_citation(retrieved)}"
        if not answer and retrieved:
            # Deterministic extractive answer, cited to the package insert.
            answer = _build_grounded_answer(
                retrieved, question, document_chunks=self.retriever.chunks
            )
        if not answer:
            answer = _INSUFFICIENT_MESSAGE

        unsupported = not retrieved or _INSUFFICIENT_PHRASE in answer.lower()

        pages = [r.page_number for r in retrieved]
        # One line per answer with safe metadata only: which path produced the
        # answer, which provider was consulted, the cited pages and latency.
        # Never the question, the answer text, or any credential.
        if provider_error_type:
            outcome = f"provider_error:{provider_error_type}"
        elif not retrieved:
            outcome = "no_evidence"
        elif provider_accepted:
            outcome = "provider_grounded"
        elif provider_refused:
            # The provider declined; the extractive quote answered instead.
            outcome = "extractive_after_provider_refusal"
        elif provider_answer and retrieved and provider_intent == "product_info":
            outcome = "provider_rejected_grounding"
        else:
            outcome = "extractive"
        logger.info(
            "mi_answer outcome=%s provider=%s pages=%s duration_ms=%d",
            outcome,
            type(self.llm_provider).__name__,
            pages,
            int((time.perf_counter() - started) * 1000),
        )

        return MIAnswerResult(
            answer=answer,
            intent="product_info",
            source_chunks=[
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "section_heading": r.section_heading,
                    "text": r.text,
                }
                for r in retrieved
            ],
            source_page_numbers=pages,
            grounded=bool(retrieved),
            unsupported=unsupported,
            retrieved_chunk_ids=[r.chunk_id for r in retrieved],
        )
