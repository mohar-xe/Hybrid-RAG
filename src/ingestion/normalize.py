# src/ingestion/normalize.py

import re
import unicodedata
from typing import List


# ---------------------------------------------------------
# BASIC NORMALIZATION
# ---------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters into a consistent form.
    Fixes weird quotation marks, unicode spacing, etc.
    """
    text = unicodedata.normalize("NFKC", text)
    return text


def normalize_newlines(text: str) -> str:
    """
    Convert all newline styles to Unix style.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_zero_width_chars(text: str) -> str:
    """
    Remove invisible unicode characters often found in PDFs.
    """
    return re.sub(r"[\u200B-\u200D\uFEFF]", "", text)


def remove_control_chars(text: str) -> str:
    """
    Strip C0/C1 control characters (incl. NUL 0x00) except tab and newline.

    PDF extraction frequently injects NUL and other control bytes; PostgreSQL
    ``text`` columns reject NUL, so this must run before storage.
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)


# ---------------------------------------------------------
# WHITESPACE CLEANING
# ---------------------------------------------------------

def collapse_spaces(text: str) -> str:
    """
    Replace multiple spaces/tabs with a single space.
    """
    return re.sub(r"[ \t]+", " ", text)


def collapse_empty_lines(text: str, max_empty: int = 1) -> str:
    """
    Reduce excessive empty lines.
    """
    pattern = r"\n{%d,}" % (max_empty + 2)
    replacement = "\n" * (max_empty + 1)
    return re.sub(pattern, replacement, text)


def strip_lines(text: str) -> str:
    """
    Strip leading/trailing whitespace from every line.
    """
    return "\n".join(line.strip() for line in text.splitlines())


# ---------------------------------------------------------
# HYPHEN / LINE FIXES
# ---------------------------------------------------------

def fix_hyphenated_words(text: str) -> str:
    """
    Fix words split across lines.

    Example:
        "informa-\ntion" -> "information"
    """
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def join_broken_lines(text: str) -> str:
    """
    Join lines that are likely part of the same paragraph.

    Keeps actual paragraph breaks intact.
    """

    lines = text.split("\n")
    merged = []

    for i, line in enumerate(lines):
        line = line.strip()

        if not line:
            merged.append("\n")
            continue

        if (
            merged
            and merged[-1] != "\n"
            and not merged[-1].endswith((".", "!", "?", ":", ";"))
            and not line.startswith(("-", "*", "•"))
        ):
            merged[-1] += " " + line
        else:
            merged.append(line)

    return "\n".join(merged)


# ---------------------------------------------------------
# PDF ARTIFACT REMOVAL
# ---------------------------------------------------------

def remove_page_numbers(text: str) -> str:
    """
    Remove isolated page numbers.

    Examples removed:
        1
        - 2 -
        Page 3
    """

    patterns = [
        r"(?m)^\s*\d+\s*$",
        r"(?m)^\s*-\s*\d+\s*-\s*$",
        r"(?m)^\s*Page\s+\d+\s*$",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text


def remove_headers_footers(text: str, min_repetition: int = 3) -> str:
    """
    Remove repeated headers/footers heuristically.

    Works best if pages are separated by form feed (\f).
    """

    pages = text.split("\f")

    if len(pages) < min_repetition:
        return text

    first_lines = {}
    last_lines = {}

    for page in pages:
        lines = [l.strip() for l in page.splitlines() if l.strip()]

        if not lines:
            continue

        first = lines[0]
        last = lines[-1]

        first_lines[first] = first_lines.get(first, 0) + 1
        last_lines[last] = last_lines.get(last, 0) + 1

    repeated_headers = {
        k for k, v in first_lines.items() if v >= min_repetition
    }

    repeated_footers = {
        k for k, v in last_lines.items() if v >= min_repetition
    }

    cleaned_pages = []

    for page in pages:
        lines = page.splitlines()

        cleaned = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if i == 0 and stripped in repeated_headers:
                continue

            if i == len(lines) - 1 and stripped in repeated_footers:
                continue

            cleaned.append(line)

        cleaned_pages.append("\n".join(cleaned))

    return "\f".join(cleaned_pages)


# ---------------------------------------------------------
# MARKDOWN / RAG FRIENDLY
# ---------------------------------------------------------

def normalize_bullets(text: str) -> str:
    """
    Convert unicode bullets into standard markdown bullets.
    """

    bullet_chars = ["•", "◦", "▪", "■", "‣"]

    for bullet in bullet_chars:
        text = text.replace(bullet, "-")

    return text


def normalize_quotes(text: str) -> str:
    """
    Convert fancy quotes into ASCII quotes.
    """

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ---------------------------------------------------------
# MASTER PIPELINE
# ---------------------------------------------------------

def clean_pdf_text(
    text: str,
    *,
    fix_lines: bool = True,
    remove_pages: bool = True,
    remove_repeated_headers: bool = False,
) -> str:
    """
    Full PDF cleaning pipeline.
    """

    text = normalize_unicode(text)
    text = normalize_newlines(text)
    text = remove_zero_width_chars(text)
    text = remove_control_chars(text)

    text = normalize_quotes(text)
    text = normalize_bullets(text)

    text = fix_hyphenated_words(text)

    if fix_lines:
        text = join_broken_lines(text)

    text = collapse_spaces(text)
    text = strip_lines(text)

    if remove_pages:
        text = remove_page_numbers(text)

    if remove_repeated_headers:
        text = remove_headers_footers(text)

    text = collapse_empty_lines(text)

    return text.strip()


# ---------------------------------------------------------
# CHUNK HELPERS
# ---------------------------------------------------------

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split cleaned text into paragraphs.
    """
    return [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]


def remove_short_lines(text: str, min_length: int = 3) -> str:
    """
    Remove noisy tiny lines.
    """

    lines = []

    for line in text.splitlines():
        if len(line.strip()) >= min_length:
            lines.append(line)

    return "\n".join(lines)