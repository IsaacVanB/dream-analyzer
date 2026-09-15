"""Small, deterministic in-memory BM25 index for dream text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
import math
import re
import unicodedata

from dream_analysis.dates import validate_date_range
from dream_analysis.models import Dream


# A token may contain internal apostrophes, but hyphens and other punctuation
# delimit tokens. ``[^\W_]`` means a Unicode alphanumeric character without
# treating an underscore as part of a word.
_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
_APOSTROPHE_TRANSLATION = str.maketrans({"\u2018": "'", "\u2019": "'"})

# Suffix stemmers cannot connect irregular past-tense forms to their lemmas.
# Keep this deliberately small and focused on common verbs in dream reports.
_IRREGULAR_VERBS = {
    "came": "come",
    "felt": "feel",
    "found": "find",
    "ran": "run",
    "saw": "see",
    "seen": "see",
    "took": "take",
    "taken": "take",
    "went": "go",
    "woke": "wake",
    "woken": "wake",
}


def _is_consonant(character: str) -> bool:
    return character.isalpha() and character not in "aeiou"


def _is_short_consonant_vowel_consonant(word: str) -> bool:
    return (
        len(word) >= 3
        and _is_consonant(word[-3])
        and word[-2] in "aeiou"
        and _is_consonant(word[-1])
        and word[-1] not in "wxy"
    )


def stem_english_token(token: str) -> str:
    """Apply conservative English inflection stemming to one normalized token.

    Apostrophe-containing words are intentionally left whole. The stemmer
    handles common plural and verb inflections and a small explicit set of
    irregular verbs; it is not intended to be a full morphological analyzer.
    """
    if "'" in token or len(token) <= 2:
        return token
    irregular = _IRREGULAR_VERBS.get(token)
    if irregular is not None:
        return irregular

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"

    for suffix in ("ing", "ed"):
        if not token.endswith(suffix) or len(token) <= len(suffix) + 2:
            continue
        stem = token[: -len(suffix)]
        if not any(character in "aeiou" for character in stem):
            return token
        if (
            len(stem) >= 2
            and stem[-1] == stem[-2]
            and stem[-1] not in "lsz"
        ):
            stem = stem[:-1]
        elif stem.endswith(("at", "bl", "iz")) or (
            len(stem) <= 3 and _is_short_consonant_vowel_consonant(stem)
        ):
            stem += "e"
        return stem

    if token.endswith(("sses", "xes", "zes", "ches", "shes", "oes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize_english(text: str) -> tuple[str, ...]:
    """Normalize, split, and stem English text for keyword retrieval."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).translate(
        _APOSTROPHE_TRANSLATION
    )
    return tuple(
        stem_english_token(match.group(0))
        for match in _WORD_PATTERN.finditer(normalized.casefold())
    )


@dataclass(frozen=True, slots=True)
class Bm25SearchResult:
    """One scored dream returned by :class:`DreamBm25Index`."""

    dream_id: str
    date: str
    document: str
    score: float


class DreamBm25Index:
    """Immutable Okapi BM25 index over a small collection of dream texts."""

    def __init__(
        self,
        dreams: Sequence[Dream],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], Sequence[str]] = tokenize_english,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer
        self._dreams = tuple(dreams)
        dream_ids = [dream.dream_id for dream in self._dreams]
        if len(dream_ids) != len(set(dream_ids)):
            raise ValueError("dream IDs must be unique")

        self._term_frequencies = tuple(
            Counter(tokenizer(dream.text)) for dream in self._dreams
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequencies.update(frequencies.keys())
        document_count = len(self._dreams)
        self._inverse_document_frequencies = {
            term: math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Bm25SearchResult]:
        """Return positive-scoring text matches in an inclusive date range."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        validate_date_range(start_date, end_date)

        # Repetition should not let a generated query accidentally multiply a
        # term's influence. Dict order preserves the normalized query order.
        query_terms = tuple(dict.fromkeys(self.tokenizer(query)))
        if not query_terms or not self._dreams:
            return []

        scored: list[tuple[float, int, Dream]] = []
        for position, (dream, frequencies, document_length) in enumerate(
            zip(self._dreams, self._term_frequencies, self._document_lengths)
        ):
            if start_date is not None or end_date is not None:
                if dream.date_sort is None:
                    continue
                if start_date is not None and dream.date_sort < start_date:
                    continue
                if end_date is not None and dream.date_sort > end_date:
                    continue

            score = sum(
                self._term_score(
                    term,
                    term_frequency=frequencies.get(term, 0),
                    document_length=document_length,
                )
                for term in query_terms
            )
            if score > 0:
                scored.append((score, position, dream))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            Bm25SearchResult(
                dream_id=dream.dream_id,
                date=dream.date,
                document=dream.text,
                score=score,
            )
            for score, _, dream in scored[:limit]
        ]

    def _term_score(
        self,
        term: str,
        *,
        term_frequency: int,
        document_length: int,
    ) -> float:
        if term_frequency == 0:
            return 0.0
        inverse_document_frequency = self._inverse_document_frequencies.get(term)
        if inverse_document_frequency is None:
            return 0.0
        length_ratio = (
            document_length / self._average_document_length
            if self._average_document_length
            else 0.0
        )
        denominator = term_frequency + self.k1 * (
            1.0 - self.b + self.b * length_ratio
        )
        return inverse_document_frequency * (
            term_frequency * (self.k1 + 1.0) / denominator
        )
