from __future__ import annotations

import unittest
from datetime import date

from dream_analysis.bm25 import DreamBm25Index, tokenize_english
from dream_analysis.models import Dream


def make_dream(dream_id: str, text: str, day: int | None) -> Dream:
    sortable_date = date(2024, 1, day) if day is not None else None
    return Dream(
        dream_id=dream_id,
        date=f"1/{day}/2024" if day is not None else "unknown",
        text=text,
        date_sort=sortable_date,
        word_count=len(text.split()),
    )


class EnglishBm25TokenizerTests(unittest.TestCase):
    def test_hyphens_split_words_but_apostrophes_remain_inside_words(self) -> None:
        self.assertEqual(
            tokenize_english("Green-fingered; don't don t"),
            ("green", "finger", "don't", "don", "t"),
        )

    def test_curly_apostrophes_and_compatible_unicode_are_normalized(self) -> None:
        self.assertEqual(tokenize_english("DON\u2019T \uff27reen"), ("don't", "green"))

    def test_regular_and_selected_irregular_verb_forms_share_stems(self) -> None:
        self.assertEqual(tokenize_english("run running ran"), ("run", "run", "run"))
        self.assertEqual(
            tokenize_english("dream dreams dreamed dreaming"),
            ("dream", "dream", "dream", "dream"),
        )
        self.assertEqual(
            tokenize_english("house houses box boxes"),
            ("house", "house", "box", "box"),
        )


class DreamBm25IndexTests(unittest.TestCase):
    def test_search_returns_ranked_typed_results(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("one", "A fox appeared.", 1),
                make_dream("two", "A fox chased another fox.", 2),
                make_dream("three", "Only a school appeared.", 3),
            ]
        )

        matches = index.search("fox", limit=2)

        self.assertEqual([match.dream_id for match in matches], ["two", "one"])
        self.assertEqual(matches[0].document, "A fox chased another fox.")
        self.assertEqual(matches[0].date, "1/2/2024")
        self.assertGreater(matches[0].score, matches[1].score)

    def test_rare_query_term_has_more_weight_than_a_common_term(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("rare", "gorilla", 1),
                make_dream("common-one", "house", 2),
                make_dream("common-two", "house", 3),
            ],
            b=0,
        )

        matches = index.search("gorilla house")

        self.assertEqual(matches[0].dream_id, "rare")

    def test_document_length_normalization_favors_the_concise_match(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("short", "basement", 1),
                make_dream("long", "basement " + "filler " * 30, 2),
            ]
        )

        matches = index.search("basement")

        self.assertEqual([match.dream_id for match in matches], ["short", "long"])

    def test_stemming_matches_inflected_text_and_irregular_past_tense(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("running", "I was running away.", 1),
                make_dream("ran", "I ran away.", 2),
            ]
        )

        self.assertEqual(
            {match.dream_id for match in index.search("run")},
            {"running", "ran"},
        )

    def test_apostrophe_query_does_not_match_apostrophe_fragments(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("contraction", "I don't remember.", 1),
                make_dream("fragments", "Don met the letter T.", 2),
            ]
        )

        self.assertEqual(
            [match.dream_id for match in index.search("don't")],
            ["contraction"],
        )

    def test_search_returns_no_arbitrary_results_for_an_unknown_term(self) -> None:
        index = DreamBm25Index([make_dream("one", "A familiar house.", 1)])

        self.assertEqual(index.search("xylophone"), [])

    def test_repeated_query_terms_do_not_change_scores(self) -> None:
        index = DreamBm25Index([make_dream("one", "house house", 1)])

        single = index.search("house")[0]
        repeated = index.search("house house house")[0]

        self.assertEqual(single.score, repeated.score)

    def test_date_filter_is_inclusive_and_omits_undated_dreams(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("before", "house", 1),
                make_dream("inside", "house", 2),
                make_dream("undated", "house", None),
            ]
        )

        matches = index.search(
            "house",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
        )

        self.assertEqual([match.dream_id for match in matches], ["inside"])

    def test_equal_scores_preserve_corpus_order(self) -> None:
        index = DreamBm25Index(
            [
                make_dream("first", "house", 1),
                make_dream("second", "house", 2),
            ]
        )

        self.assertEqual(
            [match.dream_id for match in index.search("house")],
            ["first", "second"],
        )

    def test_validation_rejects_invalid_configuration_and_queries(self) -> None:
        dream = make_dream("duplicate", "house", 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            DreamBm25Index([dream, dream])
        with self.assertRaisesRegex(ValueError, "k1"):
            DreamBm25Index([], k1=0)
        with self.assertRaisesRegex(ValueError, "between"):
            DreamBm25Index([], b=1.1)

        index = DreamBm25Index([dream])
        with self.assertRaisesRegex(ValueError, "empty"):
            index.search(" ")
        with self.assertRaisesRegex(ValueError, "positive"):
            index.search("house", limit=0)
        with self.assertRaisesRegex(ValueError, "start_date"):
            index.search(
                "house",
                start_date=date(2024, 1, 2),
                end_date=date(2024, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
