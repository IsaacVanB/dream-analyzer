from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import basic_rag
from dream_analysis.config import Settings
from dream_analysis.models import SearchResult


class BasicRagCompatibilityTests(unittest.TestCase):
    @patch("basic_rag._make_service")
    def test_retrieval_keeps_legacy_dictionary_shape(self, make_service) -> None:
        service = Mock()
        service.retrieve.return_value = [
            SearchResult(
                dream_id="dream-1",
                document="Dream text",
                metadata={"date": "1/2/2024"},
                distance=0.25,
            )
        ]
        make_service.return_value = service

        retrieved = basic_rag.retrieve_dreams(
            "hidden room",
            top_k=3,
            chroma_path="custom-chroma",
            collection_name="custom-collection",
            embed_model="custom-embed",
        )

        make_service.assert_called_once_with(
            chroma_path="custom-chroma",
            collection_name="custom-collection",
            embed_model="custom-embed",
        )
        service.retrieve.assert_called_once_with("hidden room", top_k=3)
        self.assertEqual(
            retrieved,
            [
                {
                    "dream_id": "dream-1",
                    "date": "1/2/2024",
                    "distance": 0.25,
                    "document": "Dream text",
                }
            ],
        )

    def test_context_accepts_legacy_result_dictionaries(self) -> None:
        context = basic_rag.format_context(
            [
                {
                    "dream_id": "dream-1",
                    "date": "1/2/2024",
                    "distance": 0.25,
                    "document": "Dream text",
                }
            ]
        )

        self.assertIn("DREAM_ID: dream-1", context)
        self.assertIn("DATE: 1/2/2024", context)
        self.assertIn("Dream text", context)

    def test_expected_model_constants_remain_public(self) -> None:
        settings = Settings()
        self.assertEqual(basic_rag.CHAT_MODEL, settings.ollama.chat_model)
        self.assertEqual(basic_rag.EMBED_MODEL, settings.ollama.embedding_model)
        self.assertEqual(
            basic_rag.COLLECTION_NAME,
            settings.index.collection_name,
        )


if __name__ == "__main__":
    unittest.main()
