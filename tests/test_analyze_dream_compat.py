from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cli import analyze_dream
from dream_analysis.models import RelatedDream


class AnalyzeDreamCompatibilityTests(unittest.TestCase):
    def test_load_dream_by_id_keeps_record_dictionary_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dreams.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dream_id": "dream-1",
                        "date": "1/2/2024",
                        "date_sort": "2024-01-02",
                        "tags": ["house"],
                        "text": "Dream text",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dream = analyze_dream.load_dream_by_id(path, "dream-1")

        self.assertEqual(dream["dream_id"], "dream-1")
        self.assertEqual(dream["tags"], ["house"])
        self.assertEqual(dream["text"], "Dream text")

    @patch("cli.analyze_dream._make_service")
    def test_related_retrieval_keeps_legacy_dictionary_shape(self, make_service) -> None:
        service = Mock()
        service.find_related.return_value = [
            RelatedDream(
                dream_id="related-1",
                date="1/1/2024",
                text="Related text",
                similarity=0.75,
            )
        ]
        make_service.return_value = service

        result = analyze_dream.retrieve_related_dreams(
            "Target text",
            n_results=2,
            chroma_path="custom-chroma",
            collection_name="custom-collection",
            embed_model="custom-embed",
        )

        self.assertEqual(
            result,
            [
                {
                    "dream_id": "related-1",
                    "date": "1/1/2024",
                    "similarity": 0.75,
                    "text": "Related text",
                }
            ],
        )
        service.find_related.assert_called_once()

    def test_document_text_extraction_remains_public(self) -> None:
        document = "HEADER\n\n--- DREAM TEXT ---\n\nRaw dream"
        self.assertEqual(analyze_dream.extract_dream_text(document), "Raw dream")


if __name__ == "__main__":
    unittest.main()
