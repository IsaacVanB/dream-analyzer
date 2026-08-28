from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dream_analysis.artifacts import write_json_atomic, write_text_atomic


class ArtifactWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_text_writer_creates_parents_and_preserves_exact_content(self) -> None:
        path = self.root / "nested" / "report.md"

        result = write_text_atomic(path, "# Report\n")

        self.assertEqual(result, path)
        self.assertEqual(path.read_text(encoding="utf-8"), "# Report\n")
        self.assertEqual(list(path.parent.glob(".report.md.*.tmp")), [])

    def test_json_writer_is_utf8_and_has_one_trailing_newline(self) -> None:
        path = self.root / "artifact.json"

        write_json_atomic(path, {"dream": "café"})

        content = path.read_text(encoding="utf-8")
        self.assertTrue(content.endswith("\n"))
        self.assertFalse(content.endswith("\n\n"))
        self.assertEqual(json.loads(content), {"dream": "café"})
        self.assertIn("café", content)

    def test_failed_replace_preserves_destination_and_removes_temporary_file(self) -> None:
        path = self.root / "report.md"
        path.write_text("old content", encoding="utf-8")

        with patch(
            "dream_analysis.artifacts.os.replace",
            side_effect=OSError("replace failed"),
        ):
            with self.assertRaisesRegex(OSError, "replace failed"):
                write_text_atomic(path, "new content")

        self.assertEqual(path.read_text(encoding="utf-8"), "old content")
        self.assertEqual(list(self.root.glob(".report.md.*.tmp")), [])

    def test_text_writer_rejects_non_string_content_before_writing(self) -> None:
        path = self.root / "report.md"

        with self.assertRaisesRegex(TypeError, "content must be a string"):
            write_text_atomic(path, 123)  # type: ignore[arg-type]

        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
