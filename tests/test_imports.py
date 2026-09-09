from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dream_analysis.imports import (
    AppendOnlyImportError,
    AppendOnlyJournalImporter,
    import_new_ids,
    load_import_state,
    sync_journal_files,
)
from dream_analysis.parser import JournalParser


OLD_JOURNAL = "1/1/2024\nI walked through a quiet house.\n1/2/2024\nI missed a train.\n"
APPENDED_JOURNAL = OLD_JOURNAL + "1/2/2024\nI found a red bicycle.\n"


class AppendOnlyJournalImporterTests(unittest.TestCase):
    def test_append_preserves_old_ids_and_allocates_colliding_date_id(self) -> None:
        old = JournalParser().parse(OLD_JOURNAL)

        result = AppendOnlyJournalImporter().prepare(APPENDED_JOURNAL, old)

        self.assertEqual(
            [record["dream_id"] for record in result.records[:2]],
            [record["dream_id"] for record in old],
        )
        self.assertEqual(result.manifest["new_ids"], ["dream-2024-1-2-1"])
        self.assertEqual(result.manifest["counts"], {"unchanged": 2, "edited": 0, "new": 1})

    def test_minor_edit_is_old_but_shifted_prefix_is_rejected(self) -> None:
        old = JournalParser().parse(OLD_JOURNAL)
        edited = OLD_JOURNAL.replace("quiet house", "quite house") + "1/3/2024\nNew dream.\n"

        result = AppendOnlyJournalImporter().prepare(edited, old)

        self.assertEqual(result.manifest["edited_ids"], [old[0]["dream_id"]])
        self.assertEqual(len(result.manifest["new_ids"]), 1)

        inserted = "1/1/2024\nA completely unrelated inserted dream.\n" + OLD_JOURNAL
        with self.assertRaises(AppendOnlyImportError):
            AppendOnlyJournalImporter().prepare(inserted, old)

    def test_file_sync_is_dry_run_capable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "journal.txt"
            dreams_path = root / "dreams.jsonl"
            state_path = root / "imports.json"
            input_path.write_text(APPENDED_JOURNAL, encoding="utf-8")
            dreams_path.write_text(
                "".join(json.dumps(item) + "\n" for item in JournalParser().parse(OLD_JOURNAL)),
                encoding="utf-8",
            )

            preview = sync_journal_files(input_path, dreams_path, state_path, dry_run=True)
            self.assertEqual(len(preview.manifest["new_ids"]), 1)
            self.assertFalse(state_path.exists())

            imported = sync_journal_files(input_path, dreams_path, state_path)
            repeated = sync_journal_files(input_path, dreams_path, state_path)

            self.assertFalse(imported.repeated)
            self.assertTrue(repeated.repeated)
            self.assertEqual(import_new_ids(state_path), imported.manifest["new_ids"])
            self.assertEqual(load_import_state(state_path)["latest_import_id"], imported.manifest["import_id"])


if __name__ == "__main__":
    unittest.main()
