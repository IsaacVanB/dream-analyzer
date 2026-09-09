"""Append-oriented journal imports and persistent change manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence

from dream_analysis.artifacts import write_json_atomic, write_text_atomic
from dream_analysis.parser import JournalParser
from dream_analysis.repository import DreamRepository


IMPORT_STATE_VERSION = 1
DEFAULT_IMPORT_STATE_PATH = Path("data/dream_imports.json")
_ID_SUFFIX_RE = re.compile(r"^(.*)-(\d+)$")


class AppendOnlyImportError(ValueError):
    """Raised when a journal no longer resembles an append-only update."""


def normalize_dream_text(text: str) -> str:
    """Return a stable representation for hashing and typo comparison."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.strip().splitlines())


def dream_text_hash(text: str) -> str:
    """Hash the normalized text supplied to expensive downstream operations."""
    return hashlib.sha256(normalize_dream_text(text).encode("utf-8")).hexdigest()


def journal_hash(journal_text: str) -> str:
    normalized = unicodedata.normalize("NFC", journal_text).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _comparison_text(text: str) -> str:
    return " ".join(normalize_dream_text(text).casefold().split())


def _records_differ(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    fields = (
        "date",
        "year",
        "month",
        "day",
        "date_precision",
        "date_sort",
        "tags",
        "text",
    )
    return any(old.get(field) != new.get(field) for field in fields)


def _serialize_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )


def load_import_state(path: Path | str) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid import state at {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != IMPORT_STATE_VERSION:
        raise ValueError(f"Unsupported import state at {source}")
    return value


def latest_import(state: Mapping[str, Any]) -> dict[str, Any]:
    import_id = state.get("latest_import_id")
    imports = state.get("imports", {})
    manifest = imports.get(import_id) if isinstance(imports, dict) else None
    if not isinstance(manifest, dict):
        raise ValueError("Import state does not contain a valid latest import.")
    return manifest


def import_new_ids(path: Path | str, import_id: str = "latest") -> list[str]:
    state = load_import_state(path)
    if state is None:
        raise ValueError(f"Import state not found: {path}")
    selected_id = (
        state.get("latest_import_id") if import_id == "latest" else import_id
    )
    manifest = state.get("imports", {}).get(selected_id)
    if not isinstance(manifest, dict):
        raise ValueError(f"Import not found: {selected_id}")
    new_ids = manifest.get("new_ids")
    if not isinstance(new_ids, list) or any(not isinstance(item, str) for item in new_ids):
        raise ValueError(f"Import {selected_id} has invalid new_ids.")
    return new_ids


@dataclass(frozen=True, slots=True)
class JournalImportResult:
    records: list[dict[str, Any]]
    state: dict[str, Any]
    manifest: dict[str, Any]
    repeated: bool = False


class AppendOnlyJournalImporter:
    """Reconcile a new journal against the existing positional prefix."""

    def __init__(self, *, minimum_similarity: float = 0.9) -> None:
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        self.minimum_similarity = minimum_similarity

    def prepare(
        self,
        journal_text: str,
        existing_records: Sequence[Mapping[str, Any]],
        *,
        state: Mapping[str, Any] | None = None,
        source: str | None = None,
        dream_separator_blank_lines: int | None = None,
    ) -> JournalImportResult:
        parsed = JournalParser(
            dream_separator_blank_lines=dream_separator_blank_lines
        ).parse(journal_text)
        old = [dict(record) for record in existing_records]
        if len(parsed) < len(old):
            raise AppendOnlyImportError(
                f"Parsed journal has {len(parsed)} dreams, fewer than the existing "
                f"{len(old)}; deletion or truncation is not supported."
            )

        source_hash = journal_hash(journal_text)
        current_state = dict(state) if state is not None else self._baseline_state(old)
        imports = dict(current_state.get("imports", {}))
        for manifest in imports.values():
            if isinstance(manifest, dict) and manifest.get("source_hash") == source_hash:
                return JournalImportResult(
                    old, current_state, dict(manifest), repeated=True
                )

        records: list[dict[str, Any]] = []
        unchanged_ids: list[str] = []
        edited_ids: list[str] = []
        for index, previous in enumerate(old):
            candidate = dict(parsed[index])
            previous_text = str(previous.get("text", ""))
            candidate_text = str(candidate.get("text", ""))
            similarity = SequenceMatcher(
                None, _comparison_text(previous_text), _comparison_text(candidate_text)
            ).ratio()
            if (
                dream_text_hash(previous_text) != dream_text_hash(candidate_text)
                and similarity < self.minimum_similarity
            ):
                raise AppendOnlyImportError(
                    f"Dream {index + 1} no longer matches existing dream "
                    f"{previous.get('dream_id')!r} (text similarity {similarity:.3f}). "
                    "This looks like an insertion, deletion, reorder, or substantive edit."
                )
            candidate["dream_id"] = previous["dream_id"]
            records.append(candidate)
            if _records_differ(previous, candidate):
                edited_ids.append(str(previous["dream_id"]))
            else:
                unchanged_ids.append(str(previous["dream_id"]))

        used_ids = {str(record["dream_id"]) for record in old}
        new_ids: list[str] = []
        for candidate in parsed[len(old) :]:
            candidate = dict(candidate)
            candidate["dream_id"] = self._allocate_id(candidate, used_ids)
            used_ids.add(candidate["dream_id"])
            new_ids.append(candidate["dream_id"])
            records.append(candidate)

        import_id = f"journal-{source_hash[:12]}"
        manifest = {
            "import_id": import_id,
            "imported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "source_hash": source_hash,
            "dream_count": len(records),
            "unchanged_ids": unchanged_ids,
            "edited_ids": edited_ids,
            "new_ids": new_ids,
            "counts": {
                "unchanged": len(unchanged_ids),
                "edited": len(edited_ids),
                "new": len(new_ids),
            },
        }
        catalog = dict(current_state.get("dreams", {}))
        for record in records:
            dream_id = str(record["dream_id"])
            entry = dict(catalog.get(dream_id, {}))
            entry.setdefault("first_seen_import", import_id)
            entry["current_text_hash"] = dream_text_hash(str(record["text"]))
            catalog[dream_id] = entry
        imports[import_id] = manifest
        updated_state = {
            "version": IMPORT_STATE_VERSION,
            "latest_import_id": import_id,
            "dreams": catalog,
            "imports": imports,
        }
        return JournalImportResult(records, updated_state, manifest)

    @staticmethod
    def _baseline_state(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "version": IMPORT_STATE_VERSION,
            "latest_import_id": None,
            "dreams": {
                str(record["dream_id"]): {
                    "first_seen_import": "baseline",
                    "current_text_hash": dream_text_hash(str(record.get("text", ""))),
                }
                for record in records
            },
            "imports": {},
        }

    @staticmethod
    def _allocate_id(record: Mapping[str, Any], used_ids: set[str]) -> str:
        proposed = str(record["dream_id"])
        if proposed not in used_ids:
            return proposed
        match = _ID_SUFFIX_RE.match(proposed)
        if match is None:
            base = proposed
        else:
            base = match.group(1)
        suffixes = [
            int(match.group(1))
            for dream_id in used_ids
            if (match := re.match(rf"^{re.escape(base)}-(\d+)$", dream_id))
        ]
        return f"{base}-{max(suffixes, default=-1) + 1}"


def sync_journal_files(
    input_path: Path | str,
    dreams_path: Path | str,
    state_path: Path | str = DEFAULT_IMPORT_STATE_PATH,
    *,
    dry_run: bool = False,
    minimum_similarity: float = 0.9,
    dream_separator_blank_lines: int | None = None,
) -> JournalImportResult:
    """Prepare and optionally persist one append-only journal import."""
    input_path = Path(input_path)
    dreams_path = Path(dreams_path)
    existing = DreamRepository(dreams_path).records() if dreams_path.exists() else []
    importer = AppendOnlyJournalImporter(minimum_similarity=minimum_similarity)
    result = importer.prepare(
        input_path.read_text(encoding="utf-8"),
        existing,
        state=load_import_state(state_path),
        source=str(input_path),
        dream_separator_blank_lines=dream_separator_blank_lines,
    )
    if not dry_run and not result.repeated:
        write_text_atomic(dreams_path, _serialize_jsonl(result.records))
        write_json_atomic(state_path, result.state)
    return result
