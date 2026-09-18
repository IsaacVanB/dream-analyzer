"""Character context and mention-analysis tools for the dream agent."""

from __future__ import annotations

from datetime import date
from typing import Any

from dream_analysis.characters import CharacterLookupService
from dream_analysis.dates import parse_date_bound, validate_date_range
from dream_analysis.tool_protocols import (
    AnalyticalToolResult,
    CharacterRecordRepository,
    DreamCollectionRepository,
    StructuredDreamRecordRepository,
)


class CharacterContextTool:
    """Expose manually curated character context by canonical name or alias."""

    name = "get_character_context"
    max_names = 20
    max_name_chars = 100
    max_aliases = 20
    max_history_entries = 20

    def __init__(
        self,
        repository: CharacterRecordRepository,
        *,
        max_relationship_chars: int = 500,
        max_context_chars: int = 2000,
    ) -> None:
        if max_relationship_chars < 1:
            raise ValueError("max_relationship_chars must be positive")
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")
        self.repository = repository
        self.max_relationship_chars = max_relationship_chars
        self.max_context_chars = max_context_chars

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Look up manually curated background about named characters "
                    "in the private dream journal. Match canonical names and "
                    "aliases case-insensitively. Use this after a dream retrieval "
                    "mentions a named character, or when the user asks who that "
                    "person is. An optional dream_date selects relationship context "
                    "that applied on that date. This is contextual reference data, "
                    "not evidence that the person appeared in a particular dream."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "description": "Character names or aliases to look up.",
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_name_chars,
                            },
                            "minItems": 1,
                            "maxItems": self.max_names,
                            "uniqueItems": True,
                        },
                        "dream_date": {
                            "type": "string",
                            "format": "date",
                            "description": (
                                "Optional date of the dream in YYYY-MM-DD format, "
                                "used to select date-bounded relationship history."
                            ),
                        },
                    },
                    "required": ["names"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[AnalyticalToolResult, AnalyticalToolResult]:
        unexpected = sorted(set(arguments) - {"names", "dream_date"})
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        names = self._names(arguments.get("names"))
        dream_date = parse_date_bound(
            arguments.get("dream_date"), argument_name="dream_date"
        )
        records = self.repository.all()
        index = self._index(records)

        characters: list[dict[str, Any]] = []
        missing_names: list[str] = []
        ambiguous_names: list[str] = []
        for requested_name in names:
            candidates = index.get(requested_name.casefold(), [])
            if not candidates:
                missing_names.append(requested_name)
                continue
            if len(candidates) > 1:
                ambiguous_names.append(requested_name)
            for character, matched_by, matched_value in candidates:
                characters.append(
                    self._result(
                        character,
                        requested_name=requested_name,
                        matched_by=matched_by,
                        matched_value=matched_value,
                        dream_date=dream_date,
                    )
                )

        parameters = {
            "names": names,
            "dream_date": dream_date.isoformat() if dream_date else None,
        }
        analysis = {
            "dictionary_character_count": len(records),
            "requested_name_count": len(names),
            "matched_name_count": len(names) - len(missing_names),
            "missing_names": missing_names,
            "ambiguous_names": ambiguous_names,
            "characters": characters,
        }
        warnings = self._warnings(
            missing_names=missing_names,
            ambiguous_names=ambiguous_names,
            characters=characters,
            dream_date=dream_date,
        )
        report: AnalyticalToolResult = {
            "evidence_type": "character_context",
            "parameters": parameters,
            "analysis": analysis,
            "warnings": warnings,
        }

        bounded_characters = [self._bounded(character) for character in characters]
        bounded_analysis = {**analysis, "characters": bounded_characters}
        bounded_warnings = list(warnings)
        truncated = sum(
            bool(character.get("content_truncated"))
            for character in bounded_characters
        )
        if truncated:
            bounded_warnings.append(
                f"Context was truncated for {truncated} character result(s) in "
                "model evidence; the saved report retains complete content."
            )
        bounded: AnalyticalToolResult = {
            "evidence_type": "character_context",
            "parameters": parameters,
            "analysis": bounded_analysis,
            "warnings": bounded_warnings,
        }
        return bounded, report

    def _names(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("names must be a non-empty array of strings")
        if len(value) > self.max_names:
            raise ValueError(f"names cannot contain more than {self.max_names} items")
        if any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("names must contain non-empty strings")
        names = [name.strip() for name in value]
        if any(len(name) > self.max_name_chars for name in names):
            raise ValueError(
                f"each name cannot exceed {self.max_name_chars} characters"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("names must not contain case-insensitive duplicates")
        return names

    @staticmethod
    def _index(
        records: list[dict[str, Any]],
    ) -> dict[str, list[tuple[dict[str, Any], str, str]]]:
        index: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
        for character in records:
            values = [("name", str(character["name"]))]
            values.extend(("alias", str(alias)) for alias in character.get("aliases", []))
            seen_values: set[str] = set()
            for matched_by, value in values:
                identity = value.strip().casefold()
                if identity in seen_values:
                    continue
                seen_values.add(identity)
                index.setdefault(identity, []).append(
                    (character, matched_by, value.strip())
                )
        return index

    def _result(
        self,
        character: dict[str, Any],
        *,
        requested_name: str,
        matched_by: str,
        matched_value: str,
        dream_date: date | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requested_name": requested_name,
            "matched_by": matched_by,
            "matched_value": matched_value,
            "id": character["id"],
            "name": character["name"],
            "aliases": list(character.get("aliases", [])),
        }
        if "relationship_history" in character:
            history = [dict(entry) for entry in character["relationship_history"]]
            if dream_date is not None:
                history = [
                    entry
                    for entry in history
                    if self._history_applies(entry, dream_date)
                ]
            result["relationship_history"] = history
        else:
            result["relationship"] = character.get("relationship", "")
            result["context"] = character.get("context", "")
        if isinstance(character.get("mentions"), dict):
            mentions = character["mentions"]
            result["mentions"] = {
                key: mentions.get(key)
                for key in ("count", "first_date", "last_date")
                if key in mentions
            }
        return result

    @staticmethod
    def _history_applies(entry: dict[str, Any], dream_date: date) -> bool:
        start = parse_date_bound(entry.get("start_date"), argument_name="start_date")
        end = parse_date_bound(entry.get("end_date"), argument_name="end_date")
        return (start is None or start <= dream_date) and (
            end is None or dream_date <= end
        )

    def _bounded(self, character: dict[str, Any]) -> dict[str, Any]:
        bounded = dict(character)
        truncated = False
        aliases = list(character.get("aliases", []))
        bounded["aliases"] = aliases[: self.max_aliases]
        if len(aliases) > self.max_aliases:
            bounded["aliases_omitted"] = len(aliases) - self.max_aliases
            truncated = True
        for field, maximum in (
            ("relationship", self.max_relationship_chars),
            ("context", self.max_context_chars),
        ):
            if isinstance(character.get(field), str) and len(character[field]) > maximum:
                bounded[field] = character[field][:maximum] + "\n[TRUNCATED]"
                truncated = True
        if "relationship_history" in character:
            history = []
            full_history = list(character["relationship_history"])
            for entry in full_history[: self.max_history_entries]:
                bounded_entry = dict(entry)
                for field, maximum in (
                    ("relationship", self.max_relationship_chars),
                    ("context", self.max_context_chars),
                ):
                    if len(bounded_entry.get(field, "")) > maximum:
                        bounded_entry[field] = (
                            bounded_entry[field][:maximum] + "\n[TRUNCATED]"
                        )
                        truncated = True
                history.append(bounded_entry)
            bounded["relationship_history"] = history
            if len(full_history) > self.max_history_entries:
                bounded["relationship_history_entries_omitted"] = (
                    len(full_history) - self.max_history_entries
                )
                truncated = True
        bounded["content_truncated"] = truncated
        return bounded

    @staticmethod
    def _warnings(
        *,
        missing_names: list[str],
        ambiguous_names: list[str],
        characters: list[dict[str, Any]],
        dream_date: date | None,
    ) -> list[str]:
        warnings: list[str] = []
        if missing_names:
            warnings.append(
                "No character dictionary entry was found for: "
                + ", ".join(missing_names)
                + "."
            )
        if ambiguous_names:
            warnings.append(
                "Multiple character entries matched: "
                + ", ".join(ambiguous_names)
                + ". Treat them as ambiguous."
            )
        if dream_date is not None:
            without_history = [
                character["name"]
                for character in characters
                if "relationship_history" in character
                and not character["relationship_history"]
            ]
            if without_history:
                warnings.append(
                    f"No relationship history applies on {dream_date.isoformat()} "
                    "for: " + ", ".join(without_history) + "."
                )
        return warnings


class CharacterMentionsTool:
    """Expose named-character mentions from structured dream records."""

    name = "get_character_mentions"
    max_names = 20
    max_name_chars = 100
    max_limit = 50
    max_dream_ids_per_character = 20

    def __init__(
        self,
        dream_repository: DreamCollectionRepository,
        structured_repository: StructuredDreamRecordRepository,
    ) -> None:
        self.dream_repository = dream_repository
        self.structured_repository = structured_repository

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Count explicit proper-name character mentions extracted from "
                    "structured dream records. Use this for who appears most often, "
                    "how often named characters appear, first or last appearances, "
                    "or named-character activity within an inclusive date range. "
                    "Results cover named_characters only and report structuring "
                    "coverage; they do not use manually edited character profiles."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "description": (
                                "Optional exact character names matched "
                                "case-insensitively. Omit to rank all names."
                            ),
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_name_chars,
                            },
                            "minItems": 1,
                            "maxItems": self.max_names,
                            "uniqueItems": True,
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive lower date bound.",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive upper date bound.",
                        },
                        "minimum_mentions": {
                            "type": "integer",
                            "minimum": 1,
                            "description": (
                                "Minimum distinct-dream mentions required."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.max_limit,
                            "description": "Maximum ranked characters to return.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[AnalyticalToolResult, AnalyticalToolResult]:
        allowed = {
            "names",
            "start_date",
            "end_date",
            "minimum_mentions",
            "limit",
        }
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        names = self._names(arguments.get("names"))
        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        validate_date_range(start_date, end_date)
        minimum_mentions = self._positive_integer(
            arguments.get("minimum_mentions", 1),
            name="minimum_mentions",
        )
        limit = self._positive_integer(
            arguments.get("limit", 25),
            name="limit",
            maximum=self.max_limit,
        )
        parameters = {
            "names": names,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "minimum_mentions": minimum_mentions,
            "limit": limit,
        }

        dreams = self.dream_repository.all()
        current_by_id = {dream.dream_id: dream for dream in dreams}
        structured = self.structured_repository.all()
        current_structured = [
            record
            for record in structured
            if str(record.get("dream_id")) in current_by_id
        ]
        coverage = self._coverage(
            parsed_count=len(dreams),
            structured_count=len(structured),
            current_structured_count=len(current_structured),
        )

        selected_records: list[dict[str, Any]] = []
        excluded_unknown_dates = 0
        for record in current_structured:
            dream = current_by_id[str(record["dream_id"])]
            if start_date is not None or end_date is not None:
                if dream.date_sort is None:
                    excluded_unknown_dates += 1
                    continue
                if start_date is not None and dream.date_sort < start_date:
                    continue
                if end_date is not None and dream.date_sort > end_date:
                    continue
            normalized = dict(record)
            normalized["date_sort"] = (
                dream.date_sort.isoformat() if dream.date_sort else None
            )
            selected_records.append(normalized)

        characters = CharacterLookupService().aggregate(
            selected_records,
            temporal_context=False,
        )
        requested = {name.casefold() for name in names or []}
        available = {character["name"].casefold() for character in characters}
        if requested:
            characters = [
                character
                for character in characters
                if character["name"].casefold() in requested
            ]
        characters = [
            self._character_result(character)
            for character in characters
            if int(character["mentions"]["count"]) >= minimum_mentions
        ]
        characters.sort(
            key=lambda character: (
                -int(character["mentions"]["count"]),
                str(character["name"]).casefold(),
            )
        )
        character_count = len(characters)
        characters = characters[:limit]
        missing_names = [
            name for name in names or [] if name.casefold() not in available
        ]
        analysis = {
            "coverage": coverage,
            "selected_structured_dream_count": len(selected_records),
            "excluded_unknown_date_count": excluded_unknown_dates,
            "start_date": parameters["start_date"],
            "end_date": parameters["end_date"],
            "minimum_mentions": minimum_mentions,
            "character_count": character_count,
            "characters_omitted_by_limit": max(0, character_count - limit),
            "missing_names": missing_names,
            "characters": characters,
        }
        report_warnings = self._warnings(
            coverage=coverage,
            missing_names=missing_names,
            excluded_unknown_dates=excluded_unknown_dates,
            characters_omitted=max(0, character_count - limit),
        )
        report: AnalyticalToolResult = {
            "evidence_type": "character_mentions",
            "parameters": parameters,
            "analysis": analysis,
            "warnings": report_warnings,
        }

        bounded_characters = [
            self._bound_character(character) for character in characters
        ]
        bounded_analysis = {**analysis, "characters": bounded_characters}
        truncated_lists = sum(
            character["mentions"]["dream_ids_omitted"] > 0
            for character in bounded_characters
        )
        warnings = list(report_warnings)
        if truncated_lists:
            warnings.append(
                f"Dream ID lists are truncated for {truncated_lists} "
                "character(s) in model evidence; the saved report retains all IDs."
            )
        bounded: AnalyticalToolResult = {
            "evidence_type": "character_mentions",
            "parameters": parameters,
            "analysis": bounded_analysis,
            "warnings": warnings,
        }
        return bounded, report

    def _names(self, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError("names must be a non-empty array of strings")
        if len(value) > self.max_names:
            raise ValueError(f"names cannot contain more than {self.max_names} items")
        if any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("names must contain non-empty strings")
        names = [name.strip() for name in value]
        if any(len(name) > self.max_name_chars for name in names):
            raise ValueError(
                f"each name cannot exceed {self.max_name_chars} characters"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("names must not contain case-insensitive duplicates")
        return names

    @staticmethod
    def _positive_integer(
        value: Any,
        *,
        name: str,
        maximum: int | None = None,
    ) -> int:
        if type(value) is not int or value < 1 or (
            maximum is not None and value > maximum
        ):
            detail = "a positive integer"
            if maximum is not None:
                detail = f"between 1 and {maximum}"
            raise ValueError(f"{name} must be {detail}")
        return value

    @staticmethod
    def _coverage(
        *,
        parsed_count: int,
        structured_count: int,
        current_structured_count: int,
    ) -> dict[str, Any]:
        return {
            "parsed_dream_count": parsed_count,
            "structured_record_count": structured_count,
            "structured_current_dream_count": current_structured_count,
            "unstructured_dream_count": max(
                0, parsed_count - current_structured_count
            ),
            "orphaned_structured_record_count": max(
                0, structured_count - current_structured_count
            ),
            "coverage_percentage": round(
                (current_structured_count / parsed_count) * 100, 2
            )
            if parsed_count
            else 0.0,
        }

    @staticmethod
    def _character_result(character: dict[str, Any]) -> dict[str, Any]:
        mentions = dict(character["mentions"])
        return {
            "id": character["id"],
            "name": character["name"],
            "mentions": mentions,
        }

    def _bound_character(self, character: dict[str, Any]) -> dict[str, Any]:
        mentions = dict(character["mentions"])
        dream_ids = list(mentions["dream_ids"])
        mentions["dream_ids"] = dream_ids[: self.max_dream_ids_per_character]
        mentions["dream_ids_omitted"] = max(
            0, len(dream_ids) - self.max_dream_ids_per_character
        )
        return {**character, "mentions": mentions}

    @staticmethod
    def _warnings(
        *,
        coverage: dict[str, Any],
        missing_names: list[str],
        excluded_unknown_dates: int,
        characters_omitted: int,
    ) -> list[str]:
        warnings: list[str] = []
        if coverage["coverage_percentage"] < 100:
            warnings.append(
                "Character results cover "
                f"{coverage['structured_current_dream_count']} of "
                f"{coverage['parsed_dream_count']} current dreams "
                f"({coverage['coverage_percentage']}%). Unstructured dreams "
                "cannot contribute named-character mentions."
            )
        if coverage["orphaned_structured_record_count"]:
            warnings.append(
                f"{coverage['orphaned_structured_record_count']} structured "
                "record(s) do not match a current dream ID and were excluded."
            )
        if missing_names:
            warnings.append(
                "No structured mention was found in the selected range for: "
                + ", ".join(missing_names)
                + "."
            )
        if excluded_unknown_dates:
            warnings.append(
                f"{excluded_unknown_dates} structured current dream(s) with an "
                "unknown date were excluded by the requested date range."
            )
        if characters_omitted:
            warnings.append(
                f"{characters_omitted} character(s) were omitted by the requested "
                "limit."
            )
        return warnings
