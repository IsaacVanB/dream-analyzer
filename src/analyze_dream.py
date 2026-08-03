#!/usr/bin/env python3
"""Analyze one dream with an Ollama chat model."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama


DREAMS_PATH = Path("data/dreams.jsonl")
OUTPUT_DIR = Path("outputs/analysis")
CHAT_MODEL = "qwen3:8b"


def load_dream_by_id(path: Path, dream_id: str) -> dict[str, Any]:
    """Return the JSONL record matching dream_id."""
    with path.open("r", encoding="utf-8") as dream_file:
        for line_number, line in enumerate(dream_file, start=1):
            if not line.strip():
                continue
            try:
                dream = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if dream.get("dream_id") == dream_id:
                return dream

    raise ValueError(f"Dream ID not found in {path}: {dream_id}")


def analyze_dream(
    text: str,
    *,
    chat_model: str = CHAT_MODEL,
    dream_id: str | None = None,
    date: str | None = None,
    tags: list[str] | None = None,
    num_ctx: int = 4096,
    num_predict: int = 900,
    temperature: float = 0.2,
) -> str:
    """Ask Ollama for a close, evidence-based analysis of one dream."""
    if not text.strip():
        raise ValueError("Dream text cannot be empty.")

    system_prompt = (
        "You analyze an individual dream as a narrative and subjective mental "
        "experience. Begin with what is concretely happening, then make careful "
        "interpretive hypotheses grounded in the supplied text. Discuss taboo, "
        "sexual, violent, shameful, disturbing, or contradictory material "
        "directly when it is present; do not sanitize it or avoid it. Do not try "
        "to validate, reassure, comfort, flatter, or morally judge the dreamer. "
        "Do not use universal dream dictionaries, fixed symbolic meanings, or "
        "claims such as 'X always symbolizes Y.' Treat interpretations as "
        "possibilities rather than facts, and distinguish evidence from "
        "inference. Do not diagnose mental illness or infer real-world events "
        "that the dream does not establish."
    )

    metadata_lines = []
    if dream_id is not None:
        metadata_lines.append(f"DREAM_ID: {dream_id}")
    if date is not None:
        metadata_lines.append(f"DATE: {date}")
    if tags:
        metadata_lines.append(f"JOURNAL_TAGS: {', '.join(tags)}")
    metadata = "\n".join(metadata_lines) or "SOURCE: text supplied directly"

    user_prompt = f"""
/no_think

{metadata}

DREAM TEXT:
{text}

Analyze this dream using these sections:
1. What happens: a concise account of the events, shifts, characters, and setting.
2. Emotional and relational dynamics: tensions, desires, fears, power relations,
   contradictions, and changes in the dreamer's position.
3. Themes and motifs: the strongest recurring ideas or images, with evidence.
4. Interpretation: several plausible readings tied closely to details in the
   dream, including uncomfortable readings when supported.
5. Uncertainties: details whose meaning depends on personal context, plus a few
   focused questions that would help distinguish between interpretations.
"""

    response = ollama.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )

    content = response["message"]["content"]
    return content or "[No analysis returned by chat model.]"


def save_analysis(
    analysis: str,
    *,
    dream_id: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    timestamp: datetime | None = None,
) -> Path:
    """Save an analysis and return its output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datetime_text = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")

    if dream_id is None:
        filename = f"{datetime_text}.txt"
    else:
        safe_dream_id = re.sub(r"[^A-Za-z0-9._-]", "_", dream_id)
        filename = f"{safe_dream_id}_{datetime_text}.txt"

    output_path = output_dir / filename
    output_path.write_text(analysis.rstrip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a dream selected by ID or supplied as text."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dream-id",
        help="ID of a dream to load from the JSONL file.",
    )
    source_group.add_argument(
        "--text",
        help="Dream text to analyze directly.",
    )
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=DREAMS_PATH,
        help="Path to parsed dream JSONL records.",
    )
    parser.add_argument(
        "--chat-model",
        default=CHAT_MODEL,
        help="Ollama chat model name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the analysis text file is saved.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Ollama context window option.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=900,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the chat model.",
    )
    args = parser.parse_args()

    if args.dream_id is not None:
        dream = load_dream_by_id(args.dreams_path, args.dream_id)
        text = dream.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Dream {args.dream_id} has no valid text field.")
        dream_id = args.dream_id
        date = dream.get("date")
        tags = dream.get("tags")
        print(f"Analyzing {dream_id} | {date or 'date unknown'}\n")
    else:
        text = args.text
        dream_id = None
        date = None
        tags = None

    analysis = analyze_dream(
        text,
        chat_model=args.chat_model,
        dream_id=dream_id,
        date=date,
        tags=tags,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
    )
    print(analysis)
    output_path = save_analysis(
        analysis,
        dream_id=dream_id,
        output_dir=args.output_dir,
    )
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
