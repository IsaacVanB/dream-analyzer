"""Shared, atomic writers for generated text and JSON artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path | str, content: str) -> Path:
    """Atomically replace a text file after writing it beside the destination."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if temporary_path is None:
            raise RuntimeError("temporary artifact was not created")
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_json_atomic(
    path: Path | str,
    value: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Serialize JSON with a trailing newline and replace its destination."""
    content = json.dumps(value, indent=indent, ensure_ascii=ensure_ascii) + "\n"
    return write_text_atomic(path, content)
