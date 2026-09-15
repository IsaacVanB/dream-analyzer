[![CI](https://github.com/IsaacVanB/dream-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/IsaacVanB/dream-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-red)

# Dream Journal Analyzer

A privacy-first local LLM application for semantic retrieval, longitudinal analysis, and natural-language exploration of a personal dream journal.  

Dream Journal Analyzer combines local LLMs and embeddings through Ollama, ChromaDB vector retrieval, deterministic analytical services, structured feature extraction, clustering, and a tool-calling agent. Journal data and model inference remain local.  

The project is designed around a simple goal: turn a large, long-running journal into a dataset that can be searched and analyzed through natural-language questions without sending private journal text to external AI services.  

# Features
- Semantic dream retrieval with local embeddings and ChromaDB
- Retrieval-augmented analysis with cited dream IDs and dates
- Tool-calling LLM agent for:
  - semantic search
  - exact dream lookup
  - tag filtering
  - date-range retrieval
  - journal statistics
  - tag-trend analysis
  - character mentions and context
- Structured extraction of dream content into validated records
- UMAP + HDBSCAN clustering for recurring-theme discovery
- Longitudinal statistics and normalized tag-frequency analysis
- Incremental journal/index synchronization without rebuilding unchanged data
- Retrieval and local-model comparison utilities
- Reusable Python services separated from CLI orchestration

A synthetic journal with 176 dreams from 2022–2026 is included for testing and demonstration.

## Setup

```bash
python -m pip install --editable ".[dev]"
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

The editable development install includes the runtime dependencies, pytest, and
Ruff. Omit the `dev` extra (`python -m pip install --editable .`) for a
runtime-only installation. Both forms make the reusable `dream_analysis`
package importable and install the consolidated `dream-analyzer` command. Run
`dream-analyzer --help` to list its subcommands, or
`dream-analyzer <subcommand> --help` for the options of a specific command.

## Quick start

Ollama must be running locally at `http://localhost:11434` with at least one
embedding model and one chat model available.

The included synthetic journal is the default input, so the basic workflow can
be run without additional arguments:

```bash
dream-analyzer parse
dream-analyzer index
dream-analyzer ask "What patterns appear in dreams about hidden rooms?"
```

To use your own journal, pass its path and choose an output file:

```bash
dream-analyzer parse path/to/journal.txt data/my_dreams.jsonl
```

See [example_usages.md](example_usages.md) for detailed workflows, command
arguments, retrieval-scoring notes, journal synchronization, structured feature
extraction, evaluation, and model comparison.

## Journal format

Start each group with a date in `M/D/YY` or `M/D/YYYY` format. Optional tags go
on `#tag`-only lines at the beginning of a dream, and blank lines separate
multiple dreams on the same date.

```text
1/22/2022
#house #recurring
There was an extra room behind the pantry.

I found a notebook in a freezer.

2/3/22
#animal
A white dog followed me through a grocery store.
```

The parser also supports partial dates and configurable blank-line separators.
See [example_usages.md](example_usages.md) for the full format and parsing
options.

## Main commands

```bash
dream-analyzer parse       # Parse a journal into JSONL
dream-analyzer index       # Synchronize the ChromaDB index
dream-analyzer ask         # Ask natural-language questions
dream-analyzer analyze     # Analyze one dream
dream-analyzer stats       # Compute journal statistics
dream-analyzer trends      # Plot tag trends over time
dream-analyzer cluster     # Discover recurring themes
```

Run `dream-analyzer <subcommand> --help` for command-specific options. Additional
utilities under `src/cli/` cover journal synchronization, structured extraction,
character lookup, retrieval evaluation, and model comparison.
