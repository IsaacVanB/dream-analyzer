[![CI](https://github.com/IsaacVanB/dream-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/IsaacVanB/dream-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-red)

# Dream Journal Analyzer

A privacy-first local AI system for searching, analyzing, and exploring a long-running dream journal.

Dream Journal Analyzer combines local LLMs, dense and keyword retrieval, agentic tool use, and longitudinal analysis to answer natural-language questions about a personal text corpus without sending journal contents to external AI services.

The project is also an experimental platform for evaluating retrieval strategies—including semantic search, BM25, hybrid retrieval, and LLM-directed search—on a labeled synthetic benchmark.

## What it does

- **Natural-language journal exploration** through a tool-calling local LLM agent
- **Semantic search** with Ollama embeddings and ChromaDB
- **BM25 keyword search** for names, places, exact phrases, and rare terms
- **Hybrid retrieval** using reciprocal-rank fusion
- **Evidence-grounded responses** with retrieved dream IDs and dates
- **Structured filtering** by tags and date ranges
- **Journal statistics and longitudinal tag trends**
- **Recurring-theme discovery** with UMAP + HDBSCAN clustering
- **Structured dream feature extraction**
- **Incremental index synchronization** without rebuilding unchanged data
- **Retrieval benchmarking** across fixed and agent-driven strategies

All model inference and journal processing can run locally through [Ollama](https://ollama.com/).

## Architecture

```text
                     ┌── Semantic search ── ChromaDB + embeddings ─┐
                     │                                             │
Journal → Parser ────┼── BM25 search ───── lexical index ─────────┼──→ Tools
                     │                                             │      │
                     ├── Tags / dates / statistics ───────────────┤      ▼
                     │                                             │   Local LLM
                     └── Embeddings ── UMAP + HDBSCAN ────────────┘      │
                                                                         ▼
                                                               Grounded response
```

The system separates reusable retrieval and analytical services from CLI and agent orchestration. The LLM can choose among search and deterministic analysis tools rather than attempting to answer every question directly from model context.

## Retrieval evaluation

Retrieval is evaluated on a 56-query labeled benchmark built from the included synthetic journal. Each query has a fine-grained category plus multi-label attributes for topical, lexical, entity, compositional, temporal, and low-overlap analysis. The benchmark includes proper names, rare objects, exact places, inflected variants, separated phrase terms, mixed entity/concept queries, and queries with little or no lexical overlap.  

| Strategy | Embedding model | Chat model | R-precision | R@5 | R@10 | Time/query |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Agent | qwen3-embedding | gemma4:12b | **0.608** | **0.676** | 0.763 | 50.793 s |
| Agent | nomic-embed-text | qwen3:8b | 0.537 | 0.659 | **0.783** | 9.920 s |
| BM25 baseline | — | — | 0.596 | 0.602 | 0.677 | **<0.001 s** |
| Agent | qwen3-embedding | qwen3:8b | 0.574 | 0.652 | 0.746 | 16.389 s |
| Agent | nomic-embed-text | gemma4:12b | 0.554 | 0.656 | 0.716 | 24.840 s |
| Fixed hybrid | qwen3-embedding | — | 0.444 | 0.589 | 0.680 | 0.359 s |
| Fixed hybrid | nomic-embed-text | — | 0.434 | 0.636 | 0.736 | 0.103 s |
| Semantic baseline | nomic-embed-text | — | 0.416 | 0.411 | 0.469 | 0.094 s |
| Semantic baseline | qwen3-embedding | — | 0.357 | 0.403 | 0.445 | 0.458 s |

The qwen3-embedding and gemma4:12b agent has the highest R-precision and R@5,
while the nomic-embed-text and qwen3:8b agent has the highest R@10. BM25 remains
the fastest strategy and has the second-highest R-precision.  

See [`leaderboard.md`](outputs/retrieval_evaluations/leaderboard.md) for more detailed evaluation results.  

Benchmark reports include per-query results, aggregate metrics, experiment metadata, errors, runtime, and descriptive agent tool traces.
For agent-mode evaluation, every labeled input is wrapped as an explicit request
to retrieve relevant dreams. This makes bare names and places unambiguously
retrieval queries without revealing their category, attributes, or known
relevant dream IDs to the agent.
The running leaderboard places each configured embedding/chat-model combination
in its own table. Query-suite fingerprints remain separate so unlike label sets
are never scored together, while every suite evaluated against the current
parsed dream corpus remains under **Current corpus experiments**. Runs against a
different or unknown corpus appear in older/incompatible groups.

```bash
dream-analyzer evaluate-retrieval \
  --retrieval-mode hybrid \
  --experiment-name "hybrid baseline" \
  --experiment-note "Verbatim semantic and BM25 retrieval fused with RRF"
```

## Quick start

Requires Python 3.10+ and a local [Ollama](https://ollama.com/) installation.

```bash
git clone https://github.com/IsaacVanB/dream-analyzer.git
cd dream-analyzer

python -m pip install --editable ".[dev]"

ollama pull nomic-embed-text
ollama pull qwen3:8b
```

A synthetic journal containing 176 dreams from 2022–2026 is included under
`examples/`, so the basic pipeline can be tested without providing personal
data. The parsed reference corpus and its labeled retrieval suite live under
`benchmarks/synthetic/`.

```bash
dream-analyzer parse \
  examples/mock_dream_journal.txt \
  data/sample/dreams.jsonl
dream-analyzer index \
  --dreams-path data/sample/dreams.jsonl \
  --chroma-path data/sample/chroma_db

dream-analyzer ask \
  "What patterns appear in dreams about hidden rooms?" \
  --dreams-path data/sample/dreams.jsonl \
  --chroma-path data/sample/chroma_db
```

To work with your own journal, copy it to the default private path and run the
same commands without path arguments:

```bash
mkdir -p data
cp path/to/journal.txt data/dream_journal.txt
dream-analyzer parse
dream-analyzer index
```

The entire project-local `data/` directory is ignored by Git. It is reserved
for private source journals, parsed records, character notes, import state, and
the local Chroma index, so pulls cannot replace those files. Back up this
directory separately; cloning the repository does not restore it. Commands
still accept explicit paths when a different location is preferred.

See [`example_usages.md`](example_usages.md) for detailed command options and workflows.

## CLI

```text
dream-analyzer parse               Parse journal text into structured JSONL
dream-analyzer index               Build or synchronize the retrieval index
dream-analyzer ask                 Ask a natural-language question
dream-analyzer analyze             Analyze an individual dream
dream-analyzer stats               Compute journal statistics
dream-analyzer trends              Analyze and plot tag trends
dream-analyzer cluster             Discover recurring themes
dream-analyzer evaluate-retrieval  Benchmark retrieval strategies
```

Run `dream-analyzer <command> --help` for command-specific options.

## Tech stack

- Runtime: Python 3.10+, Ollama, ChromaDB, NumPy, Pandas, scikit-learn,
  UMAP, HDBSCAN, Matplotlib, and Plotly
- Retrieval and AI: local LLMs and embeddings, RAG, tool calling, an in-project
  BM25 implementation, and reciprocal-rank fusion
- Development: pytest, Ruff, and GitHub Actions

The project uses a `src` package layout with automated tests, linting, and CI.
