[![CI](https://github.com/IsaacVanB/dream-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/IsaacVanB/dream-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
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

Retrieval is evaluated on a 64-query labeled benchmark built from the included synthetic journal. The benchmark includes semantic queries as well as cases designed to stress lexical retrieval: proper names, rare objects, exact places, inflected variants, separated phrase terms, mixed entity/concept queries, and queries with little or no lexical overlap.

| Strategy | R-precision | R@5 | R@10 | Routing accuracy | Time/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 baseline | 0.595 | 0.597 | 0.676 | — | <0.001 s |
| Fixed hybrid (BM25 + semantic) | 0.459 | 0.639 | 0.731 | — | 0.093 s |
| Semantic baseline | 0.418 | 0.412 | 0.466 | — | 0.165 s |
| Expanded batched agent | 0.477 | 0.537 | 0.701 | 0.531 | 11.369 s |
| One-shot expanded agent | 0.376 | 0.383 | 0.459 | 0.469 | 2.174 s |

The results show different strengths across retrieval methods. BM25 currently gives the highest R-precision, while the fixed hybrid retriever gives the highest recall at both 5 and 10 results. Agent-driven retrieval is evaluated separately to measure how effectively the LLM can select and combine retrieval strategies.

See [`leaderboard.md`](outputs/retrieval_evaluations/leaderboard.md) for more detailed evaluation results.  

Benchmark reports include per-query results, aggregate metrics, experiment metadata, errors, runtime, and agent routing accuracy.

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

A synthetic journal containing 176 dreams from 2022–2026 is included, so the basic pipeline can be tested without providing personal data.

```bash
dream-analyzer parse
dream-analyzer index

dream-analyzer ask \
  "What patterns appear in dreams about hidden rooms?"
```

To work with another journal:

```bash
dream-analyzer parse path/to/journal.txt data/my_dreams.jsonl
```

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
