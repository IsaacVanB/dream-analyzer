# Dream Analyzer

Small local pipeline for parsing a dream journal, embedding dreams with Ollama,
storing them in ChromaDB, and asking retrieval-augmented questions.  
The goal is to analyze a private dream journal over time while keeping all raw journal data local. The project is designed to support semantic dream retrieval, theme extraction, longitudinal analysis, graphing, and eventually a CLI/GUI interface where a user can ask natural-language questions about their dream journal.  
🚧🛠️ This project is a work in progress. 🛠️🚧

## Setup

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

At least one embedding model and chat model are needed. Ollama must be running locally at `http://localhost:11434`.

## Workflow

Format the journal as plain text, with each group of dreams beginning with a date
in `M/D/YY` or `M/D/YYYY` format. Put optional tags on one or more `#tag`-only
lines at the beginning of a dream. Separate multiple dreams from the same date
with blank lines; a new date also begins a new dream.

```text
1/22/2022
#house #recurring
There was an extra room behind the pantry.

I found a notebook in a freezer.

2/3/22
#animal
A white dog followed me through a grocery store.
```

The parser automatically detects whether one or two consecutive blank lines
separate dreams. If a journal uses a different convention, set it explicitly
with `--dream-separator-blank-lines`.

The included `data/mock_dream_journal.txt` contains synthetic example data for testing and
demonstrating this format. It is the default input, so you can run the code
without supplying your own journal:

```bash
python3 src/parse_dreams.py
```

To parse your own journal, provide input and output paths:

```bash
python3 src/parse_dreams.py path/to/journal.txt data/my_dreams.jsonl
```

Build the ChromaDB index:

```bash
python3 src/build_chroma_db.py
```

Embeddings are generated from dream text only; metadata is stored separately.

## Current Tools

Retrieve similar dreams:

```bash
python3 src/retrieve_dreams.py "hidden room water" --top-k 5
```

Plot tag frequency over time:

```bash
python3 src/plot_tags.py --tags house recurring school
```

Cluster the existing dream embeddings and generate theme evidence, static plots,
an interactive map, and per-dream assignments:

```bash
python3 src/cluster_dreams.py
python3 src/cluster_dreams.py --label-clusters
```

Cluster labels summarize recurring content and are not psychological diagnoses.
The optional `--label-clusters` flag uses the local Ollama chat model; clustering
itself does not require new model calls.

Compute summary stats:

```bash
python3 src/compute_stats.py --freq Y
```

Analyze one dream:

```bash
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0 --related-dreams 5
```

Extract structured features for every dream or one dream:

```bash
python3 src/structure_dreams.py
python3 src/structure_dreams.py --dream-id dream-2022-1-22-0 --overwrite
```

Build a fillable character lookup from those structured records without more
LLM calls:

```bash
python3 src/build_character_lookup.py --temporal-context
```

Ask a RAG question:

```bash
python3 src/basic_rag.py "What patterns appear in dreams about hidden rooms?"
```

Ask a broader question using LLM-planned multi-query retrieval:

```bash
python3 src/open_ended_rag.py \
  "How have dreams about responsibility and unfinished obligations changed over time?"
```

Compare the four chat/embedding combinations (qwen3:8b and gemma3:12b for chat, nomic-embed-text and qwen3-embedding for embedding) with a fixed retrieval query:

```bash
python3 src/compare_models.py rag \
  "What patterns appear in dreams about hidden rooms?" \
  --retrieval-query "hidden room hallway extra room concealed door behind wall"
```

Evaluate each embedding model's top retrievals with Gemma:

```bash
python3 src/evaluate_retrieval.py "hidden room hallway concealed door" --top-k 8
python3 src/evaluate_retrieval.py \
  --dream-id dream-2022-1-22-0 \
  --focus "discovering a hidden room that reveals a disturbing secret" \
  --top-k 8
```

In dream-ID mode, retrieval always embeds the complete target dream. `--focus`
changes only Gemma's evaluation criterion. Reports include the complete target
and retrieved dream texts.

Use a manual retrieval query when the question has extra analysis language:

```bash
python3 src/basic_rag.py \
  "What patterns appear in dreams about hidden rooms?" \
  --retrieval-query "hidden room hallway extra room concealed door behind wall"
```

See `example_usages.md` for more detailed command examples and arguments.

## Example Results

The following results were generated from the included example journal, which
contains 151 dreams dated from 2022 through 2026.

### Longitudinal Tag Analysis

The normalized view shows how often each of the most common tags appears as a
percentage of dreams recorded that month.

![Normalized monthly dream-tag frequency](outputs/plots/tags_by_month_normalized.png)

Summary statistics generated by `compute_stats.py`:

| year | dreams |
|---:|---:|
| 2022 | 38 |
| 2023 | 36 |
| 2024 | 30 |
| 2025 | 27 |
| 2026 | 20 |

The three most common tags were `house` (15 dreams), `animal` (13), and
`school` (12).

### Retrieval-Augmented Analysis

For the question *“What patterns appear in dreams about hidden rooms?”*, the
pipeline retrieved dreams involving concealed rooms and hallways, doors behind
walls or appliances, and recurring objects such as keys, maps, photographs, and
cryptic labels. The generated synthesis identified two broader patterns:

- Hidden spaces tend to appear inside otherwise familiar places.
- Discovery is frequently paired with hesitation, warnings, or a sense of being
  watched.

### Model Comparison

`compare_models.py` ran the same retrieval prompt across four local
chat/embedding combinations. These are the recorded timings from that run;
performance will vary by hardware.

| chat model | embedding model | retrieval | generation | total |
|---|---|---:|---:|---:|
| qwen3:8b | nomic-embed-text | 0.158 s | 23.206 s | 23.364 s |
| gemma3:12b | nomic-embed-text | 0.158 s | 89.822 s | 89.980 s |
| qwen3:8b | qwen3-embedding | 5.445 s | 16.662 s | 22.108 s |
| gemma3:12b | qwen3-embedding | 5.445 s | 69.323 s | 74.769 s |
