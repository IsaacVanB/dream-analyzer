# Dream Analyzer

Small local pipeline for parsing a dream journal, embedding dreams with Ollama,
storing them in ChromaDB, and asking retrieval-augmented questions.  
The goal is to analyze a private dream journal over time while keeping all raw journal data local. The project is designed to support semantic dream retrieval, theme extraction, longitudinal analysis, graphing, and eventually a CLI/GUI interface where a user can ask natural-language questions about their dream journal.  
🚧🛠️ This project is a work in progress. 🛠️🚧

## Setup

```bash
pip install -r requirements.txt
pip install -e .
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

The editable install makes both the reusable `dream_analysis` package and the
commands under `src/cli/` importable while developing from the repository.

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

Calendar dates are validated before any output is written. Impossible dates,
such as `4/32/2021`, stop parsing with an error containing the original date and
journal line number so the source can be corrected manually. The supported
partial-date placeholders remain `0/0/YYYY` (year only), `M/0/YYYY` (month
only), and `0/0/00` or `0/0/0000` (unknown date).

The same parsing behavior is reusable in Python through
`dream_analysis.parser.JournalParser`; the CLI only handles file input and
JSONL output.

The included `data/mock_dream_journal.txt` contains synthetic example data for testing and
demonstrating this format. It is the default input, so you can run the code
without supplying your own journal:

```bash
python3 src/cli/parse_dreams.py
```

To parse your own journal, provide input and output paths:

```bash
python3 src/cli/parse_dreams.py path/to/journal.txt data/my_dreams.jsonl
```

For an ongoing journal where new dreams are appended at the end, use the
append-oriented synchronizer after the initial parse. Preview the changes first:

```bash
python3 src/cli/sync_dream_journal.py path/to/journal.txt --dry-run
python3 src/cli/sync_dream_journal.py path/to/journal.txt
```

The synchronizer preserves IDs for the existing positional prefix, tolerates
small corrections to old entries, assigns IDs only to the appended tail, and
records each import in `data/dream_imports.json`. It refuses truncation,
reordering, insertions into the old prefix, and substantive changes so those
cannot silently attach derived data to the wrong dream. Repeating an already
recorded journal version is idempotent.

Synchronize the ChromaDB index:

```bash
python3 src/cli/build_chroma_db.py
```

Embeddings are generated from dream text only; metadata is stored separately.
The default synchronization embeds only IDs absent from the collection. For an
edited old dream, it retains the existing vector while updating the displayed
document and recording both the embedded and current text hashes. Use
`--rebuild` when intentionally changing the embedding model or embedding logic.
If a date correction changed a legacy date-derived ID, synchronization reuses
the old vector when the orphaned and current records have exactly matching dream
text. Remaining orphaned IDs are reported but retained unless `--prune` is
supplied explicitly.

## Retrieval scoring

Dream retrieval currently follows two scoring paths. The scores have opposite
directions and should not be compared directly.

| Command | Retrieval score | Behavior |
|---|---|---|
| `retrieve_dreams.py` | Chroma distance | Lower is closer. |
| `basic_rag.py` | Chroma distance | The generated or supplied retrieval query is ranked by Chroma. |
| `dream_agent.py` | Chroma distance | `search_dreams` uses Chroma ranking; optional dates filter that ranked search. |
| `evaluate_retrieval.py` | Configurable | Defaults to Chroma distance; `--retrieval-metric cosine` uses cosine similarity and `both` compares them. |
| `compare_models.py rag` | Chroma distance | Uses the same retrieval path as `basic_rag.py`. |
| `analyze_dream.py --related-dreams ...` | Cosine similarity | Higher is more similar; `--similarity-threshold` is a cosine threshold. |
| `compare_models.py analyze` | Cosine similarity | Uses the same related-dream path as `analyze_dream.py`. |

Collections built by this repository do not explicitly configure Chroma's
distance space, so Chroma's current default applies: squared L2 distance. The
explicit related-dream path instead loads stored embeddings, calculates cosine
similarity in Python, filters by the similarity threshold, and sorts from
highest to lowest. Because the project does not normalize embeddings before
storage, the two paths can return different rankings for the same input.

`cluster_dreams.py` is not a retrieval command. It L2-normalizes stored vectors,
uses cosine distance for UMAP, uses cosine similarity for representative
selection, and uses Euclidean distance for HDBSCAN clustering.

## Current Tools

Retrieve similar dreams:

```bash
python3 src/cli/retrieve_dreams.py "hidden room water" --top-k 5
```

Plot tag frequency over time:

```bash
python3 src/cli/plot_tags.py --tags house recurring school
```

Cluster the existing dream embeddings and generate theme evidence, static plots,
an interactive map, and per-dream assignments:

```bash
python3 src/cli/cluster_dreams.py
python3 src/cli/cluster_dreams.py --label-clusters
```

Cluster labels summarize recurring content and are not psychological diagnoses.
The optional `--label-clusters` flag uses the local Ollama chat model; clustering
itself does not require new model calls.

The command is a thin adapter over reusable package services:
`dream_analysis.clustering.DreamEmbeddingRepository` loads stored vectors,
`DreamClusteringService` performs projection, clustering, evidence selection,
and optional Ollama labeling, and
`dream_analysis.cluster_reporting.ClusterReportService` generates the CSV,
HTML, PNG, and Markdown artifacts. These services can be used independently
without constructing an `argparse.Namespace` or invoking the CLI.

Compute summary stats:

```bash
python3 src/cli/compute_stats.py --freq Y
```

Analyze one dream:

```bash
python3 src/cli/analyze_dream.py --dream-id dream-2022-1-22-0
python3 src/cli/analyze_dream.py --dream-id dream-2022-1-22-0 --related-dreams 5
```

Extract structured features for every dream or one dream:

```bash
python3 src/cli/structure_dreams.py
python3 src/cli/structure_dreams.py --dream-id dream-2022-1-22-0 --overwrite
```

After synchronizing an appended journal, structure only the dreams introduced
by the latest import:

```bash
python3 src/cli/structure_dreams.py --scope new
```

Use `--import-id journal-...` to select an earlier recorded import. This scope
is based on the import manifest, so it works even when no older dream has ever
been structured.

Build a fillable character lookup from those structured records without more
LLM calls:

```bash
python3 src/cli/build_character_lookup.py --temporal-context
```

Ask a RAG question:

```bash
python3 src/cli/basic_rag.py "What patterns appear in dreams about hidden rooms?"
```

Or let a tool-capable Ollama model choose the semantic search query and use the
results in an agent loop:

```bash
python3 src/cli/dream_agent.py "What patterns appear in dreams about hidden rooms?"
python3 src/cli/dream_agent.py "What are common themes in dreams from last month?"
python3 src/cli/dream_agent.py "How many dreams did I record in 2025?"
python3 src/cli/dream_agent.py "What were my most common journal tags in July 2025?"
python3 src/cli/dream_agent.py "Did school-tagged dreams become more common during 2025?"
python3 src/cli/dream_agent.py "Who appears most often in my structured dreams?"
python3 src/cli/dream_agent.py "Who is Maya?"
python3 src/cli/dream_agent.py \
  "What are common themes in dreams about school? Use only dreams from last month."
python3 src/cli/dream_agent.py "What patterns recur in school dreams?" \
  --output outputs/agent/school_patterns.md
python3 src/cli/dream_agent.py "Compare house and school dreams" \
  --max-tool-calls 3 \
  --debug \
  --output outputs/agent/comparison_trace.md
```

The agent exposes eight read-only tools. `search_dreams` performs semantic
retrieval. `get_dreams_by_date_range` exhaustively returns every dream within
two inclusive calendar dates, making it suitable for questions such as common
themes within a month. `get_dream_statistics` deterministically calculates
counts, entries by month/quarter/year, journal-tag frequencies, dream-length
statistics, and common non-stopword vocabulary, optionally within inclusive
date bounds. `analyze_tag_trends` compares exact journal-tag frequencies across
months, quarters, or years using normalized percentages by default. It supports
explicit tags or automatically selects the most frequent tags. `get_dream_by_id` retrieves one exact dream for requests
such as `Get dream-2025-1-9-0 and analyze it`. `get_dreams_by_tags` returns every dream
containing all requested exact tags; matching is case-insensitive, preserves
punctuation such as `lucid?`, and treats multiple tags as an AND combination.
`get_character_mentions` ranks named characters by the number of distinct
structured dreams that mention them, or looks up specified names
case-insensitively. It supports inclusive date bounds and reports first and last
dated mentions plus matching dream IDs.
`get_character_context` reads the manually curated character dictionary and
looks up background information by canonical name or alias. An optional dream
date selects the applicable date-bounded relationship history.
All tools implement the shared `AgentTool` protocol and are supplied through an
ordered registry, so adding another tool does not require changing agent
dispatch, schema collection, or retry-reminder logic.
The semantic and tag tools accept optional, inclusive `start_date` and
`end_date` bounds in `YYYY-MM-DD` format. The agent
translates relative language into those bounds using the current date; "last
month" means the previous calendar month. Topic terms such as "school" remain
part of the semantic query while the date bounds filter the results. Search
queries, result counts, and per-dream context are bounded before being returned
to the model. The date-range tool requires both bounds and does not rank or
sample its results semantically. `--output` optionally saves the question, settings, searches,
selected date ranges, retrieved citations, the full text of every unique dream
returned across the searches, and the answer as Markdown. Repeated results remain
listed in their search tables, but their full text appears only once. The full
report text is captured from the same search but is not added to the model's
bounded context.

Exact duplicate tool calls reuse the first result instead of querying Chroma
again, but still count toward `--max-tool-calls`. Every answer is produced by a
fresh final request with tools disabled, rather than allowing all accumulated
tool results to clog the answer prompt. Results from distinct searches are
deduplicated and ranked with reciprocal-rank fusion; duplicate queries do not
increase a dream's score. At most `--max-synthesis-dreams` unique dreams
(default `10`) enter the final prompt. The evidence builder preserves at least
105 words from every included dream, includes complete text where space permits,
and drops lower-ranked dreams rather than shrinking included evidence below that
minimum. Its context budget is derived from `--num-ctx`, now `8192` by default.
Calls beyond the remaining tool budget are marked unexecuted and shown in the
console and report.

Deterministic aggregate tools can use the shared `AnalyticalToolResult` shape:
an `evidence_type`, normalized `parameters`, an `analysis` object, and a list of
`warnings`. The agent validates this shape, includes bounded analytical evidence
in final synthesis, preserves the full report variant in Markdown, and adapts
its final instructions when aggregate results do not contain individual dreams.

The model-facing statistics result caps common words and tags and abbreviates
very long period series with explicit warnings. Saved Markdown reports retain
the complete tag and period results. Common-word counts are token occurrences;
tag counts represent the number of dreams containing each tag. Dreams with
unknown dates are excluded from temporal statistics and reported in a warning.

Tag trends match explicit tags case-insensitively and count a tag at most once
per dream. Periods with no dated dreams are included with a zero dream count so
gaps cannot be mistaken for omitted data. Missing tags and unknown-date dreams
produce warnings. Long timelines are abbreviated only in model evidence; saved
reports retain every period.

Character mentions are freshly aggregated from
`outputs/structured_dreams/dream_features.jsonl`; the tool does not read the
manually enrichable `data/characters.json`. Every result reports parsed-journal
and structured-record coverage because dreams that have not been structured
cannot contribute named characters. Structured records whose IDs no longer
occur in the current parsed journal are excluded and reported. Date filtering
uses current parsed-journal dates, so a corrected date does not require
restructuring the dream. Model-facing character results retain at most 20 dream
IDs per character, while saved reports retain complete lists. Use
`--structured-dreams-path` to select another structured JSONL file.

Character context defaults to `data/characters.json`; use `--characters-path`
to select another dictionary. Both a top-level character array and the
`{"characters": [...]}` envelope generated by `build_character_lookup.py` are
accepted. Dictionary names and aliases match case-insensitively. Missing names,
aliases shared by multiple entries, and dates with no applicable relationship
history produce explicit warnings. Context is treated as reference information,
not evidence that a person appeared in a particular dream.

The retrieval prompt asks the model to return `SEARCH_COMPLETE` rather than
drafting an answer when it has enough searches. Any other content that ends the
search phase is discarded, and the ranked no-tools synthesis is used for the
final answer. If synthesis is empty, the command saves
a partial report when `--output` is supplied and exits with status 2. Add
`--debug` to print and save every normalized assistant message together with
Ollama diagnostics such as `done_reason`, prompt and generation token counts,
durations, available thinking content, and the exact final-synthesis prompt and
evidence packet. Debug tracing is opt-in because model messages may contain
private journal details.

The deterministic reporting logic is also available independently of the CLI
and Ollama. `DreamStatisticsService`, `TagTrendService`, and
`CharacterLookupService` accept validated records and return JSON-compatible
results. The command-line scripts remain responsible only for loading files,
plotting, and saving outputs; these service boundaries can later be exposed as
read-only LLM tools.

Structured feature extraction is implemented by `DreamStructuringService` in
`dream_analysis.structuring`. That module owns the extraction schema and prompts,
response validation, versioned record construction, selection/resume behavior,
and structured JSONL decoding. `structure_dreams.py` handles CLI orchestration,
progress reporting, and atomic checkpoint saves.

Compare the four chat/embedding combinations (qwen3:8b and gemma3:12b for chat, nomic-embed-text and qwen3-embedding for embedding) with a fixed retrieval query:

```bash
python3 src/cli/compare_models.py rag \
  "What patterns appear in dreams about hidden rooms?" \
  --retrieval-query "hidden room hallway extra room concealed door behind wall"
```

Evaluate each embedding model's top retrievals with Gemma:

```bash
python3 src/cli/evaluate_retrieval.py "hidden room hallway concealed door" --top-k 8
python3 src/cli/evaluate_retrieval.py \
  "hidden room hallway concealed door" \
  --top-k 8 \
  --retrieval-metric both
python3 src/cli/evaluate_retrieval.py \
  --dream-id dream-2022-1-22-0 \
  --focus "discovering a hidden room that reveals a disturbing secret" \
  --retrieval-metric both \
  --top-k 8
```

In dream-ID mode, retrieval always embeds the complete target dream. `--focus`
changes only Gemma's evaluation criterion. Reports include the complete target
and retrieved dream texts. With `--retrieval-metric both`, each embedding model
is evaluated once with Chroma distance and once with explicit cosine similarity.
The report includes shared top-k counts, Jaccard overlap, dreams unique to each
metric, and the cosine-minus-Chroma difference in mean judged relevance. Unique
candidates are pooled and judged once without exposing their retrieval source.

Use a manual retrieval query when the question has extra analysis language:

```bash
python3 src/cli/basic_rag.py \
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
