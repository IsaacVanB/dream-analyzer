# Example Usages

These examples assume the project has been installed in editable mode with
`pip install -e .` as described in the README.

The preferred interface is the installed `dream-analyzer` command. It provides
help at both levels:

```bash
dream-analyzer --help
dream-analyzer parse --help
dream-analyzer index --help
dream-analyzer ask --help
dream-analyzer analyze --help
dream-analyzer stats --help
dream-analyzer trends --help
dream-analyzer cluster --help
```

The matching scripts under `src/cli/` remain available temporarily for
backward compatibility. Utilities that do not yet have a consolidated
subcommand continue to use their script entry points below.

## `dream-analyzer parse`

Parses the raw dream journal text file into JSON Lines, one JSON object per dream.

```bash
dream-analyzer parse
dream-analyzer parse data/mock_dream_journal.txt data/dreams.jsonl
dream-analyzer parse other_journal.txt data/other_dreams.jsonl --dream-separator-blank-lines 2
```

Arguments:

- `input`: optional path to the raw journal text file. Defaults to `data/mock_dream_journal.txt`.
- `output`: optional path for parsed JSONL output. Defaults to `data/dreams.jsonl`.
- `--dream-separator-blank-lines`: number of consecutive blank lines that separates dreams. Defaults to auto-detection.

Output fields include `dream_id`, `date`, `year`, `month`, `day`, `date_precision`, `date_sort`, `tags`, `text`, and `word_count`.

The parser validates calendar dates before writing the JSONL file. An impossible
date stops the command and identifies the original value and journal line number
for manual correction. Zero placeholders remain valid for supported partial or
unknown dates: `0/0/YYYY`, `M/0/YYYY`, and `0/0/00` or `0/0/0000`.

## `src/cli/sync_dream_journal.py`

Synchronizes an ongoing append-only journal with an existing parsed JSONL file
and records the new IDs for downstream work:

```bash
python3 src/cli/sync_dream_journal.py path/to/journal.txt --dry-run
python3 src/cli/sync_dream_journal.py path/to/journal.txt
```

Small corrections to the old positional prefix retain their IDs. Truncation,
reordering, insertion into that prefix, or a substantive edit stops the import.
Use `--minimum-similarity` to adjust the conservative edit threshold and
`--state` to choose a non-default import-state file.

## `src/cli/check_dates.py`

Checks parsed dreams for suspicious years, duplicate dates, and duplicate dream IDs. It is helpful for finding human errors in dream journal dates and does not modify the data. Dates containing `0` placeholders are ignored during date checks, but their dream IDs are still checked for duplicates.

```bash
python3 src/cli/check_dates.py
python3 src/cli/check_dates.py --dreams-path data/other_dreams.jsonl
python3 src/cli/check_dates.py --max-isolated-dates 5 --json
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--max-isolated-dates`: largest surrounded year run to flag. Defaults to `3` dates.
- `--max-year-jump`: largest adjacent year change not flagged. Defaults to `1`.
- `--json`: print machine-readable JSON instead of the text report.

## `dream-analyzer index`

Embeds only each dream's text with Ollama `nomic-embed-text` and saves the vectors to a persistent ChromaDB collection. Dates, tags, and other metadata remain available for display and filtering but do not affect similarity.

```bash
dream-analyzer index
dream-analyzer index --dreams-path data/dreams.jsonl --chroma-path data/chroma_db
dream-analyzer index --embed-model qwen3-embedding --collection-name dreams_qwen3_embedding --batch-size 16
dream-analyzer index --rebuild
dream-analyzer index --prune
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--chroma-path`: directory for the persistent ChromaDB database. Defaults to `data/chroma_db`.
- `--collection-name`: ChromaDB collection to synchronize. Defaults to `dreams_nomic_embed_text`.
- `--embed-model`: Ollama embedding model. Defaults to `nomic-embed-text`.
- `--batch-size`: number of new dream texts sent to Ollama per embedding request. Defaults to `32`.
- `--rebuild`: replace the collection and regenerate every vector. Without it,
  the command adds missing dream IDs and preserves existing embeddings.
- `--prune`: delete indexed IDs absent from the current parsed journal. Exact
  text matches are automatically migrated from an orphaned legacy ID to a new
  corrected ID without generating another embedding.

Requires Ollama running locally at `http://localhost:11434`. After changing the
embedding logic, run this command with `--rebuild` to regenerate the index.

## Retrieval scoring reference

The commands currently use two different scoring paths:

| Command | Score used for retrieval | Score direction |
|---|---|---|
| `src/cli/retrieve_dreams.py` | Chroma distance | Lower is closer. |
| `src/cli/basic_rag.py` | Chroma distance | Lower is closer. |
| `dream-analyzer ask` | Chroma distance | Lower is closer; date bounds optionally filter the ranked search. |
| `src/cli/evaluate_retrieval.py` | Selectable; agent default | Runs the labeled benchmark through the dream agent or direct query embeddings. |
| `src/cli/evaluate_retrieval_llm.py` | Configurable | Chroma distance by default; cosine similarity or both are selectable. |
| `src/cli/compare_models.py rag` | Chroma distance | Lower is closer. |
| `dream-analyzer analyze` with related dreams enabled | Cosine similarity | Higher is more similar. |
| `src/cli/compare_models.py analyze` | Cosine similarity | Higher is more similar. |

Collections created by `dream-analyzer index` do not specify a Chroma distance
space, so Chroma's current default applies: squared L2 distance. In contrast,
the related-dream analysis path reads the stored vectors and calculates cosine
similarity in Python. These rankings and numeric scores are not interchangeable,
and they can select different dreams because stored embeddings are not
explicitly normalized.

`dream-analyzer cluster` also performs cosine-based calculations, but it does not
retrieve related dreams: it normalizes vectors, uses cosine distance for UMAP,
uses cosine similarity for representative selection, and uses Euclidean
distance for HDBSCAN.

## `dream-analyzer trends`

Plots dream tag frequency over time and saves the image to `outputs/plots/`.

```bash
dream-analyzer trends
dream-analyzer trends --tags house recurring school --freq M
dream-analyzer trends --tag school --normalize
dream-analyzer trends --freq Y --normalize --output outputs/plots/tags_by_year.png
dream-analyzer trends --start-date 2023-01-01 --end-date 2023-12-31
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--output`: path where the plot image should be saved. Defaults to `outputs/plots/tag_frequency.png`.
- `--tags` / `--tag`: specific tags to plot. Defaults to the top tags.
- `--top-n`: number of top tags to plot when `--tags` is omitted. Defaults to `10`.
- `--freq`: time grouping frequency: `M`, `Q`, or `Y`. Defaults to `M`.
- `--start-date`: only include dreams on or after this date.
- `--end-date`: only include dreams on or before this date.
- `--normalize`: plot percent of dreams per period instead of raw counts.
- `--title`: optional plot title.
- `--show`: display the plot interactively after saving.

## `dream-analyzer stats`

Computes dream counts, tag frequencies, and word-count statistics, then prints and saves JSON.

```bash
dream-analyzer stats
dream-analyzer stats --freq Q --start-date 2023-01-01 --end-date 2023-12-31
dream-analyzer stats --freq Y --output outputs/stats/yearly_stats.json
dream-analyzer stats --common-words 50 --stopwords-path data/stopwords.txt
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--output`: path where JSON stats should be saved. Defaults to `outputs/stats/dream_stats.json`.
- `--freq`: time grouping frequency for entry counts: `M`, `Q`, or `Y`. Defaults to `M`.
- `--start-date`: only include dreams on or after this date.
- `--end-date`: only include dreams on or before this date.
- `--common-words`: number of most common non-trivial words to include. Defaults to `20`.
- `--stopwords-path`: optional text file of additional stopwords, one per line.
- `--min-word-length`: minimum word length for common-word stats. Defaults to `3`.

## `src/cli/retrieve_dreams.py`

Embeds a text query and returns the closest dreams from the ChromaDB collection.

```bash
python3 src/cli/retrieve_dreams.py "hidden room water"
python3 src/cli/retrieve_dreams.py "school anxiety" --top-k 5
```

Arguments:

- `query`: required text query to search for.
- `--top-k`: number of closest dreams to return. Defaults to `10`.
- `--chroma-path`: path to the persistent ChromaDB database. Defaults to `data/chroma_db`.
- `--collection-name`: ChromaDB collection name to query. Defaults to `dreams`.
- `--embed-model`: Ollama embedding model. Defaults to `nomic-embed-text`.
- `--preview-chars`: maximum preview length per result. Defaults to `300`.

Requires an existing ChromaDB index and Ollama running locally at `http://localhost:11434`.

## `dream-analyzer cluster`

Clusters embeddings already stored in ChromaDB. It writes a per-dream CSV, an
interactive HTML map, two PNG maps, and a Markdown evidence report under
`outputs/clusters/`. It does not regenerate embeddings.

The CLI delegates its work to `dream_analysis.clustering` and
`dream_analysis.cluster_reporting`. Use `DreamEmbeddingRepository` and
`DreamClusteringService` when another Python workflow needs computed cluster
data without files, then pass the resulting `ClusterAnalysis` to
`ClusterReportService` only when artifacts are wanted.

```bash
dream-analyzer cluster
dream-analyzer cluster --collection-name dreams_qwen3_embedding
dream-analyzer cluster --min-cluster-size 7 --n-neighbors 10
dream-analyzer cluster --label-clusters --label-model qwen3:8b
```

Important options include `--collection-name`, `--output-dir`,
`--pca-dimensions`, `--n-neighbors`, `--min-dist`, `--min-cluster-size`,
`--min-samples`, and `--random-state`. The exploratory default for
`--min-samples` is `1`; increasing it makes HDBSCAN more conservative and will
usually mark more dreams as noise. Use `--label-clusters` to replace the
automatic evidence-based labels with short labels generated by a local Ollama
chat model. The report includes representative dreams, distinctive TF-IDF
terms, overrepresented tags, noise fraction, and non-noise silhouette score.

## `src/cli/basic_rag.py`

Retrieves relevant dreams for a question and asks an Ollama chat model to answer using only those entries.

```bash
python3 src/cli/basic_rag.py "What patterns appear in dreams about hidden rooms?"
python3 src/cli/basic_rag.py "What symbols recur in school dreams?" --top-k 5 --chat-model qwen3:8b
python3 src/cli/basic_rag.py "What patterns appear in dreams about hidden rooms?" --retrieval-query "hidden room hidden hallway secret room"
```

Arguments:

- `question`: required question or prompt to answer.
- `--top-k`: number of dream entries to retrieve. Defaults to `8`.
- `--retrieval-query`: optional focused query to embed for retrieval. If omitted, the chat model generates one from `question`.
- `--chroma-path`: path to the persistent ChromaDB database. Defaults to `data/chroma_db`.
- `--collection-name`: ChromaDB collection name to query. Defaults to `dreams`.
- `--embed-model`: Ollama embedding model. Defaults to `nomic-embed-text`.
- `--chat-model`: Ollama chat model. Defaults to `qwen3:8b`.
- `--max-chars-per-dream`: maximum context characters per retrieved dream. Defaults to `2500`.
- `--num-ctx`: Ollama context window option. Defaults to `4096`.
- `--num-predict`: maximum generated tokens. Defaults to `700`.
- `--temperature`: sampling temperature. Defaults to `0.1`.

Requires an existing ChromaDB index and Ollama running locally at `http://localhost:11434`.

## `dream-analyzer ask`

Answers a question through Ollama's tool-calling loop. The model can call one
of eight read-only tools: semantic `search_dreams`, exhaustive
`get_dreams_by_date_range`, deterministic `get_dream_statistics`, deterministic
`analyze_tag_trends`, dictionary-backed `get_character_context`, structured
`get_character_mentions`, exact
`get_dream_by_id`, or exact
`get_dreams_by_tags`. The date-range tool is intended for questions about all
dreams or common patterns within a period and requires inclusive `start_date`
and `end_date` values. The ID tool supports prompts such as
`Get dream-2025-1-9-0 and analyze it`. The tag tool returns dreams containing
every supplied tag (AND matching), so prompts such as
`Get all dreams tagged school` and
`Get all dreams tagged school and lucid?` use exact journal tags. The model
inspects bounded results and then produces a
grounded answer with dream IDs and dates.

Internally, these tools implement a common `AgentTool` protocol and are passed
to the agent as an ordered registry. New retrieval tools can therefore be added
without extending hardcoded dispatch branches.

```bash
dream-analyzer ask "What patterns appear in dreams about hidden rooms?"
dream-analyzer ask "What are common themes in dreams from last month?"
dream-analyzer ask "How many dreams did I record in 2025?"
dream-analyzer ask "What words occurred most often last year?"
dream-analyzer ask "Did school-tagged dreams become more common during 2025?"
dream-analyzer ask "Who appears most often in my structured dreams?"
dream-analyzer ask "Who is Maya?"
dream-analyzer ask \
  "What are common themes in dreams about school? Use only dreams from last month."
dream-analyzer ask "How do school anxiety dreams appear?" \
  --top-k 5 \
  --chat-model qwen3:8b \
  --embed-model nomic-embed-text \
  --output outputs/agent/school_anxiety.md

dream-analyzer ask "Compare house and school dreams" \
  --max-tool-calls 3 \
  --debug \
  --output outputs/agent/comparison_trace.md
```

Arguments:

- `question`: required question to answer from the journal.
- `--top-k`: maximum results returned by each search call. Defaults to `8` and is capped at `20`.
- `--max-tool-calls`: maximum tool calls permitted for one answer. Defaults to `3`.
- `--max-synthesis-dreams`: maximum unique dreams included in the final answer prompt. Defaults to `10` and is capped at `20`.
- `--max-chars-per-dream`: maximum journal characters per search result returned to the model. The Markdown report still contains the full text. Defaults to `2500`.
- `--output`: optional path for a Markdown report containing the question, settings, searches, citations, full text of every unique retrieved dream, and answer. Repeated dreams are listed in each applicable search table, but their full text appears once and is not added to the model's bounded context.
- `--debug`: print assistant messages and Ollama response diagnostics and include them in the Markdown report.
- `--chroma-path`, `--collection-name`, and `--embed-model`: select the existing vector index.
- `--chat-model`: select the tool-capable Ollama model without changing `config.py`.
- `--dreams-path`: parsed journal JSONL used for exact and analytical tools.
- `--structured-dreams-path`: structured feature JSONL used by
  `get_character_mentions`. Defaults to
  `outputs/structured_dreams/dream_features.jsonl`.
- `--characters-path`: manually curated character dictionary used by
  `get_character_context`. Defaults to `data/characters.json`.
- `--num-ctx`, `--num-predict`, and `--temperature`: control chat generation. `--num-ctx` defaults to `8192` so ten average-length dreams fit comfortably.

The `search_dreams` tool supports optional inclusive `start_date` and `end_date`
arguments in `YYYY-MM-DD` format. The agent resolves relative wording using the
current date and treats "last month" as the previous calendar month. Date bounds
are applied together with semantic terms, so a question about school dreams from
last month searches for school content only inside that month. Before executing
the call, the agent removes date bounds if the original user question contains
no explicit calendar or relative-time constraint. Corrected calls are labeled
in debug output and Markdown tool traces.

`get_character_mentions` uses the `named_characters` extracted by
`structure_dreams.py`, not the manually edited character lookup. Names match
case-insensitively and counts are deduplicated within each dream. Its results
always report the share of current parsed dreams that have matching structured
records; unstructured dreams cannot contribute mentions. Date filters use the
current parsed-journal dates, and structured records for IDs no longer present
in that journal are excluded. Model evidence caps long dream-ID lists, while an
`--output` Markdown report preserves the complete lists.

`get_character_context` looks up canonical names and aliases
case-insensitively in the character dictionary. It returns manually curated
relationship and background context, plus a compact mention summary when one is
present. If the dictionary uses `relationship_history`, the tool can use a
dream's date to select entries whose inclusive date bounds apply. Ambiguous
aliases, missing names, and missing date-specific history are reported rather
than guessed. The dictionary may be either a top-level array or the envelope
written by `build_character_lookup.py`.

Exact duplicate calls reuse their cached result without another Chroma query,
although each model request still consumes one slot in `--max-tool-calls`. Every
answer uses a fresh tools-disabled final request. The agent deduplicates the
candidate pool and combines rankings from distinct queries with reciprocal-rank
fusion, while repeated queries do not add ranking weight. Exhaustive date-range
and exact-tag sets are semantically reranked without dropping matches before
they enter that fusion. Only the highest-ranked
`--max-synthesis-dreams` entries are eligible for the final prompt. At least 105
words are retained for every included dream (or its complete text when shorter),
and lower-ranked dreams are omitted when the context cannot preserve that
minimum. Remaining space is assigned to higher-ranked dreams first. Any calls
beyond the tool budget are recorded as unexecuted.

The retrieval prompt asks the model to return `SEARCH_COMPLETE` when it has
enough evidence. That response is treated only as a signal to start ranked
synthesis; any draft answer is discarded. If the final
synthesis response is empty, the CLI prints the partial state, saves a
partial report when `--output` is present, and exits with status 2. `--debug`
prints and saves a turn-by-turn trace containing normalized assistant messages,
whether tools were enabled, and Ollama response diagnostics including
`done_reason`, token counts, durations, any available thinking content, and the
exact forced prompt with its evidence packet. Because that trace can contain
private journal details, enable it only when needed and protect the saved report
accordingly.

Requires an existing matching ChromaDB index, a tool-capable chat model, and
Ollama running locally.

## `dream-analyzer analyze`

Loads one dream by ID or accepts dream text directly, then asks an Ollama chat model for a close analysis of its events, dynamics, themes, and possible interpretations.
The analysis is printed and saved under `outputs/analysis/`. Saved files include
the complete target dream, complete retrieved related dreams, and generated
analysis. ID-based filenames use `<dream_id>_<datetime>.txt`; direct-text
filenames use `<datetime>.txt`.

```bash
dream-analyzer analyze dream-2022-1-22-0
dream-analyzer analyze --text "I opened a door and found another kitchen."
dream-analyzer analyze dream-2022-1-22-0 --chat-model qwen3:8b
dream-analyzer analyze dream-2022-1-22-0 --related-dreams 5 --similarity-threshold 0.55
dream-analyzer analyze dream-2022-10-10-0 --related-dreams 5 --start-date 2022-04-10 --end-date 2022-10-10
```

Arguments:

- `dream-id`: positional dream ID to load from JSONL. The legacy `--dream-id`
  option remains available temporarily. Mutually exclusive with `--text`.
- `--text`: dream text to analyze directly. Mutually exclusive with either
  dream-ID form.
- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--chat-model`: Ollama chat model. Defaults to `qwen3:8b`.
- `--related-dreams`: maximum number of similar indexed dreams to use as context. Defaults to `0` (disabled).
- `--similarity-threshold`: minimum cosine similarity for related dreams. Defaults to `0.5`.
- `--start-date`: earliest reference dream date to include, in `YYYY-MM-DD` format.
- `--end-date`: latest reference dream date to include, in `YYYY-MM-DD` format.
- `--chroma-path`: path to the persistent ChromaDB database. Defaults to `data/chroma_db`.
- `--collection-name`: ChromaDB collection name. Defaults to `dreams`.
- `--embed-model`: Ollama embedding model. Defaults to `nomic-embed-text`.
- `--max-chars-per-related-dream`: maximum context characters per related dream. Defaults to `1500`.
- `--output-dir`: directory where analysis files are saved. Defaults to `outputs/analysis`.
- `--num-ctx`: Ollama context window option. Defaults to `8192`.
- `--num-predict`: maximum generated tokens. Defaults to `2500`.
- `--temperature`: sampling temperature. Defaults to `0.2`.

Related-dream context requires an existing ChromaDB index. Ollama must be running locally.

## `src/cli/compare_models.py`

Runs the same task across these four combinations:

- `qwen3:8b` with `nomic-embed-text`
- `gemma3:12b` with `nomic-embed-text`
- `qwen3:8b` with `qwen3-embedding`
- `gemma3:12b` with `qwen3-embedding`

It expects collections named `dreams_nomic_embed_text` and
`dreams_qwen3_embedding`. Retrieval is performed once per embedding model and
the exact retrieved context is reused for both chat models. Results are saved
as JSON and Markdown under `outputs/model_comparisons/`. Analyze-mode reports
include the complete target dream, and both modes include complete retrieved
dream texts.

Compare RAG answers. A fixed retrieval query is required so chat models do not
generate different search queries:

```bash
python3 src/cli/compare_models.py rag \
  "What patterns appear in dreams about hidden rooms?" \
  --retrieval-query "hidden room hallway extra room concealed door behind wall"
```

Compare analysis of one dream with five related dreams:

```bash
python3 src/cli/compare_models.py analyze \
  --dream-id dream-2022-1-22-0 \
  --related-dreams 5
```

The runner defaults to temperature `0` for a controlled first comparison. Run
`python3 src/cli/compare_models.py rag --help` or
`python3 src/cli/compare_models.py analyze --help` for task-specific options.

## `src/cli/evaluate_retrieval.py`

Runs every query in `data/retrieval_eval_queries.json` through the same tool
planner used by `dream-analyzer ask`. The agent can generate semantic queries,
use date filters, retrieve exact tags or date ranges, and call character or
analytical tools. Results from dream-returning calls are combined with the same
reciprocal-rank fusion used for answer synthesis. Exhaustive date and exact-tag
sets are semantically reranked first; every match is retained, and unindexed
matches are placed after indexed matches. Any `search_dreams` date bounds that
are unsupported by the original evaluation query are removed before execution
and identified in the tool trace. The final prose answer is not generated
during evaluation. A fail-fast preflight first checks the Chroma path,
collection, embedding-model metadata, and tool data files. Missing structured
data is allowed and disables only `get_character_mentions`; malformed structured
data remains a preflight error. Tool or agent failures are marked as query errors
and excluded from metric averages. The JSON and Markdown reports include the
tool trace, recall and precision at 5 and 10, R-precision, macro and category
averages, and the maximum possible precision at each cutoff based on the number
of known relevant dreams.

```bash
python3 src/cli/evaluate_retrieval.py
python3 src/cli/evaluate_retrieval.py --retrieval-mode embedding
python3 src/cli/evaluate_retrieval.py \
  --collection-name dreams_qwen3_embedding \
  --embed-model qwen3-embedding
```

Agent retrieval is the default. The `embedding` mode provides a non-agentic
baseline: it embeds each labeled query verbatim, retrieves `max(10, R)` results
directly from Chroma, and never invokes the chat model or agent tools. It also
does not require parsed, structured, or character data files.

Each semantic search returns 10 dreams by default, and each query may use up to
three tool calls. These settings can be changed with `--top-k` (10–20) and
`--max-tool-calls`. Recall and R-precision are shown as `n/a` for queries with
no known relevant dreams.

## `src/cli/evaluate_retrieval_llm.py`

Embeds one retrieval prompt with both `nomic-embed-text` and
`qwen3-embedding`, retrieves the top-k dreams from their matching collections,
and asks `gemma3:12b` to score every result from 1 (irrelevant) to 5 (directly
relevant). The judge sees the prompt and dream content but not the embedding
model name. JSON and Markdown reports are saved under
`outputs/retrieval_evaluations/`.

```bash
python3 src/cli/evaluate_retrieval_llm.py \
  "hidden room hallway extra room concealed door behind wall" \
  --top-k 8
```

Compare the current Chroma-distance and explicit-cosine retrieval paths for
both embedding models:

```bash
python3 src/cli/evaluate_retrieval_llm.py \
  "hidden room hallway extra room concealed door behind wall" \
  --retrieval-metric both \
  --top-k 8
```

`--retrieval-metric` accepts `chroma`, `cosine`, or `both` and defaults to
`chroma` for compatibility. In `both` mode the report contains four retrieval
runs: two metrics for each embedding model. It reports shared top-k count,
Jaccard overlap, metric-exclusive dream IDs, and the cosine-minus-Chroma change
in mean LLM-judged relevance. All unique candidates are pooled and judged once,
blind to retrieval metric and original rank, so shared dreams receive the same
relevance score in both result lists.

Use the complete text of an existing dream as the retrieval and relevance
target. The target dream itself is excluded from the results:

```bash
python3 src/cli/evaluate_retrieval_llm.py \
  --dream-id dream-2022-1-22-0 \
  --retrieval-metric both \
  --top-k 8
```

For a long or multi-scene dream, specify which part should control the LLM's
relevance judgment:

```bash
python3 src/cli/evaluate_retrieval_llm.py \
  --dream-id dream-2022-1-22-0 \
  --focus "discovering a hidden room that reveals a disturbing family secret" \
  --top-k 8
```

Alternatively, use an exact passage from the dream or generate an evaluation
focus:

```bash
python3 src/cli/evaluate_retrieval_llm.py \
  --dream-id dream-2022-1-22-0 \
  --focus-passage "I opened the concealed door behind the wall"

python3 src/cli/evaluate_retrieval_llm.py \
  --dream-id dream-2022-1-22-0 \
  --generate-focus
```

In all dream-ID modes, both embedding models search with the complete target
dream text. Focus options affect only Gemma's evaluation. The evaluator pools
and deduplicates both models' results, hides their source and rank, and has
Gemma score every unique dream once. Focal relevance is the primary 1-5 score;
generic overlap is reported separately and does not increase focal relevance.
The JSON and Markdown reports include the complete target dream near the
beginning, followed by the complete text of every retrieved dream.

Unique candidates are judged in batches of 12 by default so a four-run metric
comparison does not overfill one context window. Use `--judge-batch-size` to
adjust this. Other useful options include `--retrieval-metric`, `--judge-model`,
`--max-chars-per-dream`, `--max-target-chars`, `--num-ctx`, `--chroma-path`, and
`--output-dir`.

## `src/cli/structure_dreams.py`

Uses structured Ollama output to extract grounded, descriptive fields from
every dream and saves one JSON object per line to
`outputs/structured_dreams/dream_features.jsonl`:

The reusable schema, prompts, validation, record construction, selection/resume
logic, and JSONL decoding live in `dream_analysis.structuring` behind
`DreamStructuringService`. The CLI handles arguments, progress, failure
reporting, and atomic checkpoint saves.

```bash
python3 src/cli/structure_dreams.py
```

Process or regenerate one dream:

```bash
python3 src/cli/structure_dreams.py --dream-id dream-2022-1-22-0
python3 src/cli/structure_dreams.py --dream-id dream-2022-1-22-0 --overwrite
python3 src/cli/structure_dreams.py --scope new
```

`--scope new` reads `data/dream_imports.json` and processes only IDs introduced
by the latest append-oriented journal import. Select another manifest with
`--import-id`, or another state file with `--import-state`.

Existing dream IDs with the current schema version are skipped unless
`--overwrite` is supplied; records from an older schema are regenerated
automatically. Successful records are saved after each model response, so a
long run can be resumed. The
output includes source metadata plus settings, characters, emotions, themes,
objects, actions, sensory details, dream mechanics, tone,
lucidity, violence, sexual content, social conflict, threat, agency,
bizarreness, perspective, ending, memory quality, and a factual summary.
Social conflict and bizarreness use `none`, `low`, `moderate`, and `high`.

`characters` contains unnamed roles, while `named_characters` contains only
explicit proper names and preserves their capitalization. Both are produced in
the same model call.

## `src/cli/build_character_lookup.py`

Aggregates `named_characters` from structured dream records and creates a
fillable lookup without making any additional LLM calls:

```bash
python3 src/cli/build_character_lookup.py
```

Add date-bounded relationship templates when a relationship may change over
time:

```bash
python3 src/cli/build_character_lookup.py --temporal-context
```

The default output is `data/characters.json`. Each entry includes the detected
name, aliases, mention count, first and last mention dates, and contributing
dream IDs. With temporal context, fill in or duplicate entries under
`relationship_history`, using inclusive `start_date` and `end_date` values in
`YYYY-MM-DD` format. Null boundaries mean open-ended. Without temporal context,
fill in the single `relationship` and `context` fields.

The script refuses to replace an existing lookup by default because it may
contain manual edits. Use `--overwrite` only when you intentionally want to
regenerate it.
