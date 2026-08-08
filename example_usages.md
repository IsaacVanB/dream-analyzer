# Example Usages

## `src/parse_dreams.py`

Parses the raw dream journal text file into JSON Lines, one JSON object per dream.

```bash
python3 src/parse_dreams.py
python3 src/parse_dreams.py data/mock_dream_journal.txt data/dreams.jsonl
python3 src/parse_dreams.py other_journal.txt data/other_dreams.jsonl --dream-separator-blank-lines 2
```

Arguments:

- `input`: optional path to the raw journal text file. Defaults to `data/mock_dream_journal.txt`.
- `output`: optional path for parsed JSONL output. Defaults to `data/dreams.jsonl`.
- `--dream-separator-blank-lines`: number of consecutive blank lines that separates dreams. Defaults to auto-detection.

Output fields include `dream_id`, `date`, `year`, `month`, `day`, `date_precision`, `date_sort`, `tags`, `text`, and `word_count`.

## `src/check_dates.py`

Checks parsed dreams for suspicious years, duplicate dates, and duplicate dream IDs. It is helpful for finding human errors in dream journal dates and does not modify the data. Dates containing `0` placeholders are ignored during date checks, but their dream IDs are still checked for duplicates.

```bash
python3 src/check_dates.py
python3 src/check_dates.py --dreams-path data/other_dreams.jsonl
python3 src/check_dates.py --max-isolated-dates 5 --json
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--max-isolated-dates`: largest surrounded year run to flag. Defaults to `3` dates.
- `--max-year-jump`: largest adjacent year change not flagged. Defaults to `1`.
- `--json`: print machine-readable JSON instead of the text report.

## `src/build_chroma_db.py`

Embeds only each dream's text with Ollama `nomic-embed-text` and saves the vectors to a persistent ChromaDB collection. Dates, tags, and other metadata remain available for display and filtering but do not affect similarity.

```bash
python3 src/build_chroma_db.py
python3 src/build_chroma_db.py --dreams-path data/dreams.jsonl --chroma-path data/chroma_db
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--chroma-path`: directory for the persistent ChromaDB database. Defaults to `data/chroma_db`.
- `--collection-name`: ChromaDB collection name to recreate. Defaults to `dreams`.
- `--embed-model`: Ollama embedding model. Defaults to `nomic-embed-text`.

Requires Ollama running locally at `http://localhost:11434`.
Run this command again after changing the embedding logic so the existing index is rebuilt with text-only embeddings.

## `src/plot_tags.py`

Plots dream tag frequency over time and saves the image to `outputs/plots/`.

```bash
python3 src/plot_tags.py
python3 src/plot_tags.py --tags house recurring school --freq M
python3 src/plot_tags.py --freq Y --normalize --output outputs/plots/tags_by_year.png
python3 src/plot_tags.py --start-date 2023-01-01 --end-date 2023-12-31
```

Arguments:

- `--dreams-path`: path to parsed dream JSONL records. Defaults to `data/dreams.jsonl`.
- `--output`: path where the plot image should be saved. Defaults to `outputs/plots/tag_frequency.png`.
- `--tags`: specific tags to plot. Defaults to the top tags.
- `--top-n`: number of top tags to plot when `--tags` is omitted. Defaults to `10`.
- `--freq`: time grouping frequency: `M`, `Q`, or `Y`. Defaults to `M`.
- `--start-date`: only include dreams on or after this date.
- `--end-date`: only include dreams on or before this date.
- `--normalize`: plot percent of dreams per period instead of raw counts.
- `--title`: optional plot title.
- `--show`: display the plot interactively after saving.

## `src/compute_stats.py`

Computes dream counts, tag frequencies, and word-count statistics, then prints and saves JSON.

```bash
python3 src/compute_stats.py
python3 src/compute_stats.py --freq Q --start-date 2023-01-01 --end-date 2023-12-31
python3 src/compute_stats.py --freq Y --output outputs/stats/yearly_stats.json
python3 src/compute_stats.py --common-words 50 --stopwords-path data/stopwords.txt
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

## `src/retrieve_dreams.py`

Embeds a text query and returns the closest dreams from the ChromaDB collection.

```bash
python3 src/retrieve_dreams.py "hidden room water"
python3 src/retrieve_dreams.py "school anxiety" --top-k 5
```

Arguments:

- `query`: required text query to search for.
- `--top-k`: number of closest dreams to return. Defaults to `10`.
- `--chroma-path`: path to the persistent ChromaDB database. Defaults to `data/chroma_db`.
- `--collection-name`: ChromaDB collection name to query. Defaults to `dreams`.
- `--embed-model`: Ollama embedding model. Defaults to `nomic-embed-text`.
- `--preview-chars`: maximum preview length per result. Defaults to `300`.

Requires an existing ChromaDB index and Ollama running locally at `http://localhost:11434`.

## `src/basic_rag.py`

Retrieves relevant dreams for a question and asks an Ollama chat model to answer using only those entries.

```bash
python3 src/basic_rag.py "What patterns appear in dreams about hidden rooms?"
python3 src/basic_rag.py "What symbols recur in school dreams?" --top-k 5 --chat-model qwen3:8b
python3 src/basic_rag.py "What patterns appear in dreams about hidden rooms?" --retrieval-query "hidden room hidden hallway secret room"
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

## `src/analyze_dream.py`

Loads one dream by ID or accepts dream text directly, then asks an Ollama chat model for a close analysis of its events, dynamics, themes, and possible interpretations.
The analysis is printed and saved under `outputs/analysis/`. ID-based filenames use `<dream_id>_<datetime>.txt`; direct-text filenames use `<datetime>.txt`.

```bash
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0
python3 src/analyze_dream.py --text "I opened a door and found another kitchen."
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0 --chat-model qwen3:8b
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0 --related-dreams 5 --similarity-threshold 0.55
python3 src/analyze_dream.py --dream-id dream-2022-10-10-0 --related-dreams 5 --start-date 2022-04-10 --end-date 2022-10-10
```

Arguments:

- `--dream-id`: dream ID to load from JSONL. Mutually exclusive with `--text`.
- `--text`: dream text to analyze directly. Mutually exclusive with `--dream-id`.
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

## `src/compare_models.py`

Runs the same task across these four combinations:

- `qwen3:8b` with `nomic-embed-text`
- `gemma3:12b` with `nomic-embed-text`
- `qwen3:8b` with `qwen3-embedding`
- `gemma3:12b` with `qwen3-embedding`

It expects collections named `dreams_nomic_embed_text` and
`dreams_qwen3_embedding`. Retrieval is performed once per embedding model and
the exact retrieved context is reused for both chat models. Results are saved
as JSON and Markdown under `outputs/model_comparisons/`.

Compare RAG answers. A fixed retrieval query is required so chat models do not
generate different search queries:

```bash
python3 src/compare_models.py rag \
  "What patterns appear in dreams about hidden rooms?" \
  --retrieval-query "hidden room hallway extra room concealed door behind wall"
```

Compare analysis of one dream with five related dreams:

```bash
python3 src/compare_models.py analyze \
  --dream-id dream-2022-1-22-0 \
  --related-dreams 5
```

The runner defaults to temperature `0` for a controlled first comparison. Run
`python3 src/compare_models.py rag --help` or
`python3 src/compare_models.py analyze --help` for task-specific options.

## `src/evaluate_retrieval.py`

Embeds one retrieval prompt with both `nomic-embed-text` and
`qwen3-embedding`, retrieves the top-k dreams from their matching collections,
and asks `gemma3:12b` to score every result from 1 (irrelevant) to 5 (directly
relevant). The judge sees the prompt and dream content but not the embedding
model name. JSON and Markdown reports are saved under
`outputs/retrieval_evaluations/`.

```bash
python3 src/evaluate_retrieval.py \
  "hidden room hallway extra room concealed door behind wall" \
  --top-k 8
```

Use the complete text of an existing dream as the retrieval and relevance
target. The target dream itself is excluded from the results:

```bash
python3 src/evaluate_retrieval.py \
  --dream-id dream-2022-1-22-0 \
  --top-k 8
```

Useful options include `--judge-model`, `--max-chars-per-dream`, `--num-ctx`,
`--chroma-path`, and `--output-dir`.
