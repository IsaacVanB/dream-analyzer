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

Output fields include `dream_id`, `date`, `year`, `month`, `tags`, `text`, and `word_count`.

## `src/build_chroma_db.py`

Embeds parsed dreams with Ollama `nomic-embed-text` and saves them to a persistent ChromaDB collection.

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
