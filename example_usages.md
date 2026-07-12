# Example Usages

## `src/parse_dreams.py`

Parses the raw dream journal text file into JSON Lines, one JSON object per dream.

```bash
python3 src/parse_dreams.py
python3 src/parse_dreams.py data/mock_dream_journal.txt data/dreams.jsonl
```

Arguments:

- `input`: optional path to the raw journal text file. Defaults to `data/mock_dream_journal.txt`.
- `output`: optional path for parsed JSONL output. Defaults to `data/dreams.jsonl`.

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