# Dream Analysis

Small local pipeline for parsing a dream journal, embedding dreams with Ollama,
storing them in ChromaDB, and asking retrieval-augmented questions.

## Setup

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Ollama must be running locally at `http://localhost:11434`.

## Workflow

Parse the raw journal:

```bash
python3 src/parse_dreams.py
```

Build the ChromaDB index:

```bash
python3 src/build_chroma_db.py
```

Retrieve similar dreams:

```bash
python3 src/retrieve_dreams.py "hidden room water" --top-k 5
```

Plot tag frequency over time:

```bash
python3 src/plot_tags.py --tags house recurring school
```

Compute summary stats:

```bash
python3 src/compute_stats.py --freq Y
```

Analyze one dream:

```bash
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0
python3 src/analyze_dream.py --dream-id dream-2022-1-22-0 --related-dreams 5
```

Ask a RAG question:

```bash
python3 src/basic_rag.py "What patterns appear in dreams about hidden rooms?"
```

Use a manual retrieval query when the question has extra analysis language:

```bash
python3 src/basic_rag.py \
  "What patterns appear in dreams about hidden rooms?" \
  --retrieval-query "hidden room hallway extra room concealed door behind wall"
```

See `example_usages.md` for more detailed command examples and arguments.
