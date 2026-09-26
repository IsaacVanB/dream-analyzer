# Retrieval evaluation leaderboard

- Generated: `2026-09-26T01:55:33-04:00`
- Experiments: `24`
- Bold values are best within each query-suite, dream-corpus, and model table.

## Current best results

- R-precision: **0.608** — agent / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; chat: `gemma4:12b`)
- R@5: **0.676** — agent / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; chat: `gemma4:12b`)
- R@10: **0.783** — agent / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; chat: `qwen3:8b`)
- Errors: **0** — agent / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; chat: `gemma4:12b`), fixed-hybrid-baseline / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; no chat model), BM25-baseline / qwen3-embedding / gemma4:12b (no embedding or chat model), semantic-baseline / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; no chat model), agent / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; chat: `qwen3:8b`), fixed-hybrid-baseline / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; no chat model), semantic-baseline / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; no chat model), BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model), agent / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; chat: `gemma4:12b`), fixed-hybrid-baseline / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; no chat model), BM25-baseline / nomic-embed-text / gemma4:12b (no embedding or chat model), semantic-baseline / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; no chat model), agent / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; chat: `qwen3:8b`), fixed-hybrid-baseline / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; no chat model), BM25-baseline / nomic-embed-text / qwen3:8b (no embedding or chat model), semantic-baseline / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; no chat model)
- Mean seconds/query: **0.000** — BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model)

### Current R-precision winners by category

| Category | Best R-precision | Experiment and models |
|---|---:|---|
| character | 0.956 | agent / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; chat: `qwen3:8b`) |
| conceptual | 0.277 | fixed-hybrid-baseline / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; no chat model); fixed-hybrid-baseline / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; no chat model) |
| direct_semantic | 0.748 | agent / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; chat: `qwen3:8b`) |
| exact_places_and_labels | 1 | BM25-baseline / qwen3-embedding / gemma4:12b (no embedding or chat model); BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model); BM25-baseline / nomic-embed-text / gemma4:12b (no embedding or chat model); BM25-baseline / nomic-embed-text / qwen3:8b (no embedding or chat model) |
| exact_proper_names | 1 | agent / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; chat: `gemma4:12b`); BM25-baseline / qwen3-embedding / gemma4:12b (no embedding or chat model); agent / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; chat: `qwen3:8b`); BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model); agent / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; chat: `gemma4:12b`); BM25-baseline / nomic-embed-text / gemma4:12b (no embedding or chat model); semantic-baseline / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; no chat model); agent / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; chat: `qwen3:8b`); fixed-hybrid-baseline / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; no chat model); BM25-baseline / nomic-embed-text / qwen3:8b (no embedding or chat model); semantic-baseline / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; no chat model) |
| inflected_word_variants | 1 | fixed-hybrid-baseline / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; no chat model); BM25-baseline / qwen3-embedding / gemma4:12b (no embedding or chat model); semantic-baseline / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; no chat model); agent / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; chat: `qwen3:8b`); fixed-hybrid-baseline / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; no chat model); semantic-baseline / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; no chat model); BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model); BM25-baseline / nomic-embed-text / gemma4:12b (no embedding or chat model); BM25-baseline / nomic-embed-text / qwen3:8b (no embedding or chat model) |
| mixed_entity_plus_concept | 0.850 | agent / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; chat: `gemma4:12b`); semantic-baseline / nomic-embed-text / gemma4:12b (embedding: `nomic-embed-text`; no chat model); semantic-baseline / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; no chat model) |
| multi_concept | 0.512 | agent / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; chat: `qwen3:8b`) |
| rare_literal_objects | 0.800 | BM25-baseline / qwen3-embedding / gemma4:12b (no embedding or chat model); BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model); BM25-baseline / nomic-embed-text / gemma4:12b (no embedding or chat model); BM25-baseline / nomic-embed-text / qwen3:8b (no embedding or chat model) |
| recurring_event | 0.413 | fixed-hybrid-baseline / nomic-embed-text / qwen3:8b (embedding: `nomic-embed-text`; no chat model) |
| semantic_without_lexical_match | 0.750 | agent / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; chat: `gemma4:12b`); semantic-baseline / qwen3-embedding / gemma4:12b (embedding: `qwen3-embedding`; no chat model); semantic-baseline / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; no chat model) |
| separated_phrase_terms | 0.950 | BM25-baseline / qwen3-embedding / gemma4:12b (no embedding or chat model); BM25-baseline / qwen3-embedding / qwen3:8b (no embedding or chat model); BM25-baseline / nomic-embed-text / gemma4:12b (no embedding or chat model); BM25-baseline / nomic-embed-text / qwen3:8b (no embedding or chat model) |
| temporal | 0.562 | agent / qwen3-embedding / qwen3:8b (embedding: `qwen3-embedding`; chat: `qwen3:8b`) |

## Current corpus experiments

### Current query suite

Compatibility key: `f0e9598df0a7-7f3937edabb4`

- Query fingerprint: `f0e9598df0a77b1ff91b37411bc0e94960034395f183ca47b62e16c608761fd3`
- Dream corpus fingerprint: `7f3937edabb456288c50e31a55b10b264bb613e303886588b0ff7467256180dd`

#### Embedding: `qwen3-embedding`; chat: `gemma4:12b`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [agent / qwen3-embedding / gemma4:12b](<benchmark_agent_2026-09-26_01-55-33-505972.md>) | agent | 2026-09-26T01:55:33-04:00 | 56 | **0.608** | **0.676** | **0.763** | **0** | 50.793 | Corrected embedding/chat-model retrieval matrix |
| [fixed-hybrid-baseline / qwen3-embedding / gemma4:12b](<benchmark_hybrid_2026-09-26_01-08-07-736232.md>) | hybrid | 2026-09-26T01:08:07-04:00 | 56 | 0.444 | 0.589 | 0.680 | **0** | 0.359 | Corrected embedding/chat-model retrieval matrix |
| [BM25-baseline / qwen3-embedding / gemma4:12b](<benchmark_bm25_2026-09-26_01-07-46-194636.md>) | bm25 | 2026-09-26T01:07:46-04:00 | 56 | 0.596 | 0.602 | 0.677 | **0** | **0.000** | Corrected embedding/chat-model retrieval matrix |
| [semantic-baseline / qwen3-embedding / gemma4:12b](<benchmark_embedding_2026-09-26_01-07-44-986134.md>) | embedding | 2026-09-26T01:07:44-04:00 | 56 | 0.357 | 0.403 | 0.445 | **0** | 0.458 | Corrected embedding/chat-model retrieval matrix |

#### Embedding: `qwen3-embedding`; chat: `qwen3:8b`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [agent / qwen3-embedding / qwen3:8b](<benchmark_agent_2026-09-26_01-07-17-981034.md>) | agent | 2026-09-26T01:07:17-04:00 | 56 | 0.574 | **0.652** | **0.746** | **0** | 16.389 | Corrected embedding/chat-model retrieval matrix |
| [fixed-hybrid-baseline / qwen3-embedding / qwen3:8b](<benchmark_hybrid_2026-09-26_00-51-59-378785.md>) | hybrid | 2026-09-26T00:51:59-04:00 | 56 | 0.444 | 0.589 | 0.680 | **0** | 0.156 | Corrected embedding/chat-model retrieval matrix |
| [semantic-baseline / qwen3-embedding / qwen3:8b](<benchmark_embedding_2026-09-26_00-51-49-084113.md>) | embedding | 2026-09-26T00:51:49-04:00 | 56 | 0.357 | 0.403 | 0.445 | **0** | 0.292 | Corrected embedding/chat-model retrieval matrix |
| [BM25-baseline / qwen3-embedding / qwen3:8b](<benchmark_bm25_2026-09-26_00-51-49-831334.md>) | bm25 | 2026-09-26T00:51:49-04:00 | 56 | **0.596** | 0.602 | 0.677 | **0** | **0.000** | Corrected embedding/chat-model retrieval matrix |

#### Embedding: `nomic-embed-text`; chat: `gemma4:12b`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [agent / nomic-embed-text / gemma4:12b](<benchmark_agent_2026-09-26_00-51-31-770944.md>) | agent | 2026-09-26T00:51:31-04:00 | 56 | 0.554 | **0.656** | 0.716 | **0** | 24.840 | Corrected embedding/chat-model retrieval matrix |
| [fixed-hybrid-baseline / nomic-embed-text / gemma4:12b](<benchmark_hybrid_2026-09-26_00-28-18-904054.md>) | hybrid | 2026-09-26T00:28:18-04:00 | 56 | 0.434 | 0.636 | **0.736** | **0** | 0.103 | Corrected embedding/chat-model retrieval matrix |
| [BM25-baseline / nomic-embed-text / gemma4:12b](<benchmark_bm25_2026-09-26_00-28-11-308046.md>) | bm25 | 2026-09-26T00:28:11-04:00 | 56 | **0.596** | 0.602 | 0.677 | **0** | **0.001** | Corrected embedding/chat-model retrieval matrix |
| [semantic-baseline / nomic-embed-text / gemma4:12b](<benchmark_embedding_2026-09-26_00-28-09-526490.md>) | embedding | 2026-09-26T00:28:09-04:00 | 56 | 0.416 | 0.411 | 0.469 | **0** | 0.094 | Corrected embedding/chat-model retrieval matrix |

#### Embedding: `nomic-embed-text`; chat: `qwen3:8b`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [agent / nomic-embed-text / qwen3:8b](<benchmark_agent_2026-09-26_00-24-50-171917.md>) | agent | 2026-09-26T00:24:50-04:00 | 56 | 0.537 | **0.659** | **0.783** | **0** | 9.920 | Full embedding/chat-model retrieval matrix |
| [fixed-hybrid-baseline / nomic-embed-text / qwen3:8b](<benchmark_hybrid_2026-09-26_00-15-33-040126.md>) | hybrid | 2026-09-26T00:15:33-04:00 | 56 | 0.456 | 0.636 | 0.740 | **0** | 0.094 | Full embedding/chat-model retrieval matrix |
| [BM25-baseline / nomic-embed-text / qwen3:8b](<benchmark_bm25_2026-09-26_00-15-26-571327.md>) | bm25 | 2026-09-26T00:15:26-04:00 | 56 | **0.596** | 0.602 | 0.677 | **0** | **0.000** | Full embedding/chat-model retrieval matrix |
| [semantic-baseline / nomic-embed-text / qwen3:8b](<benchmark_embedding_2026-09-26_00-15-25-416906.md>) | embedding | 2026-09-26T00:15:25-04:00 | 56 | 0.416 | 0.411 | 0.469 | **0** | 0.182 | Full embedding/chat-model retrieval matrix |

### Other query suite `38728809d53d20ace0924b6830740e44e28ad8871f11822ca175f7b94b3e0e9a`

Compatibility key: `38728809d53d-7f3937edabb4`

- Query fingerprint: `38728809d53d20ace0924b6830740e44e28ad8871f11822ca175f7b94b3e0e9a`
- Dream corpus fingerprint: `7f3937edabb456288c50e31a55b10b264bb613e303886588b0ff7467256180dd`

#### Embedding: `nomic-embed-text`; chat: `qwen3:8b`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [no-redundant-results](<benchmark_agent_2026-09-21_19-57-02-802607.md>) | agent | 2026-09-21T19:57:02-04:00 | 64 | 0.448 | 0.628 | **0.732** | **0** | 10.375 | Prevented duplicate queries skewing RRF results |
| [fixed-hybrid-baseline](<benchmark_hybrid_2026-09-15_21-38-05-626331.md>) | hybrid | 2026-09-15T21:38:05-04:00 | 64 | 0.459 | **0.639** | 0.731 | **0** | 0.093 | Verbatim semantic and BM25 retrieval fused using reciprocal-rank fusion |
| [BM25 baseline](<benchmark_bm25_2026-09-15_21-37-14-119953.md>) | bm25 | 2026-09-15T21:37:14-04:00 | 64 | **0.595** | 0.597 | 0.676 | **0** | **0.000** | Evaluation queries sent verbatim to the in-memory BM25 index |
| [semantic-baseline](<benchmark_embedding_2026-09-15_21-36-43-290945.md>) | embedding | 2026-09-15T21:36:43-04:00 | 64 | 0.418 | 0.412 | 0.466 | **0** | 0.165 | Evaluation queries embedded verbatim and ranked by Chroma |
| [one-shot-expanded-agent](<benchmark_agent_2026-09-15_19-02-45-183907.md>) | agent | 2026-09-15T19:02:45-04:00 | 64 | 0.376 | 0.383 | 0.459 | **0** | 2.174 | One-shot upfront retrieval plan with expanded 6-10 word semantic rewrites and normalized duplicate rejection. |
| [expanded-batched-agent](<benchmark_agent_2026-09-15_18-55-09-469453.md>) | agent | 2026-09-15T18:55:09-04:00 | 64 | 0.477 | 0.537 | 0.701 | **0** | 11.369 | Expanded 6-10 word semantic rewrites, upfront complementary query batching, and rejected duplicate calls. |


## Older or incompatible experiment groups

### `legacy-2861f0268f07-unknown-corpus`

Compatibility key: `legacy-2861f0268f07-unknown-corpus`

- Query fingerprint: `2861f0268f07ddc64a5e3c7fc3378288e089942add3f937abb1a59cbf48c69bc`
- Dream corpus fingerprint: `unknown`

#### Embedding: `nomic-embed-text`; chat: `qwen3:8b`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [benchmark_agent_2026-09-11_19-46-31-129173](<benchmark_agent_2026-09-11_19-46-31-129173.md>) | agent | 2026-09-11T19:46:31-04:00 | 30 | **0.450** | **0.401** | **0.518** | **0** | 10.825 |  |
| [benchmark_embedding_2026-09-11_19-28-13-792910](<benchmark_embedding_2026-09-11_19-28-13-792910.md>) | embedding | 2026-09-11T19:28:13-04:00 | 30 | 0.417 | 0.366 | 0.455 | **0** | **0.249** |  |


## Skipped reports

- `benchmark_2026-09-11_18-00-01-005571.json`: not a retrieval benchmark report
