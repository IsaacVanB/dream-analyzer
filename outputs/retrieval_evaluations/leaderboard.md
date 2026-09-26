# Retrieval evaluation leaderboard

- Generated: `2026-09-26T00:10:19-04:00`
- Experiments: `8`
- Bold values are best within each query-suite, dream-corpus, and model table.

## Current best results

No experiments match the currently configured query suite and dream corpus.


## Current corpus experiments

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
