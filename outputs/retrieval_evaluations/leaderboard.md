# Retrieval evaluation leaderboard

- Generated: `2026-09-15T21:38:05-04:00`
- Experiments: `7`
- Bold values are best within a compatible query-suite and dream-corpus group.
- Routing accuracy applies only to agent runs.

## Current best results

- R-precision: **0.595** — BM25 baseline
- R@5: **0.639** — fixed-hybrid-baseline
- R@10: **0.731** — fixed-hybrid-baseline
- Routing accuracy: **0.531** — expanded-batched-agent
- Errors: **0** — fixed-hybrid-baseline, BM25 baseline, semantic-baseline, one-shot-expanded-agent, expanded-batched-agent
- Mean seconds/query: **0.000** — BM25 baseline

## Current compatible experiments

Compatibility key: `38728809d53d-7f3937edabb4`

- Query fingerprint: `38728809d53d20ace0924b6830740e44e28ad8871f11822ca175f7b94b3e0e9a`
- Dream corpus fingerprint: `7f3937edabb456288c50e31a55b10b264bb613e303886588b0ff7467256180dd`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Routing accuracy | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [fixed-hybrid-baseline](<benchmark_hybrid_2026-09-15_21-38-05-626331.md>) | hybrid | 2026-09-15T21:38:05-04:00 | 64 | 0.459 | **0.639** | **0.731** | n/a | **0** | 0.093 | Verbatim semantic and BM25 retrieval fused using reciprocal-rank fusion |
| [BM25 baseline](<benchmark_bm25_2026-09-15_21-37-14-119953.md>) | bm25 | 2026-09-15T21:37:14-04:00 | 64 | **0.595** | 0.597 | 0.676 | n/a | **0** | **0.000** | Evaluation queries sent verbatim to the in-memory BM25 index |
| [semantic-baseline](<benchmark_embedding_2026-09-15_21-36-43-290945.md>) | embedding | 2026-09-15T21:36:43-04:00 | 64 | 0.418 | 0.412 | 0.466 | n/a | **0** | 0.165 | Evaluation queries embedded verbatim and ranked by Chroma |
| [one-shot-expanded-agent](<benchmark_agent_2026-09-15_19-02-45-183907.md>) | agent | 2026-09-15T19:02:45-04:00 | 64 | 0.376 | 0.383 | 0.459 | 0.469 | **0** | 2.174 | One-shot upfront retrieval plan with expanded 6-10 word semantic rewrites and normalized duplicate rejection. |
| [expanded-batched-agent](<benchmark_agent_2026-09-15_18-55-09-469453.md>) | agent | 2026-09-15T18:55:09-04:00 | 64 | 0.477 | 0.537 | 0.701 | **0.531** | **0** | 11.369 | Expanded 6-10 word semantic rewrites, upfront complementary query batching, and rejected duplicate calls. |

### Current R-precision winners by category

| Category | Best R-precision | Experiment |
|---|---:|---|
| character | 0.905 | fixed-hybrid-baseline |
| conceptual | 0.279 | expanded-batched-agent |
| direct_semantic | 0.533 | BM25 baseline |
| exact_places_and_labels | 1 | BM25 baseline |
| exact_proper_names | 1 | fixed-hybrid-baseline, BM25 baseline, semantic-baseline, expanded-batched-agent |
| inflected_word_variants | 1 | BM25 baseline |
| mixed_entity_plus_concept | 0.900 | semantic-baseline |
| multi_concept | 0.472 | expanded-batched-agent |
| rare_literal_objects | 0.800 | BM25 baseline |
| recurring_event | 0.409 | fixed-hybrid-baseline |
| semantic_without_lexical_match | 0.250 | fixed-hybrid-baseline, semantic-baseline, one-shot-expanded-agent, expanded-batched-agent |
| separated_phrase_terms | 0.950 | BM25 baseline |
| temporal | 0.542 | expanded-batched-agent |

## Older or incompatible experiment groups

### `legacy-54d048e4c1a9-unknown-corpus`

Compatibility key: `legacy-54d048e4c1a9-unknown-corpus`

- Query fingerprint: `54d048e4c1a93ebd7687a09248457c9c003fcb9df3558d54e739210df641efea`
- Dream corpus fingerprint: `unknown`

| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Routing accuracy | Errors | Mean seconds/query | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [benchmark_agent_2026-09-11_19-46-31-129173](<benchmark_agent_2026-09-11_19-46-31-129173.md>) | agent | 2026-09-11T19:46:31-04:00 | 30 | **0.450** | **0.401** | **0.518** | n/a | **0** | 10.825 |  |
| [benchmark_embedding_2026-09-11_19-28-13-792910](<benchmark_embedding_2026-09-11_19-28-13-792910.md>) | embedding | 2026-09-11T19:28:13-04:00 | 30 | 0.417 | 0.366 | 0.455 | n/a | **0** | **0.249** |  |


## Skipped reports

- `benchmark_2026-09-11_18-00-01-005571.json`: not a retrieval benchmark report
