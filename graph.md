# Knowledge Graph Summary

## Problem Statement
TokenSmith’s retrieval pipeline originally had no persistent knowledge graph layer. Even after hybrid ranking was improved, retrieval remained limited to chunk-level semantic, lexical, and textbook-index signals. That meant the system could not explicitly capture entity and relation structure from the corpus or use graph neighborhoods to surface related evidence across chunks, which was one of the core goals in `CS6423.pdf`.

## Changes Made
- Added graph artifact models and persistence helpers in [src/graph/store.py](/NAS/School/CS6423/TokenSmith/src/graph/store.py).
- Added a versioned, human-reviewable graph artifact format containing:
  - `GraphNode`
  - `GraphEdge`
  - `GraphChunkLink`
  - `GraphArtifact`
- Added heuristic entity and relation extraction in [src/graph/extraction.py](/NAS/School/CS6423/TokenSmith/src/graph/extraction.py).
- Added graph query scoring and one-hop neighborhood retrieval in [src/graph/retrieval.py](/NAS/School/CS6423/TokenSmith/src/graph/retrieval.py).
- Exported graph helpers from [src/graph/__init__.py](/NAS/School/CS6423/TokenSmith/src/graph/__init__.py).
- Updated [src/index_builder.py](/NAS/School/CS6423/TokenSmith/src/index_builder.py) so indexing now builds and saves `{index_prefix}_graph.json` alongside FAISS, BM25, chunk, source, and metadata artifacts.
- Updated [src/config.py](/NAS/School/CS6423/TokenSmith/src/config.py) so `graph` is a first-class retriever with configurable artifact path and alias expansion behavior.
- Updated [src/retriever.py](/NAS/School/CS6423/TokenSmith/src/retriever.py) so artifact loading understands the graph artifact and `build_retrievers()` can instantiate `GraphRetriever`.
- Updated [src/main.py](/NAS/School/CS6423/TokenSmith/src/main.py) and [src/api_server.py](/NAS/School/CS6423/TokenSmith/src/api_server.py) so graph artifacts are loaded and passed into the live retrieval pipeline.
- Added graph-specific coverage in [tests/test_graph_indexing.py](/NAS/School/CS6423/TokenSmith/tests/test_graph_indexing.py), [tests/test_graph_extraction.py](/NAS/School/CS6423/TokenSmith/tests/test_graph_extraction.py), [tests/test_graph_retrieval.py](/NAS/School/CS6423/TokenSmith/tests/test_graph_retrieval.py), and expanded hybrid/end-to-end checks in [tests/test_hybrid_retrieval.py](/NAS/School/CS6423/TokenSmith/tests/test_hybrid_retrieval.py) and [tests/test_end_to_end.py](/NAS/School/CS6423/TokenSmith/tests/test_end_to_end.py).

## Implementation
The implemented graph flow is:
1. Build chunks and per-chunk metadata during indexing.
2. Run deterministic heuristic extraction over each chunk to identify normalized entities and supported relations.
3. Aggregate those results into a persistent `GraphArtifact` with node IDs, edge IDs, chunk links, and evidence metadata.
4. Save the graph artifact as JSON so it can be inspected and reloaded separately from FAISS and BM25.
5. Load the graph artifact at retrieval time together with the other corpus artifacts.
6. Build `GraphRetriever` as another active retrieval source when `graph` is enabled in config.
7. Score chunks by matching query tokens against graph nodes and then expanding across one-hop graph edges.
8. Return graph chunk scores in the same `Dict[int, float]` format as the other retrievers so the existing ensemble ranker can fuse them without special handling.

## Challenges
- The graph layer needed a stable artifact contract before it could be wired into indexing or retrieval, otherwise every later phase would have depended on an undefined storage format.
- The extraction logic had to stay deterministic and offline-friendly, which ruled out relying on a heavy or non-reproducible external extraction service for the first implementation.
- Normalization was tricky even in the heuristic version because entity forms like singular/plural variants needed to collapse to stable graph IDs.
- Retrieval integration needed explicit graph artifact validation so enabling `graph` would fail clearly when the graph JSON was missing instead of silently degrading.
- A circular import appeared once graph retrieval reused BM25 tokenization indirectly through `src/index_builder.py`; this had to be removed so indexing, retrieval, and graph modules stayed import-safe.
- Graph retrieval needed to fit the existing fusion contract cleanly, so the graph scorer had to emit bounded chunk scores instead of inventing a special ranking path outside `EnsembleRanker`.

## Future Work
- Improve entity and relation extraction beyond the current heuristic pattern set so the graph covers more textbook concepts and relation types.
- Add richer alias generation and normalization so queries with paraphrases or abbreviations match graph nodes more reliably.
- Tune graph retrieval weights and scoring coefficients on the benchmark suite rather than relying only on hand-chosen defaults.
- Add retrieval-debug instrumentation showing which graph nodes and edges contributed to each retrieved chunk.
- Expand graph retrieval beyond one-hop neighborhoods when benchmarks show it helps without introducing too much noise.
- Support human editing or post-processing of the persisted graph artifact and verify reload behavior after edits.
