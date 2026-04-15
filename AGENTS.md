# Scope / Project Overview

TokenSmith already has the first half of the CS6423 retrieval upgrade in place:

- Structure-aware chunking exists in `src/preprocessing/chunking.py` and is wired through `src/config.py` and `src/index_builder.py`.
- The hybrid ranking function described in `ranking.md` is implemented: FAISS, BM25, and textbook index keyword retrieval now contribute explicit nonzero signals, and `EnsembleRanker` fuses only active retrieval sources.

The remaining gap, relative to `CS6423.pdf`, is the LightRAG-style knowledge graph layer. The next implementation step is to add a persistent, human-reviewable graph of entities and relations extracted from chunks, then use that graph as a fourth retrieval source at query time. The final retrieval stack should therefore be:

1. FAISS semantic retrieval
2. BM25 lexical retrieval
3. Textbook index keyword retrieval
4. LightRAG-style graph retrieval over entity and relation neighborhoods

The graph implementation should sit on top of the current ranking work, not replace it. The intended end state is:

- Indexing builds chunk artifacts, FAISS, BM25, page-to-chunk mappings, and graph artifacts together.
- Query-time retrieval loads graph artifacts alongside the existing indexes.
- `EnsembleRanker` receives raw score dictionaries from all enabled sources, including the graph retriever.
- Configuration allows graph retrieval to be enabled/disabled and weighted just like the existing retrievers.
- Tests verify graph construction, persistence, loading, and contribution to final hybrid ranking.


# Files To Edit / Add

Edit these existing files:

- `src/config.py`
  - Add graph-related config fields such as enablement, artifact paths, extraction model settings if needed, and default `ranker_weights` / `enabled_retrievers` support for a `graph` retriever.
- `src/index_builder.py`
  - Extend indexing to extract graph entities/relations from chunks and persist graph artifacts in the same artifact directory as FAISS/BM25/chunk metadata.
- `src/retriever.py`
  - Extend artifact loading, retriever construction, and query-time retrieval to support a LightRAG-style graph retriever.
- `src/main.py`
  - Ensure index mode builds graph artifacts and chat mode loads and uses the graph retriever when enabled.
- `src/api_server.py`
  - Mirror the same graph artifact loading and retrieval pipeline used in `src/main.py`.
- `src/ranking/ranker.py`
  - Likely no algorithmic rewrite is needed, but confirm the current active-retriever validation works cleanly once `graph` is added as another source name.
- `tests/test_hybrid_retrieval.py`
  - Extend retrieval/ranking coverage to include `graph`.
- `tests/test_api.py`
  - Update config, retriever construction, and ranker expectations for the extra source.
- `tests/test_end_to_end.py`
  - Add or update end-to-end checks so graph retrieval participates in the full pipeline.

Add these new files:

- `src/graph/__init__.py`
  - Package for graph-specific logic.
- `src/graph/extraction.py`
  - Entity/relation extraction from chunk text.
- `src/graph/store.py`
  - Persistent graph artifact format and load/save helpers.
- `src/graph/retrieval.py`
  - Query-time graph retrieval logic that maps graph matches back to chunk IDs with scores.
- `tests/test_graph_indexing.py`
  - Unit tests for graph extraction, graph persistence, and chunk linkage.
- `tests/test_graph_retrieval.py`
  - Unit tests for graph neighborhood retrieval and score generation.

If the implementation stays small, `src/graph/extraction.py`, `src/graph/store.py`, and `src/graph/retrieval.py` could be collapsed into fewer files. The important part is to keep graph extraction/persistence separate from the generic retriever glue.


# Step-By-Step Implementation Plan

1. Define the graph artifact contract.
   - Choose exactly what gets persisted per node, edge, and chunk link.
   - At minimum persist:
     - normalized entity nodes
     - relation edges
     - source chunk IDs for every extracted fact
     - optional aliases / lexical forms for matching
   - Store the graph in a human-reviewable format such as JSON or JSON + pickle only where necessary.

2. Add graph configuration support.
   - Update `RAGConfig` in `src/config.py` so `graph` is a supported retriever name.
   - Give `graph` a nonzero default weight in `ranker_weights` once the implementation is ready.
   - Extend `_resolve_enabled_retrievers()` and `get_active_ranker_weights()` so `graph` behaves exactly like `faiss`, `bm25`, and `index_keywords`.
   - Add graph artifact path fields if artifact names are configurable.

3. Implement entity and relation extraction over chunks.
   - Build a lightweight extraction stage in `src/graph/extraction.py` that processes each indexed chunk and emits normalized entities and relations tied back to `chunk_id`.
   - Keep normalization deterministic where possible so tests remain stable.
   - Prefer a simple, inspectable first version over an opaque extraction pipeline.

4. Persist the LightRAG-style graph during indexing.
   - Extend `build_index()` in `src/index_builder.py` after chunk creation so it builds the graph from chunk text and metadata.
   - Save graph artifacts into the same artifact directory and prefix scheme used for FAISS/BM25.
   - The persisted graph must survive reloads and support human inspection/editing, matching the assignment requirements.

5. Load graph artifacts with the rest of the retrieval state.
   - Extend `load_artifacts()` in `src/retriever.py` so missing graph files are either:
     - required when `graph` retrieval is enabled, or
     - optional when `graph` is disabled.
   - Keep failure modes explicit; do not silently run with an enabled graph retriever and no graph artifact.

6. Implement graph retrieval as a first-class retriever.
   - Add a `GraphRetriever` that follows the same interface as the other retrievers and returns `Dict[int, float]`.
   - Query flow should:
     - extract or normalize query entities
     - match them against graph nodes / aliases
     - walk relevant relations / neighborhoods
     - accumulate chunk scores from linked facts and neighboring evidence
   - Keep scoring deterministic and bounded so it fuses cleanly with the existing ranker.

7. Wire graph retrieval into the current hybrid pipeline.
   - Update `build_retrievers()` so enabled retrieval sources can now include `graph`.
   - Ensure `src/main.py` and `src/api_server.py` pass the graph raw scores into `EnsembleRanker` exactly the same way they already do for other retrievers.
   - Do not special-case graph ranking outside the ensemble unless a clearly documented bonus score is required.

8. Validate ranking compatibility.
   - Confirm `src/ranking/ranker.py` needs no structural change beyond accepting the new source name.
   - If score distributions are too different, normalize in the graph retriever or rely on the existing ranker normalization rather than adding ad hoc fusion logic.

9. Add deterministic tests for graph indexing and retrieval.
   - Verify entities and relations are extracted and linked to the correct `chunk_id`s.
   - Verify persisted graph artifacts reload without losing chunk linkage.
   - Verify `build_retrievers()` includes `graph` when enabled.
   - Verify the ensemble ranker accepts four-source raw score dictionaries and that graph retrieval can alter the final ordering.

10. Re-run benchmark-oriented validation.
    - Reuse the existing benchmark/test harness to compare retrieval quality before and after graph integration.
    - Focus on Recall@k, NDCG, and final aggregate score, consistent with `CS6423.pdf`.
    - Also track indexing time, query latency, and artifact size so graph gains do not come with unreasonable overhead.

11. Keep the implementation aligned with the assignment boundaries.
    - The graph is meant to improve retrieval, not become a separate answer-generation system.
    - The deliverable is a LightRAG-style graph-enhanced hybrid retriever inside TokenSmith’s existing architecture.
    - Preserve the current structure-aware chunking and ranking improvements; extend them with graph artifacts and graph retrieval instead of reworking them.
