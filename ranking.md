# Hybrid Retrieval And Ranking Summary

## Problem Statement
TokenSmith originally had only a partial retrieval fusion setup. The baseline code supported FAISS, BM25, and textbook index keyword retrieval components, but the intended hybrid retrieval design was not fully realized because source participation was not explicit, the default configuration could effectively disable some sources, and graph-based retrieval was not wired into the query-time pipeline.

## Changes Made
- Updated retrieval-source configuration in [src/config.py](/NAS/School/CS6423/TokenSmith/src/config.py) so enabled retrievers are derived explicitly instead of being implied only by older defaults.
- Changed default hybrid weights so FAISS, BM25, and textbook index keyword retrieval all contribute nonzero weight.
- Added active-retriever helpers in [src/config.py](/NAS/School/CS6423/TokenSmith/src/config.py) to expose the enabled retrieval set and active ranker weights cleanly.
- Updated retriever construction in [src/retriever.py](/NAS/School/CS6423/TokenSmith/src/retriever.py) so the query pipeline builds retrievers from the enabled source list.
- Tightened fusion logic in [src/ranking/ranker.py](/NAS/School/CS6423/TokenSmith/src/ranking/ranker.py) so ranking operates only over the active retrievers and validates that active weights are present and positive.
- Preserved deterministic `rrf` and `linear` fusion modes in [src/ranking/ranker.py](/NAS/School/CS6423/TokenSmith/src/ranking/ranker.py).
- Updated [src/main.py](/NAS/School/CS6423/TokenSmith/src/main.py) and [src/api_server.py](/NAS/School/CS6423/TokenSmith/src/api_server.py) so retrieval gathers per-source score dictionaries first and then passes them into the ensemble ranker.
- Added hybrid retrieval coverage in [tests/test_hybrid_retrieval.py](/NAS/School/CS6423/TokenSmith/tests/test_hybrid_retrieval.py), [tests/test_api.py](/NAS/School/CS6423/TokenSmith/tests/test_api.py), and [tests/test_end_to_end.py](/NAS/School/CS6423/TokenSmith/tests/test_end_to_end.py).

## Implementation
The implemented ranking flow is:
1. Load the retrieval artifacts for the indexed corpus.
2. Build the enabled retrievers from configuration.
3. Collect raw score dictionaries from each active retrieval source independently.
4. Pass those per-source scores into `EnsembleRanker`.
5. Fuse the source scores with deterministic `rrf` or `linear` ranking.
6. Select the top-ranked chunk IDs and pass the resulting chunks into the existing answer-generation pipeline.

## Challenges
- The original branch state already had part of the retrieval stack, so the main difficulty was tightening and documenting hybrid behavior without breaking the current FAISS/BM25 pipeline.
- Retrieval correctness depends on keeping source scores logically separate before fusion, which required clearer active-source validation and more explicit configuration behavior.
- Index keyword retrieval depends on external textbook index artifacts, so source enablement and failure behavior needed to stay predictable.
- The proposed approach also expects graph-based retrieval, but this branch does not yet have a completed persisted graph artifact contract wired into retrieval-time loading.
- Deterministic testing is important here because hybrid ranking can otherwise become difficult to debug when candidate overlap and tie behavior vary across sources.

## Future Work
- Implement graph-based retrieval in [src/retriever.py](/NAS/School/CS6423/TokenSmith/src/retriever.py) using persisted graph neighborhood artifacts.
- Extend artifact loading so graph retrieval dependencies are validated explicitly instead of being handled indirectly.
- Add retrieval-debug logging that makes per-source contributions easier to inspect in chat mode and API requests.
- Add focused tests for graph retrieval, mixed overlapping/non-overlapping candidates, and disabled-source behavior.
- Run manual validation on representative queries to confirm FAISS, BM25, index keyword, and graph retrieval each affect the final ranking when enabled.
