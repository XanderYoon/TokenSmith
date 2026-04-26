import pytest

from src.ranking.ranker import EnsembleRanker


pytestmark = pytest.mark.unit


def test_ensemble_ranker_accepts_graph_source():
    ranker = EnsembleRanker(
        ensemble_method="rrf",
        weights={"faiss": 0.4, "bm25": 0.3, "index_keywords": 0.1, "graph": 0.2},
        rrf_k=60,
    )

    ordered_ids, scores = ranker.rank(
        {
            "faiss": {0: 0.9, 1: 0.6},
            "bm25": {1: 0.9, 2: 0.3},
            "index_keywords": {2: 1.0},
            "graph": {0: 0.7, 2: 0.8},
        }
    )

    assert ordered_ids
    assert len(ordered_ids) == len(scores)
