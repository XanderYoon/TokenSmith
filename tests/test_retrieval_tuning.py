import pathlib

import pytest

from src.config import RAGConfig
from src.ranking.tuning import (
    RetrievalBenchmark,
    generate_weight_grid,
    ndcg_at_k,
    recall_at_k,
    retrieval_objective,
    tune_retrieval_weights,
)


@pytest.mark.unit
def test_weight_grid_respects_simplex_and_minimum_weight():
    grid = generate_weight_grid(
        ["faiss", "bm25", "index_keywords"],
        step=0.5,
        minimum_weight=0.0,
    )

    assert {"faiss": 0.0, "bm25": 0.0, "index_keywords": 1.0} in grid
    assert all(abs(sum(weights.values()) - 1.0) < 1e-6 for weights in grid)


@pytest.mark.unit
def test_retrieval_objective_rewards_ideal_ordering():
    ideal = [10, 11, 12]
    better = [10, 11, 50]
    worse = [50, 10, 11]

    better_ndcg, better_recall, better_obj = retrieval_objective(better, ideal, 3)
    worse_ndcg, worse_recall, worse_obj = retrieval_objective(worse, ideal, 3)

    assert better_ndcg > worse_ndcg
    assert better_recall == worse_recall
    assert better_obj > worse_obj


@pytest.mark.unit
def test_ndcg_and_recall_edge_cases():
    assert ndcg_at_k([], [1, 2], 5) == 0.0
    assert recall_at_k([], [1, 2], 5) == 0.0
    assert ndcg_at_k([1, 2], [], 5) == 0.0
    assert recall_at_k([1, 2], [], 5) == 0.0


@pytest.mark.unit
def test_tune_retrieval_weights_prefers_source_matching_benchmark(monkeypatch):
    cfg = RAGConfig(
        top_k=2,
        num_candidates=3,
        ensemble_method="linear",
        ranker_weights={"faiss": 0.34, "bm25": 0.33, "index_keywords": 0.33},
        enabled_retrievers={"faiss": True, "bm25": True, "index_keywords": True},
    )
    benchmarks = [
        RetrievalBenchmark(
            benchmark_id="b1",
            question="q1",
            ideal_retrieved_chunks=[1, 2],
        )
    ]

    class StubRetriever:
        def __init__(self, name, scores):
            self.name = name
            self._scores = scores

        def get_scores(self, query, pool_size, chunks):
            return self._scores

    monkeypatch.setattr(
        "src.ranking.tuning.load_artifacts",
        lambda artifacts_dir, index_prefix: (object(), object(), ["c0", "c1", "c2"], [], []),
    )
    monkeypatch.setattr(
        "src.ranking.tuning.build_retrievers",
        lambda cfg, faiss_index, bm25_index: [
            StubRetriever("faiss", {0: 1.0}),
            StubRetriever("bm25", {1: 1.0, 2: 0.8}),
            StubRetriever("index_keywords", {1: 1.0}),
        ],
    )

    result = tune_retrieval_weights(
        cfg,
        benchmarks,
        artifacts_dir=pathlib.Path("index/sections"),
        index_prefix="textbook_index",
        step=0.5,
        minimum_weight=0.0,
    )

    assert result.weights["bm25"] >= result.weights["faiss"]
