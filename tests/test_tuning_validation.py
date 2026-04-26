from pathlib import Path

import pytest

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.tuning.backend import TuningSession, baseline_tuning_state
from src.tuning.validation import validate_queries_against_baseline


pytestmark = pytest.mark.unit


class _MockRetriever:
    def __init__(self, name, score_by_query):
        self.name = name
        self._score_by_query = score_by_query

    def get_scores(self, query, pool_size, chunks):
        return dict(self._score_by_query.get(query, {}))


def _build_session(initial_weights):
    model_state = baseline_tuning_state(RAGConfig(ranker_weights=initial_weights))
    return TuningSession(
        cfg=RAGConfig(top_k=2, ensemble_method="linear", ranker_weights=initial_weights),
        index_prefix="test",
        chunks=["chunk-a", "chunk-b", "chunk-c"],
        metadata=[{}, {}, {}],
        retrievers=[
            _MockRetriever(
                "faiss",
                {
                    "q1": {0: 0.9, 1: 0.1},
                    "q2": {1: 0.8, 2: 0.2},
                },
            ),
            _MockRetriever(
                "bm25",
                {
                    "q1": {1: 0.9, 0: 0.2},
                    "q2": {2: 0.8, 1: 0.3},
                },
            ),
            _MockRetriever(
                "index_keywords",
                {
                    "q1": {1: 0.2},
                    "q2": {2: 0.4},
                },
            ),
            _MockRetriever(
                "graph",
                {
                    "q1": {1: 0.7},
                    "q2": {2: 0.6},
                },
            ),
        ],
        ranker=EnsembleRanker(ensemble_method="linear", weights=model_state.weights),
        model_state=model_state,
        baseline_weights={"faiss": 0.8, "bm25": 0.1, "index_keywords": 0.05, "graph": 0.05},
        active_weight_source="last_saved_learned",
        demonstrations_path=Path("demonstrations.jsonl"),
        dataset_path=Path("prompt_chunk_dataset.jsonl"),
        model_state_path=Path("learned_weights.json"),
        manifest_path=Path("manifest.json"),
        export_path=Path("ranker_weights.export.yaml"),
        learned_weights_loaded=True,
        last_status_message="",
    )


def test_validation_detects_ranking_change_against_baseline():
    session = _build_session({"faiss": 0.1, "bm25": 0.3, "index_keywords": 0.1, "graph": 0.5})
    summary = validate_queries_against_baseline(session, ["q1", "q2"], top_k=2)

    assert summary.total_queries == 2
    assert summary.top1_changed_queries >= 1
    assert 0.0 <= summary.avg_overlap_at_k <= 1.0


def test_validation_handles_empty_query_list():
    session = _build_session({"faiss": 0.25, "bm25": 0.25, "index_keywords": 0.25, "graph": 0.25})
    summary = validate_queries_against_baseline(session, [], top_k=2)

    assert summary.total_queries == 0
    assert summary.per_query == []
