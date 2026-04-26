from pathlib import Path

import pytest

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.tuning.backend import (
    TuningCandidate,
    TuningQueryResult,
    TuningSession,
    apply_training_example,
    baseline_tuning_state,
    run_tuning_query,
    training_example_from_query_result,
)
from src.tuning.model import example_feature_vectors, pairwise_weight_update


pytestmark = pytest.mark.unit


class _MockRetriever:
    def __init__(self, name, scores):
        self.name = name
        self._scores = scores

    def get_scores(self, query, pool_size, chunks):
        return dict(self._scores)


def test_example_feature_vectors_include_all_retrievers():
    result = TuningQueryResult(
        query="q",
        top_k=2,
        weights={"faiss": 0.25, "bm25": 0.25, "index_keywords": 0.25, "graph": 0.25},
        candidates=[
            TuningCandidate(1, 3, 0.9, {"faiss": 2.0, "bm25": 1.0, "index_keywords": 0.0, "graph": 0.5}, "a"),
            TuningCandidate(2, 7, 0.5, {"faiss": 1.0, "bm25": 0.5, "index_keywords": 0.3, "graph": 0.1}, "b"),
        ],
        raw_scores_by_retriever={},
    )
    example = training_example_from_query_result(result, selected_chunk_ids=[3])
    feature_vectors = example_feature_vectors(example)

    assert set(feature_vectors[3]) == {"faiss", "bm25", "index_keywords", "graph"}
    assert feature_vectors[3]["faiss"] > feature_vectors[7]["faiss"]


def test_pairwise_weight_update_keeps_weights_bounded_and_normalized():
    model_state = baseline_tuning_state(RAGConfig())
    result = TuningQueryResult(
        query="q",
        top_k=3,
        weights=model_state.weights,
        candidates=[
            TuningCandidate(1, 0, 0.7, {"faiss": 0.9, "bm25": 0.1, "index_keywords": 0.0, "graph": 0.0}, "a"),
            TuningCandidate(2, 1, 0.6, {"faiss": 0.2, "bm25": 0.7, "index_keywords": 0.0, "graph": 0.1}, "b"),
            TuningCandidate(3, 2, 0.4, {"faiss": 0.1, "bm25": 0.2, "index_keywords": 0.8, "graph": 0.4}, "c"),
        ],
        raw_scores_by_retriever={},
    )
    example = training_example_from_query_result(result, selected_chunk_ids=[2])

    summary = pairwise_weight_update(model_state, example, learning_rate=0.3)

    assert abs(sum(summary.updated_weights.values()) - 1.0) < 1e-6
    assert all(weight >= 0.0 for weight in summary.updated_weights.values())
    assert summary.pair_count > 0


def test_apply_training_example_updates_session_and_persists(tmp_path):
    cfg = RAGConfig(top_k=3, ensemble_method="linear")
    model_state = baseline_tuning_state(cfg)
    session = TuningSession(
        cfg=cfg,
        index_prefix="test",
        chunks=["chunk a", "chunk b", "chunk c"],
        metadata=[{}, {}, {}],
        retrievers=[
            _MockRetriever("faiss", {0: 0.2, 1: 0.1, 2: 0.9}),
            _MockRetriever("bm25", {0: 0.1, 1: 0.2, 2: 0.4}),
            _MockRetriever("index_keywords", {0: 0.0, 1: 0.3, 2: 0.7}),
            _MockRetriever("graph", {0: 0.0, 1: 0.1, 2: 0.8}),
        ],
        ranker=EnsembleRanker(ensemble_method="linear", weights=model_state.weights),
        model_state=model_state,
        baseline_weights=dict(model_state.weights),
        active_weight_source="last_saved_learned",
        demonstrations_path=tmp_path / "demonstrations.jsonl",
        dataset_path=tmp_path / "prompt_chunk_dataset.jsonl",
        model_state_path=tmp_path / "learned_weights.json",
        manifest_path=tmp_path / "manifest.json",
        export_path=tmp_path / "ranker_weights.export.yaml",
        learned_weights_loaded=False,
        last_status_message="",
    )

    result = run_tuning_query(session, "database transactions", top_k=3)
    example = training_example_from_query_result(result, selected_chunk_ids=[2])
    previous_weights = dict(session.model_state.weights)

    summary = apply_training_example(session, example, learning_rate=0.4, persist=True)

    assert session.model_state.demonstrations_seen == 1
    assert session.model_state.weights != previous_weights
    assert summary.updated_weights == session.model_state.weights
    assert session.demonstrations_path.exists()
    assert session.dataset_path.exists() is False
    assert session.model_state_path.exists()


def test_positive_selection_moves_preferred_chunk_upward():
    cfg = RAGConfig(top_k=2, ensemble_method="linear")
    initial_weights = {"faiss": 0.7, "bm25": 0.1, "index_keywords": 0.1, "graph": 0.1}
    session = TuningSession(
        cfg=cfg,
        index_prefix="test",
        chunks=["faiss-heavy chunk", "graph-heavy chunk"],
        metadata=[{}, {}],
        retrievers=[
            _MockRetriever("faiss", {0: 0.9, 1: 0.1}),
            _MockRetriever("bm25", {0: 0.0, 1: 0.2}),
            _MockRetriever("index_keywords", {0: 0.0, 1: 0.3}),
            _MockRetriever("graph", {0: 0.0, 1: 0.9}),
        ],
        ranker=EnsembleRanker(ensemble_method="linear", weights=initial_weights),
        model_state=baseline_tuning_state(RAGConfig(ranker_weights=initial_weights)),
        baseline_weights=baseline_tuning_state(RAGConfig(ranker_weights=initial_weights)).weights,
        active_weight_source="last_saved_learned",
        demonstrations_path=Path("demonstrations.jsonl"),
        dataset_path=Path("prompt_chunk_dataset.jsonl"),
        model_state_path=Path("learned_weights.json"),
        manifest_path=Path("manifest.json"),
        export_path=Path("ranker_weights.export.yaml"),
        learned_weights_loaded=False,
        last_status_message="",
    )

    before = run_tuning_query(session, "q", top_k=2)
    assert before.candidates[0].chunk_id == 0

    example = training_example_from_query_result(before, selected_chunk_ids=[1])
    apply_training_example(session, example, learning_rate=0.8, persist=False)
    after = run_tuning_query(session, "q", top_k=2)

    assert after.candidates[0].chunk_id == 1


def test_persisted_weights_can_be_reloaded_after_training(tmp_path, monkeypatch):
    cfg = RAGConfig(top_k=2, ensemble_method="linear")
    model_state = baseline_tuning_state(cfg)
    session = TuningSession(
        cfg=cfg,
        index_prefix="test",
        chunks=["chunk a", "chunk b"],
        metadata=[{}, {}],
        retrievers=[
            _MockRetriever("faiss", {0: 0.1, 1: 0.9}),
            _MockRetriever("bm25", {0: 0.2, 1: 0.6}),
            _MockRetriever("index_keywords", {0: 0.0, 1: 0.3}),
            _MockRetriever("graph", {0: 0.0, 1: 0.8}),
        ],
        ranker=EnsembleRanker(ensemble_method="linear", weights=model_state.weights),
        model_state=model_state,
        baseline_weights=dict(model_state.weights),
        active_weight_source="last_saved_learned",
        demonstrations_path=tmp_path / "demonstrations.jsonl",
        dataset_path=tmp_path / "prompt_chunk_dataset.jsonl",
        model_state_path=tmp_path / "learned_weights.json",
        manifest_path=tmp_path / "manifest.json",
        export_path=tmp_path / "ranker_weights.export.yaml",
        learned_weights_loaded=False,
        last_status_message="",
    )
    result = run_tuning_query(session, "q", top_k=2)
    example = training_example_from_query_result(result, selected_chunk_ids=[1])
    apply_training_example(
        session,
        example,
        learning_rate=0.5,
        dataset_record={"query": "q", "selected_chunk_ids": [1], "selected_chunks": []},
        persist=True,
    )

    monkeypatch.setattr(cfg, "get_artifacts_directory", lambda: tmp_path)
    monkeypatch.setattr(cfg, "get_tuning_state_path", lambda index_prefix: tmp_path / "learned_weights.json")
    monkeypatch.setattr(cfg, "get_tuning_examples_path", lambda index_prefix: tmp_path / "demonstrations.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_dataset_path", lambda index_prefix: tmp_path / "prompt_chunk_dataset.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_manifest_path", lambda index_prefix: tmp_path / "manifest.json")
    monkeypatch.setattr(cfg, "get_tuning_export_path", lambda index_prefix: tmp_path / "ranker_weights.export.yaml")
    monkeypatch.setattr(
        "src.tuning.backend.load_artifacts",
        lambda **kwargs: ("faiss", "bm25", ["chunk a", "chunk b"], ["source"], [{}, {}], "graph"),
    )
    monkeypatch.setattr(
        "src.tuning.backend.build_retrievers",
        lambda cfg, **kwargs: session.retrievers,
    )

    from src.tuning.backend import load_tuning_session

    restored = load_tuning_session(cfg, index_prefix="test", starting_weight_source="last_saved_learned")

    assert restored.learned_weights_loaded is True
    assert restored.model_state.weights == session.model_state.weights
    assert restored.dataset_path == tmp_path / "prompt_chunk_dataset.jsonl"
    assert session.dataset_path.exists()
