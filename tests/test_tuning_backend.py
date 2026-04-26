import json
from pathlib import Path

import pytest

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.tuning.backend import (
    append_dataset_record,
    append_training_example,
    baseline_tuning_state,
    dataset_record_from_query_result,
    load_dataset_records,
    export_ranker_settings,
    list_available_tuning_indices,
    load_training_examples,
    load_tuning_manifest,
    load_tuning_session,
    load_tuning_state,
    run_tuning_query,
    save_exported_ranker_settings,
    save_tuning_manifest,
    save_tuning_state,
    TuningSession,
)
from src.tuning.contracts import ChunkFeedbackRecord, TrainingExample


pytestmark = pytest.mark.unit


class _MockRetriever:
    def __init__(self, name, scores):
        self.name = name
        self._scores = scores

    def get_scores(self, query, pool_size, chunks):
        return dict(self._scores)


def test_save_and_load_tuning_state_round_trip(tmp_path):
    state = baseline_tuning_state(RAGConfig())
    output_path = tmp_path / "learned_weights.json"

    save_tuning_state(output_path, state)
    restored = load_tuning_state(output_path)

    assert restored is not None
    assert restored.weights == state.weights
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert "weights" in payload


def test_append_and_load_training_examples(tmp_path):
    example = TrainingExample(
        query="What is ACID?",
        candidates=[
            ChunkFeedbackRecord(
                chunk_id=1,
                fused_score=0.9,
                raw_scores={"faiss": 0.6, "bm25": 0.3, "index_keywords": 0.0, "graph": 0.1},
            )
        ],
        selected_chunk_ids=[1],
    )
    examples_path = tmp_path / "demonstrations.jsonl"

    append_training_example(examples_path, example)
    loaded = load_training_examples(examples_path)

    assert len(loaded) == 1
    assert loaded[0].query == "What is ACID?"
    assert loaded[0].selected_chunk_ids == [1]


def test_run_tuning_query_returns_structured_top_k():
    model_state = baseline_tuning_state(RAGConfig())
    session = TuningSession(
        cfg=RAGConfig(top_k=10, ensemble_method="linear"),
        index_prefix="test_index",
        chunks=["chunk zero", "chunk one", "chunk two"],
        metadata=[{"page_numbers": [1]}, {"page_numbers": [2]}, {"page_numbers": [3]}],
        retrievers=[
            _MockRetriever("faiss", {0: 0.9, 1: 0.2}),
            _MockRetriever("bm25", {1: 0.8, 2: 0.5}),
            _MockRetriever("index_keywords", {2: 1.0}),
            _MockRetriever("graph", {0: 0.6, 2: 0.4}),
        ],
        ranker=EnsembleRanker(
            ensemble_method="linear",
            weights=model_state.weights,
        ),
        model_state=model_state,
        baseline_weights=dict(model_state.weights),
        active_weight_source="last_saved_learned",
        demonstrations_path=Path("demonstrations.jsonl"),
        dataset_path=Path("prompt_chunk_dataset.jsonl"),
        model_state_path=Path("weights.json"),
        manifest_path=Path("manifest.json"),
        export_path=Path("ranker_weights.export.yaml"),
        learned_weights_loaded=False,
        last_status_message="",
    )

    result = run_tuning_query(session, "test query", top_k=2)

    assert result.query == "test query"
    assert len(result.candidates) == 2
    assert all("chunk_text" in candidate.to_dict() for candidate in result.candidates)
    assert set(result.raw_scores_by_retriever) == {"faiss", "bm25", "index_keywords", "graph"}
    assert result.candidates[0].rank == 1
    assert "page_numbers" in result.candidates[0].metadata
    assert set(result.candidates[0].raw_scores) == {"faiss", "bm25", "index_keywords", "graph"}


def test_load_tuning_session_prefers_saved_weights(tmp_path, monkeypatch):
    cfg = RAGConfig()

    monkeypatch.setattr(cfg, "get_artifacts_directory", lambda: tmp_path)
    monkeypatch.setattr(cfg, "get_tuning_state_path", lambda index_prefix: tmp_path / "learned_weights.json")
    monkeypatch.setattr(cfg, "get_tuning_examples_path", lambda index_prefix: tmp_path / "demonstrations.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_dataset_path", lambda index_prefix: tmp_path / "prompt_chunk_dataset.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_manifest_path", lambda index_prefix: tmp_path / "manifest.json")
    monkeypatch.setattr(cfg, "get_tuning_export_path", lambda index_prefix: tmp_path / "ranker_weights.export.yaml")

    save_tuning_state(
        tmp_path / "learned_weights.json",
        baseline_tuning_state(RAGConfig(ranker_weights={"faiss": 0.7, "bm25": 0.1, "index_keywords": 0.1, "graph": 0.1})),
    )

    monkeypatch.setattr(
        "src.tuning.backend.load_artifacts",
        lambda **kwargs: ("faiss", "bm25", ["chunk"], ["source"], [{"page_numbers": [1]}], "graph"),
    )
    monkeypatch.setattr(
        "src.tuning.backend.build_retrievers",
        lambda cfg, **kwargs: [
            _MockRetriever("faiss", {0: 0.9}),
            _MockRetriever("bm25", {0: 0.1}),
            _MockRetriever("index_keywords", {0: 0.0}),
            _MockRetriever("graph", {0: 0.2}),
        ],
    )

    session = load_tuning_session(cfg, index_prefix="unit_index")

    assert session.model_state.weights["faiss"] == pytest.approx(0.7)
    assert session.demonstrations_path == tmp_path / "demonstrations.jsonl"
    assert session.dataset_path == tmp_path / "prompt_chunk_dataset.jsonl"
    assert session.model_state_path == tmp_path / "learned_weights.json"
    assert session.manifest_path == tmp_path / "manifest.json"
    assert session.export_path == tmp_path / "ranker_weights.export.yaml"
    assert session.learned_weights_loaded is True


def test_manifest_tracks_separate_demo_and_state_files(tmp_path):
    examples_path = tmp_path / "demonstrations.jsonl"
    dataset_path = tmp_path / "prompt_chunk_dataset.jsonl"
    state_path = tmp_path / "learned_weights.json"
    manifest_path = tmp_path / "manifest.json"

    save_tuning_state(state_path, baseline_tuning_state(RAGConfig()))
    append_training_example(
        examples_path,
        TrainingExample(
            query="What is ACID?",
            candidates=[
                ChunkFeedbackRecord(
                    chunk_id=1,
                    fused_score=0.9,
                    raw_scores={"faiss": 0.6, "bm25": 0.3, "index_keywords": 0.0, "graph": 0.1},
                )
            ],
            selected_chunk_ids=[1],
        ),
    )
    append_dataset_record(
        dataset_path,
        {
            "query": "What is ACID?",
            "selected_chunk_ids": [1],
            "selected_chunks": [{"chunk_id": 1, "chunk_text": "transactions chunk"}],
        },
    )
    save_tuning_manifest(
        manifest_path,
        demonstrations_path=examples_path,
        dataset_path=dataset_path,
        model_state_path=state_path,
        export_path=tmp_path / "ranker_weights.export.yaml",
        demonstration_count=1,
        dataset_count=1,
        baseline_weights={"faiss": 0.55, "bm25": 0.2, "index_keywords": 0.1, "graph": 0.15},
        active_weight_source="last_saved_learned",
        active_weights={"faiss": 0.6, "bm25": 0.2, "index_keywords": 0.1, "graph": 0.1},
    )

    manifest = load_tuning_manifest(manifest_path)

    assert manifest is not None
    assert manifest["paths"]["demonstrations"] == "demonstrations.jsonl"
    assert manifest["paths"]["prompt_chunk_dataset"] == "prompt_chunk_dataset.jsonl"
    assert manifest["paths"]["model_state"] == "learned_weights.json"
    assert manifest["paths"]["exported_ranker_settings"] == "ranker_weights.export.yaml"
    assert manifest["counts"]["demonstrations"] == 1
    assert manifest["counts"]["dataset_records"] == 1
    assert manifest_path.read_text().startswith("{\n")


def test_load_tuning_session_can_start_from_baseline(tmp_path, monkeypatch):
    cfg = RAGConfig(ranker_weights={"faiss": 0.4, "bm25": 0.3, "index_keywords": 0.2, "graph": 0.1})
    monkeypatch.setattr(cfg, "get_artifacts_directory", lambda: tmp_path)
    monkeypatch.setattr(cfg, "get_tuning_state_path", lambda index_prefix: tmp_path / "learned_weights.json")
    monkeypatch.setattr(cfg, "get_tuning_examples_path", lambda index_prefix: tmp_path / "demonstrations.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_dataset_path", lambda index_prefix: tmp_path / "prompt_chunk_dataset.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_manifest_path", lambda index_prefix: tmp_path / "manifest.json")
    monkeypatch.setattr(cfg, "get_tuning_export_path", lambda index_prefix: tmp_path / "ranker_weights.export.yaml")

    save_tuning_state(
        tmp_path / "learned_weights.json",
        baseline_tuning_state(RAGConfig(ranker_weights={"faiss": 0.7, "bm25": 0.1, "index_keywords": 0.1, "graph": 0.1})),
    )
    monkeypatch.setattr(
        "src.tuning.backend.load_artifacts",
        lambda **kwargs: ("faiss", "bm25", ["chunk"], ["source"], [{"page_numbers": [1]}], "graph"),
    )
    monkeypatch.setattr(
        "src.tuning.backend.build_retrievers",
        lambda cfg, **kwargs: [
            _MockRetriever("faiss", {0: 0.9}),
            _MockRetriever("bm25", {0: 0.1}),
            _MockRetriever("index_keywords", {0: 0.0}),
            _MockRetriever("graph", {0: 0.2}),
        ],
    )

    session = load_tuning_session(cfg, index_prefix="unit_index", starting_weight_source="baseline_config")

    assert session.active_weight_source == "baseline_config"
    assert session.model_state.weights["faiss"] == pytest.approx(0.4)
    assert "baseline config weights" in session.last_status_message.lower()


def test_export_ranker_settings_is_config_friendly(tmp_path):
    model_state = baseline_tuning_state(RAGConfig(ranker_weights={"faiss": 0.6, "bm25": 0.2, "index_keywords": 0.1, "graph": 0.1}))
    session = TuningSession(
        cfg=RAGConfig(),
        index_prefix="test_index",
        chunks=[],
        metadata=[],
        retrievers=[],
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

    payload = export_ranker_settings(session)
    export_path = save_exported_ranker_settings(session)

    assert "enabled_retrievers" in payload
    assert "ranker_weights" in payload
    assert payload["ranker_weights"]["faiss"] == pytest.approx(0.6)
    assert export_path.exists()
    assert "ranker_weights:" in export_path.read_text()


def test_training_example_creation_rejects_empty_and_duplicate_selection():
    from src.tuning.backend import TuningCandidate, TuningQueryResult, training_example_from_query_result

    result = TuningQueryResult(
        query="q",
        top_k=2,
        weights={"faiss": 0.25, "bm25": 0.25, "index_keywords": 0.25, "graph": 0.25},
        candidates=[
            TuningCandidate(1, 1, 0.9, {"faiss": 0.9, "bm25": 0.2, "index_keywords": 0.0, "graph": 0.1}, "a"),
            TuningCandidate(2, 2, 0.5, {"faiss": 0.2, "bm25": 0.8, "index_keywords": 0.1, "graph": 0.0}, "b"),
        ],
        raw_scores_by_retriever={},
    )

    with pytest.raises(ValueError, match="at least one preferred chunk"):
        training_example_from_query_result(result, selected_chunk_ids=[])

    with pytest.raises(ValueError, match="Duplicate selected"):
        training_example_from_query_result(result, selected_chunk_ids=[1, 1])


def test_load_tuning_session_falls_back_cleanly_when_learned_weights_missing(tmp_path, monkeypatch):
    cfg = RAGConfig(ranker_weights={"faiss": 0.5, "bm25": 0.2, "index_keywords": 0.2, "graph": 0.1})
    monkeypatch.setattr(cfg, "get_artifacts_directory", lambda: tmp_path)
    monkeypatch.setattr(cfg, "get_tuning_state_path", lambda index_prefix: tmp_path / "missing_learned_weights.json")
    monkeypatch.setattr(cfg, "get_tuning_examples_path", lambda index_prefix: tmp_path / "demonstrations.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_dataset_path", lambda index_prefix: tmp_path / "prompt_chunk_dataset.jsonl")
    monkeypatch.setattr(cfg, "get_tuning_manifest_path", lambda index_prefix: tmp_path / "manifest.json")
    monkeypatch.setattr(cfg, "get_tuning_export_path", lambda index_prefix: tmp_path / "ranker_weights.export.yaml")
    monkeypatch.setattr(
        "src.tuning.backend.load_artifacts",
        lambda **kwargs: ("faiss", "bm25", ["chunk"], ["source"], [{"page_numbers": [1]}], "graph"),
    )
    monkeypatch.setattr(
        "src.tuning.backend.build_retrievers",
        lambda cfg, **kwargs: [
            _MockRetriever("faiss", {0: 0.9}),
            _MockRetriever("bm25", {0: 0.1}),
            _MockRetriever("index_keywords", {0: 0.0}),
            _MockRetriever("graph", {0: 0.2}),
        ],
    )

    session = load_tuning_session(cfg, index_prefix="unit_index", starting_weight_source="last_saved_learned")

    assert session.learned_weights_loaded is False
    assert session.active_weight_source == "baseline_config"
    assert "falling back to baseline" in session.last_status_message.lower()


def test_tuning_state_save_load_has_no_drift_across_multiple_round_trips(tmp_path):
    state = baseline_tuning_state(
        RAGConfig(ranker_weights={"faiss": 0.61, "bm25": 0.14, "index_keywords": 0.1, "graph": 0.15})
    )
    output_path = tmp_path / "learned_weights.json"

    save_tuning_state(output_path, state)
    restored_once = load_tuning_state(output_path)
    save_tuning_state(output_path, restored_once)
    restored_twice = load_tuning_state(output_path)

    assert restored_once is not None
    assert restored_twice is not None
    assert restored_once.weights == restored_twice.weights
    assert restored_once.demonstrations_seen == restored_twice.demonstrations_seen


def test_dataset_record_saves_query_and_selected_chunk_text(tmp_path):
    result = run_tuning_query(
        TuningSession(
            cfg=RAGConfig(top_k=10, ensemble_method="linear"),
            index_prefix="test_index",
            chunks=["chunk zero", "chunk one", "chunk two"],
            metadata=[{"page_numbers": [1]}, {"page_numbers": [2]}, {"page_numbers": [3]}],
            retrievers=[
                _MockRetriever("faiss", {0: 0.9, 1: 0.2}),
                _MockRetriever("bm25", {1: 0.8, 2: 0.5}),
                _MockRetriever("index_keywords", {2: 1.0}),
                _MockRetriever("graph", {0: 0.6, 2: 0.4}),
            ],
            ranker=EnsembleRanker(
                ensemble_method="linear",
                weights=baseline_tuning_state(RAGConfig()).weights,
            ),
            model_state=baseline_tuning_state(RAGConfig()),
            baseline_weights=dict(baseline_tuning_state(RAGConfig()).weights),
            active_weight_source="last_saved_learned",
            demonstrations_path=tmp_path / "demonstrations.jsonl",
            dataset_path=tmp_path / "prompt_chunk_dataset.jsonl",
            model_state_path=tmp_path / "learned_weights.json",
            manifest_path=tmp_path / "manifest.json",
            export_path=tmp_path / "ranker_weights.export.yaml",
            learned_weights_loaded=False,
            last_status_message="",
        ),
        "acid transactions",
        top_k=2,
    )

    record = dataset_record_from_query_result(
        result,
        selected_chunk_ids=[result.candidates[0].chunk_id],
        session_id="streamlit",
    )

    assert record["query"] == "acid transactions"
    assert record["selected_chunk_ids"] == [result.candidates[0].chunk_id]
    assert len(record["selected_chunks"]) == 1
    assert record["selected_chunks"][0]["chunk_text"] == result.candidates[0].chunk_text
    assert len(record["all_candidates"]) == 2


def test_dataset_records_can_be_saved_and_loaded(tmp_path):
    dataset_path = tmp_path / "prompt_chunk_dataset.jsonl"
    record = {
        "query": "What is normalization?",
        "selected_chunk_ids": [3],
        "selected_chunks": [{"chunk_id": 3, "chunk_text": "normal forms"}],
    }

    append_dataset_record(dataset_path, record)
    restored = load_dataset_records(dataset_path)

    assert dataset_path.exists()
    assert len(restored) == 1
    assert restored[0]["query"] == "What is normalization?"
    assert restored[0]["selected_chunks"][0]["chunk_text"] == "normal forms"


def test_list_available_tuning_indices_reads_saved_chunk_sets(tmp_path, monkeypatch):
    cfg = RAGConfig()
    monkeypatch.setattr(cfg, "get_artifacts_directory", lambda: tmp_path)
    monkeypatch.setattr(cfg, "get_artifacts_root_directory", lambda: tmp_path)

    (tmp_path / "course_a_chunks.pkl").write_bytes(b"chunks-a")
    (tmp_path / "course_b_chunks.pkl").write_bytes(b"chunks-b")
    (tmp_path / "course_a_ingestion_metadata.json").write_text(
        json.dumps(
            {
                "source": {"source_file": "/tmp/course_a.md"},
                "chunks": {"chunk_count": 42},
                "chunking": {
                    "method_label": "structure_aware",
                    "method_config": "chunk_mode=sections+recursive",
                },
            }
        ),
        encoding="utf-8",
    )

    indices = list_available_tuning_indices(cfg)

    assert [entry["index_prefix"] for entry in indices] == ["course_a", "course_b"]
    assert indices[0]["chunk_count"] == 42
    assert indices[0]["chunking_method"] == "structure_aware"
    assert indices[0]["source_file"] == "/tmp/course_a.md"
    assert "chunk_count" not in indices[1]
