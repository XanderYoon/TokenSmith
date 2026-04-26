from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.tuning.backend import TuningSession, baseline_tuning_state
from src.tuning.benchmarking import (
    _dataset_records_from_ragas_samples,
    _fit_contexts_to_generation_budget,
    load_benchmark_runs,
    run_automated_ragas_benchmark,
    run_ragas_benchmark,
)


pytestmark = pytest.mark.unit


class _MockRetriever:
    def __init__(self, name, scores):
        self.name = name
        self._scores = scores

    def get_scores(self, query, pool_size, chunks):
        return dict(self._scores.get(query, {}))


def test_run_ragas_benchmark_persists_metadata_and_scores(tmp_path, monkeypatch):
    cfg = RAGConfig(top_k=2, ensemble_method="linear")
    model_state = baseline_tuning_state(cfg)
    dataset_path = tmp_path / "prompt_chunk_dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-23T12:00:00+00:00",
                        "query": "What is ACID?",
                        "selected_chunk_ids": [1],
                        "selected_chunks": [{"chunk_id": 1, "chunk_text": "ACID chunk"}],
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-23T12:05:00+00:00",
                        "query": "What is normalization?",
                        "selected_chunk_ids": [0],
                        "selected_chunks": [{"chunk_id": 0, "chunk_text": "normal forms chunk"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ingestion_path = tmp_path / "unit_index_ingestion_metadata.json"
    ingestion_path.write_text(
        json.dumps(
            {
                "chunking": {
                    "method_label": "structure_aware",
                    "method_config": "chunk_mode=sections+recursive",
                },
                "source": {"source_file": "/tmp/textbook.md"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cfg, "get_tuning_benchmark_directory", lambda index_prefix: tmp_path / "benchmarks")
    monkeypatch.setattr(cfg, "get_ingestion_metadata_path", lambda index_prefix: ingestion_path)

    session = TuningSession(
        cfg=cfg,
        index_prefix="unit_index",
        chunks=["normal forms chunk", "ACID chunk"],
        metadata=[{}, {}],
        retrievers=[
            _MockRetriever("faiss", {"What is ACID?": {1: 0.9}, "What is normalization?": {0: 0.8}}),
            _MockRetriever("bm25", {"What is ACID?": {1: 0.7}, "What is normalization?": {0: 0.6}}),
            _MockRetriever("index_keywords", {"What is ACID?": {1: 0.3}, "What is normalization?": {0: 0.2}}),
            _MockRetriever("graph", {"What is ACID?": {1: 0.4}, "What is normalization?": {0: 0.1}}),
        ],
        ranker=EnsembleRanker(ensemble_method="linear", weights=model_state.weights),
        model_state=model_state,
        baseline_weights=dict(model_state.weights),
        active_weight_source="last_saved_learned",
        demonstrations_path=tmp_path / "demonstrations.jsonl",
        dataset_path=dataset_path,
        model_state_path=tmp_path / "learned_weights.json",
        manifest_path=tmp_path / "manifest.json",
        export_path=tmp_path / "ranker_weights.export.yaml",
        learned_weights_loaded=True,
        last_status_message="",
    )

    monkeypatch.setattr(
        "src.tuning.benchmarking._generate_answer_from_contexts",
        lambda session, **kwargs: f"answer for {kwargs['query']}",
    )
    monkeypatch.setattr(
        "src.tuning.benchmarking._evaluate_with_ragas",
        lambda rows, metric_names: {
            "status": "completed",
            "metric_names": list(metric_names),
            "aggregated_scores": {"faithfulness": 0.8, "answer_relevancy": 0.7},
            "per_prompt_scores": [
                {"faithfulness": 0.75, "answer_relevancy": 0.65},
                {"faithfulness": 0.85, "answer_relevancy": 0.75},
            ],
        },
    )

    summary = run_ragas_benchmark(
        session,
        top_k=2,
        system_prompt_mode="tutor",
        use_double_prompt=False,
        max_examples=2,
        ragas_metric_names=["faithfulness", "answer_relevancy"],
        benchmark_label="unit benchmark",
    )

    assert summary.output_path.exists()
    payload = json.loads(summary.output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset"]["dataset_records_total"] == 2
    assert payload["metadata"]["dataset"]["dataset_records_used"] == 2
    assert payload["metadata"]["chunk_set"]["chunking_method"] == "structure_aware"
    assert payload["metadata"]["parameters"]["weights"]["faiss"] == pytest.approx(model_state.weights["faiss"])
    assert payload["aggregated_metrics"]["prompt_count"] == 2
    assert payload["aggregated_metrics"]["ragas"]["aggregated_scores"]["faithfulness"] == pytest.approx(0.8)
    assert payload["prompts"][0]["ragas_scores"]["faithfulness"] == pytest.approx(0.75)


def test_load_benchmark_runs_returns_saved_runs(tmp_path):
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "20260423T120000Z_first.json").write_text(
        json.dumps({"run_id": "first", "output_path": "a.json"}),
        encoding="utf-8",
    )
    (benchmark_dir / "20260423T120100Z_second.json").write_text(
        json.dumps({"run_id": "second", "output_path": "b.json"}),
        encoding="utf-8",
    )

    runs = load_benchmark_runs(benchmark_dir)

    assert [run["run_id"] for run in runs] == ["second", "first"]


def test_dataset_records_from_ragas_samples_maps_reference_contexts_to_chunks(tmp_path):
    cfg = RAGConfig(top_k=2, ensemble_method="linear")
    model_state = baseline_tuning_state(cfg)

    session = TuningSession(
        cfg=cfg,
        index_prefix="unit_index",
        chunks=["normal forms chunk", "ACID chunk"],
        metadata=[{"section": "NF"}, {"section": "Transactions"}],
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
        learned_weights_loaded=True,
        last_status_message="",
    )

    records = _dataset_records_from_ragas_samples(
        session,
        [
            {
                "user_input": "What is ACID?",
                "reference": "Atomicity, consistency, isolation, durability.",
                "reference_contexts": ["ACID chunk"],
                "reference_context_ids": [1],
                "synthesizer_name": "single_hop",
                "query_style": "specific",
                "query_length": "short",
            }
        ],
    )

    assert len(records) == 1
    assert records[0]["query"] == "What is ACID?"
    assert records[0]["selected_chunk_ids"] == [1]
    assert records[0]["selected_chunks"][0]["chunk_text"] == "ACID chunk"
    assert records[0]["reference_answer"] == "Atomicity, consistency, isolation, durability."


def test_run_automated_ragas_benchmark_uses_generated_dataset(tmp_path, monkeypatch):
    cfg = RAGConfig(top_k=2, ensemble_method="linear")
    model_state = baseline_tuning_state(cfg)
    session = TuningSession(
        cfg=cfg,
        index_prefix="unit_index",
        chunks=["normal forms chunk", "ACID chunk"],
        metadata=[{}, {}],
        retrievers=[
            _MockRetriever("faiss", {"What is ACID?": {1: 0.9}}),
            _MockRetriever("bm25", {"What is ACID?": {1: 0.7}}),
            _MockRetriever("index_keywords", {"What is ACID?": {1: 0.3}}),
            _MockRetriever("graph", {"What is ACID?": {1: 0.4}}),
        ],
        ranker=EnsembleRanker(ensemble_method="linear", weights=model_state.weights),
        model_state=model_state,
        baseline_weights=dict(model_state.weights),
        active_weight_source="last_saved_learned",
        demonstrations_path=tmp_path / "demonstrations.jsonl",
        dataset_path=tmp_path / "manual_prompt_chunk_dataset.jsonl",
        model_state_path=tmp_path / "learned_weights.json",
        manifest_path=tmp_path / "manifest.json",
        export_path=tmp_path / "ranker_weights.export.yaml",
        learned_weights_loaded=True,
        last_status_message="",
    )

    generated_dataset_path = tmp_path / "automated_dataset.jsonl"
    generated_dataset_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-25T12:00:00+00:00",
                "query": "What is ACID?",
                "selected_chunk_ids": [1],
                "selected_chunks": [{"chunk_id": 1, "chunk_text": "ACID chunk"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.tuning.benchmarking.generate_automated_probe_dataset",
        lambda *args, **kwargs: generated_dataset_path,
    )
    monkeypatch.setattr(
        "src.tuning.benchmarking._generate_answer_from_contexts",
        lambda session, **kwargs: f"answer for {kwargs['query']}",
    )
    monkeypatch.setattr(
        "src.tuning.benchmarking._evaluate_with_ragas",
        lambda rows, metric_names: {
            "status": "completed",
            "metric_names": list(metric_names),
            "aggregated_scores": {"faithfulness": 0.9},
            "per_prompt_scores": [{"faithfulness": 0.9}],
        },
    )

    summary = run_automated_ragas_benchmark(
        session,
        testset_size=1,
        top_k=2,
        system_prompt_mode="tutor",
        use_double_prompt=False,
        ragas_metric_names=["faithfulness"],
        dataset_label="auto probes",
        benchmark_label="auto benchmark",
    )

    payload = json.loads(summary.output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset"]["dataset_path"] == str(generated_dataset_path)
    assert payload["aggregated_metrics"]["prompt_count"] == 1
    assert payload["aggregated_metrics"]["ragas"]["aggregated_scores"]["faithfulness"] == pytest.approx(0.9)


def test_fit_contexts_to_generation_budget_trims_long_contexts(tmp_path):
    cfg = RAGConfig(top_k=2, ensemble_method="linear", max_gen_tokens=1200)
    model_state = baseline_tuning_state(cfg)
    session = TuningSession(
        cfg=cfg,
        index_prefix="unit_index",
        chunks=["chunk"],
        metadata=[{}],
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
        learned_weights_loaded=True,
        last_status_message="",
    )

    fitted_contexts, max_generation_tokens = _fit_contexts_to_generation_budget(
        session,
        query="Explain latch-free updates.",
        contexts=["A" * 12000, "B" * 12000],
        system_prompt_mode="tutor",
    )

    assert fitted_contexts
    assert len("".join(fitted_contexts)) < 24000
    assert max_generation_tokens <= 1200
