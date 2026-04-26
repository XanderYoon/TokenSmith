from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import yaml

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import build_retrievers, load_artifacts
from src.tuning.contracts import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_TUNING_TOP_K,
    TUNING_RETRIEVERS,
    TUNING_WEIGHT_SOURCES,
    ChunkFeedbackRecord,
    TrainingExample,
    TuningModelState,
    utc_now_iso,
)
from src.tuning.model import WeightUpdateSummary, pairwise_weight_update


@dataclass
class TuningCandidate:
    rank: int
    chunk_id: int
    fused_score: float
    raw_scores: Dict[str, float]
    chunk_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "chunk_id": int(self.chunk_id),
            "fused_score": float(self.fused_score),
            "raw_scores": {
                name: float(self.raw_scores.get(name, 0.0))
                for name in TUNING_RETRIEVERS
            },
            "chunk_text": self.chunk_text,
            "metadata": dict(self.metadata),
        }


@dataclass
class TuningQueryResult:
    query: str
    top_k: int
    weights: Dict[str, float]
    candidates: List[TuningCandidate]
    raw_scores_by_retriever: Dict[str, Dict[int, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k": int(self.top_k),
            "weights": {
                name: float(self.weights.get(name, 0.0))
                for name in TUNING_RETRIEVERS
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "raw_scores_by_retriever": {
                name: {int(chunk_id): float(score) for chunk_id, score in scores.items()}
                for name, scores in self.raw_scores_by_retriever.items()
            },
        }


@dataclass
class TuningSession:
    cfg: RAGConfig
    index_prefix: str
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    retrievers: List[Any]
    ranker: EnsembleRanker
    model_state: TuningModelState
    baseline_weights: Dict[str, float]
    active_weight_source: str
    demonstrations_path: Path
    dataset_path: Path
    model_state_path: Path
    manifest_path: Path
    export_path: Path
    learned_weights_loaded: bool
    last_status_message: str = ""


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {
        name: max(0.0, float(weights.get(name, 0.0)))
        for name in TUNING_RETRIEVERS
    }
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Tuning weights must include at least one positive value.")
    return {name: value / total for name, value in normalized.items()}


def baseline_tuning_state(cfg: RAGConfig) -> TuningModelState:
    return TuningModelState(weights=_normalize_weights(cfg.ranker_weights))


def load_tuning_state(path: Path) -> Optional[TuningModelState]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TuningModelState.from_dict(payload)


def save_tuning_state(path: Path, state: TuningModelState) -> None:
    _atomic_write_json(path, state.to_dict())


def _build_manifest_payload(
    *,
    demonstrations_path: Path,
    dataset_path: Path,
    model_state_path: Path,
    export_path: Path,
    demonstration_count: int,
    dataset_count: int,
    baseline_weights: Dict[str, float],
    active_weight_source: str,
    active_weights: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "storage_format": {
            "demonstrations": "jsonl",
            "prompt_chunk_dataset": "jsonl",
            "model_state": "json",
            "exported_ranker_settings": "yaml",
        },
        "paths": {
            "demonstrations": demonstrations_path.name,
            "prompt_chunk_dataset": dataset_path.name,
            "model_state": model_state_path.name,
            "exported_ranker_settings": export_path.name,
        },
        "counts": {
            "demonstrations": int(demonstration_count),
            "dataset_records": int(dataset_count),
        },
        "weight_sources": {
            "baseline_config": dict(baseline_weights),
            "active_source": active_weight_source,
            "active_weights": dict(active_weights),
        },
        "updated_at_utc": utc_now_iso(),
    }


def save_tuning_manifest(
    path: Path,
    *,
    demonstrations_path: Path,
    dataset_path: Path,
    model_state_path: Path,
    export_path: Path,
    demonstration_count: int,
    dataset_count: int,
    baseline_weights: Dict[str, float],
    active_weight_source: str,
    active_weights: Dict[str, float],
) -> None:
    _atomic_write_json(
        path,
        _build_manifest_payload(
            demonstrations_path=demonstrations_path,
            dataset_path=dataset_path,
            model_state_path=model_state_path,
            export_path=export_path,
            demonstration_count=demonstration_count,
            dataset_count=dataset_count,
            baseline_weights=baseline_weights,
            active_weight_source=active_weight_source,
            active_weights=active_weights,
        ),
    )


def load_tuning_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_training_example(path: Path, example: TrainingExample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: List[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            existing_lines = [line.rstrip("\n") for line in handle if line.strip()]
    existing_lines.append(json.dumps(example.to_dict(), sort_keys=True))
    _atomic_write_text(path, "\n".join(existing_lines) + "\n")


def append_dataset_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: List[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            existing_lines = [line.rstrip("\n") for line in handle if line.strip()]
    existing_lines.append(json.dumps(record, sort_keys=True))
    _atomic_write_text(path, "\n".join(existing_lines) + "\n")


def load_training_examples(path: Path) -> List[TrainingExample]:
    if not path.exists():
        return []
    examples: List[TrainingExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(TrainingExample.from_dict(json.loads(stripped)))
    return examples


def load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(dict(json.loads(stripped)))
    return records


def session_stats(session: TuningSession) -> Dict[str, Any]:
    dataset_records = load_dataset_records(session.dataset_path)
    return {
        "index_prefix": session.index_prefix,
        "artifact_dir": str(session.demonstrations_path.parent.parent),
        "demonstrations_path": str(session.demonstrations_path),
        "dataset_path": str(session.dataset_path),
        "model_state_path": str(session.model_state_path),
        "manifest_path": str(session.manifest_path),
        "export_path": str(session.export_path),
        "num_chunks": len(session.chunks),
        "demonstrations_seen": int(session.model_state.demonstrations_seen),
        "dataset_records": len(dataset_records),
        "baseline_weights": dict(session.baseline_weights),
        "active_weight_source": session.active_weight_source,
        "learned_weights_loaded": bool(session.learned_weights_loaded),
        "weights": dict(session.model_state.weights),
        "updated_at_utc": session.model_state.updated_at_utc,
        "last_status_message": session.last_status_message,
    }


def list_available_tuning_indices(cfg: RAGConfig) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for artifacts_dir in cfg.discover_artifact_directories():
        for chunks_path in sorted(artifacts_dir.glob("*_chunks.pkl")):
            index_prefix = chunks_path.name[: -len("_chunks.pkl")]
            entry: Dict[str, Any] = {
                "index_prefix": index_prefix,
                "chunks_path": str(chunks_path),
                "artifacts_dir": str(artifacts_dir),
            }

            ingestion_metadata_path = cfg.get_ingestion_metadata_path(index_prefix, artifacts_dir=artifacts_dir)
            if ingestion_metadata_path.exists():
                try:
                    with open(ingestion_metadata_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    source_info = payload.get("source", {})
                    chunk_info = payload.get("chunks", {})
                    chunking_info = payload.get("chunking", {})
                    entry.update(
                        {
                            "source_file": source_info.get("source_file"),
                            "chunk_count": chunk_info.get("chunk_count"),
                            "chunking_method": chunking_info.get("method_label"),
                            "chunking_config": chunking_info.get("method_config"),
                        }
                    )
                except (OSError, json.JSONDecodeError):
                    entry["metadata_error"] = str(ingestion_metadata_path)

            results.append(entry)

    return results


def load_tuning_session(
    cfg: RAGConfig,
    *,
    index_prefix: str,
    artifacts_dir: os.PathLike | None = None,
    starting_weight_source: str = "last_saved_learned",
) -> TuningSession:
    if starting_weight_source not in TUNING_WEIGHT_SOURCES:
        raise ValueError(f"Unsupported starting_weight_source: {starting_weight_source}")

    explicit_artifacts_dir = artifacts_dir
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else cfg.resolve_artifacts_directory(index_prefix)
    if explicit_artifacts_dir is None:
        model_state_path = cfg.get_tuning_state_path(index_prefix)
        demonstrations_path = cfg.get_tuning_examples_path(index_prefix)
        dataset_path = cfg.get_tuning_dataset_path(index_prefix)
        manifest_path = cfg.get_tuning_manifest_path(index_prefix)
        export_path = cfg.get_tuning_export_path(index_prefix)
    else:
        model_state_path = cfg.get_tuning_state_path(index_prefix, artifacts_dir=artifacts_dir)
        demonstrations_path = cfg.get_tuning_examples_path(index_prefix, artifacts_dir=artifacts_dir)
        dataset_path = cfg.get_tuning_dataset_path(index_prefix, artifacts_dir=artifacts_dir)
        manifest_path = cfg.get_tuning_manifest_path(index_prefix, artifacts_dir=artifacts_dir)
        export_path = cfg.get_tuning_export_path(index_prefix, artifacts_dir=artifacts_dir)

    baseline_state = baseline_tuning_state(cfg)
    saved_state = load_tuning_state(model_state_path)
    learned_weights_loaded = saved_state is not None
    if starting_weight_source == "baseline_config" or saved_state is None:
        model_state = TuningModelState(
            weights=dict(baseline_state.weights),
            demonstrations_seen=saved_state.demonstrations_seen if saved_state else 0,
        )
        active_weight_source = "baseline_config"
    else:
        model_state = saved_state
        active_weight_source = "last_saved_learned"

    learned_weights = _normalize_weights(model_state.weights)
    demonstration_count = len(load_training_examples(demonstrations_path))
    dataset_count = len(load_dataset_records(dataset_path))
    save_tuning_manifest(
        manifest_path,
        demonstrations_path=demonstrations_path,
        dataset_path=dataset_path,
        model_state_path=model_state_path,
        export_path=export_path,
        demonstration_count=demonstration_count,
        dataset_count=dataset_count,
        baseline_weights=baseline_state.weights,
        active_weight_source=active_weight_source,
        active_weights=learned_weights,
    )

    tuning_cfg = RAGConfig(**cfg.get_config_state())
    tuning_cfg.enabled_retrievers = list(TUNING_RETRIEVERS)
    tuning_cfg.ranker_weights = dict(learned_weights)

    try:
        faiss_index, bm25_index, chunks, _sources, metadata, graph_store = load_artifacts(
            artifacts_dir=artifacts_dir,
            index_prefix=index_prefix,
            cfg=tuning_cfg,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing retrieval artifacts for index_prefix '{index_prefix}' in '{artifacts_dir}'. "
            f"Run indexing first or verify the selected index prefix and artifact directory. "
            f"Original error: {exc}"
        ) from exc
    retrievers = build_retrievers(
        tuning_cfg,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        graph_store=graph_store,
    )
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights=learned_weights,
        rrf_k=int(cfg.rrf_k),
    )
    if starting_weight_source == "last_saved_learned" and not learned_weights_loaded:
        last_status_message = (
            "No saved learned weights were found. Falling back to baseline config weights."
        )
    elif starting_weight_source == "baseline_config":
        last_status_message = "Session started from baseline config weights."
    else:
        last_status_message = "Session started from saved learned weights."

    return TuningSession(
        cfg=tuning_cfg,
        index_prefix=index_prefix,
        chunks=chunks,
        metadata=metadata,
        retrievers=retrievers,
        ranker=ranker,
        model_state=TuningModelState(
            weights=learned_weights,
            demonstrations_seen=max(model_state.demonstrations_seen, demonstration_count),
            updated_at_utc=model_state.updated_at_utc,
            schema_version=model_state.schema_version,
            model_type=model_state.model_type,
            update_rule=model_state.update_rule,
        ),
        baseline_weights=dict(baseline_state.weights),
        active_weight_source=active_weight_source,
        demonstrations_path=demonstrations_path,
        dataset_path=dataset_path,
        model_state_path=model_state_path,
        manifest_path=manifest_path,
        export_path=export_path,
        learned_weights_loaded=learned_weights_loaded,
        last_status_message=last_status_message,
    )


def export_ranker_settings(session: TuningSession) -> Dict[str, Any]:
    return session.cfg.export_ranker_settings(
        weights=session.model_state.weights,
        enabled_retrievers=list(TUNING_RETRIEVERS),
    )


def save_exported_ranker_settings(session: TuningSession) -> Path:
    payload = export_ranker_settings(session)
    _atomic_write_text(session.export_path, yaml.safe_dump(payload, sort_keys=False))
    return session.export_path


def run_tuning_query(
    session: TuningSession,
    query: str,
    *,
    top_k: int = DEFAULT_TUNING_TOP_K,
) -> TuningQueryResult:
    pool_n = max(session.cfg.num_candidates, top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in session.retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, session.chunks)

    ordered_ids, ordered_scores = session.ranker.rank(raw_scores)
    selected_ids = ordered_ids[:top_k]
    selected_scores = ordered_scores[:top_k]

    candidates: List[TuningCandidate] = []
    for rank, (chunk_id, fused_score) in enumerate(zip(selected_ids, selected_scores), start=1):
        metadata = {}
        if 0 <= chunk_id < len(session.metadata):
            metadata = dict(session.metadata[chunk_id])
        candidates.append(
            TuningCandidate(
                rank=rank,
                chunk_id=int(chunk_id),
                fused_score=float(fused_score),
                raw_scores={
                    name: float(raw_scores.get(name, {}).get(chunk_id, 0.0))
                    for name in TUNING_RETRIEVERS
                },
                chunk_text=session.chunks[chunk_id],
                metadata=metadata,
            )
        )

    return TuningQueryResult(
        query=query,
        top_k=top_k,
        weights=dict(session.model_state.weights),
        candidates=candidates,
        raw_scores_by_retriever=raw_scores,
    )


def training_example_from_query_result(
    result: TuningQueryResult,
    *,
    selected_chunk_ids: List[int],
    rejected_chunk_ids: Optional[List[int]] = None,
    session_id: str = "default",
) -> TrainingExample:
    if not selected_chunk_ids:
        raise ValueError("Select at least one preferred chunk before creating a training example.")
    if len(selected_chunk_ids) != len(set(int(chunk_id) for chunk_id in selected_chunk_ids)):
        raise ValueError("Duplicate selected chunk IDs are not allowed.")
    if rejected_chunk_ids and len(rejected_chunk_ids) != len(set(int(chunk_id) for chunk_id in rejected_chunk_ids)):
        raise ValueError("Duplicate rejected chunk IDs are not allowed.")

    selected_set = {int(chunk_id) for chunk_id in selected_chunk_ids}
    rejected_set = {int(chunk_id) for chunk_id in (rejected_chunk_ids or [])}

    candidates = [
        ChunkFeedbackRecord(
            chunk_id=candidate.chunk_id,
            fused_score=candidate.fused_score,
            raw_scores=dict(candidate.raw_scores),
            selected=candidate.chunk_id in selected_set,
            rejected=candidate.chunk_id in rejected_set,
            chunk_text_preview=candidate.chunk_text[:300],
            metadata=dict(candidate.metadata),
        )
        for candidate in result.candidates
    ]
    return TrainingExample(
        query=result.query,
        candidates=candidates,
        selected_chunk_ids=sorted(selected_set),
        rejected_chunk_ids=sorted(rejected_set),
        session_id=session_id,
        top_k=result.top_k,
    )


def dataset_record_from_query_result(
    result: TuningQueryResult,
    *,
    selected_chunk_ids: List[int],
    rejected_chunk_ids: Optional[List[int]] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    if not selected_chunk_ids:
        raise ValueError("Select at least one preferred chunk before creating a dataset record.")
    if len(selected_chunk_ids) != len(set(int(chunk_id) for chunk_id in selected_chunk_ids)):
        raise ValueError("Duplicate selected chunk IDs are not allowed.")
    if rejected_chunk_ids and len(rejected_chunk_ids) != len(set(int(chunk_id) for chunk_id in rejected_chunk_ids)):
        raise ValueError("Duplicate rejected chunk IDs are not allowed.")

    selected_set = {int(chunk_id) for chunk_id in selected_chunk_ids}
    rejected_set = {int(chunk_id) for chunk_id in (rejected_chunk_ids or [])}
    candidate_by_id = {candidate.chunk_id: candidate for candidate in result.candidates}

    return {
        "schema_version": 1,
        "timestamp_utc": utc_now_iso(),
        "session_id": session_id,
        "query": result.query,
        "top_k": int(result.top_k),
        "weights": {
            name: float(result.weights.get(name, 0.0))
            for name in TUNING_RETRIEVERS
        },
        "selected_chunk_ids": sorted(selected_set),
        "rejected_chunk_ids": sorted(rejected_set),
        "selected_chunks": [
            candidate_by_id[chunk_id].to_dict()
            for chunk_id in sorted(selected_set)
            if chunk_id in candidate_by_id
        ],
        "all_candidates": [candidate.to_dict() for candidate in result.candidates],
    }


def apply_training_example(
    session: TuningSession,
    example: TrainingExample,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dataset_record: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> WeightUpdateSummary:
    update_summary = pairwise_weight_update(
        session.model_state,
        example,
        learning_rate=learning_rate,
    )
    session.model_state.weights = dict(update_summary.updated_weights)
    session.model_state.demonstrations_seen += 1
    session.model_state.updated_at_utc = utc_now_iso()
    session.cfg.ranker_weights = dict(update_summary.updated_weights)
    session.active_weight_source = "last_saved_learned"
    session.ranker = EnsembleRanker(
        ensemble_method="linear",
        weights=dict(update_summary.updated_weights),
        rrf_k=int(session.cfg.rrf_k),
    )
    if persist:
        append_training_example(session.demonstrations_path, example)
        if dataset_record is not None:
            append_dataset_record(session.dataset_path, dataset_record)
        save_tuning_state(session.model_state_path, session.model_state)
        save_exported_ranker_settings(session)
        save_tuning_manifest(
            session.manifest_path,
            demonstrations_path=session.demonstrations_path,
            dataset_path=session.dataset_path,
            model_state_path=session.model_state_path,
            export_path=session.export_path,
            demonstration_count=session.model_state.demonstrations_seen,
            dataset_count=len(load_dataset_records(session.dataset_path)),
            baseline_weights=session.baseline_weights,
            active_weight_source=session.active_weight_source,
            active_weights=session.model_state.weights,
        )
        session.last_status_message = (
            f"Saved feedback example #{session.model_state.demonstrations_seen}, updated learned weights, "
            f"and appended the prompt/chunk dataset."
        )
    return update_summary


def reset_session_to_baseline(
    session: TuningSession,
    *,
    persist: bool = True,
) -> TuningModelState:
    session.model_state.weights = dict(session.baseline_weights)
    session.model_state.updated_at_utc = utc_now_iso()
    session.cfg.ranker_weights = dict(session.baseline_weights)
    session.active_weight_source = "baseline_config"
    session.ranker = EnsembleRanker(
        ensemble_method="linear",
        weights=dict(session.baseline_weights),
        rrf_k=int(session.cfg.rrf_k),
    )
    if persist:
        save_tuning_state(session.model_state_path, session.model_state)
        save_exported_ranker_settings(session)
        save_tuning_manifest(
            session.manifest_path,
            demonstrations_path=session.demonstrations_path,
            dataset_path=session.dataset_path,
            model_state_path=session.model_state_path,
            export_path=session.export_path,
            demonstration_count=session.model_state.demonstrations_seen,
            dataset_count=len(load_dataset_records(session.dataset_path)),
            baseline_weights=session.baseline_weights,
            active_weight_source=session.active_weight_source,
            active_weights=session.model_state.weights,
        )
        session.last_status_message = "Reset learned weights to baseline config weights and saved the change."
    return session.model_state
