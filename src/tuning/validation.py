from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from src.ranking.ranker import EnsembleRanker
from src.tuning.backend import (
    TuningQueryResult,
    TuningSession,
    run_tuning_query,
)
from src.tuning.contracts import TUNING_RETRIEVERS, TuningModelState


@dataclass
class QueryValidationResult:
    query: str
    baseline_top_ids: List[int]
    learned_top_ids: List[int]
    top1_changed: bool
    overlap_at_k: float


@dataclass
class ValidationSummary:
    total_queries: int
    top1_changed_queries: int
    avg_overlap_at_k: float
    baseline_weights: Dict[str, float]
    learned_weights: Dict[str, float]
    per_query: List[QueryValidationResult]


def _normalized_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {name: max(0.0, float(weights.get(name, 0.0))) for name in TUNING_RETRIEVERS}
    total = sum(normalized.values()) or 1.0
    return {name: value / total for name, value in normalized.items()}


def session_with_weights(
    session: TuningSession,
    weights: Dict[str, float],
    *,
    weight_source: str,
) -> TuningSession:
    normalized = _normalized_weights(weights)
    return TuningSession(
        cfg=session.cfg,
        index_prefix=session.index_prefix,
        chunks=session.chunks,
        metadata=session.metadata,
        retrievers=session.retrievers,
        ranker=EnsembleRanker(
            ensemble_method="linear",
            weights=normalized,
            rrf_k=int(session.cfg.rrf_k),
        ),
        model_state=TuningModelState(
            weights=normalized,
            demonstrations_seen=session.model_state.demonstrations_seen,
            updated_at_utc=session.model_state.updated_at_utc,
            schema_version=session.model_state.schema_version,
            model_type=session.model_state.model_type,
            update_rule=session.model_state.update_rule,
        ),
        baseline_weights=dict(session.baseline_weights),
        active_weight_source=weight_source,
        demonstrations_path=session.demonstrations_path,
        dataset_path=session.dataset_path,
        model_state_path=session.model_state_path,
        manifest_path=session.manifest_path,
        export_path=session.export_path,
        learned_weights_loaded=session.learned_weights_loaded,
        last_status_message=session.last_status_message,
    )


def compare_query_results(
    baseline_result: TuningQueryResult,
    learned_result: TuningQueryResult,
) -> QueryValidationResult:
    baseline_top_ids = [candidate.chunk_id for candidate in baseline_result.candidates]
    learned_top_ids = [candidate.chunk_id for candidate in learned_result.candidates]
    overlap = len(set(baseline_top_ids) & set(learned_top_ids)) / max(len(learned_top_ids), 1)
    return QueryValidationResult(
        query=learned_result.query,
        baseline_top_ids=baseline_top_ids,
        learned_top_ids=learned_top_ids,
        top1_changed=(baseline_top_ids[:1] != learned_top_ids[:1]),
        overlap_at_k=float(overlap),
    )


def validate_queries_against_baseline(
    session: TuningSession,
    queries: Sequence[str],
    *,
    top_k: int,
) -> ValidationSummary:
    baseline_session = session_with_weights(
        session,
        session.baseline_weights,
        weight_source="baseline_config",
    )
    learned_session = session_with_weights(
        session,
        session.model_state.weights,
        weight_source=session.active_weight_source,
    )

    per_query: List[QueryValidationResult] = []
    for query in queries:
        stripped = query.strip()
        if not stripped:
            continue
        baseline_result = run_tuning_query(baseline_session, stripped, top_k=top_k)
        learned_result = run_tuning_query(learned_session, stripped, top_k=top_k)
        per_query.append(compare_query_results(baseline_result, learned_result))

    if not per_query:
        return ValidationSummary(
            total_queries=0,
            top1_changed_queries=0,
            avg_overlap_at_k=0.0,
            baseline_weights=dict(session.baseline_weights),
            learned_weights=dict(session.model_state.weights),
            per_query=[],
        )

    return ValidationSummary(
        total_queries=len(per_query),
        top1_changed_queries=sum(1 for result in per_query if result.top1_changed),
        avg_overlap_at_k=float(mean(result.overlap_at_k for result in per_query)),
        baseline_weights=dict(session.baseline_weights),
        learned_weights=dict(session.model_state.weights),
        per_query=per_query,
    )
