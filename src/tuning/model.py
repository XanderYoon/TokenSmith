from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.tuning.contracts import (
    DEFAULT_LEARNING_RATE,
    TUNING_RETRIEVERS,
    TrainingExample,
    TuningModelState,
)


@dataclass
class WeightUpdateSummary:
    previous_weights: Dict[str, float]
    updated_weights: Dict[str, float]
    pair_count: int
    applied_delta: Dict[str, float]


def normalize_feature_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not rows:
        return []

    normalized_rows: List[Dict[str, float]] = []
    mins = {name: min(float(row.get(name, 0.0)) for row in rows) for name in TUNING_RETRIEVERS}
    maxs = {name: max(float(row.get(name, 0.0)) for row in rows) for name in TUNING_RETRIEVERS}

    for row in rows:
        normalized: Dict[str, float] = {}
        for name in TUNING_RETRIEVERS:
            value = float(row.get(name, 0.0))
            min_value = mins[name]
            max_value = maxs[name]
            if max_value <= min_value:
                normalized[name] = 0.0
            else:
                normalized[name] = (value - min_value) / (max_value - min_value)
        normalized_rows.append(normalized)
    return normalized_rows


def example_feature_vectors(example: TrainingExample) -> Dict[int, Dict[str, float]]:
    raw_rows = [candidate.raw_scores for candidate in example.candidates]
    normalized_rows = normalize_feature_rows(raw_rows)
    return {
        candidate.chunk_id: normalized_row
        for candidate, normalized_row in zip(example.candidates, normalized_rows)
    }


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    bounded = {
        name: max(0.0, float(weights.get(name, 0.0)))
        for name in TUNING_RETRIEVERS
    }
    total = sum(bounded.values())
    if total <= 0:
        uniform = 1.0 / len(TUNING_RETRIEVERS)
        return {name: uniform for name in TUNING_RETRIEVERS}
    return {name: value / total for name, value in bounded.items()}


def pairwise_weight_update(
    model_state: TuningModelState,
    example: TrainingExample,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> WeightUpdateSummary:
    if not example.selected_chunk_ids:
        raise ValueError("At least one selected chunk is required to update the tuning model.")

    feature_vectors = example_feature_vectors(example)
    positive_ids = list(example.selected_chunk_ids)
    implicit_negative_ids = [
        candidate.chunk_id
        for candidate in example.candidates
        if candidate.chunk_id not in set(example.selected_chunk_ids)
    ]
    negative_ids = list(example.rejected_chunk_ids) or implicit_negative_ids
    negative_ids = [chunk_id for chunk_id in negative_ids if chunk_id not in set(positive_ids)]
    if not negative_ids:
        raise ValueError("At least one non-selected candidate is required for pairwise updates.")

    previous_weights = _normalize_weights(model_state.weights)
    delta = {name: 0.0 for name in TUNING_RETRIEVERS}
    pair_count = 0

    for pos_id in positive_ids:
        pos_features = feature_vectors[pos_id]
        for neg_id in negative_ids:
            neg_features = feature_vectors[neg_id]
            pair_count += 1
            for name in TUNING_RETRIEVERS:
                delta[name] += pos_features.get(name, 0.0) - neg_features.get(name, 0.0)

    averaged_delta = {
        name: (delta[name] / pair_count) if pair_count else 0.0
        for name in TUNING_RETRIEVERS
    }
    updated_weights = {
        name: previous_weights[name] + float(learning_rate) * averaged_delta[name]
        for name in TUNING_RETRIEVERS
    }
    normalized_updated = _normalize_weights(updated_weights)

    return WeightUpdateSummary(
        previous_weights=previous_weights,
        updated_weights=normalized_updated,
        pair_count=pair_count,
        applied_delta=averaged_delta,
    )
