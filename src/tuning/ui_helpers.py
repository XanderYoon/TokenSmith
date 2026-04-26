from __future__ import annotations

from typing import Dict, List

from src.tuning.contracts import TUNING_RETRIEVERS


def weight_table(weights: Dict[str, float]) -> List[Dict[str, float]]:
    return [
        {"retriever": name, "weight": float(weights.get(name, 0.0))}
        for name in TUNING_RETRIEVERS
    ]


def candidate_label(candidate) -> str:
    pages = candidate.metadata.get("page_numbers", [])
    page_suffix = f" | pages={pages}" if pages else ""
    return f"#{candidate.rank} chunk {candidate.chunk_id} | fused={candidate.fused_score:.4f}{page_suffix}"
