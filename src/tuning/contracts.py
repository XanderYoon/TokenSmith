from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal


DEFAULT_TUNING_TOP_K = 10
FEEDBACK_SCHEMA_VERSION = 1
DEFAULT_LEARNING_RATE = 0.2
TUNING_WEIGHT_SOURCES = ("baseline_config", "last_saved_learned")
TUNING_RETRIEVERS = ("faiss", "bm25", "index_keywords", "graph")
WORKFLOW_STEPS = (
    "enter_query",
    "run_retrieval",
    "inspect_top_10",
    "select_preferred_chunks",
    "submit_feedback",
    "update_weights",
    "review_revised_weights",
)

LEARNING_OBJECTIVE = {
    "model_type": "linear_retriever_weights",
    "feature_names": list(TUNING_RETRIEVERS),
    "preference_type": "pairwise_preferences_within_top_k",
    "update_rule": "deterministic_online_pairwise_weight_update",
    "constraints": {
        "nonnegative_weights": True,
        "normalize_to_sum_one": True,
        "interpretable_weights": True,
    },
    "default_learning_rate": DEFAULT_LEARNING_RATE,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChunkFeedbackRecord:
    chunk_id: int
    fused_score: float
    raw_scores: Dict[str, float]
    selected: bool = False
    rejected: bool = False
    chunk_text_preview: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["chunk_id"] = int(self.chunk_id)
        payload["fused_score"] = float(self.fused_score)
        payload["raw_scores"] = {
            name: float(self.raw_scores.get(name, 0.0))
            for name in TUNING_RETRIEVERS
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkFeedbackRecord":
        return cls(
            chunk_id=int(payload["chunk_id"]),
            fused_score=float(payload.get("fused_score", 0.0)),
            raw_scores={
                name: float(payload.get("raw_scores", {}).get(name, 0.0))
                for name in TUNING_RETRIEVERS
            },
            selected=bool(payload.get("selected", False)),
            rejected=bool(payload.get("rejected", False)),
            chunk_text_preview=str(payload.get("chunk_text_preview", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class TrainingExample:
    query: str
    candidates: List[ChunkFeedbackRecord]
    selected_chunk_ids: List[int]
    rejected_chunk_ids: List[int] = field(default_factory=list)
    session_id: str = "default"
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: int = FEEDBACK_SCHEMA_VERSION
    top_k: int = DEFAULT_TUNING_TOP_K
    workflow: List[str] = field(default_factory=lambda: list(WORKFLOW_STEPS))
    learning_objective: Dict[str, Any] = field(default_factory=lambda: dict(LEARNING_OBJECTIVE))

    def __post_init__(self) -> None:
        self.selected_chunk_ids = sorted({int(chunk_id) for chunk_id in self.selected_chunk_ids})
        self.rejected_chunk_ids = sorted({int(chunk_id) for chunk_id in self.rejected_chunk_ids})
        if not self.candidates:
            raise ValueError("TrainingExample must include at least one candidate chunk.")
        if len(self.candidates) > DEFAULT_TUNING_TOP_K:
            raise ValueError(f"TrainingExample supports at most {DEFAULT_TUNING_TOP_K} candidates.")
        candidate_ids = {candidate.chunk_id for candidate in self.candidates}
        if not set(self.selected_chunk_ids).issubset(candidate_ids):
            raise ValueError("selected_chunk_ids must be drawn from candidate chunk IDs.")
        if not set(self.rejected_chunk_ids).issubset(candidate_ids):
            raise ValueError("rejected_chunk_ids must be drawn from candidate chunk IDs.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query": self.query,
            "top_k": self.top_k,
            "workflow": list(self.workflow),
            "learning_objective": dict(self.learning_objective),
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "rejected_chunk_ids": list(self.rejected_chunk_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingExample":
        return cls(
            query=str(payload["query"]),
            candidates=[
                ChunkFeedbackRecord.from_dict(candidate_payload)
                for candidate_payload in payload.get("candidates", [])
            ],
            selected_chunk_ids=[int(chunk_id) for chunk_id in payload.get("selected_chunk_ids", [])],
            rejected_chunk_ids=[int(chunk_id) for chunk_id in payload.get("rejected_chunk_ids", [])],
            session_id=str(payload.get("session_id", "default")),
            timestamp_utc=str(payload.get("timestamp_utc", utc_now_iso())),
            schema_version=int(payload.get("schema_version", FEEDBACK_SCHEMA_VERSION)),
            top_k=int(payload.get("top_k", DEFAULT_TUNING_TOP_K)),
            workflow=list(payload.get("workflow", list(WORKFLOW_STEPS))),
            learning_objective=dict(payload.get("learning_objective", dict(LEARNING_OBJECTIVE))),
        )


@dataclass
class TuningModelState:
    weights: Dict[str, float]
    model_type: Literal["linear_retriever_weights"] = "linear_retriever_weights"
    update_rule: Literal["deterministic_online_pairwise_weight_update"] = (
        "deterministic_online_pairwise_weight_update"
    )
    schema_version: int = FEEDBACK_SCHEMA_VERSION
    updated_at_utc: str = field(default_factory=utc_now_iso)
    demonstrations_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "update_rule": self.update_rule,
            "updated_at_utc": self.updated_at_utc,
            "demonstrations_seen": int(self.demonstrations_seen),
            "weights": {
                name: float(self.weights.get(name, 0.0))
                for name in TUNING_RETRIEVERS
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TuningModelState":
        return cls(
            weights={
                name: float(payload.get("weights", {}).get(name, 0.0))
                for name in TUNING_RETRIEVERS
            },
            model_type=payload.get("model_type", "linear_retriever_weights"),
            update_rule=payload.get(
                "update_rule",
                "deterministic_online_pairwise_weight_update",
            ),
            schema_version=int(payload.get("schema_version", FEEDBACK_SCHEMA_VERSION)),
            updated_at_utc=str(payload.get("updated_at_utc", utc_now_iso())),
            demonstrations_seen=int(payload.get("demonstrations_seen", 0)),
        )
