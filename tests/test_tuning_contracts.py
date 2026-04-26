import pytest

from src.config import RAGConfig
from src.tuning.contracts import (
    DEFAULT_TUNING_TOP_K,
    LEARNING_OBJECTIVE,
    TUNING_RETRIEVERS,
    ChunkFeedbackRecord,
    TrainingExample,
    TuningModelState,
)


pytestmark = pytest.mark.unit


def test_training_example_round_trip():
    example = TrainingExample(
        query="What is a transaction?",
        candidates=[
            ChunkFeedbackRecord(
                chunk_id=4,
                fused_score=0.91,
                raw_scores={"faiss": 0.8, "bm25": 0.6, "index_keywords": 0.2, "graph": 0.4},
                selected=True,
                chunk_text_preview="Transactions group reads and writes...",
                metadata={"page_numbers": [10]},
            ),
            ChunkFeedbackRecord(
                chunk_id=9,
                fused_score=0.51,
                raw_scores={"faiss": 0.3, "bm25": 0.4, "index_keywords": 0.1, "graph": 0.0},
            ),
        ],
        selected_chunk_ids=[4],
        rejected_chunk_ids=[9],
        session_id="session-1",
    )

    restored = TrainingExample.from_dict(example.to_dict())

    assert restored.query == example.query
    assert restored.selected_chunk_ids == [4]
    assert restored.rejected_chunk_ids == [9]
    assert restored.top_k == DEFAULT_TUNING_TOP_K
    assert restored.learning_objective["model_type"] == LEARNING_OBJECTIVE["model_type"]


def test_training_example_rejects_unknown_selected_chunk():
    with pytest.raises(ValueError, match="selected_chunk_ids"):
        TrainingExample(
            query="test",
            candidates=[
                ChunkFeedbackRecord(chunk_id=1, fused_score=0.3, raw_scores={}),
            ],
            selected_chunk_ids=[99],
        )


def test_tuning_model_state_round_trip():
    state = TuningModelState(
        weights={"faiss": 0.4, "bm25": 0.3, "index_keywords": 0.1, "graph": 0.2},
        demonstrations_seen=5,
    )
    restored = TuningModelState.from_dict(state.to_dict())

    assert set(restored.weights) == set(TUNING_RETRIEVERS)
    assert restored.weights["graph"] == 0.2
    assert restored.demonstrations_seen == 5


def test_config_exposes_fixed_tuning_storage_paths():
    cfg = RAGConfig()
    examples_path = cfg.get_tuning_examples_path("textbook_index")
    state_path = cfg.get_tuning_state_path("textbook_index")

    assert examples_path.name == "demonstrations.jsonl"
    assert state_path.name == "learned_weights.json"
    assert examples_path.parent == state_path.parent
