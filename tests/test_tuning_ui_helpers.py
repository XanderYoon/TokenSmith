import pytest

from src.tuning.ui_helpers import candidate_label, weight_table


pytestmark = pytest.mark.unit


class _Candidate:
    def __init__(self):
        self.rank = 2
        self.chunk_id = 17
        self.fused_score = 0.625
        self.metadata = {"page_numbers": [8, 9]}


def test_weight_table_orders_known_retrievers():
    rows = weight_table({"graph": 0.2, "faiss": 0.5, "bm25": 0.2, "index_keywords": 0.1})

    assert [row["retriever"] for row in rows] == ["faiss", "bm25", "index_keywords", "graph"]
    assert rows[0]["weight"] == 0.5


def test_candidate_label_includes_rank_chunk_and_pages():
    label = candidate_label(_Candidate())

    assert "#2" in label
    assert "chunk 17" in label
    assert "pages=[8, 9]" in label
