import pytest

from src.graph.retrieval import GraphRetriever
from src.graph.store import GraphStore


pytestmark = pytest.mark.unit


def test_graph_retriever_scores_entity_matches():
    chunks = [
        "Transactions use locks to preserve isolation.",
        "Buffers cache recently accessed pages.",
        "Lock managers track transaction conflicts.",
    ]
    retriever = GraphRetriever(GraphStore.from_chunks(chunks))

    scores = retriever.get_scores("How do locks help transactions?", pool_size=3, chunks=chunks)

    assert scores
    assert max(scores, key=scores.get) in {0, 2}
    assert 1 not in list(scores)[:1]


def test_graph_retriever_uses_relation_neighborhoods():
    chunks = [
        "A transaction acquires a lock before updating a row.",
        "Rows are stored on disk pages.",
        "Lock compatibility determines whether transactions can proceed.",
    ]
    retriever = GraphRetriever(GraphStore.from_chunks(chunks))

    scores = retriever.get_scores("transaction lock", pool_size=5, chunks=chunks)

    assert 0 in scores
    assert 2 in scores
    assert scores[0] >= scores[2]
