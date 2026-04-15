import pytest

from src.graph import GraphArtifact, GraphEdge, GraphNode, GraphQueryScorer


pytestmark = pytest.mark.unit


def test_graph_query_scorer_expands_one_hop_neighbors():
    artifact = GraphArtifact(
        document_id="test_index",
        nodes=[
            GraphNode(node_id="entity:foreign-key", label="foreign key", chunk_ids=[1]),
            GraphNode(node_id="entity:primary-key", label="primary key", chunk_ids=[2]),
            GraphNode(node_id="entity:relation", label="relation", chunk_ids=[3]),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:foreign-key-references-primary-key",
                source_id="entity:foreign-key",
                target_id="entity:primary-key",
                relation="references",
                chunk_ids=[1],
            )
        ],
    )

    scorer = GraphQueryScorer(artifact)
    scores = scorer.score_chunks("What does a foreign key reference?", 10)

    assert list(scores.keys())[0] == 1
    assert 2 in scores
    assert 3 not in scores


def test_graph_query_scorer_returns_empty_for_unmatched_query():
    artifact = GraphArtifact(
        nodes=[GraphNode(node_id="entity:database-schema", label="database schema", chunk_ids=[0])]
    )

    scorer = GraphQueryScorer(artifact)
    assert scorer.score_chunks("network congestion control", 10) == {}
