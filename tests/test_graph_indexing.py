import json

import pytest

from src.graph.extraction import build_graph_artifact
from src.graph.store import (
    GRAPH_ARTIFACT_VERSION,
    GraphArtifact,
    GraphArtifactError,
    GraphChunkLink,
    GraphEdge,
    GraphNode,
    load_graph_artifact,
    save_graph_artifact,
)


pytestmark = pytest.mark.unit


def test_graph_artifact_round_trip_json(tmp_path):
    artifact = GraphArtifact(
        document_id="textbook_index",
        metadata={"source": "unit-test"},
        nodes=[
            GraphNode(
                node_id="entity:database schema",
                label="database schema",
                aliases=["schema", "database schema", "schema"],
                chunk_ids=[4, 2, 4],
            ),
            GraphNode(
                node_id="entity:relation",
                label="relation",
                chunk_ids=[2],
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:schema-defines-relation",
                source_id="entity:database schema",
                target_id="entity:relation",
                relation="defines",
                chunk_ids=[2, 2],
                metadata={"confidence": "high"},
            )
        ],
        chunk_links=[
            GraphChunkLink(
                chunk_id=2,
                node_ids=["entity:relation", "entity:database schema"],
                edge_ids=["edge:schema-defines-relation"],
            ),
            GraphChunkLink(
                chunk_id=4,
                node_ids=["entity:database schema"],
                edge_ids=[],
            ),
        ],
    )

    output_path = tmp_path / "graph.json"
    save_graph_artifact(output_path, artifact)

    reloaded = load_graph_artifact(output_path)

    assert reloaded.version == GRAPH_ARTIFACT_VERSION
    assert reloaded.document_id == "textbook_index"
    assert reloaded.nodes[0].chunk_ids == [2, 4]
    assert reloaded.nodes[0].aliases == ["database schema", "schema"]
    assert reloaded.edges[0].chunk_ids == [2]
    assert [link.chunk_id for link in reloaded.chunk_links] == [2, 4]

    raw = json.loads(output_path.read_text())
    assert raw["nodes"][0]["node_id"] == "entity:database schema"
    assert raw["edges"][0]["edge_id"] == "edge:schema-defines-relation"


def test_graph_artifact_rejects_unknown_edge_references():
    with pytest.raises(GraphArtifactError, match="unknown edge_ids"):
        GraphArtifact(
            nodes=[GraphNode(node_id="entity:a", label="A")],
            chunk_links=[GraphChunkLink(chunk_id=0, node_ids=["entity:a"], edge_ids=["edge:x"])],
        )


def test_graph_artifact_rejects_edges_with_unknown_nodes():
    with pytest.raises(GraphArtifactError, match="unknown node_ids"):
        GraphArtifact(
            nodes=[GraphNode(node_id="entity:a", label="A")],
            edges=[
                GraphEdge(
                    edge_id="edge:a-b",
                    source_id="entity:a",
                    target_id="entity:b",
                    relation="related_to",
                )
            ],
        )


def test_graph_artifact_rejects_duplicate_chunk_links():
    with pytest.raises(GraphArtifactError, match="duplicate chunk_links"):
        GraphArtifact(
            nodes=[GraphNode(node_id="entity:a", label="A")],
            chunk_links=[
                GraphChunkLink(chunk_id=1, node_ids=["entity:a"]),
                GraphChunkLink(chunk_id=1, node_ids=["entity:a"]),
            ],
        )


def test_build_graph_artifact_merges_labels_with_same_slugified_node_id():
    artifact = build_graph_artifact(
        ["B+-tree", "B tree"],
        document_id="textbook_index",
    )

    assert len(artifact.nodes) == 1
    assert artifact.nodes[0].node_id == "entity:b-tree"
    assert artifact.nodes[0].aliases == ["b tree", "b+-tree"]
    assert artifact.nodes[0].chunk_ids == [0, 1]
