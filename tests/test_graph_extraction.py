import pytest

from src.graph.extraction import build_graph_artifact, extract_chunk_graph


pytestmark = pytest.mark.unit


def test_extract_chunk_graph_finds_entities_and_relations():
    extracted = extract_chunk_graph(
        "A database schema is the logical design of a database. "
        "A foreign key references a primary key.",
        chunk_id=7,
    )

    assert extracted["chunk_id"] == 7
    assert "entity:database-schema" in extracted["node_ids"]
    assert "entity:logical-design-of-a-database" in extracted["node_ids"]
    assert "entity:foreign-key" in extracted["node_ids"]
    assert "entity:primary-key" in extracted["node_ids"]
    assert extracted["edge_ids"] == [
        "edge:database-schema-is_a-logical-design-of-a-database",
        "edge:foreign-key-references-primary-key",
    ]


def test_build_graph_artifact_dedupes_nodes_across_chunks():
    artifact = build_graph_artifact(
        [
            "A database schema is the logical design of a database.",
            "A foreign key references a primary key.",
            "A database schema contains relations.",
        ],
        document_id="textbook_index",
        metadata_by_chunk=[
            {"section": "Intro", "section_path": "Chapter 1 Intro"},
            {"section": "Keys", "section_path": "Chapter 1 Keys"},
            {"section": "Intro", "section_path": "Chapter 1 Intro"},
        ],
    )

    node_map = {node.node_id: node for node in artifact.nodes}
    edge_map = {edge.edge_id: edge for edge in artifact.edges}
    chunk_link_map = {link.chunk_id: link for link in artifact.chunk_links}

    assert artifact.document_id == "textbook_index"
    assert node_map["entity:database-schema"].chunk_ids == [0, 2]
    assert node_map["entity:relation"].chunk_ids == [2]
    assert edge_map["edge:database-schema-is_a-logical-design-of-a-database"].chunk_ids == [0]
    assert edge_map["edge:foreign-key-references-primary-key"].chunk_ids == [1]
    assert edge_map["edge:database-schema-contains-relation"].chunk_ids == [2]
    assert chunk_link_map[1].metadata["section"] == "Keys"
    assert "entity:database-schema" in chunk_link_map[0].node_ids


def test_build_graph_artifact_rejects_misaligned_chunk_metadata():
    with pytest.raises(ValueError, match="align 1:1"):
        build_graph_artifact(
            ["A database schema is the logical design of a database."],
            metadata_by_chunk=[],
        )
