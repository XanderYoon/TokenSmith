import json

import pytest

from src.graph.extraction import extract_chunk_graph, normalize_phrase
from src.graph.store import GraphStore, load_graph_store, save_graph_store


pytestmark = pytest.mark.unit


def test_extract_chunk_graph_links_entities_and_relations():
    extracted = extract_chunk_graph(
        "Transactions coordinate locks and logs. Locks protect transactions.",
        chunk_id=7,
        max_entities=12,
    )

    entity_ids = {entity.id for entity in extracted.entities}
    assert extracted.chunk_id == 7
    assert "transaction" in entity_ids
    assert "lock" in entity_ids
    assert any("lock" in {relation.source, relation.target} for relation in extracted.relations)


def test_graph_store_persists_chunk_linkage(tmp_path):
    chunks = [
        "Transactions use locks to preserve isolation.",
        "Indexes accelerate query execution plans.",
    ]
    graph_store = GraphStore.from_chunks(chunks, max_entities_per_chunk=12)
    output_path = tmp_path / "graph.json"

    save_graph_store(graph_store, output_path)
    reloaded = load_graph_store(output_path)

    assert output_path.exists()
    assert reloaded.chunk_to_entities[0]
    assert "transaction" in reloaded.chunk_to_entities[0]
    assert reloaded.nodes["transaction"].chunk_ids == [0]


def test_graph_json_is_human_reviewable(tmp_path):
    graph_store = GraphStore.from_chunks(["Deadlocks involve wait-for graphs."])
    output_path = tmp_path / "graph.json"
    save_graph_store(graph_store, output_path)

    payload = json.loads(output_path.read_text())

    assert "nodes" in payload
    assert "edges" in payload
    assert "chunk_to_entities" in payload
    assert "deadlock" in payload["nodes"]
    assert normalize_phrase("Deadlocks") == "deadlock"
