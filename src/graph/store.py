from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.graph.extraction import extract_chunk_graph


@dataclass
class GraphNode:
    id: str
    label: str
    aliases: List[str] = field(default_factory=list)
    chunk_ids: List[int] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    chunk_ids: List[int] = field(default_factory=list)


@dataclass
class GraphStore:
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    chunk_to_entities: Dict[int, List[str]]
    chunk_to_edges: Dict[int, List[int]]

    @classmethod
    def from_chunks(cls, chunks: Iterable[str], max_entities_per_chunk: int = 24) -> "GraphStore":
        nodes: Dict[str, GraphNode] = {}
        edge_map: Dict[tuple[str, str, str], GraphEdge] = {}
        chunk_to_entities: Dict[int, List[str]] = {}
        chunk_to_edges: Dict[int, List[int]] = {}

        for chunk_id, chunk in enumerate(chunks):
            extracted = extract_chunk_graph(
                text=chunk,
                chunk_id=chunk_id,
                max_entities=max_entities_per_chunk,
            )
            entity_ids = [entity.id for entity in extracted.entities]
            chunk_to_entities[chunk_id] = entity_ids

            for entity in extracted.entities:
                node = nodes.setdefault(
                    entity.id,
                    GraphNode(id=entity.id, label=entity.surface, aliases=[entity.surface], chunk_ids=[]),
                )
                if chunk_id not in node.chunk_ids:
                    node.chunk_ids.append(chunk_id)
                if entity.surface not in node.aliases:
                    node.aliases.append(entity.surface)

            edge_indices: List[int] = []
            for relation in extracted.relations:
                edge_key = (relation.source, relation.relation, relation.target)
                edge = edge_map.get(edge_key)
                if edge is None:
                    edge = GraphEdge(
                        source=relation.source,
                        target=relation.target,
                        relation=relation.relation,
                        chunk_ids=[],
                    )
                    edge_map[edge_key] = edge
                if chunk_id not in edge.chunk_ids:
                    edge.chunk_ids.append(chunk_id)
                edge_indices.append(edge_key)

            chunk_to_edges[chunk_id] = edge_indices

        edges = list(edge_map.values())
        edge_index_lookup = {
            (edge.source, edge.relation, edge.target): idx
            for idx, edge in enumerate(edges)
        }
        normalized_chunk_to_edges = {
            chunk_id: [edge_index_lookup[key] for key in edge_keys if key in edge_index_lookup]
            for chunk_id, edge_keys in chunk_to_edges.items()
        }

        for node in nodes.values():
            node.chunk_ids.sort()
            node.aliases.sort()
        for edge in edges:
            edge.chunk_ids.sort()

        return cls(
            nodes=nodes,
            edges=edges,
            chunk_to_entities=chunk_to_entities,
            chunk_to_edges=normalized_chunk_to_edges,
        )

    def to_dict(self) -> dict:
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "label": node.label,
                    "aliases": node.aliases,
                    "chunk_ids": node.chunk_ids,
                }
                for node_id, node in sorted(self.nodes.items())
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "chunk_ids": edge.chunk_ids,
                }
                for edge in self.edges
            ],
            "chunk_to_entities": {
                str(chunk_id): entity_ids
                for chunk_id, entity_ids in sorted(self.chunk_to_entities.items())
            },
            "chunk_to_edges": {
                str(chunk_id): edge_ids
                for chunk_id, edge_ids in sorted(self.chunk_to_edges.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GraphStore":
        return cls(
            nodes={
                node_id: GraphNode(
                    id=node_payload["id"],
                    label=node_payload.get("label", node_id),
                    aliases=list(node_payload.get("aliases", [])),
                    chunk_ids=[int(chunk_id) for chunk_id in node_payload.get("chunk_ids", [])],
                )
                for node_id, node_payload in payload.get("nodes", {}).items()
            },
            edges=[
                GraphEdge(
                    source=edge_payload["source"],
                    target=edge_payload["target"],
                    relation=edge_payload.get("relation", "co_occurs"),
                    chunk_ids=[int(chunk_id) for chunk_id in edge_payload.get("chunk_ids", [])],
                )
                for edge_payload in payload.get("edges", [])
            ],
            chunk_to_entities={
                int(chunk_id): list(entity_ids)
                for chunk_id, entity_ids in payload.get("chunk_to_entities", {}).items()
            },
            chunk_to_edges={
                int(chunk_id): [int(edge_id) for edge_id in edge_ids]
                for chunk_id, edge_ids in payload.get("chunk_to_edges", {}).items()
            },
        )


def save_graph_store(graph_store: GraphStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(graph_store.to_dict(), handle, indent=2, sort_keys=True)


def load_graph_store(path: Path) -> GraphStore:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GraphStore.from_dict(payload)
