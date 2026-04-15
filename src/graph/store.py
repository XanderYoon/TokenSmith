from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


GRAPH_ARTIFACT_VERSION = 1


class GraphArtifactError(ValueError):
    """Raised when a graph artifact is malformed."""


def _dedupe_sorted_strs(values: Iterable[str]) -> List[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _dedupe_sorted_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(value) for value in values})


@dataclass
class GraphNode:
    node_id: str
    label: str
    node_type: str = "entity"
    aliases: List[str] = field(default_factory=list)
    chunk_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_id = str(self.node_id).strip()
        self.label = str(self.label).strip()
        self.node_type = str(self.node_type).strip() or "entity"
        self.aliases = _dedupe_sorted_strs(self.aliases)
        self.chunk_ids = _dedupe_sorted_ints(self.chunk_ids)
        self.metadata = dict(self.metadata)

        if not self.node_id:
            raise GraphArtifactError("GraphNode.node_id must be non-empty")
        if not self.label:
            raise GraphArtifactError("GraphNode.label must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "aliases": list(self.aliases),
            "chunk_ids": list(self.chunk_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(
            node_id=data["node_id"],
            label=data["label"],
            node_type=data.get("node_type", "entity"),
            aliases=data.get("aliases", []),
            chunk_ids=data.get("chunk_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    chunk_ids: List[int] = field(default_factory=list)
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edge_id = str(self.edge_id).strip()
        self.source_id = str(self.source_id).strip()
        self.target_id = str(self.target_id).strip()
        self.relation = str(self.relation).strip()
        self.chunk_ids = _dedupe_sorted_ints(self.chunk_ids)
        self.weight = float(self.weight)
        self.metadata = dict(self.metadata)

        if not self.edge_id:
            raise GraphArtifactError("GraphEdge.edge_id must be non-empty")
        if not self.source_id or not self.target_id:
            raise GraphArtifactError("GraphEdge source_id and target_id must be non-empty")
        if not self.relation:
            raise GraphArtifactError("GraphEdge.relation must be non-empty")
        if self.weight <= 0:
            raise GraphArtifactError("GraphEdge.weight must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "chunk_ids": list(self.chunk_ids),
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=data["relation"],
            chunk_ids=data.get("chunk_ids", []),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GraphChunkLink:
    chunk_id: int
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_id = int(self.chunk_id)
        self.node_ids = _dedupe_sorted_strs(self.node_ids)
        self.edge_ids = _dedupe_sorted_strs(self.edge_ids)
        self.metadata = dict(self.metadata)

        if self.chunk_id < 0:
            raise GraphArtifactError("GraphChunkLink.chunk_id must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphChunkLink":
        return cls(
            chunk_id=data["chunk_id"],
            node_ids=data.get("node_ids", []),
            edge_ids=data.get("edge_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GraphArtifact:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    chunk_links: List[GraphChunkLink] = field(default_factory=list)
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = GRAPH_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        self.version = int(self.version)
        self.document_id = None if self.document_id is None else str(self.document_id)
        self.nodes = list(self.nodes)
        self.edges = list(self.edges)
        self.chunk_links = sorted(list(self.chunk_links), key=lambda link: link.chunk_id)
        self.metadata = dict(self.metadata)
        self.validate()

    def validate(self) -> None:
        if self.version != GRAPH_ARTIFACT_VERSION:
            raise GraphArtifactError(
                f"Unsupported graph artifact version {self.version}; "
                f"expected {GRAPH_ARTIFACT_VERSION}"
            )

        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)):
            raise GraphArtifactError("GraphArtifact contains duplicate node_ids")
        if len(edge_ids) != len(set(edge_ids)):
            raise GraphArtifactError("GraphArtifact contains duplicate edge_ids")

        known_nodes = set(node_ids)
        known_edges = set(edge_ids)

        for edge in self.edges:
            if edge.source_id not in known_nodes or edge.target_id not in known_nodes:
                raise GraphArtifactError(
                    f"GraphEdge '{edge.edge_id}' references unknown node_ids"
                )

        chunk_ids_in_links = set()
        for link in self.chunk_links:
            if link.chunk_id in chunk_ids_in_links:
                raise GraphArtifactError(
                    f"GraphArtifact contains duplicate chunk_links for chunk_id={link.chunk_id}"
                )
            chunk_ids_in_links.add(link.chunk_id)

            missing_nodes = [node_id for node_id in link.node_ids if node_id not in known_nodes]
            if missing_nodes:
                raise GraphArtifactError(
                    f"Chunk link {link.chunk_id} references unknown node_ids: {missing_nodes}"
                )

            missing_edges = [edge_id for edge_id in link.edge_ids if edge_id not in known_edges]
            if missing_edges:
                raise GraphArtifactError(
                    f"Chunk link {link.chunk_id} references unknown edge_ids: {missing_edges}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "document_id": self.document_id,
            "metadata": dict(self.metadata),
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda node: node.node_id)],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda edge: edge.edge_id)],
            "chunk_links": [link.to_dict() for link in self.chunk_links],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphArtifact":
        return cls(
            version=data.get("version", GRAPH_ARTIFACT_VERSION),
            document_id=data.get("document_id"),
            metadata=data.get("metadata", {}),
            nodes=[GraphNode.from_dict(item) for item in data.get("nodes", [])],
            edges=[GraphEdge.from_dict(item) for item in data.get("edges", [])],
            chunk_links=[
                GraphChunkLink.from_dict(item) for item in data.get("chunk_links", [])
            ],
        )


def save_graph_artifact(path: Path | str, artifact: GraphArtifact) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.validate()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, indent=2, sort_keys=False)


def load_graph_artifact(path: Path | str) -> GraphArtifact:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return GraphArtifact.from_dict(data)
