from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from src.graph.store import GraphArtifact, GraphChunkLink, GraphEdge, GraphNode


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ID_CHARS_RE = re.compile(r"[^a-z0-9]+")
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_TRAILING_CLAUSE_RE = re.compile(
    r"\b(?:that|which|who|because|including|such as|especially|where|when)\b.*$",
    re.IGNORECASE,
)

_RELATION_PATTERNS = (
    ("is_a", re.compile(r"^(?P<subject>.+?)\s+is\s+(?P<object>.+)$", re.IGNORECASE)),
    ("is_a", re.compile(r"^(?P<subject>.+?)\s+are\s+(?P<object>.+)$", re.IGNORECASE)),
    (
        "references",
        re.compile(r"^(?P<subject>.+?)\s+references\s+(?P<object>.+)$", re.IGNORECASE),
    ),
    (
        "consists_of",
        re.compile(r"^(?P<subject>.+?)\s+consists of\s+(?P<object>.+)$", re.IGNORECASE),
    ),
    (
        "contains",
        re.compile(r"^(?P<subject>.+?)\s+contains\s+(?P<object>.+)$", re.IGNORECASE),
    ),
)


def _normalize_spaces(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _clean_phrase(text: str) -> str:
    phrase = _normalize_spaces(text.strip(" \t\r\n.,;:()[]{}\"'`"))
    phrase = _LEADING_ARTICLE_RE.sub("", phrase)
    phrase = _TRAILING_CLAUSE_RE.sub("", phrase).strip(" ,;:.")
    return _normalize_spaces(phrase)


def _normalize_entity_label(text: str) -> Optional[str]:
    phrase = _clean_phrase(text)
    if not phrase:
        return None
    words = phrase.split()
    if len(words) > 10:
        phrase = " ".join(words[:10])
        words = phrase.split()
    if words:
        last_word = words[-1]
        if last_word.endswith("s") and not last_word.endswith("ss") and len(last_word) > 3:
            words[-1] = last_word[:-1]
            phrase = " ".join(words)
    if len(phrase) < 2:
        return None
    return phrase.lower()


def _slugify(text: str) -> str:
    slug = _NON_ID_CHARS_RE.sub("-", text.lower()).strip("-")
    return slug or "unknown"


def _entity_id(label: str) -> str:
    return f"entity:{_slugify(label)}"


def _edge_id(source_label: str, relation: str, target_label: str) -> str:
    return f"edge:{_slugify(source_label)}-{relation}-{_slugify(target_label)}"


def _sentence_candidates(text: str) -> List[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(_normalize_spaces(text)) if part.strip()]


def _extract_relation(sentence: str) -> Optional[Dict[str, str]]:
    sentence = sentence.strip().strip(".!?")
    for relation, pattern in _RELATION_PATTERNS:
        match = pattern.match(sentence)
        if not match:
            continue
        subject = _normalize_entity_label(match.group("subject"))
        obj = _normalize_entity_label(match.group("object"))
        if not subject or not obj or subject == obj:
            return None
        return {
            "source_label": subject,
            "target_label": obj,
            "relation": relation,
            "evidence": sentence,
        }
    return None


def _extract_standalone_entities(chunk_text: str) -> List[str]:
    entities: List[str] = []
    for sentence in _sentence_candidates(chunk_text):
        relation = _extract_relation(sentence)
        if relation is not None:
            entities.extend([relation["source_label"], relation["target_label"]])
            continue

        cleaned = _normalize_entity_label(sentence)
        if cleaned and len(cleaned.split()) <= 6:
            entities.append(cleaned)
    return sorted(set(entities))


def extract_chunk_graph(chunk_text: str, *, chunk_id: int) -> Dict[str, Any]:
    node_labels = set()
    edge_records: List[Dict[str, str]] = []

    for sentence_index, sentence in enumerate(_sentence_candidates(chunk_text)):
        relation = _extract_relation(sentence)
        if relation is None:
            continue
        relation["sentence_index"] = sentence_index
        edge_records.append(relation)
        node_labels.add(relation["source_label"])
        node_labels.add(relation["target_label"])

    if not edge_records:
        node_labels.update(_extract_standalone_entities(chunk_text))

    node_ids = sorted(_entity_id(label) for label in node_labels)
    edge_ids = sorted(
        _edge_id(record["source_label"], record["relation"], record["target_label"])
        for record in edge_records
    )

    return {
        "chunk_id": int(chunk_id),
        "node_labels": sorted(node_labels),
        "node_ids": node_ids,
        "edges": edge_records,
        "edge_ids": edge_ids,
    }


def build_graph_artifact(
    chunks: Iterable[str],
    *,
    document_id: Optional[str] = None,
    metadata_by_chunk: Optional[Iterable[Dict[str, Any]]] = None,
) -> GraphArtifact:
    chunk_list = list(chunks)
    metadata_list = list(metadata_by_chunk) if metadata_by_chunk is not None else [None] * len(chunk_list)
    if len(metadata_list) != len(chunk_list):
        raise ValueError("metadata_by_chunk must align 1:1 with chunks")

    node_chunks: Dict[str, set[int]] = defaultdict(set)
    node_aliases: Dict[str, set[str]] = defaultdict(set)
    edge_chunks: Dict[str, set[int]] = defaultdict(set)
    edge_metadata: Dict[str, Dict[str, Any]] = {}
    chunk_links: List[GraphChunkLink] = []

    for chunk_id, chunk_text in enumerate(chunk_list):
        extracted = extract_chunk_graph(chunk_text, chunk_id=chunk_id)
        if not extracted["node_ids"] and not extracted["edge_ids"]:
            continue

        for label in extracted["node_labels"]:
            node_chunks[label].add(chunk_id)
            node_aliases[label].add(label)

        for edge in extracted["edges"]:
            edge_id = _edge_id(edge["source_label"], edge["relation"], edge["target_label"])
            edge_chunks[edge_id].add(chunk_id)
            edge_metadata.setdefault(
                edge_id,
                {
                    "source_label": edge["source_label"],
                    "target_label": edge["target_label"],
                    "relation": edge["relation"],
                    "evidence": [],
                },
            )
            edge_metadata[edge_id]["evidence"].append(
                {
                    "chunk_id": chunk_id,
                    "sentence_index": edge["sentence_index"],
                    "text": edge["evidence"],
                }
            )

        link_metadata = {}
        if metadata_list[chunk_id] is not None:
            chunk_meta = dict(metadata_list[chunk_id])
            link_metadata["section"] = chunk_meta.get("section")
            link_metadata["section_path"] = chunk_meta.get("section_path")
        chunk_links.append(
            GraphChunkLink(
                chunk_id=chunk_id,
                node_ids=extracted["node_ids"],
                edge_ids=extracted["edge_ids"],
                metadata=link_metadata,
            )
        )

    nodes = [
        GraphNode(
            node_id=_entity_id(label),
            label=label,
            aliases=sorted(node_aliases[label]),
            chunk_ids=sorted(node_chunks[label]),
            metadata={"extractor": "heuristic_v1"},
        )
        for label in sorted(node_chunks)
    ]

    edges = [
        GraphEdge(
            edge_id=edge_id,
            source_id=_entity_id(edge_metadata[edge_id]["source_label"]),
            target_id=_entity_id(edge_metadata[edge_id]["target_label"]),
            relation=edge_metadata[edge_id]["relation"],
            chunk_ids=sorted(edge_chunks[edge_id]),
            metadata={
                "extractor": "heuristic_v1",
                "evidence": edge_metadata[edge_id]["evidence"],
            },
        )
        for edge_id in sorted(edge_chunks)
    ]

    return GraphArtifact(
        document_id=document_id,
        metadata={"extractor": "heuristic_v1"},
        nodes=nodes,
        edges=edges,
        chunk_links=chunk_links,
    )
