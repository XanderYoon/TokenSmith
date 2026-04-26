from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "what", "when", "where", "which", "who", "why",
    "with", "within", "without",
}


@dataclass(frozen=True)
class ExtractedEntity:
    id: str
    surface: str


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class ExtractedChunkGraph:
    chunk_id: int
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]


def normalize_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r"[^a-z0-9_-]", "", token)
    if len(token) <= 2:
        return ""
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith(("sses", "shes", "ches", "xes", "zes")) and len(token) > 5:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    return token


def normalize_phrase(text: str) -> str:
    tokens = [normalize_token(token) for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)]
    normalized = [token for token in tokens if token and token not in STOPWORDS and not token.isdigit()]
    return " ".join(normalized)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_chunk_graph(text: str, chunk_id: int, max_entities: int = 24) -> ExtractedChunkGraph:
    sentence_entities: List[List[str]] = []
    entity_surfaces: dict[str, str] = {}

    for sentence in re.split(r"[\n\r.!?;:]+", text):
        raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", sentence)
        normalized_tokens = [normalize_token(token) for token in raw_tokens]
        filtered = [
            token
            for token in normalized_tokens
            if token and token not in STOPWORDS and not token.isdigit()
        ]
        if not filtered:
            continue

        sentence_terms: List[str] = []
        for idx, token in enumerate(filtered):
            sentence_terms.append(token)
            if idx + 1 < len(filtered):
                sentence_terms.append(f"{token} {filtered[idx + 1]}")

        ordered_terms = _dedupe_preserve_order(sentence_terms)[:max_entities]
        if not ordered_terms:
            continue

        sentence_entities.append(ordered_terms)
        for term in ordered_terms:
            entity_surfaces.setdefault(term, term)

    all_entities = _dedupe_preserve_order(
        [entity for entities in sentence_entities for entity in entities]
    )[:max_entities]
    relations: List[ExtractedRelation] = []
    seen_relations = set()

    for entities in sentence_entities:
        for left_idx, source in enumerate(entities):
            for right_idx in range(left_idx + 1, min(len(entities), left_idx + 3)):
                target = entities[right_idx]
                if source == target:
                    continue
                edge = tuple(sorted((source, target)))
                if edge in seen_relations:
                    continue
                seen_relations.add(edge)
                relations.append(
                    ExtractedRelation(source=edge[0], target=edge[1], relation="co_occurs")
                )

    return ExtractedChunkGraph(
        chunk_id=chunk_id,
        entities=[ExtractedEntity(id=entity_id, surface=entity_surfaces[entity_id]) for entity_id in all_entities],
        relations=relations,
    )
