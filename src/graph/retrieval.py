from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, List, Set

from src.graph.extraction import extract_chunk_graph, normalize_phrase
from src.graph.store import GraphStore


class GraphRetriever:
    name = "graph"

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        self.alias_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self.token_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self.edges_by_node: Dict[str, List[int]] = defaultdict(list)

        for node_id, node in self.graph_store.nodes.items():
            aliases = set(node.aliases) | {node.label, node_id}
            for alias in aliases:
                normalized = normalize_phrase(alias)
                if not normalized:
                    continue
                self.alias_to_nodes[normalized].add(node_id)
                for token in normalized.split():
                    self.token_to_nodes[token].add(node_id)

        for idx, edge in enumerate(self.graph_store.edges):
            self.edges_by_node[edge.source].append(idx)
            self.edges_by_node[edge.target].append(idx)

    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        query_graph = extract_chunk_graph(query, chunk_id=-1, max_entities=12)
        query_entities = [entity.id for entity in query_graph.entities]
        if not query_entities:
            query_entities = self._fallback_query_entities(query)

        node_matches = self._match_nodes(query_entities)
        if not node_matches:
            return {}

        chunk_scores: Dict[int, float] = defaultdict(float)
        query_entity_set = set(query_entities)

        for node_id, match_score in node_matches.items():
            node = self.graph_store.nodes.get(node_id)
            if node is None:
                continue

            for chunk_id in node.chunk_ids:
                if 0 <= chunk_id < len(chunks):
                    chunk_scores[chunk_id] += 1.5 * match_score

            for edge_idx in self.edges_by_node.get(node_id, []):
                edge = self.graph_store.edges[edge_idx]
                neighbor_id = edge.target if edge.source == node_id else edge.source

                for chunk_id in edge.chunk_ids:
                    if 0 <= chunk_id < len(chunks):
                        edge_bonus = 1.0
                        if edge.source in query_entity_set and edge.target in query_entity_set:
                            edge_bonus += 0.75
                        chunk_scores[chunk_id] += edge_bonus * match_score

                neighbor = self.graph_store.nodes.get(neighbor_id)
                if neighbor is None:
                    continue
                for chunk_id in neighbor.chunk_ids:
                    if 0 <= chunk_id < len(chunks):
                        chunk_scores[chunk_id] += 0.35 * match_score

        if not chunk_scores:
            return {}

        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:pool_size]
        max_score = ranked[0][1] if ranked else 1.0
        if max_score <= 0:
            return {}
        return {int(chunk_id): float(score / max_score) for chunk_id, score in ranked}

    def _match_nodes(self, query_entities: List[str]) -> Dict[str, float]:
        node_scores: Dict[str, float] = defaultdict(float)
        for entity in query_entities:
            direct_matches = self.alias_to_nodes.get(entity, set())
            if direct_matches:
                for node_id in direct_matches:
                    node_scores[node_id] += 1.0
                continue

            entity_tokens = set(entity.split())
            candidate_nodes: Set[str] = set()
            for token in entity_tokens:
                candidate_nodes.update(self.token_to_nodes.get(token, set()))

            for node_id in candidate_nodes:
                node_tokens = set(node_id.split())
                overlap = len(entity_tokens & node_tokens)
                if not overlap:
                    continue
                node_scores[node_id] += overlap / max(len(entity_tokens), len(node_tokens))
        return node_scores

    @staticmethod
    def _fallback_query_entities(query: str) -> List[str]:
        tokens = [normalize_phrase(token) for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)]
        normalized_tokens = [token for token in tokens if token]
        entities = []
        for idx, token in enumerate(normalized_tokens):
            entities.append(token)
            if idx + 1 < len(normalized_tokens):
                entities.append(f"{token} {normalized_tokens[idx + 1]}")
        seen = set()
        ordered = []
        for entity in entities:
            if entity in seen:
                continue
            seen.add(entity)
            ordered.append(entity)
        return ordered[:12]
