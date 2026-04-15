from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from src.graph.store import GraphArtifact, GraphEdge, GraphNode


_TOKEN_RE = re.compile(r"[^a-z0-9_'#+-]")


def _tokenize_text(text: str) -> Set[str]:
    normalized = _TOKEN_RE.sub(" ", text.lower())
    return {token for token in normalized.split() if token}


def _overlap_score(query_tokens: Set[str], candidate_tokens: Set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(candidate_tokens)


class GraphQueryScorer:
    """Scores chunks using graph node matches plus one-hop edge expansion."""

    def __init__(self, artifact: GraphArtifact, *, use_aliases: bool = True) -> None:
        self.artifact = artifact
        self.use_aliases = bool(use_aliases)
        self.nodes_by_id: Dict[str, GraphNode] = {node.node_id: node for node in artifact.nodes}
        self.node_tokens: Dict[str, Set[str]] = {}
        self.edges_by_node: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.relation_tokens: Dict[str, Set[str]] = {}

        for node in artifact.nodes:
            lexical_forms = [node.label]
            if self.use_aliases:
                lexical_forms.extend(node.aliases)
            combined_tokens: Set[str] = set()
            for form in lexical_forms:
                combined_tokens.update(_tokenize_text(form))
            self.node_tokens[node.node_id] = combined_tokens

        for edge in artifact.edges:
            self.edges_by_node[edge.source_id].append(edge)
            self.edges_by_node[edge.target_id].append(edge)
            self.relation_tokens[edge.edge_id] = _tokenize_text(edge.relation.replace("_", " "))

    def score_chunks(self, query: str, pool_size: int) -> Dict[int, float]:
        query_tokens = _tokenize_text(query)
        if not query_tokens:
            return {}

        matched_nodes = self._match_nodes(query_tokens)
        if not matched_nodes:
            return {}

        chunk_scores: Dict[int, float] = defaultdict(float)

        for node_id, match_score in matched_nodes.items():
            node = self.nodes_by_id[node_id]
            for chunk_id in node.chunk_ids:
                chunk_scores[chunk_id] += match_score

        for node_id, match_score in matched_nodes.items():
            for edge in self.edges_by_node.get(node_id, []):
                relation_match = _overlap_score(query_tokens, self.relation_tokens.get(edge.edge_id, set()))
                other_node_id = edge.target_id if edge.source_id == node_id else edge.source_id
                other_match = matched_nodes.get(other_node_id, 0.0)
                edge_score = (
                    0.6 * match_score
                    + 0.3 * other_match
                    + 0.4 * relation_match
                ) * edge.weight
                if edge_score <= 0.0:
                    continue

                for chunk_id in edge.chunk_ids:
                    chunk_scores[chunk_id] += 1.2 * edge_score

                # One-hop expansion to neighboring evidence even when only one endpoint matches.
                if other_match == 0.0:
                    for chunk_id in self.nodes_by_id[other_node_id].chunk_ids:
                        chunk_scores[chunk_id] += 0.35 * match_score

        if not chunk_scores:
            return {}

        max_score = max(chunk_scores.values())
        normalized = {
            int(chunk_id): float(score) / max_score
            for chunk_id, score in chunk_scores.items()
            if max_score > 0.0
        }
        sorted_hits = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
        return dict(sorted_hits[:pool_size])

    def _match_nodes(self, query_tokens: Set[str]) -> Dict[str, float]:
        matches: Dict[str, float] = {}
        for node_id, candidate_tokens in self.node_tokens.items():
            overlap_score = _overlap_score(query_tokens, candidate_tokens)
            if overlap_score <= 0.0:
                continue
            matches[node_id] = overlap_score
        return matches
