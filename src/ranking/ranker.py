"""
ranker.py

This module supports deterministic fusion of retrieval scores.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

Candidate = int


class EnsembleRanker:
    """
    Computes weighted reciprocal rank fusion (RRF) or weighted linear fusion.
    Active retrievers are the only sources considered during validation and fusion.
    """

    def __init__(
        self,
        ensemble_method: str,
        weights: Dict[str, float],
        rrf_k: int = 60,
        active_retrievers: Optional[Iterable[str]] = None,
        normalization: str = "minmax",
    ):
        self.ensemble_method = ensemble_method.lower().strip()
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.rrf_k = int(rrf_k)
        self.normalization = normalization.lower().strip()
        self.active_retrievers = list(active_retrievers or self._infer_active_retrievers())
        self.linear_score_weight = 0.75
        self.linear_rank_weight = 0.25
        self.linear_agreement_bonus = 0.2

        if self.ensemble_method not in {"linear", "weighted", "rrf"}:
            raise ValueError(f"Unsupported ranking method '{self.ensemble_method}'")
        if self.normalization not in {"minmax"}:
            raise ValueError(f"Unsupported normalization policy '{self.normalization}'")
        if not self.active_retrievers:
            raise ValueError("At least one retriever must be active")

        active_weights = self._active_weights()
        total = sum(active_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights for active retrievers must sum to 1.0. Current sum: {total}"
            )
        missing = [name for name in self.active_retrievers if active_weights.get(name, 0.0) <= 0.0]
        if missing:
            raise ValueError(
                f"Active retrievers must have positive weights. Missing positive weights for: {missing}"
            )

    def rank(self, raw_scores: Dict[str, Dict[Candidate, float]]) -> Tuple[List[int], List[float]]:
        per_retriever_scores = self._validate_raw_scores(raw_scores)
        if not any(per_retriever_scores.values()):
            return [], []

        if self.ensemble_method == "rrf":
            return self._weighted_rrf_fuse(per_retriever_scores)
        return self._weighted_linear_fuse(per_retriever_scores)

    def _weighted_rrf_fuse(
        self,
        per_retriever_scores: Dict[str, Dict[Candidate, float]],
    ) -> Tuple[List[int], List[float]]:
        fused_scores: Dict[Candidate, float] = defaultdict(float)
        per_retriever_ranks = {
            name: self.scores_to_ranks(scores)
            for name, scores in per_retriever_scores.items()
        }
        all_candidates = sorted({cand for scores in per_retriever_scores.values() for cand in scores})

        for cand in all_candidates:
            for name, ranks in per_retriever_ranks.items():
                rank = ranks.get(cand)
                if rank is None:
                    continue
                fused_scores[cand] += self.weights[name] * (1.0 / (self.rrf_k + rank))

        return self._sort_fused_scores(fused_scores)

    def _weighted_linear_fuse(
        self,
        per_retriever_scores: Dict[str, Dict[Candidate, float]],
    ) -> Tuple[List[int], List[float]]:
        fused_scores: Dict[Candidate, float] = defaultdict(float)
        normalized_scores = {
            name: self.normalize(scores)
            for name, scores in per_retriever_scores.items()
        }
        rank_scores = {
            name: self._rank_score_map(scores)
            for name, scores in per_retriever_scores.items()
        }
        all_candidates = sorted({cand for scores in normalized_scores.values() for cand in scores})
        total_sources = max(len(self.active_retrievers), 1)

        for cand in all_candidates:
            source_hits = 0
            for name, scores in normalized_scores.items():
                if cand in scores:
                    source_hits += 1
                    blended_score = (
                        self.linear_score_weight * scores[cand]
                        + self.linear_rank_weight * rank_scores[name][cand]
                    )
                    fused_scores[cand] += self.weights[name] * blended_score
            fused_scores[cand] += self.linear_agreement_bonus * (source_hits / total_sources)

        return self._sort_fused_scores(fused_scores)

    def _validate_raw_scores(
        self,
        raw_scores: Dict[str, Dict[Candidate, float]],
    ) -> Dict[str, Dict[Candidate, float]]:
        missing = [name for name in self.active_retrievers if name not in raw_scores]
        if missing:
            raise ValueError(f"Missing score dictionaries for active retrievers: {missing}")

        return {
            name: {
                int(candidate): float(score)
                for candidate, score in raw_scores.get(name, {}).items()
            }
            for name in self.active_retrievers
        }

    def _infer_active_retrievers(self) -> List[str]:
        return [name for name, weight in self.weights.items() if weight > 0]

    def _active_weights(self) -> Dict[str, float]:
        return {
            name: float(self.weights.get(name, 0.0))
            for name in self.active_retrievers
        }

    @staticmethod
    def _sort_fused_scores(fused_scores: Dict[Candidate, float]) -> Tuple[List[int], List[float]]:
        sorted_items = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return (
            [int(candidate) for candidate, _ in sorted_items],
            [float(score) for _, score in sorted_items],
        )

    @staticmethod
    def scores_to_ranks(scores: Dict[Candidate, float]) -> Dict[Candidate, int]:
        if not scores:
            return {}
        sorted_candidates = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return {
            int(candidate): rank
            for rank, (candidate, _) in enumerate(sorted_candidates, start=1)
        }

    @staticmethod
    def normalize(scores: Dict[Candidate, float]) -> Dict[Candidate, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val <= min_val:
            return {int(candidate): 1.0 for candidate in scores}
        return {
            int(candidate): (float(score) - min_val) / (max_val - min_val)
            for candidate, score in scores.items()
        }

    @staticmethod
    def _rank_score_map(scores: Dict[Candidate, float]) -> Dict[Candidate, float]:
        ranks = EnsembleRanker.scores_to_ranks(scores)
        if not ranks:
            return {}
        max_rank = max(ranks.values())
        if max_rank <= 1:
            return {candidate: 1.0 for candidate in ranks}
        return {
            candidate: 1.0 - ((rank - 1) / (max_rank - 1))
            for candidate, rank in ranks.items()
        }
