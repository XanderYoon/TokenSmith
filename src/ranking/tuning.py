from __future__ import annotations

import argparse
import itertools
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import build_retrievers, load_artifacts


@dataclass(frozen=True)
class RetrievalBenchmark:
    benchmark_id: str
    question: str
    ideal_retrieved_chunks: List[int]


@dataclass(frozen=True)
class TuningResult:
    weights: Dict[str, float]
    mean_ndcg: float
    mean_recall: float
    objective: float


def load_retrieval_benchmarks(path: pathlib.Path) -> List[RetrievalBenchmark]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    benchmarks = []
    for item in payload.get("benchmarks", []):
        ideal_chunks = item.get("ideal_retrieved_chunks") or []
        if not ideal_chunks:
            continue
        benchmarks.append(
            RetrievalBenchmark(
                benchmark_id=str(item["id"]),
                question=str(item["question"]),
                ideal_retrieved_chunks=[int(chunk_id) for chunk_id in ideal_chunks],
            )
        )
    return benchmarks


def dcg_at_k(retrieved: Sequence[int], ideal: Sequence[int], k: int) -> float:
    ideal_set = set(ideal[:k])
    return sum(
        1.0 / _log2(rank + 2)
        for rank, chunk_id in enumerate(retrieved[:k])
        if chunk_id in ideal_set
    )


def ndcg_at_k(retrieved: Sequence[int], ideal: Sequence[int], k: int) -> float:
    if not ideal:
        return 0.0
    actual = dcg_at_k(retrieved, ideal, k)
    ideal_best = sum(1.0 / _log2(rank + 2) for rank in range(min(k, len(ideal))))
    if ideal_best == 0.0:
        return 0.0
    return actual / ideal_best


def recall_at_k(retrieved: Sequence[int], ideal: Sequence[int], k: int) -> float:
    if not ideal:
        return 0.0
    ideal_set = set(ideal)
    hits = sum(1 for chunk_id in retrieved[:k] if chunk_id in ideal_set)
    return hits / len(ideal_set)


def retrieval_objective(retrieved: Sequence[int], ideal: Sequence[int], k: int) -> Tuple[float, float, float]:
    ndcg = ndcg_at_k(retrieved, ideal, k)
    recall = recall_at_k(retrieved, ideal, k)
    objective = (0.7 * ndcg) + (0.3 * recall)
    return ndcg, recall, objective


def generate_weight_grid(
    sources: Sequence[str],
    step: float = 0.05,
    minimum_weight: float = 0.05,
) -> List[Dict[str, float]]:
    if not sources:
        return []

    units = int(round(1.0 / step))
    minimum_units = int(round(minimum_weight / step))
    weights = []

    for combo in itertools.product(range(minimum_units, units + 1), repeat=len(sources)):
        if sum(combo) != units:
            continue
        weights.append({
            source: weight_units / units
            for source, weight_units in zip(sources, combo)
        })

    return weights


def tune_retrieval_weights(
    cfg: RAGConfig,
    benchmarks: Sequence[RetrievalBenchmark],
    *,
    artifacts_dir: pathlib.Path,
    index_prefix: str,
    ensemble_method: str | None = None,
    step: float = 0.05,
    minimum_weight: float = 0.05,
) -> TuningResult:
    faiss_index, bm25_index, chunks, _, _ = load_artifacts(artifacts_dir, index_prefix)
    retrievers = build_retrievers(cfg, faiss_index=faiss_index, bm25_index=bm25_index)
    raw_scores_by_query = {
        benchmark.benchmark_id: {
            retriever.name: retriever.get_scores(benchmark.question, cfg.num_candidates, chunks)
            for retriever in retrievers
        }
        for benchmark in benchmarks
    }

    sources = [retriever.name for retriever in retrievers]
    best_result: TuningResult | None = None

    for weights in generate_weight_grid(sources, step=step, minimum_weight=minimum_weight):
        active_sources = [source for source, weight in weights.items() if weight > 0.0]
        ranker = EnsembleRanker(
            ensemble_method=ensemble_method or cfg.ensemble_method,
            weights=weights,
            active_retrievers=active_sources,
            rrf_k=int(cfg.rrf_k),
            normalization=cfg.score_normalization,
        )

        ndcgs: List[float] = []
        recalls: List[float] = []
        objectives: List[float] = []
        for benchmark in benchmarks:
            ordered_ids, _ = ranker.rank(raw_scores_by_query[benchmark.benchmark_id])
            ndcg, recall, objective = retrieval_objective(
                ordered_ids,
                benchmark.ideal_retrieved_chunks,
                cfg.top_k,
            )
            ndcgs.append(ndcg)
            recalls.append(recall)
            objectives.append(objective)

        result = TuningResult(
            weights=weights,
            mean_ndcg=sum(ndcgs) / len(ndcgs),
            mean_recall=sum(recalls) / len(recalls),
            objective=sum(objectives) / len(objectives),
        )
        if best_result is None or result.objective > best_result.objective:
            best_result = result

    if best_result is None:
        raise ValueError("No weight combinations were generated for tuning")
    return best_result


def _log2(value: int) -> float:
    import math

    return math.log2(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune retrieval weights against ideal_retrieved_chunks.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--minimum-weight", type=float, default=0.05)
    args = parser.parse_args()

    cfg = RAGConfig.from_yaml(args.config)
    benchmarks = load_retrieval_benchmarks(pathlib.Path(args.benchmarks))
    result = tune_retrieval_weights(
        cfg,
        benchmarks,
        artifacts_dir=pathlib.Path(cfg.get_artifacts_directory()),
        index_prefix=args.index_prefix,
        step=args.step,
        minimum_weight=args.minimum_weight,
    )
    print(
        {
            "weights": result.weights,
            "mean_ndcg": round(result.mean_ndcg, 4),
            "mean_recall": round(result.mean_recall, 4),
            "objective": round(result.objective, 4),
        }
    )


if __name__ == "__main__":
    main()
