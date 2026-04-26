from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

from src.generator import answer, dedupe_generated_text, double_answer, format_prompt
from src.tuning.backend import TuningSession, load_dataset_records, run_tuning_query


DEFAULT_RAGAS_METRICS = ("faithfulness", "answer_relevancy")
GENERATION_CONTEXT_WINDOW_TOKENS = 4096
GENERATION_CONTEXT_SAFETY_MARGIN_TOKENS = 64
_RAGAS_EVALUATOR_SESSION: TuningSession | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _estimate_tokens(text: str) -> int:
    return max(0, (len(text) + 3) // 4)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _dataset_metadata(
    dataset_path: Path,
    all_records: Sequence[Dict[str, Any]],
    used_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    timestamps = [
        str(record.get("timestamp_utc"))
        for record in all_records
        if record.get("timestamp_utc")
    ]
    return {
        "dataset_path": str(dataset_path),
        "dataset_records_total": len(all_records),
        "dataset_records_used": len(used_records),
        "dataset_created_range": {
            "first_record_utc": min(timestamps) if timestamps else None,
            "last_record_utc": max(timestamps) if timestamps else None,
        },
    }


def _generate_answer_from_contexts(
    session: TuningSession,
    *,
    query: str,
    contexts: Sequence[str],
    system_prompt_mode: str,
    use_double_prompt: bool,
) -> str:
    bounded_contexts, max_generation_tokens = _fit_contexts_to_generation_budget(
        session,
        query=query,
        contexts=contexts,
        system_prompt_mode=system_prompt_mode,
    )
    stream = (
        double_answer(
            query,
            bounded_contexts,
            session.cfg.gen_model,
            max_tokens=max_generation_tokens,
            system_prompt_mode=system_prompt_mode,
        )
        if use_double_prompt
        else answer(
            query,
            bounded_contexts,
            session.cfg.gen_model,
            max_tokens=max_generation_tokens,
            system_prompt_mode=system_prompt_mode,
        )
    )
    return dedupe_generated_text("".join(stream))


def _fit_contexts_to_generation_budget(
    session: TuningSession,
    *,
    query: str,
    contexts: Sequence[str],
    system_prompt_mode: str,
) -> tuple[List[str], int]:
    max_generation_tokens = min(
        int(session.cfg.max_gen_tokens),
        GENERATION_CONTEXT_WINDOW_TOKENS // 2,
    )
    prompt_budget = (
        GENERATION_CONTEXT_WINDOW_TOKENS
        - max_generation_tokens
        - GENERATION_CONTEXT_SAFETY_MARGIN_TOKENS
    )

    fitted_contexts: List[str] = []
    for context in contexts:
        candidate_contexts = fitted_contexts + [str(context)]
        prompt = format_prompt(
            candidate_contexts,
            query,
            system_prompt_mode=system_prompt_mode,
        )
        if _estimate_tokens(prompt) <= prompt_budget:
            fitted_contexts = candidate_contexts
            continue
        if not fitted_contexts:
            remaining_chars = max(prompt_budget * 4, 256)
            truncated_context = str(context)[:remaining_chars]
            fitted_contexts = [truncated_context]
        break

    prompt_tokens = _estimate_tokens(
        format_prompt(
            fitted_contexts,
            query,
            system_prompt_mode=system_prompt_mode,
        )
    )
    available_generation_tokens = max(
        64,
        GENERATION_CONTEXT_WINDOW_TOKENS
        - prompt_tokens
        - GENERATION_CONTEXT_SAFETY_MARGIN_TOKENS,
    )
    return fitted_contexts, min(max_generation_tokens, available_generation_tokens)


def _retrieval_overlap(
    retrieved_chunk_ids: Sequence[int],
    selected_chunk_ids: Sequence[int],
) -> Dict[str, Any]:
    retrieved = {int(chunk_id) for chunk_id in retrieved_chunk_ids}
    selected = {int(chunk_id) for chunk_id in selected_chunk_ids}
    overlap = retrieved & selected
    recall = float(len(overlap) / len(selected)) if selected else 0.0
    precision = float(len(overlap) / len(retrieved)) if retrieved else 0.0
    return {
        "retrieved_chunk_ids": sorted(retrieved),
        "selected_chunk_ids": sorted(selected),
        "overlap_chunk_ids": sorted(overlap),
        "selected_recall_at_k": recall,
        "selected_precision_at_k": precision,
    }


def _resolve_ragas_metrics(metric_names: Sequence[str]) -> tuple[Any, List[Any], List[str]]:
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "ragas is not installed in this environment. Install ragas and datasets to run this benchmark."
        ) from exc

    resolved_names = [str(name).strip().lower() for name in metric_names if str(name).strip()]
    if not resolved_names:
        resolved_names = list(DEFAULT_RAGAS_METRICS)

    try:
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            NonLLMContextRecall,
            NonLLMContextPrecisionWithReference,
            RougeScore,
            BleuScore,
            SemanticSimilarity,
        )

        metric_map = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "response_relevancy": ResponseRelevancy,
            "nonllm_context_recall": NonLLMContextRecall,
            "nonllm_context_precision": NonLLMContextPrecisionWithReference,
            "nonllm_context_precision_with_reference": NonLLMContextPrecisionWithReference,
            "rouge": RougeScore,
            "rouge_score": RougeScore,
            "bleu": BleuScore,
            "bleu_score": BleuScore,
            "semantic_similarity": SemanticSimilarity,
        }
        metrics = [metric_map[name]() for name in resolved_names]
        return evaluate, metrics, resolved_names
    except ImportError:
        try:
            from ragas.metrics import (
                answer_relevancy,
                faithfulness,
                non_llm_context_recall,
                non_llm_context_precision_with_reference,
                rouge_score,
                bleu_score,
                semantic_similarity,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Could not resolve compatible RAGAS metrics for this installed ragas version."
            ) from exc

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "response_relevancy": answer_relevancy,
            "nonllm_context_recall": non_llm_context_recall,
            "nonllm_context_precision": non_llm_context_precision_with_reference,
            "nonllm_context_precision_with_reference": non_llm_context_precision_with_reference,
            "rouge": rouge_score,
            "rouge_score": rouge_score,
            "bleu": bleu_score,
            "bleu_score": bleu_score,
            "semantic_similarity": semantic_similarity,
        }
        metrics = [metric_map[name] for name in resolved_names]
        return evaluate, metrics, resolved_names


def _evaluate_with_ragas(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_names: Sequence[str],
) -> Dict[str, Any]:
    evaluate, metrics, resolved_metric_names = _resolve_ragas_metrics(metric_names)
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not installed in this environment. Install datasets to run RAGAS evaluation."
        ) from exc

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
            "ground_truth": [row["reference_answer"] for row in rows],
            "reference_contexts": [row.get("reference_contexts", []) for row in rows],
        }
    )
    evaluate_kwargs: Dict[str, Any] = {}
    metrics_requiring_local_models = {
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "semantic_similarity",
    }
    if _RAGAS_EVALUATOR_SESSION is not None and any(
        name in metrics_requiring_local_models for name in resolved_metric_names
    ):
        evaluator_llm, evaluator_embeddings = _build_ragas_generator_models(_RAGAS_EVALUATOR_SESSION)
        evaluate_kwargs["llm"] = evaluator_llm
        evaluate_kwargs["embeddings"] = evaluator_embeddings

    result = evaluate(dataset, metrics=metrics, **evaluate_kwargs)

    per_prompt: List[Dict[str, Any]] = []
    aggregated: Dict[str, float] = {}
    column_aliases = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "response_relevancy"],
        "response_relevancy": ["response_relevancy", "answer_relevancy"],
        "nonllm_context_recall": ["non_llm_context_recall", "nonllm_context_recall"],
        "nonllm_context_precision": [
            "non_llm_context_precision_with_reference",
            "nonllm_context_precision",
            "nonllm_context_precision_with_reference",
        ],
        "nonllm_context_precision_with_reference": [
            "non_llm_context_precision_with_reference",
            "nonllm_context_precision_with_reference",
            "nonllm_context_precision",
        ],
        "rouge": ["rouge", "rouge_score"],
        "rouge_score": ["rouge_score", "rouge"],
        "bleu": ["bleu", "bleu_score"],
        "bleu_score": ["bleu_score", "bleu"],
        "semantic_similarity": ["semantic_similarity"],
    }

    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        metric_columns: Dict[str, str] = {}
        for name in resolved_metric_names:
            for candidate in column_aliases.get(name, [name]):
                if candidate in dataframe.columns:
                    metric_columns[name] = candidate
                    break
        for _, row in dataframe.iterrows():
            per_prompt.append(
                {
                    name: float(row[column])
                    for name, column in metric_columns.items()
                    if row[column] is not None
                }
            )
        aggregated = {
            name: float(dataframe[column].mean())
            for name, column in metric_columns.items()
            if column in dataframe
        }
    elif isinstance(result, dict):
        aggregated = {
            str(name): float(value)
            for name, value in result.items()
            if isinstance(value, (int, float))
        }
    else:
        for name in resolved_metric_names:
            value = getattr(result, name, None)
            if isinstance(value, (int, float)):
                aggregated[name] = float(value)

    return {
        "status": "completed",
        "metric_names": resolved_metric_names,
        "aggregated_scores": aggregated,
        "per_prompt_scores": per_prompt,
    }


def _benchmark_filename(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip()) or "benchmark"
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe}.json"


def _automated_probe_filename(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip()) or "automated_probes"
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe}.jsonl"


def _ollama_model_available(model_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return model_name in result.stdout


def _preferred_ollama_generation_model() -> Optional[str]:
    for model_name in (
        "qwen2.5:14b",
        "qwen2.5:7b-instruct-q4_0",
        "qwen3:8b",
    ):
        if _ollama_model_available(model_name):
            return model_name
    return None


def _build_ragas_generator_models(session: TuningSession) -> tuple[Any, Any]:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.llms import Ollama

        generation_model = _preferred_ollama_generation_model()
        if generation_model and _ollama_model_available("nomic-embed-text:latest"):
            return (
                Ollama(model=generation_model, temperature=0.1),
                OllamaEmbeddings(model="nomic-embed-text:latest"),
            )
    except ImportError:
        pass

    from langchain_community.embeddings import LlamaCppEmbeddings
    from langchain_community.llms import LlamaCpp

    return (
        LlamaCpp(
            model_path=session.cfg.gen_model,
            n_ctx=4096,
            max_tokens=min(512, int(session.cfg.max_gen_tokens)),
            temperature=0.2,
            verbose=False,
        ),
        LlamaCppEmbeddings(
            model_path=session.cfg.embed_model,
            n_ctx=4096,
            verbose=False,
        ),
    )


def _resolve_reference_chunk_ids(
    session: TuningSession,
    *,
    reference_contexts: Sequence[str],
    reference_context_ids: Sequence[Any],
) -> List[int]:
    resolved_ids: List[int] = []

    for raw_id in reference_context_ids:
        try:
            chunk_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if 0 <= chunk_id < len(session.chunks):
            resolved_ids.append(chunk_id)

    if resolved_ids:
        return sorted(dict.fromkeys(resolved_ids))

    chunk_id_by_text: Dict[str, int] = {}
    for idx, chunk_text in enumerate(session.chunks):
        chunk_id_by_text.setdefault(str(chunk_text), idx)

    for context in reference_contexts:
        chunk_id = chunk_id_by_text.get(str(context))
        if chunk_id is not None:
            resolved_ids.append(chunk_id)

    return sorted(dict.fromkeys(resolved_ids))


def _dataset_records_from_ragas_samples(
    session: TuningSession,
    samples: Sequence[Dict[str, Any]],
    *,
    session_id: str = "ragas_automated",
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for sample in samples:
        query = str(sample.get("user_input", "")).strip()
        if not query:
            continue

        reference_contexts = [
            str(context).strip()
            for context in sample.get("reference_contexts", []) or []
            if str(context).strip()
        ]
        reference_context_ids = sample.get("reference_context_ids", []) or []
        selected_chunk_ids = _resolve_reference_chunk_ids(
            session,
            reference_contexts=reference_contexts,
            reference_context_ids=reference_context_ids,
        )

        selected_chunks = []
        for idx, context in enumerate(reference_contexts):
            chunk_id = selected_chunk_ids[idx] if idx < len(selected_chunk_ids) else None
            selected_chunks.append(
                {
                    "chunk_id": int(chunk_id) if chunk_id is not None else -1,
                    "chunk_text": context,
                    "metadata": {
                        "source": "ragas_reference_context",
                        "synthesizer_name": sample.get("synthesizer_name"),
                        "reference_answer": sample.get("reference"),
                        "query_style": sample.get("query_style"),
                        "query_length": sample.get("query_length"),
                    },
                }
            )

        records.append(
            {
                "schema_version": 1,
                "timestamp_utc": _utc_now_iso(),
                "session_id": session_id,
                "query": query,
                "top_k": int(session.cfg.top_k),
                "weights": {
                    name: float(session.model_state.weights.get(name, 0.0))
                    for name in session.model_state.weights
                },
                "selected_chunk_ids": selected_chunk_ids,
                "rejected_chunk_ids": [],
                "selected_chunks": selected_chunks,
                "reference_answer": str(sample.get("reference", "")).strip(),
                "all_candidates": [],
                "automated_probe_metadata": {
                    "synthesizer_name": sample.get("synthesizer_name"),
                    "query_style": sample.get("query_style"),
                    "query_length": sample.get("query_length"),
                    "reference_context_count": len(reference_contexts),
                },
            }
        )

    return records


def generate_automated_probe_dataset(
    session: TuningSession,
    *,
    testset_size: int,
    dataset_label: str = "automated_ragas_probes",
    session_id: str = "ragas_automated",
) -> Path:
    try:
        from langchain_core.documents import Document
        from ragas.testset import TestsetGenerator
    except ImportError as exc:
        raise RuntimeError(
            "Automated RAGAS probe generation requires ragas, langchain, and local llama.cpp integrations."
        ) from exc

    docs = [
        Document(
            page_content=str(chunk),
            metadata={
                **dict(session.metadata[idx] or {}),
                "chunk_id": idx,
            },
        )
        for idx, chunk in enumerate(session.chunks)
    ]

    generator_llm, generator_embeddings = _build_ragas_generator_models(session)

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=int(testset_size),
        raise_exceptions=False,
    )
    samples = testset.to_list()
    dataset_records = _dataset_records_from_ragas_samples(
        session,
        samples,
        session_id=session_id,
    )
    if not dataset_records:
        raise ValueError("RAGAS did not generate any usable automated probe records.")

    automated_dir = session.demonstrations_path.parent / "automated_datasets"
    dataset_path = automated_dir / _automated_probe_filename(dataset_label)
    raw_testset_path = dataset_path.with_suffix(".raw.jsonl")
    _atomic_write_jsonl(dataset_path, dataset_records)
    _atomic_write_jsonl(raw_testset_path, list(samples))
    return dataset_path


@dataclass
class BenchmarkPromptResult:
    query: str
    answer: str
    contexts: List[str]
    selected_contexts: List[str]
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    total_latency_seconds: float
    estimated_prompt_tokens: int
    estimated_context_tokens: int
    estimated_answer_tokens: int
    estimated_total_tokens: int
    overlap: Dict[str, Any]
    ragas_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": list(self.contexts),
            "selected_contexts": list(self.selected_contexts),
            "retrieval_latency_seconds": float(self.retrieval_latency_seconds),
            "generation_latency_seconds": float(self.generation_latency_seconds),
            "total_latency_seconds": float(self.total_latency_seconds),
            "estimated_prompt_tokens": int(self.estimated_prompt_tokens),
            "estimated_context_tokens": int(self.estimated_context_tokens),
            "estimated_answer_tokens": int(self.estimated_answer_tokens),
            "estimated_total_tokens": int(self.estimated_total_tokens),
            "overlap": dict(self.overlap),
            "ragas_scores": {str(name): float(value) for name, value in self.ragas_scores.items()},
        }


@dataclass
class BenchmarkRunSummary:
    run_id: str
    output_path: Path
    metadata: Dict[str, Any]
    aggregated_metrics: Dict[str, Any]
    prompts: List[BenchmarkPromptResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_path": str(self.output_path),
            "metadata": dict(self.metadata),
            "aggregated_metrics": dict(self.aggregated_metrics),
            "prompts": [prompt.to_dict() for prompt in self.prompts],
        }


def load_benchmark_runs(benchmark_dir: Path) -> List[Dict[str, Any]]:
    if not benchmark_dir.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for path in sorted(benchmark_dir.glob("*.json"), reverse=True):
        payload = _load_json_if_exists(path)
        if payload is not None:
            runs.append(payload)
    return runs


def run_ragas_benchmark(
    session: TuningSession,
    *,
    top_k: int,
    system_prompt_mode: str,
    use_double_prompt: bool,
    max_examples: Optional[int] = None,
    ragas_metric_names: Optional[Sequence[str]] = None,
    benchmark_label: str = "ragas_benchmark",
    dataset_path: Optional[Path] = None,
) -> BenchmarkRunSummary:
    resolved_dataset_path = dataset_path or session.dataset_path
    dataset_records = load_dataset_records(resolved_dataset_path)
    if not dataset_records:
        raise ValueError("No saved prompt/chunk dataset records were found for this chunk set.")

    all_dataset_records = list(dataset_records)
    if max_examples is not None and max_examples > 0:
        dataset_records = dataset_records[:max_examples]

    started_at = _utc_now_iso()
    prompt_results: List[BenchmarkPromptResult] = []
    ragas_rows: List[Dict[str, Any]] = []

    for record in dataset_records:
        query = str(record.get("query", "")).strip()
        if not query:
            continue

        retrieval_started = time.perf_counter()
        query_result = run_tuning_query(session, query, top_k=top_k)
        retrieval_latency = time.perf_counter() - retrieval_started

        contexts = [candidate.chunk_text for candidate in query_result.candidates]
        generation_started = time.perf_counter()
        generated_answer = _generate_answer_from_contexts(
            session,
            query=query,
            contexts=contexts,
            system_prompt_mode=system_prompt_mode,
            use_double_prompt=use_double_prompt,
        )
        generation_latency = time.perf_counter() - generation_started
        total_latency = retrieval_latency + generation_latency

        selected_contexts = [
            str(chunk.get("chunk_text", ""))
            for chunk in record.get("selected_chunks", [])
            if chunk.get("chunk_text")
        ]
        overlap = _retrieval_overlap(
            [candidate.chunk_id for candidate in query_result.candidates],
            [int(chunk_id) for chunk_id in record.get("selected_chunk_ids", [])],
        )

        estimated_prompt_tokens = _estimate_tokens(query)
        estimated_context_tokens = sum(_estimate_tokens(context) for context in contexts)
        estimated_answer_tokens = _estimate_tokens(generated_answer)
        estimated_total_tokens = (
            estimated_prompt_tokens + estimated_context_tokens + estimated_answer_tokens
        )

        prompt_results.append(
            BenchmarkPromptResult(
                query=query,
                answer=generated_answer,
                contexts=contexts,
                selected_contexts=selected_contexts,
                retrieval_latency_seconds=retrieval_latency,
                generation_latency_seconds=generation_latency,
                total_latency_seconds=total_latency,
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_context_tokens=estimated_context_tokens,
                estimated_answer_tokens=estimated_answer_tokens,
                estimated_total_tokens=estimated_total_tokens,
                overlap=overlap,
            )
        )
        ragas_rows.append(
            {
                "question": query,
                "answer": generated_answer,
                "contexts": contexts,
                "reference_answer": "\n\n".join(selected_contexts),
                "reference_contexts": list(selected_contexts),
            }
        )

    if not prompt_results:
        raise ValueError("The saved dataset does not contain any valid prompts to benchmark.")

    ragas_status: Dict[str, Any]
    global _RAGAS_EVALUATOR_SESSION
    previous_evaluator_session = _RAGAS_EVALUATOR_SESSION
    _RAGAS_EVALUATOR_SESSION = session
    try:
        try:
            ragas_status = _evaluate_with_ragas(
                ragas_rows,
                metric_names=ragas_metric_names or DEFAULT_RAGAS_METRICS,
            )
            for prompt_result, score_row in zip(prompt_results, ragas_status.get("per_prompt_scores", [])):
                prompt_result.ragas_scores = {
                    str(name): float(value)
                    for name, value in score_row.items()
                    if isinstance(value, (int, float))
                }
        except Exception as exc:
            ragas_status = {
                "status": "failed",
                "metric_names": list(ragas_metric_names or DEFAULT_RAGAS_METRICS),
                "error": str(exc),
                "aggregated_scores": {},
                "per_prompt_scores": [],
            }
    finally:
        _RAGAS_EVALUATOR_SESSION = previous_evaluator_session

    benchmark_dir = session.demonstrations_path.parent / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    run_id = _benchmark_filename(benchmark_label).removesuffix(".json")
    output_path = benchmark_dir / f"{run_id}.json"
    try:
        ingestion_metadata_path = session.cfg.get_ingestion_metadata_path(
            session.index_prefix,
            artifacts_dir=session.demonstrations_path.parent.parent,
        )
    except TypeError:
        ingestion_metadata_path = session.cfg.get_ingestion_metadata_path(session.index_prefix)
    ingestion_metadata = _load_json_if_exists(ingestion_metadata_path)

    average_latency = sum(result.total_latency_seconds for result in prompt_results) / len(prompt_results)
    average_token_cost = sum(result.estimated_total_tokens for result in prompt_results) / len(prompt_results)
    average_selected_recall = sum(
        float(result.overlap.get("selected_recall_at_k", 0.0))
        for result in prompt_results
    ) / len(prompt_results)

    metadata = {
        "schema_version": 1,
        "run_started_at_utc": started_at,
        "run_completed_at_utc": _utc_now_iso(),
        "chunk_set": {
            "index_prefix": session.index_prefix,
            "chunking_method": ((ingestion_metadata or {}).get("chunking") or {}).get("method_label"),
            "chunking_config": ((ingestion_metadata or {}).get("chunking") or {}).get("method_config"),
            "source_file": ((ingestion_metadata or {}).get("source") or {}).get("source_file"),
        },
        "dataset": _dataset_metadata(resolved_dataset_path, all_dataset_records, dataset_records),
        "parameters": {
            "top_k": int(top_k),
            "system_prompt_mode": system_prompt_mode,
            "use_double_prompt": bool(use_double_prompt),
            "gen_model": session.cfg.gen_model,
            "max_gen_tokens": int(session.cfg.max_gen_tokens),
            "active_weight_source": session.active_weight_source,
            "weights": {name: float(weight) for name, weight in session.model_state.weights.items()},
            "ragas_metric_names": list(ragas_metric_names or DEFAULT_RAGAS_METRICS),
        },
    }
    aggregated_metrics = {
        "prompt_count": len(prompt_results),
        "average_latency_seconds": float(average_latency),
        "average_estimated_total_tokens": float(average_token_cost),
        "average_selected_recall_at_k": float(average_selected_recall),
        "ragas": ragas_status,
    }

    summary = BenchmarkRunSummary(
        run_id=run_id,
        output_path=output_path,
        metadata=metadata,
        aggregated_metrics=aggregated_metrics,
        prompts=prompt_results,
    )
    _atomic_write_json(output_path, summary.to_dict())
    return summary


def run_automated_ragas_benchmark(
    session: TuningSession,
    *,
    testset_size: int,
    top_k: int,
    system_prompt_mode: str,
    use_double_prompt: bool,
    ragas_metric_names: Optional[Sequence[str]] = None,
    dataset_label: str = "automated_ragas_probes",
    benchmark_label: str = "automated_ragas_benchmark",
) -> BenchmarkRunSummary:
    dataset_path = generate_automated_probe_dataset(
        session,
        testset_size=testset_size,
        dataset_label=dataset_label,
    )
    return run_ragas_benchmark(
        session,
        top_k=top_k,
        system_prompt_mode=system_prompt_mode,
        use_double_prompt=use_double_prompt,
        max_examples=testset_size,
        ragas_metric_names=ragas_metric_names,
        benchmark_label=benchmark_label,
        dataset_path=dataset_path,
    )
