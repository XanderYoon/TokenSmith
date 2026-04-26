from __future__ import annotations

import pathlib
import sys
from typing import List

# Allow `streamlit run src/tuning/streamlit_app.py` from the repo root.
_project_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from src.config import RAGConfig
from src.tuning.backend import (
    apply_training_example,
    dataset_record_from_query_result,
    export_ranker_settings,
    list_available_tuning_indices,
    load_tuning_session,
    reset_session_to_baseline,
    run_tuning_query,
    save_exported_ranker_settings,
    session_stats,
    training_example_from_query_result,
)
from src.tuning.benchmarking import DEFAULT_RAGAS_METRICS, load_benchmark_runs, run_ragas_benchmark
from src.tuning.contracts import DEFAULT_LEARNING_RATE, DEFAULT_TUNING_TOP_K, TUNING_WEIGHT_SOURCES
from src.tuning.ui_helpers import candidate_label, weight_table
from src.tuning.validation import validate_queries_against_baseline


def _default_config_path() -> pathlib.Path:
    return _project_root / "config" / "config.yaml"


def _load_cfg(config_path: str) -> RAGConfig:
    return RAGConfig.from_yaml(config_path)


@st.cache_resource(show_spinner=False)
def _cached_session(
    config_path: str,
    index_prefix: str,
    artifacts_dir: str,
    starting_weight_source: str,
):
    cfg = _load_cfg(config_path)
    return load_tuning_session(
        cfg,
        index_prefix=index_prefix,
        artifacts_dir=artifacts_dir or None,
        starting_weight_source=starting_weight_source,
    )


def _reload_session(config_path: str, index_prefix: str, artifacts_dir: str, starting_weight_source: str):
    _cached_session.clear()
    return _cached_session(config_path, index_prefix, artifacts_dir, starting_weight_source)


def _render_candidate(candidate) -> None:
    st.markdown(f"**Rank {candidate.rank}**")
    st.caption(f"Chunk ID: {candidate.chunk_id}")

    meta = candidate.metadata or {}
    meta_fields = []
    if meta.get("page_numbers"):
        meta_fields.append(f"Pages: {meta['page_numbers']}")
    if meta.get("section"):
        meta_fields.append(f"Section: {meta['section']}")
    if meta.get("section_path"):
        meta_fields.append(f"Path: {meta['section_path']}")
    if meta_fields:
        st.caption(" | ".join(meta_fields))

    st.code(candidate.chunk_text[:1200], language="text")
    st.dataframe(
        weight_table(candidate.raw_scores),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Current fused score: {candidate.fused_score:.6f}")


def _chunk_set_label(index_info: dict) -> str:
    parts = [index_info["index_prefix"]]
    if index_info.get("chunk_count") is not None:
        parts.append(f"{index_info['chunk_count']} chunks")
    if index_info.get("chunking_method"):
        parts.append(str(index_info["chunking_method"]))
    source_file = index_info.get("source_file")
    if source_file:
        parts.append(pathlib.Path(source_file).name)
    return " | ".join(parts)


def _chunk_set_key(index_info: dict) -> str:
    return f"{index_info.get('artifacts_dir', '')}::{index_info['index_prefix']}"


def main() -> None:
    st.set_page_config(page_title="TokenSmith Weight Tuner", layout="wide")
    st.title("TokenSmith Retrieval Weight Tuner")
    st.write(
        "Run retrieval with the current learned weights, inspect the top chunks, "
        "select the best chunk or chunks, and update the linear tuning model."
    )

    default_config = str(_default_config_path())
    st.session_state.setdefault("last_status", "")
    st.session_state.setdefault("selected_chunk_set_key", "")
    with st.sidebar:
        st.header("Session")
        config_path = st.text_input("Config Path", value=default_config)
        available_chunk_sets = []
        available_chunk_set_error = ""
        try:
            cfg_preview = _load_cfg(config_path)
            available_chunk_sets = list_available_tuning_indices(cfg_preview)
        except Exception as exc:
            available_chunk_set_error = str(exc)

        selected_chunk_set_key = st.session_state.get("selected_chunk_set_key", "")
        selected_artifacts_dir = ""
        if available_chunk_sets:
            available_keys = [_chunk_set_key(entry) for entry in available_chunk_sets]
            if selected_chunk_set_key not in available_keys:
                selected_chunk_set_key = available_keys[0]
            selected_chunk_set_key = st.selectbox(
                "Chunk Set",
                options=available_keys,
                index=available_keys.index(selected_chunk_set_key),
                format_func=lambda key: _chunk_set_label(
                    next(entry for entry in available_chunk_sets if _chunk_set_key(entry) == key)
                ),
                help="Choose which indexed chunk set to use for retrieval tuning and dataset collection.",
            )
            selected_info = next(
                entry for entry in available_chunk_sets if _chunk_set_key(entry) == selected_chunk_set_key
            )
            selected_discovered_prefix = selected_info["index_prefix"]
            selected_artifacts_dir = selected_info.get("artifacts_dir", "")
            st.caption(f"Selected chunk set: {_chunk_set_label(selected_info)}")
            index_prefix_override = st.text_input("Index Prefix Override", value="")
            index_prefix = index_prefix_override.strip() or selected_discovered_prefix
        else:
            selected_discovered_prefix = "textbook_index"
            index_prefix = st.text_input("Index Prefix", value=selected_discovered_prefix)
            if available_chunk_set_error:
                st.caption(f"Could not inspect available chunk sets: {available_chunk_set_error}")
            else:
                st.caption("No saved chunk sets were discovered yet. Enter an index prefix manually.")

        st.session_state["selected_chunk_set_key"] = selected_chunk_set_key
        starting_weight_source = st.selectbox(
            "Startup Weights",
            options=list(TUNING_WEIGHT_SOURCES),
            index=1,
            help="Choose whether the tuning app starts from baseline config weights or the last saved learned weights.",
        )
        session_id = st.text_input("Session ID", value="streamlit")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=DEFAULT_TUNING_TOP_K)
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=1.0,
            value=float(DEFAULT_LEARNING_RATE),
            step=0.01,
        )

        if st.button("Reload Session", use_container_width=True):
            try:
                _reload_session(config_path, index_prefix, selected_artifacts_dir, starting_weight_source)
                st.session_state.pop("last_result", None)
                st.session_state["last_status"] = "Reloaded tuning session."
                st.success("Reloaded tuning session.")
            except Exception as exc:
                st.error(f"Failed to reload session: {exc}")

        if st.button("Reset Learned Weights", use_container_width=True):
            try:
                session = _cached_session(config_path, index_prefix, selected_artifacts_dir, starting_weight_source)
                reset_session_to_baseline(session, persist=True)
                st.session_state.pop("last_result", None)
                st.session_state["last_status"] = session.last_status_message
                st.success("Reset learned weights to baseline config weights.")
            except Exception as exc:
                st.error(f"Failed to reset weights: {exc}")

        if st.button("Export Weights For Config", use_container_width=True):
            try:
                session = _cached_session(config_path, index_prefix, selected_artifacts_dir, starting_weight_source)
                export_path = save_exported_ranker_settings(session)
                st.session_state["last_status"] = f"Exported config-friendly ranker settings to {export_path}"
                st.success(f"Exported config-friendly ranker settings to {export_path}")
            except Exception as exc:
                st.error(f"Failed to export weights: {exc}")

    try:
        session = _cached_session(config_path, index_prefix, selected_artifacts_dir, starting_weight_source)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Expected artifacts include FAISS, BM25, chunk metadata, and graph artifacts for the selected index prefix.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load tuning session: {exc}")
        st.stop()

    stats = session_stats(session)
    if stats["last_status_message"]:
        st.info(stats["last_status_message"])
    if st.session_state.get("last_status"):
        st.success(st.session_state["last_status"])
        st.session_state["last_status"] = ""

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks", stats["num_chunks"])
    col2.metric("Demonstrations", stats["demonstrations_seen"])
    col3.metric("Index Prefix", stats["index_prefix"])
    col4.metric("Weight Source", stats["active_weight_source"])

    st.subheader("Current Weights")
    st.dataframe(weight_table(session.model_state.weights), use_container_width=True, hide_index=True)
    st.caption(f"Baseline config weights are kept separate from learned weights. Active source: `{stats['active_weight_source']}`.")
    with st.expander("Session Status"):
        st.write(f"Artifact directory: `{stats['artifact_dir']}`")
        st.write(f"Demonstration log: `{stats['demonstrations_path']}`")
        st.write(f"Prompt/chunk dataset: `{stats['dataset_path']}`")
        st.write(f"Benchmark directory: `{session.demonstrations_path.parent / 'benchmarks'}`")
        st.write(f"Learned weights file: `{stats['model_state_path']}`")
        st.write(f"Manifest file: `{stats['manifest_path']}`")
        st.write(f"Export file: `{stats['export_path']}`")
        st.write(f"Saved learned weights available: `{stats['learned_weights_loaded']}`")
        st.write(f"Demonstrations collected: `{stats['demonstrations_seen']}`")
        st.write(f"Dataset rows collected: `{stats['dataset_records']}`")
    with st.expander("Config Export Preview"):
        st.code(str(export_ranker_settings(session)), language="python")

    st.subheader("Validation")
    validation_queries = st.text_area(
        "Held-out Queries",
        placeholder="Enter one validation prompt per line to compare baseline vs learned weights.",
        key="validation_queries",
    )
    if st.button("Run Baseline vs Learned Validation", use_container_width=True):
        queries = [line.strip() for line in validation_queries.splitlines() if line.strip()]
        if not queries:
            st.warning("Enter at least one held-out query before running validation.")
        else:
            try:
                summary = validate_queries_against_baseline(session, queries, top_k=top_k)
                st.session_state["validation_summary"] = summary
                st.session_state["last_status"] = (
                    f"Ran validation on {summary.total_queries} held-out quer"
                    f"{'y' if summary.total_queries == 1 else 'ies'}."
                )
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

    validation_summary = st.session_state.get("validation_summary")
    if validation_summary:
        v1, v2 = st.columns(2)
        v1.metric("Validation Queries", validation_summary.total_queries)
        v2.metric("Top-1 Changed", validation_summary.top1_changed_queries)
        st.caption(f"Average overlap@{top_k}: {validation_summary.avg_overlap_at_k:.3f}")
        st.dataframe(
            [
                {
                    "query": result.query,
                    "top1_changed": result.top1_changed,
                    "overlap_at_k": result.overlap_at_k,
                    "baseline_top_ids": result.baseline_top_ids,
                    "learned_top_ids": result.learned_top_ids,
                }
                for result in validation_summary.per_query
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("RAGAS Benchmarking")
    st.write(
        "Run the saved prompt/chunk dataset for this chunk set through the current retrieval and generation "
        "pipeline, then persist latency, token-cost, overlap, and RAGAS metrics for later comparison."
    )
    if stats["dataset_records"] <= 0:
        st.info("Collect at least one prompt/chunk selection before running the benchmark.")
    else:
        benchmark_col1, benchmark_col2, benchmark_col3 = st.columns(3)
        with benchmark_col1:
            benchmark_label = st.text_input("Benchmark Label", value="ragas_benchmark")
        with benchmark_col2:
            benchmark_prompt_mode = st.selectbox(
                "Benchmark Prompt Mode",
                options=["baseline", "tutor", "concise", "detailed"],
                index=1,
            )
        with benchmark_col3:
            benchmark_max_examples = st.number_input(
                "Max Dataset Rows",
                min_value=1,
                max_value=max(1, int(stats["dataset_records"])),
                value=min(25, max(1, int(stats["dataset_records"]))),
                step=1,
            )

        benchmark_metric_names = st.multiselect(
            "RAGAS Metrics",
            options=["faithfulness", "answer_relevancy"],
            default=list(DEFAULT_RAGAS_METRICS),
            help="These metrics will be attempted with the installed ragas version. If ragas is unavailable, the run is still persisted with the failure status.",
        )
        benchmark_use_double_prompt = st.checkbox(
            "Use Double Prompt During Benchmark",
            value=bool(session.cfg.use_double_prompt),
        )

        if st.button("Run RAGAS Benchmark", use_container_width=True):
            try:
                summary = run_ragas_benchmark(
                    session,
                    top_k=top_k,
                    system_prompt_mode=benchmark_prompt_mode,
                    use_double_prompt=benchmark_use_double_prompt,
                    max_examples=int(benchmark_max_examples),
                    ragas_metric_names=benchmark_metric_names,
                    benchmark_label=benchmark_label,
                )
                st.session_state["benchmark_summary"] = summary.to_dict()
                st.session_state["last_status"] = (
                    f"Saved benchmark run {summary.run_id} to {summary.output_path}"
                )
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")

        benchmark_summary = st.session_state.get("benchmark_summary")
        if benchmark_summary:
            st.caption(f"Latest benchmark artifact: `{benchmark_summary['output_path']}`")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Benchmarked Prompts", benchmark_summary["aggregated_metrics"]["prompt_count"])
            metric_cols[1].metric(
                "Avg Latency (s)",
                f"{benchmark_summary['aggregated_metrics']['average_latency_seconds']:.3f}",
            )
            metric_cols[2].metric(
                "Avg Estimated Tokens",
                f"{benchmark_summary['aggregated_metrics']['average_estimated_total_tokens']:.1f}",
            )
            st.caption(
                f"Average selected recall@k: {benchmark_summary['aggregated_metrics']['average_selected_recall_at_k']:.3f}"
            )
            ragas_summary = benchmark_summary["aggregated_metrics"].get("ragas", {})
            if ragas_summary.get("status") == "completed":
                st.dataframe(
                    [
                        {"metric": name, "score": value}
                        for name, value in ragas_summary.get("aggregated_scores", {}).items()
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    f"RAGAS status: {ragas_summary.get('status', 'unknown')}. "
                    f"{ragas_summary.get('error', 'No additional details were recorded.')}"
                )

        recent_runs = load_benchmark_runs(session.demonstrations_path.parent / "benchmarks")[:5]
        if recent_runs:
            with st.expander("Recent Benchmark Runs"):
                st.dataframe(
                    [
                        {
                            "run_id": run.get("run_id"),
                            "completed_at_utc": run.get("metadata", {}).get("run_completed_at_utc"),
                            "prompt_count": run.get("aggregated_metrics", {}).get("prompt_count"),
                            "avg_latency_seconds": run.get("aggregated_metrics", {}).get("average_latency_seconds"),
                            "ragas_status": run.get("aggregated_metrics", {}).get("ragas", {}).get("status"),
                            "output_path": run.get("output_path"),
                        }
                        for run in recent_runs
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    query = st.text_area("Query", placeholder="Enter a prompt to retrieve the top 10 chunks...")
    run_clicked = st.button("Run Retrieval", type="primary")

    if run_clicked:
        if not query.strip():
            st.warning("Enter a query before running retrieval.")
        else:
            try:
                st.session_state["last_result"] = run_tuning_query(session, query.strip(), top_k=top_k)
                st.session_state["last_status"] = "Retrieved the current top chunks with the active weights."
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Run retrieval to inspect the current top chunks.")
        return

    st.subheader("Top Retrieved Chunks")
    selected_ids = st.multiselect(
        "Select the best chunk(s)",
        options=[candidate.chunk_id for candidate in result.candidates],
        format_func=lambda chunk_id: next(
            candidate_label(candidate)
            for candidate in result.candidates
            if candidate.chunk_id == chunk_id
        ),
    )

    candidate_cols = st.columns(2)
    for idx, candidate in enumerate(result.candidates):
        with candidate_cols[idx % 2]:
            with st.container(border=True):
                _render_candidate(candidate)

    if st.button("Submit Feedback And Update Weights", use_container_width=True):
        if not selected_ids:
            st.warning("Select at least one preferred chunk before submitting feedback.")
        else:
            try:
                example = training_example_from_query_result(
                    result,
                    selected_chunk_ids=selected_ids,
                    session_id=session_id,
                )
                dataset_record = dataset_record_from_query_result(
                    result,
                    selected_chunk_ids=selected_ids,
                    session_id=session_id,
                )
                update_summary = apply_training_example(
                    session,
                    example,
                    learning_rate=learning_rate,
                    dataset_record=dataset_record,
                    persist=True,
                )
                reranked = run_tuning_query(session, result.query, top_k=top_k)
                st.session_state["last_result"] = reranked
                st.session_state["last_status"] = session.last_status_message

                st.success("Updated the tuning model and persisted the new learned weights.")
                st.subheader("Training Summary")
                st.write(f"Pairwise comparisons applied: `{update_summary.pair_count}`")

                summary_cols = st.columns(2)
                with summary_cols[0]:
                    st.markdown("**Previous Weights**")
                    st.dataframe(
                        weight_table(update_summary.previous_weights),
                        use_container_width=True,
                        hide_index=True,
                    )
                with summary_cols[1]:
                    st.markdown("**Updated Weights**")
                    st.dataframe(
                        weight_table(update_summary.updated_weights),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.subheader("Re-ranked Top Chunks")
                for candidate in reranked.candidates:
                    with st.container(border=True):
                        _render_candidate(candidate)
            except Exception as exc:
                st.error(f"Failed to apply feedback: {exc}")


if __name__ == "__main__":
    main()
