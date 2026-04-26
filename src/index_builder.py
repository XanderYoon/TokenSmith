#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
import json
import math
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer
from src.graph.store import GraphStore, save_graph_store

from src.preprocessing.chunking import DocumentChunker, ChunkConfig, NaiveRecursiveConfig
from src.preprocessing.extraction import extract_sections_from_markdown

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']
CHECKPOINT_VERSION = 1
SINGLE_PROCESS_EMBED_BATCH_SIZE = 8
MULTIPROCESS_EMBED_BATCH_SIZE = 32


def _checkpoint_paths(artifacts_dir: os.PathLike, index_prefix: str) -> Dict[str, pathlib.Path]:
    artifacts_dir = pathlib.Path(artifacts_dir)
    return {
        "state": artifacts_dir / f"{index_prefix}_ingestion_checkpoint.pkl",
        "embeddings": artifacts_dir / f"{index_prefix}_embeddings.partial.npy",
    }


def _save_checkpoint(path: pathlib.Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(state, handle)


def _load_checkpoint(path: pathlib.Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _clear_checkpoint_files(paths: Dict[str, pathlib.Path]) -> None:
    for path in paths.values():
        if path.exists():
            path.unlink()


def _build_chunk_checkpoint_state(
    *,
    markdown_file: str,
    chunk_config: ChunkConfig,
    use_headings: bool,
    sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "stage": "chunking",
        "markdown_file": markdown_file,
        "chunk_mode": chunk_config.to_string(),
        "use_headings": use_headings,
        "next_section_idx": 0,
        "total_sections": len(sections),
        "current_page": 1,
        "heading_stack": [],
        "all_chunks": [],
        "sources": [],
        "metadata": [],
        "page_to_chunk_ids": {},
    }


def _validate_checkpoint(
    checkpoint: Dict[str, Any] | None,
    *,
    markdown_file: str,
    chunk_config: ChunkConfig,
    use_headings: bool,
) -> Dict[str, Any] | None:
    if checkpoint is None:
        return None
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        return None
    if checkpoint.get("markdown_file") != markdown_file:
        return None
    if checkpoint.get("chunk_mode") != chunk_config.to_string():
        return None
    if checkpoint.get("use_headings") != use_headings:
        return None
    return checkpoint


def _finalize_page_map(page_to_chunk_ids: Dict[int, Any]) -> Dict[int, List[int]]:
    final_map = {}
    for page, id_set in page_to_chunk_ids.items():
        final_map[int(page)] = sorted(int(chunk_id) for chunk_id in id_set)
    return final_map


def _write_base_artifacts(
    *,
    all_chunks: List[str],
    sources: List[str],
    metadata: List[Dict[str, Any]],
    page_to_chunk_ids: Dict[int, Any],
    artifacts_dir: os.PathLike,
    index_prefix: str,
    graph_max_entities_per_chunk: int,
) -> None:
    final_map = _finalize_page_map(page_to_chunk_ids)
    output_file = pathlib.Path(artifacts_dir) / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page-to-chunk map: {output_file}")

    graph_store = GraphStore.from_chunks(
        all_chunks,
        max_entities_per_chunk=graph_max_entities_per_chunk,
    )
    graph_output = pathlib.Path(artifacts_dir) / f"{index_prefix}_graph.json"
    save_graph_store(graph_store, graph_output)
    print(f"Saved graph artifacts: {graph_output}")

    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved chunk metadata artifacts for {len(all_chunks):,} chunks.")


def _ingestion_timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_source_document_stats(markdown_file: str) -> Dict[str, Any]:
    source_path = pathlib.Path(markdown_file)
    if not source_path.exists():
        return {
            "source_file": markdown_file,
            "source_file_exists": False,
            "source_file_size_bytes": None,
            "source_document_char_count": None,
        }

    try:
        content = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = source_path.read_text(encoding="utf-8", errors="replace")

    return {
        "source_file": str(source_path),
        "source_file_exists": True,
        "source_file_size_bytes": int(source_path.stat().st_size),
        "source_document_char_count": len(content),
    }


def _classify_chunking_method(chunk_mode: str) -> str:
    lowered = chunk_mode.lower()
    if "section" in lowered or "structure" in lowered:
        return "structure_aware"
    if "recursive" in lowered:
        return "naive_recursive"
    return "custom"


def _estimate_token_cost(all_chunks: List[str]) -> Dict[str, Any]:
    total_characters = sum(len(chunk) for chunk in all_chunks)
    estimated_tokens = int(math.ceil(total_characters / 4.0))
    average_tokens_per_chunk = float(estimated_tokens / len(all_chunks)) if all_chunks else 0.0
    return {
        "estimation_method": "characters_divided_by_4",
        "estimated_total_tokens": estimated_tokens,
        "average_tokens_per_chunk": average_tokens_per_chunk,
    }


def _artifact_size_summary(artifacts_dir: os.PathLike, index_prefix: str) -> Dict[str, Any]:
    base_dir = pathlib.Path(artifacts_dir)
    files = sorted(base_dir.glob(f"{index_prefix}*"))
    artifact_files = {
        path.name: int(path.stat().st_size)
        for path in files
        if path.is_file()
    }
    return {
        "artifact_files": artifact_files,
        "artifact_directory_total_size_bytes": int(sum(artifact_files.values())),
    }


def _write_ingestion_metadata(
    *,
    markdown_file: str,
    chunk_config: ChunkConfig,
    all_chunks: List[str],
    artifacts_dir: os.PathLike,
    index_prefix: str,
    ingestion_started_at_utc: str,
    ingestion_duration_seconds: float | None = None,
) -> pathlib.Path:
    source_stats = _read_source_document_stats(markdown_file)
    chunks_path = pathlib.Path(artifacts_dir) / f"{index_prefix}_chunks.pkl"
    metadata_path = pathlib.Path(artifacts_dir) / f"{index_prefix}_ingestion_metadata.json"
    chunk_mode = chunk_config.to_string()
    artifact_summary = _artifact_size_summary(artifacts_dir, index_prefix)

    payload = {
        "schema_version": 1,
        "ingestion_started_at_utc": ingestion_started_at_utc,
        "ingestion_completed_at_utc": _ingestion_timestamp_utc(),
        "ingestion_duration_seconds": (
            float(round(ingestion_duration_seconds, 3))
            if ingestion_duration_seconds is not None else None
        ),
        "source": source_stats,
        "chunks": {
            "chunk_count": len(all_chunks),
            "artifact_file": chunks_path.name,
            "artifact_size_bytes": int(chunks_path.stat().st_size) if chunks_path.exists() else None,
        },
        "chunking": {
            "method_label": _classify_chunking_method(chunk_mode),
            "method_config": chunk_mode,
        },
        "estimated_token_cost": _estimate_token_cost(all_chunks),
        "artifacts": artifact_summary,
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved ingestion metadata: {metadata_path}")
    return metadata_path


def _batch_ranges(total_items: int, batch_size: int) -> List[tuple[int, int]]:
    return [
        (start_idx, min(start_idx + batch_size, total_items))
        for start_idx in range(0, total_items, batch_size)
    ]


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _progress_time_summary(
    *,
    processed_total: int,
    processed_at_start: int,
    total_items: int,
    start_time: float,
) -> str:
    elapsed_seconds = time.perf_counter() - start_time
    processed_this_run = max(0, processed_total - processed_at_start)
    if processed_this_run <= 0:
        return f"elapsed={_format_duration(elapsed_seconds)} eta=estimating..."

    avg_seconds_per_item = elapsed_seconds / processed_this_run
    remaining_items = max(0, total_items - processed_total)
    remaining_seconds = avg_seconds_per_item * remaining_items
    estimated_total_seconds = elapsed_seconds + remaining_seconds
    return (
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"eta={_format_duration(remaining_seconds)} "
        f"est_total={_format_duration(estimated_total_seconds)}"
    )


def _build_embeddings_with_checkpoint(
    *,
    all_chunks: List[str],
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool,
    chunk_checkpoint: Dict[str, Any],
    checkpoint_paths: Dict[str, pathlib.Path],
) -> np.ndarray:
    total_chunks = len(all_chunks)
    batch_size = MULTIPROCESS_EMBED_BATCH_SIZE if use_multiprocessing else SINGLE_PROCESS_EMBED_BATCH_SIZE
    batches = _batch_ranges(total_chunks, batch_size)
    total_batches = len(batches)

    print(f"Embedding {total_chunks:,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
    print(f"Embedding batch size: {batch_size} | total batches: {total_batches}")

    checkpoint = _validate_checkpoint(
        _load_checkpoint(checkpoint_paths["state"]),
        markdown_file=chunk_checkpoint["markdown_file"],
        chunk_config=_CheckpointChunkConfig(chunk_checkpoint["chunk_mode"]),
        use_headings=chunk_checkpoint["use_headings"],
    ) or chunk_checkpoint

    completed_batches = int(checkpoint.get("embedding_completed_batches", 0))
    embedder = SentenceTransformer(embedding_model_path)
    embedding_dim = int(checkpoint.get("embedding_dim") or embedder.embedding_dimension)
    checkpoint_compatible = (
        checkpoint.get("embedding_model_path") in {None, embedding_model_path}
        and int(checkpoint.get("embedding_batch_size", batch_size)) == batch_size
        and int(checkpoint.get("total_chunks", total_chunks)) == total_chunks
    )
    if not checkpoint_compatible:
        completed_batches = 0

    embeddings_path = checkpoint_paths["embeddings"]
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if not checkpoint_compatible or embeddings.shape != (total_chunks, embedding_dim):
            embeddings_path.unlink()
            embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_chunks, embedding_dim),
            )
            completed_batches = 0
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_chunks, embedding_dim),
        )
        completed_batches = 0

    checkpoint.update(
        {
            "stage": "embedding",
            "embedding_model_path": embedding_model_path,
            "embedding_batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "embedding_completed_batches": completed_batches,
            "total_chunks": total_chunks,
            "total_batches": total_batches,
        }
    )
    _save_checkpoint(checkpoint_paths["state"], checkpoint)

    if completed_batches >= total_batches:
        print(f"Embedding checkpoint already complete at batch {completed_batches}/{total_batches}.")
        return np.asarray(embeddings)

    print(
        f"Resuming embedding from batch {completed_batches + 1}/{total_batches}"
        if completed_batches > 0 else
        f"Starting embedding from batch 1/{total_batches}"
    )
    embedding_start_time = time.perf_counter()
    completed_batches_at_start = completed_batches

    progress = tqdm(
        total=total_batches,
        initial=completed_batches,
        desc="Embedding Batches",
        unit="batch",
    )

    pool = None
    try:
        if use_multiprocessing:
            print("Starting multi-process pool for embeddings...")
            pool = embedder.start_multi_process_pool(num_workers=4)

        for batch_idx in range(completed_batches, total_batches):
            start_idx, end_idx = batches[batch_idx]
            batch_texts = all_chunks[start_idx:end_idx]

            if use_multiprocessing and pool is not None:
                batch_embeddings = embedder.encode_batch_multi_process(batch_texts, pool)
            else:
                batch_embeddings = embedder.encode_batch(batch_texts)

            embeddings[start_idx:end_idx] = batch_embeddings
            if hasattr(embeddings, "flush"):
                embeddings.flush()

            checkpoint["embedding_completed_batches"] = batch_idx + 1
            _save_checkpoint(checkpoint_paths["state"], checkpoint)

            timing_summary = _progress_time_summary(
                processed_total=batch_idx + 1,
                processed_at_start=completed_batches_at_start,
                total_items=total_batches,
                start_time=embedding_start_time,
            )
            progress.set_postfix_str(
                f"batch={batch_idx + 1}/{total_batches} "
                f"chunks={start_idx + 1}-{end_idx}/{total_chunks} "
                f"{timing_summary}"
            )
            print(
                f"Embedding progress: batch {batch_idx + 1}/{total_batches} | "
                f"chunks {start_idx + 1}-{end_idx}/{total_chunks} | {timing_summary}"
            )
            progress.update(1)
    finally:
        progress.close()
        if pool is not None:
            embedder.stop_multi_process_pool(pool)

    return np.asarray(embeddings)


class _CheckpointChunkConfig:
    def __init__(self, mode: str):
        self._mode = mode

    def to_string(self) -> str:
        return self._mode

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    graph_max_entities_per_chunk: int,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
        - {prefix}_graph.json
        - {prefix}_ingestion_metadata.json
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = _checkpoint_paths(artifacts_dir, index_prefix)
    ingestion_started_at_utc = _ingestion_timestamp_utc()
    ingestion_started_monotonic = time.perf_counter()

    # Extract sections from markdown. Exclude some with certain keywords.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )
    if isinstance(chunk_config, NaiveRecursiveConfig):
        combined_content = "\n\n".join(
            section["content"]
            for section in sections
            if section.get("heading") != "Introduction"
        )
        sections = [
            {
                "heading": "Document",
                "content": combined_content,
                "level": 1,
                "chapter": 0,
            }
        ]

    checkpoint = _validate_checkpoint(
        _load_checkpoint(checkpoint_paths["state"]),
        markdown_file=markdown_file,
        chunk_config=chunk_config,
        use_headings=use_headings,
    )
    if checkpoint is None:
        checkpoint = _build_chunk_checkpoint_state(
            markdown_file=markdown_file,
            chunk_config=chunk_config,
            use_headings=use_headings,
            sections=sections,
        )
        print(f"Starting new ingestion build for {pathlib.Path(markdown_file).name}")
    else:
        print(
            f"Resuming ingestion checkpoint at stage '{checkpoint.get('stage', 'chunking')}' "
            f"for {pathlib.Path(markdown_file).name}"
        )

    all_chunks: List[str] = checkpoint["all_chunks"]
    sources: List[str] = checkpoint["sources"]
    metadata: List[Dict] = checkpoint["metadata"]
    page_to_chunk_ids = {
        int(page): set(chunk_ids)
        for page, chunk_ids in checkpoint.get("page_to_chunk_ids", {}).items()
    }
    current_page = int(checkpoint.get("current_page", 1))
    heading_stack = list(checkpoint.get("heading_stack", []))
    start_section_idx = int(checkpoint.get("next_section_idx", 0))

    print(f"Chunking progress: section {start_section_idx}/{len(sections)} complete.")
    chunking_start_time = time.perf_counter()
    completed_sections_at_start = start_section_idx

    # Step 1: Chunk using DocumentChunker
    chunk_progress = tqdm(
        range(start_section_idx, len(sections)),
        initial=start_section_idx,
        total=len(sections),
        desc="Chunking Sections",
        unit="section",
    )
    try:
        for section_idx in chunk_progress:
            c = sections[section_idx]
            # Determine current section level
            current_level = c.get('level', 1)

            # Determine current chapter number
            chapter_num = c.get('chapter', 0)

            # Pop sections that are deeper or siblings
            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()
            
            # Push pair of (level, heading)
            if c['heading'] != "Introduction":
                heading_stack.append((current_level, c['heading']))

            # Construct section path
            path_list = [h[1] for h in heading_stack]
            full_section_path = " ".join(path_list)
            full_section_path = f"Chapter {chapter_num} " + full_section_path

            # Use DocumentChunker to recursively split this section
            sub_chunks = chunker.chunk(c['content'])

            # Regex to find page markers like "--- Page 3 ---"
            page_pattern = re.compile(r'--- Page (\d+) ---')

            # Iterate through each chunk produced from this section
            for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
                # Track all pages this specific chunk touches
                chunk_pages = set()

                # Split the sub_chunk by page markers to see if it
                # spans multiple pages.
                fragments = page_pattern.split(sub_chunk)

                # If there is content before the first page marker,
                # it belongs to the current_page.
                if fragments[0].strip():
                    chunk_pages.add(current_page)

                # Process the new pages found within this sub_chunk. 
                # Step by 2 where each pair represents (page number, text after it)
                for fragment_idx in range(1, len(fragments), 2):
                    try:
                        # Get the new page number from the marker
                        new_page = int(fragments[fragment_idx]) + 1

                        # If there is text after this marker, it belongs to the new_page.
                        if fragments[fragment_idx + 1].strip():
                            chunk_pages.add(new_page)
                        
                        current_page = new_page

                    except (IndexError, ValueError):
                        continue

                # Clean sub_chunk by removing page markers
                clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()
                
                # Skip introduction chunks for embedding
                if c["heading"] == "Introduction":
                    continue

                chunk_idx = len(all_chunks)
                for page_number in chunk_pages:
                    page_to_chunk_ids.setdefault(page_number, set()).add(chunk_idx)
                
                # Prepare metadata
                meta = {
                    "filename": markdown_file,
                    "mode": chunk_config.to_string(),
                    "char_len": len(clean_chunk),
                    "word_len": len(clean_chunk.split()),
                    "section": c['heading'],
                    "section_path": full_section_path,
                    "text_preview": clean_chunk[:100],
                    "page_numbers": sorted(list(chunk_pages)),
                    "chunk_id": chunk_idx
                }

                # Prepare chunk with prefix
                if use_headings:
                    chunk_prefix = (
                        f"Description: {full_section_path} "
                        f"Content: "
                    )
                else:
                    chunk_prefix = ""

                all_chunks.append(chunk_prefix+clean_chunk)
                sources.append(markdown_file)
                metadata.append(meta)

            checkpoint["all_chunks"] = all_chunks
            checkpoint["sources"] = sources
            checkpoint["metadata"] = metadata
            checkpoint["page_to_chunk_ids"] = {page: sorted(ids) for page, ids in page_to_chunk_ids.items()}
            checkpoint["current_page"] = current_page
            checkpoint["heading_stack"] = heading_stack
            checkpoint["next_section_idx"] = section_idx + 1
            _save_checkpoint(checkpoint_paths["state"], checkpoint)

            timing_summary = _progress_time_summary(
                processed_total=section_idx + 1,
                processed_at_start=completed_sections_at_start,
                total_items=len(sections),
                start_time=chunking_start_time,
            )
            chunk_progress.set_postfix_str(
                f"section={section_idx + 1}/{len(sections)} "
                f"chunks={len(all_chunks)} "
                f"{timing_summary}"
            )
            print(
                f"Chunking progress: section {section_idx + 1}/{len(sections)} | "
                f"chunks={len(all_chunks):,} | {timing_summary}"
            )
    finally:
        chunk_progress.close()

    print(f"Chunking complete: {len(all_chunks):,} chunks extracted.")

    _write_base_artifacts(
        all_chunks=all_chunks,
        sources=sources,
        metadata=metadata,
        page_to_chunk_ids=page_to_chunk_ids,
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        graph_max_entities_per_chunk=graph_max_entities_per_chunk,
    )

    # Step 2: Create embeddings for FAISS index
    embeddings = _build_embeddings_with_checkpoint(
        all_chunks=all_chunks,
        embedding_model_path=embedding_model_path,
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        use_multiprocessing=use_multiprocessing,
        chunk_checkpoint=checkpoint,
        checkpoint_paths=checkpoint_paths,
    )

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")
    _clear_checkpoint_files(checkpoint_paths)
    _write_ingestion_metadata(
        markdown_file=markdown_file,
        chunk_config=chunk_config,
        all_chunks=all_chunks,
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        ingestion_started_at_utc=ingestion_started_at_utc,
        ingestion_duration_seconds=time.perf_counter() - ingestion_started_monotonic,
    )
    print(f"Ingestion complete. Saved all index artifacts with prefix: {index_prefix}")

# ------------------------ Helper functions ------------------------------

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens
