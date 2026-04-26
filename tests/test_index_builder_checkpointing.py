import json
import numpy as np
import pytest


pytestmark = pytest.mark.unit


class _DummyFaissIndex:
    def __init__(self, dim):
        self.dim = dim
        self.added = None

    def add(self, embeddings):
        self.added = embeddings


class _DummyBM25:
    def __init__(self, tokenized_chunks):
        self.tokenized_chunks = tokenized_chunks


def test_build_index_resumes_chunking_from_checkpoint(tmp_path, monkeypatch):
    from src.index_builder import build_index, _checkpoint_paths, _load_checkpoint

    sections = [
        {"heading": "Section A", "content": "alpha", "level": 1, "chapter": 1},
        {"heading": "Section B", "content": "beta", "level": 1, "chapter": 1},
        {"heading": "Section C", "content": "gamma", "level": 1, "chapter": 1},
    ]
    monkeypatch.setattr("src.index_builder.extract_sections_from_markdown", lambda *a, **k: sections)

    fail_once = {"enabled": True}
    call_counter = {"count": 0}

    class FlakyChunker:
        def chunk(self, content):
            call_counter["count"] += 1
            if fail_once["enabled"] and call_counter["count"] == 2:
                raise RuntimeError("chunking interrupted")
            return [f"{content} piece {idx}" for idx in range(4)]

    class DummyChunkConfig:
        def to_string(self):
            return "dummy"

    monkeypatch.setattr(
        "src.index_builder._build_embeddings_with_checkpoint",
        lambda **kwargs: np.ones((len(kwargs["all_chunks"]), 2), dtype=np.float32),
    )
    monkeypatch.setattr("src.index_builder.GraphStore.from_chunks", lambda *a, **k: object())
    monkeypatch.setattr("src.index_builder.save_graph_store", lambda *a, **k: None)
    monkeypatch.setattr("src.index_builder.BM25Okapi", _DummyBM25)
    monkeypatch.setattr("src.index_builder.faiss.IndexFlatL2", _DummyFaissIndex)
    monkeypatch.setattr("src.index_builder.faiss.write_index", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="chunking interrupted"):
        build_index(
            markdown_file="dummy.md",
            chunker=FlakyChunker(),
            chunk_config=DummyChunkConfig(),
            graph_max_entities_per_chunk=8,
            embedding_model_path="model.gguf",
            artifacts_dir=tmp_path,
            index_prefix="resume_test",
        )

    checkpoint = _load_checkpoint(_checkpoint_paths(tmp_path, "resume_test")["state"])
    assert checkpoint is not None
    assert checkpoint["next_section_idx"] == 1
    assert len(checkpoint["all_chunks"]) == 4

    fail_once["enabled"] = False
    build_index(
        markdown_file="dummy.md",
        chunker=FlakyChunker(),
        chunk_config=DummyChunkConfig(),
        graph_max_entities_per_chunk=8,
        embedding_model_path="model.gguf",
        artifacts_dir=tmp_path,
        index_prefix="resume_test",
    )

    assert call_counter["count"] == 4
    assert (tmp_path / "resume_test_chunks.pkl").exists()
    assert (tmp_path / "resume_test_bm25.pkl").exists()
    assert not _checkpoint_paths(tmp_path, "resume_test")["state"].exists()


def test_embedding_checkpoint_resumes_from_last_completed_batch(tmp_path, monkeypatch):
    from src.index_builder import _build_embeddings_with_checkpoint, _checkpoint_paths, _load_checkpoint

    class DummyChunkConfig:
        def to_string(self):
            return "dummy"

    batch_calls = []
    fail_on_second_batch = {"enabled": True}

    class FakeSentenceTransformer:
        def __init__(self, model_path):
            self.model_path = model_path
            self.embedding_dimension = 2

        def encode_batch(self, texts):
            batch_calls.append(list(texts))
            if fail_on_second_batch["enabled"] and len(batch_calls) == 2:
                raise RuntimeError("embedding interrupted")
            return np.array([[float(len(text)), float(len(text)) + 1.0] for text in texts], dtype=np.float32)

    monkeypatch.setattr("src.index_builder.SentenceTransformer", FakeSentenceTransformer)

    checkpoint_paths = _checkpoint_paths(tmp_path, "embed_resume")
    chunk_checkpoint = {
        "version": 1,
        "stage": "chunking",
        "markdown_file": "dummy.md",
        "chunk_mode": DummyChunkConfig().to_string(),
        "use_headings": False,
    }
    chunks = [f"chunk-{idx}" for idx in range(10)]

    with pytest.raises(RuntimeError, match="embedding interrupted"):
        _build_embeddings_with_checkpoint(
            all_chunks=chunks,
            embedding_model_path="model.gguf",
            artifacts_dir=tmp_path,
            index_prefix="embed_resume",
            use_multiprocessing=False,
            chunk_checkpoint=chunk_checkpoint,
            checkpoint_paths=checkpoint_paths,
        )

    checkpoint = _load_checkpoint(checkpoint_paths["state"])
    assert checkpoint["embedding_completed_batches"] == 1
    assert checkpoint_paths["embeddings"].exists()

    partial = np.load(checkpoint_paths["embeddings"])
    assert partial.shape == (10, 2)
    assert np.allclose(partial[:8], np.array([[7.0, 8.0]] * 8, dtype=np.float32))

    fail_on_second_batch["enabled"] = False
    batch_calls.clear()
    embeddings = _build_embeddings_with_checkpoint(
        all_chunks=chunks,
        embedding_model_path="model.gguf",
        artifacts_dir=tmp_path,
        index_prefix="embed_resume",
        use_multiprocessing=False,
        chunk_checkpoint=chunk_checkpoint,
        checkpoint_paths=checkpoint_paths,
    )

    assert len(batch_calls) == 1
    assert batch_calls[0] == ["chunk-8", "chunk-9"]
    assert embeddings.shape == (10, 2)


def test_build_index_saves_ingestion_metadata_manifest(tmp_path, monkeypatch):
    from src.index_builder import build_index

    markdown_path = tmp_path / "source.md"
    markdown_path.write_text("# Chapter 1\nalpha beta gamma delta\n", encoding="utf-8")

    sections = [
        {"heading": "Section A", "content": "alpha beta", "level": 1, "chapter": 1},
        {"heading": "Section B", "content": "gamma delta", "level": 1, "chapter": 1},
    ]
    monkeypatch.setattr("src.index_builder.extract_sections_from_markdown", lambda *a, **k: sections)

    class StableChunker:
        def chunk(self, content):
            return [content, f"{content} extra"]

    class SectionStyleChunkConfig:
        def to_string(self):
            return "chunk_mode=sections+recursive, chunk_size=100, overlap=10"

    monkeypatch.setattr(
        "src.index_builder._build_embeddings_with_checkpoint",
        lambda **kwargs: np.ones((len(kwargs["all_chunks"]), 2), dtype=np.float32),
    )
    monkeypatch.setattr("src.index_builder.GraphStore.from_chunks", lambda *a, **k: object())
    monkeypatch.setattr("src.index_builder.save_graph_store", lambda *a, **k: None)
    monkeypatch.setattr("src.index_builder.BM25Okapi", _DummyBM25)
    monkeypatch.setattr("src.index_builder.faiss.IndexFlatL2", _DummyFaissIndex)
    monkeypatch.setattr("src.index_builder.faiss.write_index", lambda *a, **k: None)

    build_index(
        markdown_file=str(markdown_path),
        chunker=StableChunker(),
        chunk_config=SectionStyleChunkConfig(),
        graph_max_entities_per_chunk=8,
        embedding_model_path="model.gguf",
        artifacts_dir=tmp_path,
        index_prefix="manifest_test",
    )

    manifest_path = tmp_path / "manifest_test_ingestion_metadata.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["source"]["source_file"] == str(markdown_path)
    assert payload["source"]["source_file_size_bytes"] == markdown_path.stat().st_size
    assert payload["chunks"]["chunk_count"] == 4
    assert payload["chunks"]["artifact_file"] == "manifest_test_chunks.pkl"
    assert payload["chunks"]["artifact_size_bytes"] is not None
    assert payload["chunking"]["method_label"] == "structure_aware"
    assert "sections+recursive" in payload["chunking"]["method_config"]
    assert payload["estimated_token_cost"]["estimation_method"] == "characters_divided_by_4"
    assert payload["estimated_token_cost"]["estimated_total_tokens"] > 0
    assert payload["ingestion_started_at_utc"]
    assert payload["ingestion_completed_at_utc"]
    assert payload["ingestion_duration_seconds"] is not None
    assert payload["artifacts"]["artifact_directory_total_size_bytes"] > 0
    assert "manifest_test_bm25.pkl" in payload["artifacts"]["artifact_files"]


def test_progress_time_summary_reports_elapsed_and_eta(monkeypatch):
    from src.index_builder import _progress_time_summary

    monkeypatch.setattr("src.index_builder.time.perf_counter", lambda: 16.0)

    summary = _progress_time_summary(
        processed_total=3,
        processed_at_start=0,
        total_items=6,
        start_time=10.0,
    )

    assert "elapsed=6s" in summary
    assert "eta=6s" in summary
    assert "est_total=12s" in summary
