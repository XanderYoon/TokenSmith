import pickle
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.graph import load_graph_artifact
from src.index_builder import build_index
from src.preprocessing.chunking import DocumentChunker, StructureAwareConfig, StructureAwareStrategy


class FakeEmbedder:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def encode(self, chunks, batch_size=8, show_progress_bar=True, convert_to_numpy=True):
        return np.zeros((len(chunks), 4), dtype=np.float32)


def test_build_index_records_structure_aware_metadata(tmp_path):
    markdown = """Intro text

## 1.3 View of Data
Lead paragraph for the section.

## 1.3.1 Data Models
Subsection body line one
line two

--- Page 56 ---

## 1.3.2 Database Languages
Second subsection body.

## 1.4 Database Users
Plain paragraph in next section.
"""
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text(markdown, encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    chunker = DocumentChunker(
        strategy=StructureAwareStrategy(
            StructureAwareConfig(max_chunk_chars=500, oversize_fallback_overlap=0)
        ),
        keep_tables=True,
    )

    with patch("src.index_builder.SentenceTransformer", FakeEmbedder):
        build_index(
            markdown_file=str(markdown_file),
            chunker=chunker,
            chunk_config=StructureAwareConfig(
                max_chunk_chars=500, oversize_fallback_overlap=0
            ),
            embedding_model_path="fake-model.gguf",
            artifacts_dir=artifacts_dir,
            index_prefix="test_index",
            use_multiprocessing=False,
            use_headings=False,
        )

    meta_path = artifacts_dir / "test_index_meta.pkl"
    chunks_path = artifacts_dir / "test_index_chunks.pkl"
    page_map_path = artifacts_dir / "test_index_page_to_chunk_map.json"
    graph_path = artifacts_dir / "test_index_graph.json"

    metadata = pickle.loads(meta_path.read_bytes())
    chunks = pickle.loads(chunks_path.read_bytes())
    page_map = json.loads(page_map_path.read_text(encoding="utf-8"))
    graph_artifact = load_graph_artifact(graph_path)

    assert len(chunks) == 4
    assert [item["chunk_id"] for item in metadata] == [0, 1, 2, 3]
    assert metadata[0]["chunk_unit_type"] == "paragraph"
    assert metadata[0]["unit_heading"] is None
    assert metadata[1]["chunk_unit_type"] == "subsection"
    assert metadata[1]["unit_heading"] == "Section 1.3.1 Data Models"
    assert metadata[2]["unit_heading"] == "Section 1.3.2 Database Languages"
    assert metadata[3]["section"] == "Section 1.4 Database Users"
    assert metadata[1]["page_numbers"] == [1]
    assert metadata[2]["page_numbers"] == [57]
    assert "--- Page 56 ---" not in chunks[1]
    assert page_map == {"1": [0, 1], "57": [2, 3]}
    assert graph_artifact.document_id == "test_index"
    assert len(graph_artifact.chunk_links) >= 1
    assert any(link.chunk_id == 0 for link in graph_artifact.chunk_links)
    assert any(node.node_id == "entity:lead-paragraph-for-the-section" for node in graph_artifact.nodes)
    assert graph_artifact.chunk_links[0].metadata["section"] == "Section 1.3 View of Data"


def test_build_index_records_fallback_split_metadata(tmp_path):
    markdown = """## 1.3 View of Data
## 1.3.1 Data Models
Sentence one is long enough to require a split. Sentence two is also long enough to require a split.
"""
    markdown_file = tmp_path / "oversized.md"
    markdown_file.write_text(markdown, encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    chunker = DocumentChunker(
        strategy=StructureAwareStrategy(
            StructureAwareConfig(max_chunk_chars=60, oversize_fallback_overlap=0)
        ),
        keep_tables=True,
    )

    with patch("src.index_builder.SentenceTransformer", FakeEmbedder):
        build_index(
            markdown_file=str(markdown_file),
            chunker=chunker,
            chunk_config=StructureAwareConfig(
                max_chunk_chars=60, oversize_fallback_overlap=0
            ),
            embedding_model_path="fake-model.gguf",
            artifacts_dir=artifacts_dir,
            index_prefix="test_index",
            use_multiprocessing=False,
            use_headings=False,
        )

    metadata = pickle.loads((artifacts_dir / "test_index_meta.pkl").read_bytes())
    chunks = pickle.loads((artifacts_dir / "test_index_chunks.pkl").read_bytes())
    graph_artifact = load_graph_artifact(artifacts_dir / "test_index_graph.json")

    assert len(chunks) >= 2
    assert all(item["chunk_unit_type"] == "fallback_split" for item in metadata)
    assert all(item["unit_heading"] == "Section 1.3.1 Data Models" for item in metadata)
    assert all(item["chunk_id"] == idx for idx, item in enumerate(metadata))
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert graph_artifact.document_id == "test_index"
