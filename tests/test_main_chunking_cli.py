from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from src.config import RAGConfig
from src.main import _resolve_indexing_config, run_index_mode


pytestmark = pytest.mark.unit


def test_resolve_indexing_config_keeps_base_config_without_overrides():
    cfg = RAGConfig(chunk_mode="recursive_sections", chunk_size=2000, chunk_overlap=200)
    args = Namespace(chunking_method=None, chunk_size=None, chunk_overlap=None)

    resolved = _resolve_indexing_config(args, cfg)

    assert resolved is cfg


def test_resolve_indexing_config_switches_to_naive_mode():
    cfg = RAGConfig(chunk_mode="recursive_sections", chunk_size=2000, chunk_overlap=200)
    args = Namespace(chunking_method="naive", chunk_size=512, chunk_overlap=64)

    resolved = _resolve_indexing_config(args, cfg)

    assert resolved is not cfg
    assert resolved.chunk_mode == "recursive_naive"
    assert resolved.chunk_size == 512
    assert resolved.chunk_overlap == 64


def test_resolve_indexing_config_switches_to_structure_aware_mode():
    cfg = RAGConfig(chunk_mode="recursive_naive", chunk_size=900, chunk_overlap=50)
    args = Namespace(chunking_method="structure_aware", chunk_size=None, chunk_overlap=None)

    resolved = _resolve_indexing_config(args, cfg)

    assert resolved.chunk_mode == "recursive_sections"
    assert resolved.chunk_size == 900
    assert resolved.chunk_overlap == 50


def test_build_versioned_artifacts_directory_increments_versions(tmp_path, monkeypatch):
    cfg = RAGConfig(chunk_mode="recursive_sections")
    monkeypatch.setattr(cfg, "get_artifacts_root_directory", lambda: tmp_path)

    first = cfg.build_versioned_artifacts_directory("docs/Database Systems.md")
    second = cfg.build_versioned_artifacts_directory("docs/Database Systems.md")

    assert first.name == "database_systems_structure_1"
    assert second.name == "database_systems_structure_2"
    assert first.exists()
    assert second.exists()


def test_resolve_artifacts_directory_finds_versioned_directory(tmp_path, monkeypatch):
    cfg = RAGConfig(chunk_mode="recursive_naive")
    monkeypatch.setattr(cfg, "get_artifacts_root_directory", lambda: tmp_path)
    monkeypatch.setattr(cfg, "get_artifacts_directory", lambda: tmp_path / "legacy")

    target_dir = tmp_path / "textbook_recursive_3"
    target_dir.mkdir(parents=True)
    (target_dir / "unit_index_chunks.pkl").write_bytes(b"chunks")

    resolved = cfg.resolve_artifacts_directory("unit_index")

    assert resolved == target_dir


def test_run_index_mode_uses_explicit_markdown_file(tmp_path, monkeypatch):
    markdown_path = tmp_path / "chosen.md"
    markdown_path.write_text("# chosen", encoding="utf-8")

    cfg = RAGConfig(chunk_mode="recursive_sections")
    captured = {}

    monkeypatch.setattr(
        cfg,
        "build_versioned_artifacts_directory",
        lambda source_document: tmp_path / f"artifacts_for_{Path(source_document).stem}",
    )
    monkeypatch.setattr("src.main.build_index", lambda **kwargs: captured.update(kwargs))

    args = Namespace(
        keep_tables=False,
        chunking_method=None,
        chunk_size=None,
        chunk_overlap=None,
        markdown_file=str(markdown_path),
        index_prefix="unit_index",
        multiproc_indexing=False,
        embed_with_headings=False,
    )

    run_index_mode(args, cfg)

    assert captured["markdown_file"] == str(markdown_path)
    assert captured["index_prefix"] == "unit_index"
