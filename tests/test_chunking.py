from src.preprocessing.chunking import (
    DocumentChunker,
    StructureAwareConfig,
    StructureAwareStrategy,
)


def test_structure_aware_chunking_prefers_subsections():
    text = """Lead paragraph before subsections.

## 1.3.1 Data Models
Data models paragraph.

## 1.3.2 Database Languages
Database languages paragraph.
"""
    strategy = StructureAwareStrategy(StructureAwareConfig(max_chunk_chars=500))

    pieces = strategy.chunk_pieces(text)

    assert len(pieces) == 3
    assert pieces[0].metadata.unit_type == "paragraph"
    assert pieces[0].metadata.heading is None
    assert pieces[0].text == "Lead paragraph before subsections."
    assert pieces[1].metadata.unit_type == "subsection"
    assert pieces[1].metadata.heading == "Section 1.3.1 Data Models"
    assert pieces[1].text.startswith("## 1.3.1 Data Models")
    assert pieces[2].metadata.heading == "Section 1.3.2 Database Languages"


def test_structure_aware_chunking_falls_back_to_paragraphs():
    text = """First paragraph line one
continues here

--- Page 45 ---

Second paragraph starts on a new page.

- bullet one
- bullet two
"""
    strategy = StructureAwareStrategy(StructureAwareConfig(max_chunk_chars=500))

    pieces = strategy.chunk_pieces(text)

    assert len(pieces) == 3
    assert all(piece.metadata.unit_type == "paragraph" for piece in pieces)
    assert pieces[0].text == "First paragraph line one continues here"
    assert pieces[1].text == "--- Page 45 ---\nSecond paragraph starts on a new page."
    assert pieces[2].text == "- bullet one\n- bullet two"


def test_structure_aware_chunking_splits_only_oversized_unit():
    text = """## 1.3.1 Data Models
Sentence one is long enough to require a split. Sentence two is also long enough to require a split.
"""
    strategy = StructureAwareStrategy(
        StructureAwareConfig(max_chunk_chars=60, oversize_fallback_overlap=0)
    )

    pieces = strategy.chunk_pieces(text)

    assert len(pieces) >= 2
    assert all(piece.metadata.unit_type == "fallback_split" for piece in pieces)
    assert all(piece.metadata.heading == "Section 1.3.1 Data Models" for piece in pieces)
    assert all(len(piece.text) <= 60 for piece in pieces)


def test_structure_aware_chunking_preserves_full_coverage_without_subsections():
    text = """Alpha paragraph line one
line two

Beta paragraph line one
line two

- bullet one
- bullet two
"""
    strategy = StructureAwareStrategy(StructureAwareConfig(max_chunk_chars=500))

    pieces = strategy.chunk_pieces(text)

    assert [piece.metadata.unit_type for piece in pieces] == [
        "paragraph",
        "paragraph",
        "paragraph",
    ]
    assert [piece.text for piece in pieces] == [
        "Alpha paragraph line one line two",
        "Beta paragraph line one line two",
        "- bullet one\n- bullet two",
    ]


def test_document_chunker_restores_tables_for_structure_aware_chunks():
    text = """Paragraph intro.

<table><tr><td>cell</td></tr></table>

## 1.3.1 Data Models
Subsection body.
"""
    strategy = StructureAwareStrategy(StructureAwareConfig(max_chunk_chars=500))
    chunker = DocumentChunker(strategy=strategy, keep_tables=True)

    chunks = chunker.chunk(text)

    assert any("<table><tr><td>cell</td></tr></table>" in chunk for chunk in chunks)
