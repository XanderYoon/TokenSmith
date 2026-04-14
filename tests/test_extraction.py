from pathlib import Path
import src.preprocessing.extraction as extraction
import pytest

def test_extract_sections_preserves_subsections_within_parent_section(tmp_path):
    markdown = """Preface line

## 1.3 View of Data
Lead paragraph line 1
line 2

## 1.3.1 Data Models
Subsection paragraph

## 1.3.2 Database Languages
Another subsection paragraph

## 1.4 Database Users
Next top level section paragraph
"""
    md_path = tmp_path / "sample.md"
    md_path.write_text(markdown, encoding="utf-8")

    sections = extraction.extract_sections_from_markdown(str(md_path))

    assert len(sections) == 3
    assert sections[0]["heading"] == "Introduction"
    assert sections[1]["heading"] == "Section 1.3 View of Data"
    assert sections[1]["content"].startswith("Lead paragraph line 1 line 2")
    assert "## 1.3.1 Data Models" in sections[1]["content"]
    assert "## 1.3.2 Database Languages" in sections[1]["content"]
    assert sections[2]["heading"] == "Section 1.4 Database Users"
    assert "## 1.3.1 Data Models" not in sections[2]["content"]

def test_preprocess_extracted_section_preserves_structure():
    raw = """Paragraph first line
continues here

<!-- image -->

## 1.3.1 Data Models
Subsection text line 1
line 2

--- Page 56 ---

- bullet one
- bullet two
"""

    cleaned = extraction.preprocess_extracted_section(raw)

    assert "Paragraph first line continues here" in cleaned
    assert "<!-- image -->" not in cleaned
    assert "\n\n## 1.3.1 Data Models\n" in cleaned
    assert "Subsection text line 1 line 2" in cleaned
    assert "\n\n--- Page 56 ---\n" in cleaned
    assert "\n\n- bullet one\n- bullet two" in cleaned

def test_extract_no_pdfs_exits_with_error(tmp_path, monkeypatch, capsys):
    # Create temporary data/chapters/ with no PDFs
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "chapters").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc:
        extraction.main()

    # if no PDFs found, exit code should be 1
    assert exc.value.code == 1
    captured = capsys.readouterr()

    # error prints to stderr
    assert "No PDFs found in data/chapters/" in (captured.err + captured.out)

def test_extract_single_pdf_creates_one_md(tmp_path, monkeypatch):
    # Create temporary data/chapters/ with one PDF
    monkeypatch.chdir(tmp_path)
    chapters = tmp_path / "data" / "chapters"
    chapters.mkdir(parents=True)
    # Create a fake PDF file
    pdf_path = chapters / "Textbook.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake\n%%EOF\n")

    # Fake conversion process so we do not invoke docling in tests
    def fake_convert(input_file_path, output_file_path):
        assert Path(input_file_path).exists()
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file_path).write_text("## 1.1 Fake Section\ncontent\n", encoding="utf-8")
    # Replace the real conversion function
    monkeypatch.setattr(extraction, "convert_and_save_with_page_numbers", fake_convert)

    extraction.main()

    # Assert
    out_md = tmp_path / "data" / "Textbook--extracted_markdown.md"
    assert out_md.exists()
    assert out_md.read_text(encoding="utf-8").strip() != ""

    mds = list((tmp_path / "data").glob("*--extracted_markdown.md"))
    assert len(mds) == 1
    assert mds[0].name == "Textbook--extracted_markdown.md"

def test_extract_multiple_pdfs_creates_matching_mds(tmp_path, monkeypatch):
    # Create temporary data/chapters/ with multiple PDFs
    monkeypatch.chdir(tmp_path)
    chapters = tmp_path / "data" / "chapters"
    chapters.mkdir(parents=True)

    pdfs = [
        chapters / "chapter1.pdf",
        chapters / "blah2.pdf",
        chapters / "Z-last.pdf",
    ]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake\n%%EOF\n")

    def fake_convert(input_file_path, output_file_path):
        assert Path(input_file_path).exists()
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file_path).write_text("## 1 Fake\nx\n", encoding="utf-8")
    # Replace the real conversion function
    monkeypatch.setattr(extraction, "convert_and_save_with_page_numbers", fake_convert)

    extraction.main()

    # Assert
    expected = [
        tmp_path / "data" / "chapter1--extracted_markdown.md",
        tmp_path / "data" / "blah2--extracted_markdown.md",
        tmp_path / "data" / "Z-last--extracted_markdown.md",
    ]
    for md in expected:
        assert md.exists(), f"Missing output md: {md}"
    
    out_mds = list((tmp_path / "data").glob("*--extracted_markdown.md"))
    assert len(out_mds) == len(pdfs)

    out_names = sorted(p.name for p in out_mds)
    expected_names = sorted(p.name for p in expected)
    assert out_names == expected_names
