from pathlib import Path
import re
import json
from typing import List, Dict
import sys
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

TOP_LEVEL_HEADING_RE = re.compile(
    r"^## (?P<number>\d+\.\d+)\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)
PAGE_MARKER_RE = re.compile(r"^--- Page \d+ ---$")
HEADING_LINE_RE = re.compile(r"^#{1,6}\s+")
LIST_LINE_RE = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)")

def _get_runtime_project_root() -> Path:
    cwd_root = Path.cwd()
    if (cwd_root / "data" / "chapters").exists():
        return cwd_root
    return Path(__file__).resolve().parent.parent.parent

def extract_sections_from_markdown(
    file_path: str,
    exclusion_keywords: List[str] = None
) -> List[Dict]:
    """
    Chunks a markdown file into sections based on '##' headings.

    Args:
        file_path : The path to the markdown file.
        exclusion_keywords : List of keywords for excluding sections.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              section with 'heading' and 'content' keys.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    sections = []

    top_level_matches = list(TOP_LEVEL_HEADING_RE.finditer(content))
    if not top_level_matches:
        cleaned_content = preprocess_extracted_section(content)
        return [{"heading": "Introduction", "content": cleaned_content}] if cleaned_content else []

    intro_content = preprocess_extracted_section(content[:top_level_matches[0].start()])
    if intro_content:
        sections.append({
            'heading': 'Introduction',
            'content': intro_content
        })

    for idx, match in enumerate(top_level_matches):
        next_start = (
            top_level_matches[idx + 1].start()
            if idx + 1 < len(top_level_matches)
            else len(content)
        )
        heading = f"Section {match.group('number')} {match.group('title')}"

        if exclusion_keywords is not None:
            if any(keyword.lower() in heading.lower() for keyword in exclusion_keywords):
                continue

        section_content = preprocess_extracted_section(content[match.end():next_start])
        if not section_content:
            continue

        section_number = match.group("number")
        assert isinstance(section_number, str) and section_number.strip(), \
            f"Invalid section number extracted from heading: {heading}"
        assert all(part.isdigit() for part in section_number.split('.')), \
            f"Malformed section numbering '{section_number}' in heading: {heading}"

        try:
            chapter_num = int(section_number.split('.')[0])
        except ValueError:
            chapter_num = 0

        sections.append({
            'heading': heading,
            'content': section_content,
            'level': section_number.count('.') + 1,
            'chapter': chapter_num
        })

    return sections

def extract_index_with_range_expansion(text_content):
    """
    Extracts keywords and page numbers from the raw text of a book index,
    expands page ranges, and returns the data as a JSON string.
    """
    
    # Pre-process the text: remove source tags and page headers/footers
    text_content = re.sub(r'\\', '', text_content)
    text_content = re.sub(r'--- PAGE \d+ ---', '', text_content)
    text_content = re.sub(r'^\d+\s+Index\s*$', '', text_content, flags=re.MULTILINE)
    text_content = re.sub(r'^Index\s+\d+\s*$', '', text_content, flags=re.MULTILINE)

    # Regex to find a keyword followed by its page numbers.
    pattern = re.compile(r'^(.*?),\s*([\d,\s\-]+?)(?=\n[A-Za-z]|\Z)', re.MULTILINE | re.DOTALL)
    
    index_data = {}
    
    for match in pattern.finditer(text_content):
        # Clean up the keyword and the page number string
        keyword = match.group(1).strip().replace('\n', ' ')
        page_numbers_str = match.group(2).strip().replace('\n', ' ')

        # Skip entries that are clearly not valid keywords
        if keyword.lower() in ["mc", "graw", "hill", "education"]:
            continue

        pages = []
        # Split the string of page numbers by comma
        for part in re.split(r',\s*', page_numbers_str):
            part = part.strip()
            if not part:
                continue
            
            # Check for a page range (e.g., "805-807")
            if '-' in part:
                try:
                    start_str, end_str = part.split('-')
                    start = int(start_str)
                    end = int(end_str)
                    # Add all numbers in the range (inclusive)
                    pages.extend(range(start, end + 1))
                except ValueError:
                    # Handle cases where a part with a hyphen isn't a valid range
                    pass 
            else:
                try:
                    # It's a single page number
                    pages.append(int(part))
                except ValueError:
                    # Handle cases where a part is not a valid number
                    pass
        
        if keyword and pages:
            # Add the parsed pages to the dictionary
            if keyword in index_data:
                index_data[keyword].extend(pages)
            else:
                index_data[keyword] = pages

    # Convert the dictionary to a nicely formatted JSON string
    return json.dumps(index_data, indent=2)

def convert_and_save_with_page_numbers(input_file_path, output_file_path):
    """
    Converts a document to Markdown, iterating page by page
    to insert a custom footer with the page number after each page,
    and saves the result to a file.
    
    Args:
        input_file_path (str): The path to the source file (e.g., "/path/to/file.pdf").
        output_file_path (str): The path to the destination .md file.
    """
    
    source = Path(input_file_path)
    if not source.exists():
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        return

    # Disable OCR and table structure extraction for faster processing
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
    format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV2DocumentBackend)
        }
    )
    
    try:
        # Convert the entire document once
        result = converter.convert(source)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return
        
    doc = result.document

    num_pages = len(doc.pages)
    
    # Extract markdown and append page number footer except for the last page
    final_text = "".join(
        doc.export_to_markdown(page_no=i) + (f"\n\n--- Page {i} ---\n\n" if i < num_pages else "")
        for i in range(1, num_pages + 1)
    )

    # Write the combined markdown string to the output file
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"Successfully converted and saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing to file {output_file_path}: {e}", file=sys.stderr)


def preprocess_extracted_section(text: str) -> str:
    """
    Cleans a raw textbook section to prepare it for chunking.

    Args:
        text: The raw text of the section.

    Returns:
        str: The cleaned text.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('**', '')

    cleaned_lines = []
    paragraph_buffer: List[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            cleaned_lines.append(" ".join(paragraph_buffer))
            paragraph_buffer.clear()

    for raw_line in text.split('\n'):
        stripped = raw_line.strip()

        if not stripped:
            flush_paragraph()
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if stripped == '<!-- image -->':
            flush_paragraph()
            continue

        if (
            PAGE_MARKER_RE.match(stripped)
            or HEADING_LINE_RE.match(stripped)
            or LIST_LINE_RE.match(stripped)
        ):
            flush_paragraph()
            cleaned_lines.append(stripped)
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()

    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def main():
    # Returns all pdf files under data/chapters/
    project_root = _get_runtime_project_root()
    chapters_dir = project_root / "data/chapters"
    pdfs = sorted(chapters_dir.glob("*.pdf"))

    # Ensure at least one PDF is found
    if len(pdfs) == 0:
        print("ERROR: No PDFs found in data/chapters/. Please copy a PDF there first.", file=sys.stderr)
        sys.exit(1)

    # Convert each PDF to Markdown
    markdown_files = []
    for pdf_path in pdfs:
        pdf_name = pdf_path.stem
        output_md = Path("data") / f"{pdf_name}--extracted_markdown.md"

        print(f"Converting '{pdf_path}' to '{output_md}'...")
        convert_and_save_with_page_numbers(str(pdf_path), str(output_md))

        markdown_files.append(output_md)

    # TODO: Add logic to select which markdown file to process
    extracted_sections = extract_sections_from_markdown(markdown_files[0])
    # print(f"Processing markdown file: {markdown_files[0]}")

    if extracted_sections:
        print(f"Successfully extracted {len(extracted_sections)} sections.")
        output_filename = project_root / "data/extracted_sections.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_sections, f, indent=4, ensure_ascii=False)
        print(f"\nFull extracted content saved to '{output_filename}'")


if __name__ == '__main__':
    main()
