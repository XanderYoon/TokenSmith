import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_MARKER_LINE_RE = re.compile(r"^--- Page \d+ ---$")
SUBSECTION_HEADING_RE = re.compile(
    r"^## (?P<number>\d+\.\d+\.\d+(?:\.\d+)*)\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)


class ChunkConfig(ABC):
    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass


@dataclass
class SectionRecursiveConfig(ChunkConfig):
    """Configuration for section-based chunking with recursive splitting."""

    recursive_chunk_size: int
    recursive_overlap: int

    def to_string(self) -> str:
        return (
            "chunk_mode=sections+recursive, "
            f"chunk_size={self.recursive_chunk_size}, "
            f"overlap={self.recursive_overlap}"
        )

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"


@dataclass
class StructureAwareConfig(ChunkConfig):
    """Configuration for subsection/paragraph aware chunking."""

    max_chunk_chars: int
    oversize_fallback_overlap: int = 0

    def to_string(self) -> str:
        return (
            "chunk_mode=structure_aware, "
            f"max_chunk_chars={self.max_chunk_chars}, "
            f"oversize_fallback_overlap={self.oversize_fallback_overlap}"
        )

    def validate(self):
        assert self.max_chunk_chars > 0, "max_chunk_chars must be > 0"
        assert (
            self.oversize_fallback_overlap >= 0
        ), "oversize_fallback_overlap must be >= 0"
        assert (
            self.oversize_fallback_overlap < self.max_chunk_chars
        ), "oversize_fallback_overlap must be smaller than max_chunk_chars"


@dataclass
class ChunkMetadata:
    unit_type: str
    heading: Optional[str] = None


@dataclass
class ChunkPiece:
    text: str
    metadata: ChunkMetadata


class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def chunk_pieces(self, text: str) -> List[ChunkPiece]:
        pass

    @abstractmethod
    def artifact_folder_name(self) -> str:
        pass


class SectionRecursiveStrategy(ChunkStrategy):
    """
    Applies recursive character-based splitting to text.
    This is meant to be used on already-extracted sections.
    """

    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        self.recursive_chunk_size = config.recursive_chunk_size
        self.recursive_overlap = config.recursive_overlap

    def name(self) -> str:
        return f"sections+recursive({self.recursive_chunk_size},{self.recursive_overlap})"

    def artifact_folder_name(self) -> str:
        return "sections"

    def chunk(self, text: str) -> List[str]:
        return [piece.text for piece in self.chunk_pieces(text)]

    def chunk_pieces(self, text: str) -> List[ChunkPiece]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.recursive_chunk_size,
            chunk_overlap=self.recursive_overlap,
            separators=[". "],
        )
        return [
            ChunkPiece(text=chunk, metadata=ChunkMetadata(unit_type="recursive"))
            for chunk in splitter.split_text(text)
        ]


class StructureAwareStrategy(ChunkStrategy):
    """Chunk sections by subsection first, then by paragraph, with a local fallback."""

    def __init__(self, config: StructureAwareConfig):
        self.config = config
        self.max_chunk_chars = config.max_chunk_chars
        self.oversize_fallback_overlap = config.oversize_fallback_overlap

    def name(self) -> str:
        return f"structaware({self.max_chunk_chars},{self.oversize_fallback_overlap})"

    def artifact_folder_name(self) -> str:
        return "structure_aware"

    def chunk(self, text: str) -> List[str]:
        return [piece.text for piece in self.chunk_pieces(text)]

    def chunk_pieces(self, text: str) -> List[ChunkPiece]:
        if not text.strip():
            return []

        subsection_units = _extract_subsection_units(text)
        if subsection_units:
            pieces: List[ChunkPiece] = []
            for unit_type, heading, unit_text in subsection_units:
                pieces.extend(
                    self._emit_unit_chunks(
                        unit_text,
                        unit_type=unit_type,
                        heading=heading,
                    )
                )
            return pieces

        paragraph_units = _extract_paragraph_units(text)
        pieces = []
        for unit_text in paragraph_units:
            pieces.extend(
                self._emit_unit_chunks(
                    unit_text,
                    unit_type="paragraph",
                    heading=None,
                )
            )
        return pieces

    def _emit_unit_chunks(
        self,
        unit_text: str,
        *,
        unit_type: str,
        heading: Optional[str],
    ) -> List[ChunkPiece]:
        normalized_text = unit_text.strip()
        if not normalized_text:
            return []
        if len(normalized_text) <= self.max_chunk_chars:
            return [
                ChunkPiece(
                    text=normalized_text,
                    metadata=ChunkMetadata(unit_type=unit_type, heading=heading),
                )
            ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_chars,
            chunk_overlap=self.oversize_fallback_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        fallback_chunks = splitter.split_text(normalized_text)
        return [
            ChunkPiece(
                text=chunk.strip(),
                metadata=ChunkMetadata(unit_type="fallback_split", heading=heading),
            )
            for chunk in fallback_chunks
            if chunk.strip()
        ]


class DocumentChunker:
    """
    Chunk text via a provided strategy.
    Table blocks (<table>...</table>) are preserved within chunks.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True,
    ):
        self.strategy = strategy
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, table in enumerate(tables):
            text = text.replace(table, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, table in enumerate(tables):
            placeholder = f"[TABLE_PLACEHOLDER_{i}]"
            if placeholder in chunk:
                chunk = chunk.replace(placeholder, table)
        return chunk

    def chunk(self, text: str) -> List[str]:
        return [piece.text for piece in self.chunk_pieces(text)]

    def chunk_pieces(self, text: str) -> List[ChunkPiece]:
        if not text:
            return []
        if self.strategy is None:
            raise ValueError("No chunk strategy provided")

        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        pieces = self.strategy.chunk_pieces(work)

        if self.keep_tables and tables:
            for piece in pieces:
                piece.text = self._restore_tables(piece.text, tables)
        return pieces


def _extract_subsection_units(text: str) -> List[Tuple[str, Optional[str], str]]:
    matches = list(SUBSECTION_HEADING_RE.finditer(text))
    if not matches:
        return []

    units: List[Tuple[str, Optional[str], str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        units.extend(
            [("paragraph", None, unit) for unit in _extract_paragraph_units(preamble)]
        )

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        unit_text = text[start:end].strip()
        heading = f"Section {match.group('number')} {match.group('title')}"
        if unit_text:
            units.append(("subsection", heading, unit_text))
    return units


def _extract_paragraph_units(text: str) -> List[str]:
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    units: List[str] = []
    page_buffer: List[str] = []

    for block in raw_blocks:
        normalized_block = _normalize_block(block)
        if not normalized_block:
            continue
        if PAGE_MARKER_LINE_RE.match(normalized_block):
            page_buffer.append(normalized_block)
            continue
        if page_buffer:
            normalized_block = "\n".join(page_buffer + [normalized_block])
            page_buffer.clear()
        units.append(normalized_block)

    if page_buffer:
        units.extend(page_buffer)
    return units


def _normalize_block(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return ""
    if all(PAGE_MARKER_LINE_RE.match(line) for line in lines):
        return "\n".join(lines)
    if all(re.match(r"^(?:[-*+]\s+|\d+\.\s+)", line) for line in lines):
        return "\n".join(lines)
    if lines[0].startswith("## "):
        if len(lines) == 1:
            return lines[0]
        return "\n".join([lines[0], " ".join(lines[1:])])
    return " ".join(lines)
