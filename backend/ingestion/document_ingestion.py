"""
ingestion/document_ingestion.py
--------------------------------
Ingests PDFs, Markdown, and plain-text files into the vector store.

Chunking strategy: sentence-aware overlapping windows.
Sentences are accumulated until *chunk_size* characters is reached; the final
~chunk_overlap characters from the previous window are carried forward as
overlap into the next chunk to preserve context continuity.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF


class DocumentIngestionService:
    """
    Parses and chunks text documents (PDF / Markdown / TXT) and exposes them
    as a list of ``{text, metadata}`` dicts suitable for vector store ingestion.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: str, vector_store) -> Dict[str, int]:
        """Walk *directory*, chunk every supported file, and add to *vector_store*."""
        stats = {"files": 0, "chunks": 0}
        path = Path(directory)
        if not path.exists():
            return stats

        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                chunks = self._parse_file(str(file_path))
                for chunk in chunks:
                    vector_store.add_text(chunk["text"], chunk["metadata"])
                stats["files"]  += 1
                stats["chunks"] += len(chunks)

        return stats

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract and chunk all text pages from a PDF file."""
        doc = fitz.open(file_path)
        pages: List[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()

        full_text = "\n".join(pages)
        return self._build_chunks(full_text, file_path, content_type="pdf")

    def process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read and chunk a Markdown or plain-text file."""
        text = Path(file_path).read_text(encoding="utf-8")
        return self._build_chunks(text, file_path, content_type="text")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Route a file to the appropriate parser based on its extension."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.process_pdf(file_path)
        return self.process_text_file(file_path)

    def _build_chunks(
        self, text: str, file_path: str, content_type: str
    ) -> List[Dict[str, Any]]:
        """Produce a list of enriched chunk dicts from raw text."""
        file_name  = Path(file_path).name
        title      = self._extract_title(text) or file_name
        raw_chunks = self._chunk_text(text)

        return [
            {
                "text": chunk,
                "metadata": {
                    "source":       file_path,
                    "file_name":    file_name,
                    "title":        title,
                    "chunk_index":  i,
                    "total_chunks": len(raw_chunks),
                    "content_type": content_type,
                },
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    def _extract_title(self, text: str) -> Optional[str]:
        """Heuristically derive a document title from the first few lines."""
        for line in text.strip().splitlines()[:8]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line and len(line) < 120 and not line.startswith("#"):
                return line
        return None

    def _chunk_text(self, text: str) -> List[str]:
        """Sentence-aware overlapping chunker."""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

        chunks: List[str] = []
        window: List[str] = []
        size = 0

        for sentence in sentences:
            if size + len(sentence) > self.chunk_size and window:
                chunks.append(" ".join(window))
                # Build overlap tail from the end of the previous window
                tail: List[str] = []
                tail_size = 0
                for s in reversed(window):
                    tail_size += len(s)
                    tail.insert(0, s)
                    if tail_size >= self.chunk_overlap:
                        break
                window = tail
                size   = sum(len(s) for s in window)

            window.append(sentence)
            size += len(sentence)

        if window:
            chunks.append(" ".join(window))

        # Discard micro-chunks that carry little information
        return [c for c in chunks if len(c) > 60]
