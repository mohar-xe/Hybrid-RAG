from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader

from core.schema import MemoryUnit
from core.config import settings


class DocumentAdapter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _read_pdf(self, file_path: Path) -> dict[str, Any]:
        reader = PdfReader(file_path)
        metadata = {
            "title": reader.metadata.title if reader.metadata else None,
            "author": reader.metadata.author if reader.metadata else None,
            "subject": reader.metadata.subject if reader.metadata else None,
            "creator": reader.metadata.creator if reader.metadata else None,
            "producer": reader.metadata.producer if reader.metadata else None,
            "page_count": len(reader.pages),
        }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    def _read_pdf(self, file_path: Path) -> Dict[str, Any]:
        reader = PdfReader(file_path)
        metadata = {
            "title": reader.metadata.title if reader.metadata else None,
            "author": reader.metadata.author if reader.metadata else None,
            "subject": reader.metadata.subject if reader.metadata else None,
            "creator": reader.metadata.creator if reader.metadata else None,
            "producer": reader.metadata.producer if reader.metadata else None,
            "page_count": len(reader.pages),
        }
        full_text = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"[Page {page_num + 1}]\n{text}")
        return {"text": "\n\n".join(full_text), "metadata": metadata}

    def _read_epub(self, file_path: Path) -> Dict[str, Any]:
        text_parts = []
        with zipfile.ZipFile(file_path, "r") as epub:
            for name in epub.namelist():
                if (
                    name.endswith(".xhtml")
                    or name.endswith(".html")
                    or name.endswith(".htm")
                ):
                    try:
                        content = epub.read(name).decode("utf-8")
                        root = ET.fromstring(content)
                        for elem in root.iter():
                            if elem.text:
                                text_parts.append(elem.text.strip())
                    except Exception:
                        pass
        return {"text": "\n\n".join(text_parts), "metadata": {}}

    def _get_reader(self, file_path: Path) -> Dict[str, Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(file_path)
        elif suffix == ".txt":
            return self._read_txt(file_path)
        elif suffix == ".md":
            return self._read_md(file_path)
        elif suffix == ".epub":
            return self._read_epub(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]

            if end < text_len:
                newline_pos = chunk.rfind("\n")
                if newline_pos != -1:
                    chunk = chunk[:newline_pos]
                    end = start + newline_pos + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap if end < text_len else text_len

        return [c for c in chunks if c]

    def ingest(self, file_path: str | Path) -> List[MemoryUnit]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_data = self._get_reader(file_path)
        chunks = self.chunk_text(doc_data["text"])
        source_type = file_path.suffix.lower()[1:]

        memory_units = []
        for chunk in chunks:
            memory_unit = MemoryUnit(
                id=str(uuid.uuid4()),
                content=chunk,
                source_type=source_type,
                timestamp=datetime.now(),
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    **doc_data["metadata"],
                },
            )
            memory_units.append(memory_unit)

        return memory_units

    def ingest_directory(self, directory: str | Path) -> List[MemoryUnit]:
        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        extensions = ["*.pdf", "*.txt", "*.md", "*.epub"]
        all_memory_units = []

        for ext in extensions:
            for doc_file in directory.glob(ext):
                try:
                    units = self.ingest(doc_file)
                    all_memory_units.extend(units)
                except Exception as e:
                    print(f"Error ingesting {doc_file}: {e}")

        return all_memory_units
