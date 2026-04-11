import re
import hashlib
import logging
from ingestion.pdf_loader import parse_pdf

LOGGER = logging.getLogger(__name__)

def _split_text(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    
    if not text:
        LOGGER.warning("No text to split.")
        return []

    chunks = []
    start = 0
    text_len = len(text)
    separator_pattern = re.compile("|".join(re.escape(sep) for sep in sorted(separators, key=len, reverse=True)))

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            window = text[start:end]
            split_end = -1
            for match in separator_pattern.finditer(window):
                split_end = match.end()
            if split_end > 0:
                end = start + split_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_text(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
    
    parsed_text = parse_pdf(file_path)
    text = parsed_text["contents"]
    metadata = parsed_text["metadata"]
    doc_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

    chunks = _split_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " "],
    )
    chunks = [re.sub(r'\s+', ' ', chunk).strip() for chunk in chunks]

    structured = []
    for i, chunk in enumerate(chunks):
        structured.append(
            {
                "chunk_id": f"{doc_hash}_chunk_{i}",
                "chunk_index": i,
                "text": chunk,
                "metadata": metadata,
            }
        )
    LOGGER.info(f"Chunked {file_path} into {len(chunks)} chunks.")
    return structured