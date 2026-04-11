from pypdf import PdfReader
import logging

LOGGER = logging.getLogger(__name__)


def _load_pdf(file_path: str) -> PdfReader:
    try:
        reader = PdfReader(file_path)
        LOGGER.info(f"Successfully loaded {len(reader.pages)} pages.")
        return reader
    except Exception as exc:
        LOGGER.exception(f"Error loading PDF: {exc}")
        raise RuntimeError(f"Failed to load PDF: {exc}") from exc

def parse_pdf(file_path: str) -> dict:
    reader = _load_pdf(file_path)
#This is crazy "" is always false so if page.extract_text() is None it will use "" instead, and then we join all the pages with a space in between, and then we strip any leading or trailing whitespace from the final string.
    contents = " ".join(page.extract_text() or "" for page in reader.pages).strip()

    raw_meta = reader.metadata or {}
    metadata = {
        "title": raw_meta.get("/Title"),
        "author": raw_meta.get("/Author"),
        "source": file_path,
        "total_pages": len(reader.pages),
        "creationdate": raw_meta.get("/CreationDate"),
    }
    LOGGER.info(f"Extracted contents length: {len(contents)} characters")
    LOGGER.info(f"Extracted metadata: {metadata}")
    return {
        "contents": contents,
        "metadata": metadata,
    }