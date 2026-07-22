from pathlib import Path

from constants.exceptions import TextExtractionError
from constants.logger import setup_logger
from ingestion.normalize import clean_pdf_text

LOGGER = setup_logger(__name__)


class Extractor:
    def extract_pdf(self, path: str) -> str:
        if not Path(path).exists():
            raise TextExtractionError(f"File not found: {path}")
        if not path.endswith(".pdf"):
            raise TextExtractionError(f"Not a PDF file: {path}")
        try:
            import pymupdf
        except ImportError:
            LOGGER.error(
                "Dependency is not installed. Please install it using 'uv add -r requirements.txt'."
            )
            raise TextExtractionError(
                "Dependency is not installed. Please install it using 'uv add -r requirements.txt'."
            )
        try:
            with pymupdf.open(path) as doc:
                raw_text = "".join(page.get_text() for page in doc if page.get_text())
                cleaned = clean_pdf_text(
                    raw_text,
                    fix_lines=True,
                    remove_pages=True,
                    remove_repeated_headers=True,
                )
                return cleaned

        except pymupdf.FileDataError as e:
            LOGGER.error(f"Invalid or corrupted PDF: {e}")
            raise TextExtractionError(f"Invalid or corrupted PDF: {e}")
        except ValueError as e:
            LOGGER.error(f"Cannot open file: {e}")
            raise TextExtractionError(f"Cannot open file: {e}")
