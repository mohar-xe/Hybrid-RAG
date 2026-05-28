from pathlib import Path

from constants.exceptions import TextExtractionError
from constants.logger import setup_logger
from ingestion.normalize import clean_pdf_text

LOGGER = setup_logger(__name__)

class Extractor:
    """Later want to add docling or marker.dots.ocr use @classmethod to add more extractors and use a factory pattern to call the right extractor based on the file type or source"""
    def __init__(self) -> None:
        self._whisper = None
    #PDF will return text only no caching raw_data since it is not needed for future use
    def extract_pdf(self, path: str) -> str:
        if not Path(path).exists():
            raise TextExtractionError(f"File not found: {path}")
        if not path.endswith('.pdf'):
            raise TextExtractionError(f"Not a PDF file: {path}")
        try:
            import pymupdf
        except ImportError:
            LOGGER.error("Dependency is not installed. Please install it using 'uv add -r requirements.txt'.")
            raise TextExtractionError("Dependency is not installed. Please install it using 'uv add -r requirements.txt'.")
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
        
    def yt_subtitle_extraction(self, video_id: str) -> str:
        if not video_id or not video_id.strip():
            raise TextExtractionError("video_id cannot be empty")
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import SRTFormatter
            from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
            ytt = YouTubeTranscriptApi()
            formatter = SRTFormatter()
        except ImportError:
            LOGGER.error("Dependency is not installed. Please install it using 'uv add -r requirements.txt'.")
            raise TextExtractionError("Dependency is not installed. Please install it using 'uv add -r requirements.txt'.")
        
        try:
            transcript = ytt.fetch(video_id)
            text = formatter.format_transcript(transcript)

            return text
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            LOGGER.error(f"No transcript available: {e}")
            raise TextExtractionError(f"No transcript available: {e}")
    @property   
    def whisper(self):
        if self._whisper is None:
            import os
            try:
                from pywhispercpp.model import Model
            except Exception as e:
                LOGGER.error("Dependency is not installed. Please install it using 'uv add -r requirements.txt'.")
                raise ImportError("Dependency is not installed. Please install it using 'uv add -r requirements.txt'.")
            self._whisper = Model('base.en', n_threads=os.cpu_count())
            return self._whisper

    def reel_subtitle_extraction(self, path: str) -> str:
        if not Path(path).exists():
            raise TextExtractionError(f"File not found: {path}")
        try:
            segments = self.whisper.transcribe(path)
            text = " ".join([segment.text for segment in segments])

            return text
        except RuntimeError as e:
            LOGGER.error(f"Whisper model failed to transcribe: {e}")
            raise TextExtractionError(f"Whisper model failed to transcribe: {e}")
        except OSError as e:
            LOGGER.error(f"Cannot read audio file: {e}")
            raise TextExtractionError(f"Cannot read audio file: {e}")