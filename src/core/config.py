import os
from pathlib import Path

class Settings:
    PROJECT_NAME = "Grey"
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    RAW_PDF_PATH = "/home/malow/Code/grey/data/raw/pdf"
    RAW_TEXT_PATH = "/home/malow/Code/grey/data/raw/text"
    METADATA_PATH = "/home/malow/Code/grey/data/raw/metadata"

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    EMDED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

settings = Settings()