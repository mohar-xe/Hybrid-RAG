from typing import Annotated
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field


class DatabaseSettings(BaseSettings):
    host: str
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    db_name: str
    user: str
    password: SecretStr

class GraphSettings(BaseSettings):
    db_path: Path = Path("data/kuzu_db")

class EmbeddingSettings(BaseSettings):
    model: str = "nomic-embed-text"
    embed_dim: int = 256
    batch_size: int = 128

class GeneratorSettings(BaseSettings):
    base_url: str
    model: str
    api_key: SecretStr

class EntitySettings(BaseSettings):
    """will be published in HF so not setting now."""
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",env_nested_delimiter="__")

    database: DatabaseSettings
    graph: GraphSettings
    embedding: EmbeddingSettings
    generator: GeneratorSettings

def get_settings() -> Settings:
    return Settings()