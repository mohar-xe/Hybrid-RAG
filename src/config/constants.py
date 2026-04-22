from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL = "google/gemma-4-27b-it"
GENERATOR_MODEL = "google/gemma-4-27b-it"

class DatabaseSettings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str
    db_password: str
    db_name: str

    @computed_field
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

settings = DatabaseSettings()
print(settings.database_url)