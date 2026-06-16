from typing import Annotated, Literal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field

# Resolve the project-root .env regardless of the current working directory
# (e.g. when a module is imported from a notebook under src/graph/).
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
# Project root (the directory holding .env). Used for resolving allow-listed
# paths (e.g. the /ingest directory) independent of the current working dir.
_ROOT = _ENV_FILE.parent


class DatabaseSettings(BaseSettings):
    host: str
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    db_name: str
    user: str
    password: SecretStr

    @property
    def conninfo(self) -> str:
        """libpq connection string used by psycopg across the project.

        The ``password`` key is omitted when empty so passwordless local setups
        (unix-socket peer/trust auth) work; including ``password=`` empty would
        otherwise trip md5/scram auth with "no password supplied".
        """
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.db_name}",
            f"user={self.user}",
        ]
        secret = self.password.get_secret_value()
        if secret:
            parts.append(f"password={secret}")
        return " ".join(parts)

class GraphSettings(BaseSettings):
    db_path: Path = Path("data/kuzu_db")
    # Cosine similarity at/above which two entity-name embeddings are treated
    # as "near-same" and become merge candidates (explicit, user-invoked).
    merge_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.90
    # Relations with weight at/below this are treated as low-confidence and are
    # excluded from graph retrieval (weight also drives visual node proximity).
    min_relation_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.50

class EmbeddingSettings(BaseSettings):
    model: str = "nomic-embed-text"
    embed_dim: int = 256
    batch_size: int = 128

class GeneratorSettings(BaseSettings):
    base_url: str
    model: str
    api_key: SecretStr

class ExtractionSettings(BaseSettings):
    """Local finetuned model used for KG entity/triplet extraction (Ollama-native API).

    Self-contained: can be imported and instantiated on its own (e.g. in a notebook)
    without constructing the full application Settings (database, generator, ...).
    Reads EXTRACTION__* env vars, identical to its nested form under Settings.
    """
    # The shared project .env also holds keys for other settings groups
    # (DATABASE__*, GENERATOR__*, ...). As a prefixed, self-contained model we
    # only own EXTRACTION__* keys, so ignore everything else instead of
    # raising `extra_forbidden`.
    model_config = SettingsConfigDict(
        env_prefix="extraction__", env_file=_ENV_FILE, extra="ignore"
    )
    # Default extraction backend when no per-call / CLI override is given:
    #   "local"    -> the Ollama-hosted fine-tuned model configured below
    #   "deepseek" -> remote OpenAI-compatible API (see NERSettings)
    # Overridden per run by `ingest --extractor/-e` or `/ingest?extractor=`.
    backend: Literal["local", "deepseek"] = "deepseek"
    base_url: str = "http://localhost:11434"
    model: str = "hgr-triplet:q4"
    temperature: float = 0.0
    timeout: float = 120.0
    num_ctx: int | None = None
    # Concurrent extraction requests (used by extract_entities_batch). Ollama
    # may serialize on CPU, so gains here are smaller than for the remote API.
    concurrency: Annotated[int, Field(ge=1, le=64)] = 10
    min_triplets: Annotated[int, Field(ge=1)] = 1
    max_triplets: Annotated[int, Field(ge=1)] = 8


class NERSettings(BaseSettings):
    """Remote entity/triplet extraction via an OpenAI-compatible endpoint.

    Default provider is DeepSeek V4 Flash. Self-contained (reads ``NER__*`` env
    vars) so it can be imported and instantiated on its own, exactly like
    ExtractionSettings. ``api_key`` defaults to empty so the ``local`` backend
    keeps working without it; it is validated at call time only when the
    ``deepseek`` backend is actually used.
    """
    model_config = SettingsConfigDict(
        env_prefix="ner__", env_file=_ENV_FILE, extra="ignore"
    )
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key: SecretStr = SecretStr("")
    temperature: float = 0.0
    timeout: float = 120.0
    # Concurrent extraction requests against the remote API
    # (used by extract_entities_batch). The API parallelizes well.
    concurrency: Annotated[int, Field(ge=1, le=64)] = 10
    # Disable the model's reasoning/"thinking" mode. DeepSeek's hybrid models
    # default to thinking, which burns the token budget on reasoning and can
    # leave `content` empty — bad for structured extraction. Sends
    # {"thinking": {"type": "disabled"}} on the request.
    disable_thinking: bool = True
    min_triplets: Annotated[int, Field(ge=1)] = 1
    max_triplets: Annotated[int, Field(ge=1)] = 8


class RerankerSettings(BaseSettings):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class ContextSettings(BaseSettings):
    """Context-assembly budget for the prompt sent to the generator.

    ``max_tokens`` caps the approximate token count of retrieved chunks packed
    into the context window (rough word-count * ``token_ratio`` estimate; the
    real tokenizer lives server-side). Keep headroom below the generator's
    context length for the system prompt, the question, and the answer.
    """
    max_tokens: Annotated[int, Field(ge=256)] = 3000
    # Words -> approx tokens multiplier used to estimate chunk size without a
    # tokenizer dependency. ~1.3 is a reasonable English heuristic.
    token_ratio: Annotated[float, Field(gt=0.0)] = 1.3


class VerifierSettings(BaseSettings):
    # Off by default: the NLI cross-encoder is heavy and adds latency to every
    # /query. Enable explicitly (VERIFIER__ENABLED=true) to populate the
    # `faithfulness` score on responses.
    enabled: bool = False
    model: str = "cross-encoder/nli-deberta-v3-base"
    threshold: float = 0.6


class APISettings(BaseSettings):
    """FastAPI surface configuration (``API__*``).

    ``api_key`` gates every endpoint when set; leave it empty ONLY for trusted
    local use — the app logs a loud warning at startup when it is unset.
    ``ingest_dir`` is the allow-listed root for ``/ingest`` file paths: requests
    referencing files resolving outside it are rejected, preventing the endpoint
    from reading arbitrary server files.
    """
    api_key: SecretStr = SecretStr("")
    ingest_dir: Path = _ROOT / "data"


class Settings(BaseSettings):
    # `extra="ignore"` so unmapped keys in .env (e.g. a flat GENERATION_API_KEY)
    # don't trigger `extra_forbidden`. Required values are still validated.
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_nested_delimiter="__", extra="ignore"
    )

    database: DatabaseSettings
    graph: GraphSettings = GraphSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    generator: GeneratorSettings
    extraction: ExtractionSettings = ExtractionSettings()
    ner: NERSettings = NERSettings()
    reranker: RerankerSettings = RerankerSettings()
    context: ContextSettings = ContextSettings()
    verifier: VerifierSettings = VerifierSettings()
    api: APISettings = APISettings()

def get_settings() -> Settings:
    return Settings()