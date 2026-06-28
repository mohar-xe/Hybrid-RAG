from typing import Annotated, Literal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field, field_validator

# Resolve the project-root .env regardless of the current working directory
# (e.g. when a module is imported from a notebook under src/graph/).
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
# Project root (the directory holding .env). Used for resolving allow-listed
# paths (e.g. the /ingest directory) independent of the current working dir.
_ROOT = _ENV_FILE.parent
# Package root: the `src/` directory that holds this config package. The Kùzu
# graph DB is anchored here so it always resolves to the *same* file regardless
# of the current working directory. Historically the DB is created under
# `src/data/` because the CLI is launched from inside `src/`; anchoring keeps
# that single populated location canonical no matter where a command is run.
_SRC = Path(__file__).resolve().parents[1]


class DatabaseSettings(BaseSettings):
    host: str
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    db_name: str
    user: str
    password: SecretStr
    # When false, ``init_db()`` skips all DDL (CREATE EXTENSION/TABLE/INDEX,
    # ALTER). Set false for a read-only deployment against a FROZEN database
    # whose connecting role may lack DDL privileges (e.g. a restricted Supabase
    # role) — the schema already exists, so there is nothing to migrate.
    init_schema: bool = True

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
    # Absolute, CWD-independent path to the embedded Kùzu graph DB. Anchored to
    # the package root rather than left relative: a relative "data/kuzu_db"
    # resolves against the *current working directory*, so running `ingest` from
    # src/ but `merge-graph`/`ask` from the project root would open two different
    # files — silently creating an empty graph. Anchoring guarantees every
    # command opens the one populated DB.
    db_path: Path = _SRC / "data" / "kuzu_db"
    # Cosine similarity at/above which two entity-name embeddings are treated
    # as "near-same" and become merge candidates (explicit, user-invoked).
    merge_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.90
    # Relations with weight at/below this are treated as low-confidence and are
    # excluded from graph retrieval (weight also drives visual node proximity).
    min_relation_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.50
    # --- Multi-hop graph retrieval (get_entity_context) ---
    # How many RELATES_TO edges to follow out from each seed entity. 1 == the
    # old single-hop behavior (a seed's direct relations only); >=2 lets the
    # retriever surface "bridge" facts — A->B (hop 1) then B->C (hop 2) — that
    # multi-hop questions (e.g. HotpotQA) need but a single hop can never reach.
    max_hops: Annotated[int, Field(ge=1, le=5)] = 2
    # Hard cap on the total number of distinct facts returned, so a densely
    # connected seed can't flood the context budget as the hop count grows.
    max_facts: Annotated[int, Field(ge=1, le=500)] = 30
    # Max edges expanded per node per hop (BFS fan-out). Bounds the breadth of
    # each hop so one high-degree hub doesn't dominate the traversal.
    per_hop_neighbors: Annotated[int, Field(ge=1, le=100)] = 10

    @field_validator("db_path")
    @classmethod
    def _anchor_db_path(cls, v: Path) -> Path:
        """Anchor a relative ``db_path`` (e.g. a ``GRAPH__DB_PATH`` override) to
        the package root so it never resolves against the variable current
        working directory. Absolute paths are left untouched."""
        return v if v.is_absolute() else (_SRC / v).resolve()

class EmbeddingSettings(BaseSettings):
    """Query/document embedding configuration (``EMBEDDING__*``).

    Self-contained (own ``env_prefix``) so it can be read without building the
    full ``Settings`` — mirroring ``ExtractionSettings`` / ``NERSettings``.

    Two backends produce the same 256-d (Matryoshka-truncated) vectors:
      * ``ollama``                — local Ollama server (default). Used for
        ingestion and local dev; requires Ollama running.
      * ``sentence_transformers`` — in-process HF model, no Ollama. Used for the
        read-only / free deployment (e.g. HF Spaces) where running Ollama isn't
        practical. Loads ``st_model`` lazily on first use.
    """
    model_config = SettingsConfigDict(
        env_prefix="embedding__", env_file=_ENV_FILE, extra="ignore"
    )
    backend: Literal["ollama", "sentence_transformers"] = "ollama"
    model: str = "nomic-embed-text"                     # Ollama model name
    st_model: str = "nomic-ai/nomic-embed-text-v1.5"    # sentence-transformers model id
    embed_dim: int = 256
    batch_size: int = 128
    # Task prefix prepended to every input before embedding. Empty by default to
    # match the FROZEN corpus: Ollama's nomic-embed-text template is
    # `{{ .Prompt }}` (no prefix), so the stored vectors are prefix-free. Only
    # set this if you re-embed the corpus under a different convention — a query
    # prefix that the documents didn't use puts queries in a different subspace.
    query_prefix: str = ""

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