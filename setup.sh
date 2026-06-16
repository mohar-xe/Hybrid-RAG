#!/usr/bin/env bash
#
# Hybrid-RAG bootstrap — get a fresh clone running with minimal friction.
#
# Usage:
#   ./setup.sh           # Python deps + .env + Ollama embedding model
#   ./setup.sh --local   # also build the local fine-tuned extraction model into Ollama
#
# The default extraction backend is the remote DeepSeek API (needs NER__API_KEY).
# Pass --local to also build the bundled fine-tuned model so you can run the
# 'local' extraction backend fully offline.
#
set -euo pipefail

LOCAL=0
[[ "${1:-}" == "--local" ]] && LOCAL=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Python dependencies (uv)
# ---------------------------------------------------------------------------
info "Python dependencies (uv sync)"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv is not installed. Install it, then re-run this script:"
  warn "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv sync

# ---------------------------------------------------------------------------
# 2. .env
# ---------------------------------------------------------------------------
info ".env configuration"
if [[ -f .env ]]; then
  echo "    .env already exists — leaving it untouched."
else
  cp .env.example .env
  echo "    created .env from .env.example."
  warn "Fill in REQUIRED values: GENERATOR__* (generation), DATABASE__* (Postgres),"
  warn "and NER__API_KEY (default 'deepseek' extraction backend; skip if using --local)."
fi

# ---------------------------------------------------------------------------
# 3. Ollama + embedding model
# ---------------------------------------------------------------------------
info "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama is not installed. Install it, then re-run this script:"
  warn "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
  warn "  macOS/Windows: https://ollama.com/download"
  exit 1
fi

if ! curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "    Ollama server not reachable; starting 'ollama serve' in the background..."
  (ollama serve >/tmp/ollama-serve.log 2>&1 &) || true
  for _ in $(seq 1 30); do
    curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 \
    || { warn "Could not reach Ollama at localhost:11434 (see /tmp/ollama-serve.log)."; exit 1; }
fi

info "Embedding model: nomic-embed-text (required for all retrieval)"
ollama pull nomic-embed-text

# ---------------------------------------------------------------------------
# 4. Optional: local fine-tuned extraction model
# ---------------------------------------------------------------------------
if [[ "$LOCAL" == "1" ]]; then
  info "Local extraction model: hgr-triplet:q4"
  GGUF="model/qwen3-0.6b.F16.gguf"
  HF_REPO="mohar07/qwen3-0.6b-kg-triplets"

  if [[ ! -f "$GGUF" ]]; then
    info "Downloading GGUF from HuggingFace ($HF_REPO)"
    # Use HF_TOKEN from the environment or .env for faster/authenticated pulls
    # (the repo is public, so a token is optional but helps with rate limits).
    if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
      HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | cut -d= -f2- || true)"
    fi
    [[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN
    if ! uv run hf download "$HF_REPO" qwen3-0.6b.F16.gguf --local-dir model >/dev/null; then
      warn "HuggingFace download failed. Check your network / HF_TOKEN and re-run."
      exit 1
    fi
    echo "    downloaded $GGUF"
  else
    echo "    $GGUF already present; skipping download."
  fi

  ollama create hgr-triplet:q4 -f model/Modfile
  echo "    built hgr-triplet:q4. Use it with: --extractor local (or set EXTRACTION__BACKEND=local)."
else
  echo "    Skipping local extraction model (default backend is remote DeepSeek)."
  echo "    Re-run with './setup.sh --local' to download + build it for offline extraction."
fi

# ---------------------------------------------------------------------------
# 5. PostgreSQL reminder (cannot be fully automated — needs your DB + pgvector)
# ---------------------------------------------------------------------------
cat <<'EOF'

==> PostgreSQL (manual, one-time)
    The pipeline needs a database with the pgvector extension. Using the values
    from your .env (DATABASE__DB_NAME, DATABASE__USER):

      createdb <DATABASE__DB_NAME>
      psql -d <DATABASE__DB_NAME> -c 'CREATE EXTENSION IF NOT EXISTS vector;'

    (Table/index creation is handled automatically by init_db on first run.)

==> Run the pipeline (from src/)
      cd src
      uv run python pipeline.py ingest ../data/sample.pdf --type pdf   # store + graph
      uv run python pipeline.py reindex                                # cluster + medoids
      uv run python pipeline.py ask "your question"

EOF
info "Setup complete."
