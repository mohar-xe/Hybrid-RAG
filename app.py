"""Hugging Face Spaces entry point — read-only, no-Ollama deployment.

Serves the Gradio demo against a FROZEN Postgres (pgvector) + Kùzu graph, using
in-process sentence-transformers embeddings and DeepSeek for query NER and
answer generation. The application code lives under ``src/`` (src-layout), so it
is put on the import path before the demo is imported.

Configuration comes entirely from environment variables / Space secrets — see
``deploy/hf-space/.env.example``. This file is the Space ``app_file``; for local
development keep using ``src/demo/app.py`` or the CLI instead.
"""

import os
import sys
from pathlib import Path

# src-layout: expose the package modules (config, retrieval, demo, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Deployment posture. These are defaults only: an identically-named Space secret
# / variable overrides them via the environment.
#   * embed in-process so no Ollama server is required
#   * never run DDL against the frozen database (the schema already exists and
#     the connecting role may be read-only)
os.environ.setdefault("EMBEDDING__BACKEND", "sentence_transformers")
os.environ.setdefault("DATABASE__INIT_SCHEMA", "false")

from demo.app import create_demo  # noqa: E402  (import after path/env setup)

# Module-level handle so HF Spaces can also discover the app by attribute.
demo = create_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
