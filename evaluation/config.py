"""Evaluation configuration — every knob in one place, seed-first.

Changing ``SEED`` or ``N_QUERIES`` changes which questions are evaluated; the
selection is otherwise fully deterministic given the HotpotQA split.
"""

from dataclasses import dataclass
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parent
DATA_DIR = _EVAL_ROOT / "data"
RESULTS_DIR = _EVAL_ROOT / "results"
CACHE_DIR = DATA_DIR / "eval_cache"   # staged-phase JSON persistence

# --- Reproducibility ---
SEED = 42
N_QUERIES = 100

# --- Staged-pipeline knobs ---
# The eval runs in decoupled phases (artifacts -> retrieve -> generate ->
# report), each persisted to CACHE_DIR. Generation bundles many (question,
# context) pairs per API call; this is the batch size in pairs per call.
GENERATION_BATCH_SIZE = 40

# --- HotpotQA source (HuggingFace `datasets`) ---
HF_DATASET = "hotpotqa/hotpot_qa"   # namespaced id (datasets>=4 rejects bare "hotpot_qa")
HF_CONFIG = "distractor"      # 10 paragraphs/question: 2 gold + 8 distractors
HF_SPLIT = "validation"       # 7,405 questions; test has no public answers

# --- Retrieval knobs ---
TOP_K = 5            # final chunks fed to the generator / scored for hit@k
CANDIDATE_K = 20     # per-retriever candidate pool before fusion/rerank


@dataclass(frozen=True)
class ModeSpec:
    """A forced-retrieval configuration (excludes the closed-book `direct` run)."""

    name: str
    use_vector: bool
    use_bm25: bool
    use_graph: bool


# The three retrieval configurations requested for comparison. ``direct`` is
# handled separately (it performs no retrieval, so reranking does not apply).
RETRIEVAL_MODES: list[ModeSpec] = [
    ModeSpec("semantic", use_vector=True, use_bm25=False, use_graph=False),
    ModeSpec("semantic_bm25", use_vector=True, use_bm25=True, use_graph=False),
    ModeSpec("all_three", use_vector=True, use_bm25=True, use_graph=True),
]

DIRECT_MODE = "direct"


def selection_cache_path(n: int = N_QUERIES, seed: int = SEED) -> Path:
    """Deterministic cache filename for a given (n, seed) selection."""
    return DATA_DIR / f"selected_{HF_CONFIG}_{HF_SPLIT}_n{n}_seed{seed}.json"
