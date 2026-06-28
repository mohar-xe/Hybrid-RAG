"""HotpotQA loading, seeded query selection, and corpus construction.

The selection step is deliberately pure and isolated (``select_indices``) so it
can be unit-tested for determinism without downloading anything. The actual
download (via the optional ``datasets`` dependency) happens only in
``prepare_dataset`` and its result is cached to JSON for offline, reproducible
re-runs.

Each cached record has the shape::

    {
      "id": str,
      "question": str,
      "answer": str,
      "paragraphs": [{"title": str, "text": str}, ...],   # 10 (2 gold + 8 distractor)
      "supporting_titles": [str, ...]                      # gold paragraph titles
    }
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from evaluation import config


def select_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Deterministically choose ``n`` distinct indices from ``range(n_total)``.

    Pure and side-effect free: identical (n_total, n, seed) always yields the
    same sorted index list, which is what makes the evaluation reproducible.
    """
    if n_total <= 0:
        return []
    n = min(n, n_total)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n))


def _example_to_record(example: dict) -> dict:
    """Convert a raw HotpotQA (distractor) example into our compact record.

    Handles the canonical HF ``hotpot_qa`` schema where ``context`` is a dict
    with parallel ``title`` (list[str]) and ``sentences`` (list[list[str]])
    fields, and ``supporting_facts`` is a dict with a ``title`` list.
    """
    context = example["context"]
    titles = context["title"]
    sentences = context["sentences"]
    paragraphs = [
        {"title": title, "text": " ".join(sents).strip()}
        for title, sents in zip(titles, sentences)
    ]
    support = example["supporting_facts"]
    supporting_titles = sorted(set(support["title"]))
    return {
        "id": str(example.get("id", "")),
        "question": example["question"],
        "answer": example["answer"],
        "paragraphs": paragraphs,
        "supporting_titles": supporting_titles,
    }


def _load_hotpotqa_split():
    """Load the HotpotQA validation split via `datasets`, tolerating API changes."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "The `datasets` package is required to download HotpotQA. Install the "
            "eval extra:  uv sync --extra eval   (or  pip install 'datasets>=2.18.0')."
        ) from e

    # `datasets` >= 4 requires a namespaced repo id ("hotpotqa/hotpot_qa") and
    # rejects the bare canonical id ("hotpot_qa") with an HfUriError; older
    # versions accepted the bare id. Try the configured id first, then the
    # alternate forms, so the harness works across `datasets` versions. The inner
    # try/except handles `trust_remote_code` being required (script datasets) or
    # rejected (parquet-backed datasets).
    bare = config.HF_DATASET.split("/")[-1]
    candidates = [config.HF_DATASET, f"hotpotqa/{bare}", bare]
    seen: set[str] = set()
    last_err: Exception | None = None
    for ds_id in candidates:
        if ds_id in seen:
            continue
        seen.add(ds_id)
        try:
            try:
                return load_dataset(ds_id, config.HF_CONFIG, split=config.HF_SPLIT, trust_remote_code=True)
            except TypeError:
                return load_dataset(ds_id, config.HF_CONFIG, split=config.HF_SPLIT)
        except Exception as e:  # invalid id for this datasets version, network, etc.
            last_err = e
    raise last_err if last_err else RuntimeError("Could not load the HotpotQA split.")


def prepare_dataset(
    n: int = config.N_QUERIES,
    seed: int = config.SEED,
    *,
    force: bool = False,
) -> list[dict]:
    """Return the seeded ``n``-query selection, downloading + caching if needed.

    Cached to ``evaluation/data/selected_<config>_<split>_n<n>_seed<seed>.json``.
    Pass ``force=True`` to re-download and overwrite the cache.
    """
    cache = config.selection_cache_path(n, seed)
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    dataset = _load_hotpotqa_split()
    indices = select_indices(len(dataset), n, seed)
    records = [_example_to_record(dataset[i]) for i in indices]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def load_cached(n: int = config.N_QUERIES, seed: int = config.SEED) -> list[dict]:
    """Load the cached selection, erroring if it has not been prepared yet."""
    cache = config.selection_cache_path(n, seed)
    if not cache.exists():
        raise FileNotFoundError(
            f"No cached selection at {cache}. Run `prepare` first "
            f"(uv run --extra eval python -m evaluation.run_eval prepare)."
        )
    return json.loads(cache.read_text(encoding="utf-8"))


def build_corpus(records: list[dict]) -> dict[str, str]:
    """Union of all paragraphs across the selected questions, keyed by title.

    Distractor paragraphs are shared across the corpus, so retrieval must
    discriminate gold supporting paragraphs from ~10x as many distractors —
    a realistic retrieval setting rather than a per-question oracle.
    """
    corpus: dict[str, str] = {}
    for record in records:
        for paragraph in record["paragraphs"]:
            title = paragraph["title"]
            if title not in corpus and paragraph["text"]:
                corpus[title] = paragraph["text"]
    return corpus
