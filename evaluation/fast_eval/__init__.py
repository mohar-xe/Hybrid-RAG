"""fast_eval — separate, concurrent, fully-paid evaluation harness.

Independent from the staged ``evaluation`` pipeline (that one stays for personal
free-tier use). This suite treats the API key as paid, runs generation
per-query (interactive, not bundled), measures real latency, and checks first
whether the corpus is already ingested before doing any ingestion work.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
