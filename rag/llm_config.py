"""
Centralized LLM configuration.

Single source of truth for model names used across the project.

Production-readiness note: in production you would pin to a dated snapshot
(e.g. "gpt-4o-2024-08-06") so a silent OpenAI-side update can't cause a
false regression. Override per-environment via env vars below.
"""
from __future__ import annotations

import os


# Default models. Pin to dated snapshots in production env vars.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
SYNTHETIC_MODEL = os.getenv("SYNTHETIC_MODEL", "gpt-4.1-mini")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1-mini")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Determinism knobs - fixed seed for reproducible runs (judge especially).
DETERMINISTIC_SEED = int(os.getenv("DETERMINISTIC_SEED", "42"))
