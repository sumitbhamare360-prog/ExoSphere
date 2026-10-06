"""Seeded RNG helper.

Every stochastic step must be deterministic given a seed (AGENTS.md rule 6), and the
seed must be recordable in Provenance.retrieval_parameters["seed"].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SEED_MAX = 2**32 - 1


def make_seed() -> int:
    """Draw a random 32-bit seed (record it in provenance when used)."""
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def make_rng(seed: int) -> np.random.Generator:
    """Build a numpy Generator that is fully determined by ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    if not 0 <= seed <= _SEED_MAX:
        raise ValueError(f"seed must be in [0, {_SEED_MAX}], got {seed}")
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class SeededRNG:
    """Generator bundled with its recordable seed."""

    seed: int
    generator: np.random.Generator

    @classmethod
    def create(cls, seed: int | None = None) -> "SeededRNG":
        resolved = make_seed() if seed is None else seed
        return cls(seed=int(resolved), generator=make_rng(resolved))
