"""Tests for the seeded RNG helper."""

import json

import pytest

from exosphere.core.provenance import Provenance
from exosphere.core.rng import SeededRNG, make_rng, make_seed


def test_same_seed_is_deterministic():
    first = make_rng(123).random(5)
    second = make_rng(123).random(5)
    assert list(first) == list(second)


def test_different_seed_differs():
    assert list(make_rng(1).random(5)) != list(make_rng(2).random(5))


def test_make_seed_is_recordable_uint32():
    seed = make_seed()
    assert isinstance(seed, int)
    assert 0 <= seed <= 2**32 - 1


def test_make_seed_is_json_serializable_in_provenance():
    seed = make_seed()
    provenance = Provenance(analysis_id="EXO-000001", retrieval_parameters={"seed": seed})
    restored = Provenance.model_validate_json(json.dumps(provenance.model_dump(mode="json")))
    assert restored.retrieval_parameters["seed"] == seed


def test_seeded_rng_bundles_recordable_seed():
    rng = SeededRNG.create(seed=7)
    assert rng.seed == 7
    assert list(rng.generator.random(3)) == list(make_rng(7).random(3))
    assert SeededRNG.create().seed >= 0


@pytest.mark.parametrize("bad_seed", [-1, 2**32, True, "3"])
def test_make_seed_rejects_invalid(bad_seed):
    with pytest.raises((ValueError, TypeError)):
        make_rng(bad_seed)
