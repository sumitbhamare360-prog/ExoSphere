"""Tests for the Provenance record and analysis id generator (AGENTS.md section 5)."""

import json

import pytest
from pydantic import ValidationError

from exosphere.core.provenance import (
    AnalysisIdGenerator,
    Provenance,
    generate_analysis_id,
)


def test_generate_analysis_id_format():
    assert generate_analysis_id(1) == "EXO-000001"
    assert generate_analysis_id(42) == "EXO-000042"
    assert generate_analysis_id(0) == "EXO-000000"


def test_generate_analysis_id_rejects_out_of_range():
    with pytest.raises(ValueError, match="sequence must be in"):
        generate_analysis_id(1_000_000)
    with pytest.raises(ValueError, match="sequence must be in"):
        generate_analysis_id(-1)


def test_generate_analysis_id_rejects_non_int():
    with pytest.raises(TypeError, match="sequence must be an int"):
        generate_analysis_id("1")


def test_analysis_id_generator_is_sequential():
    generator = AnalysisIdGenerator()
    assert generator.next_id() == "EXO-000001"
    assert generator.next_id() == "EXO-000002"


def make_provenance() -> Provenance:
    return Provenance(
        analysis_id="EXO-000007",
        planet="WASP-39 b",
        observation_id="obs-001",
        telescope="JWST",
        instrument="NIRSpec PRISM",
        source_archive="MAST",
        input_data_version="v1",
        input_data_hash="abc123",
        preprocessing_version="0.1.0",
        ml_model_version="0.1.0",
        retrieval_model_version="0.1.0",
        retrieval_parameters={"sampler": "jaxns", "n_live": 400, "seed": 12345},
        result_reference="results/EXO-000007",
    )


def test_provenance_round_trips_through_json():
    provenance = make_provenance()
    payload = json.dumps(provenance.model_dump(mode="json"))
    restored = Provenance.model_validate_json(payload)
    assert restored == provenance
    assert Provenance.model_validate(json.loads(payload)) == provenance
    assert restored.retrieval_parameters["seed"] == 12345


def test_provenance_default_timestamp_is_set():
    provenance = Provenance(analysis_id="EXO-000001")
    assert provenance.timestamp


def test_provenance_rejects_invalid_analysis_id():
    with pytest.raises(ValidationError, match="EXO-000001 format"):
        Provenance(analysis_id="EXO-1")
    with pytest.raises(ValidationError, match="EXO-000001 format"):
        Provenance(analysis_id="ABC-000001")
