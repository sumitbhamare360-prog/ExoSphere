"""Regression coverage for dynesty checkpoint restoration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from exosphere.retrieval import samplers


def test_run_dynesty_restores_existing_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """An existing checkpoint must be restored, not replaced by a fresh sampler."""
    checkpoint = tmp_path / "nested.pkl"
    checkpoint.write_bytes(b"checkpoint")
    seen: dict[str, object] = {}

    class ForbiddenSampler:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a checkpointed run must not construct a new sampler")

        @staticmethod
        def restore(path):
            raise AssertionError(f"unexpected restore before test setup: {path}")


    class FakeSampler:
        ncall = 17
        results = type(
            "Results",
            (),
            {
                "logwt": np.array([0.0]),
                "logz": np.array([0.0]),
                "logzerr": np.array([0.0]),
                "niter": 1,
            },
        )()

        def run_nested(self, **kwargs):
            seen["run_kwargs"] = kwargs

    monkeypatch.setattr(
        ForbiddenSampler,
        "restore",
        staticmethod(lambda path: seen.setdefault("path", path) and FakeSampler()),
    )
    import dynesty

    monkeypatch.setattr(dynesty, "NestedSampler", ForbiddenSampler)
    monkeypatch.setattr(
        samplers.RetrievalResult,
        "from_dynesty",
        classmethod(lambda cls, *args, **kwargs: {"results": kwargs["results"]}),
    )

    config = samplers.SamplerConfig(n_live=10, checkpoint_path=str(checkpoint))
    result = samplers.run_dynesty(None, None, config, seed=42)

    assert result["results"].niter == 1
    assert seen["path"] == str(checkpoint)
    assert seen["run_kwargs"]["resume"] is True
