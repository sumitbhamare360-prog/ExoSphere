"""ExoSphere ML module (Phase 5)."""

from __future__ import annotations

from .dataset import (
    DEFAULT_N_SAMPLES,
    MOLECULES,
    N_MOLECULES,
    DatasetConfig,
    generate_dataset,
    load_dataset,
    load_target_grid,
    save_dataset,
)
from .result import MLResult

__all__ = [
    # dataset
    "DatasetConfig",
    "generate_dataset",
    "load_dataset",
    "save_dataset",
    "load_target_grid",
    "MOLECULES",
    "N_MOLECULES",
    "DEFAULT_N_SAMPLES",
    # model
    "MoleculeCNN",
    "create_model",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "set_seed",
    # train
    "Trainer",
    "create_dataloaders",
    "evaluate_test_set",
    # infer
    "MLClassifier",
    "MLResult",
    "predict_from_checkpoint",
    "predict_from_checkpoint_batch",
]


def __getattr__(name: str):
    """Load optional PyTorch-backed ML helpers only when requested."""
    if name in {"MLClassifier", "predict_from_checkpoint", "predict_from_checkpoint_batch"}:
        from . import infer

        return getattr(infer, name)
    if name in {
        "MoleculeCNN",
        "count_parameters",
        "create_model",
        "load_checkpoint",
        "save_checkpoint",
        "set_seed",
    }:
        from . import model

        return getattr(model, name)
    if name in {"Trainer", "create_dataloaders", "evaluate_test_set"}:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
