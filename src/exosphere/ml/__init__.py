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
from .infer import MLClassifier, MLResult, predict_from_checkpoint, predict_from_checkpoint_batch
from .model import (
    MoleculeCNN,
    count_parameters,
    create_model,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from .train import Trainer, create_dataloaders, evaluate_test_set

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
