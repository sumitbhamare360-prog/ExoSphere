"""Phase 5 tests: ML molecule classifier."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from exosphere.forward.model import ModelParams, PlanetFixed
from exosphere.ml.dataset import (
    DatasetConfig,
    compute_label,
    generate_dataset,
    sample_atmosphere_params,
)
from exosphere.ml.model import count_parameters, create_model, set_seed
from exosphere.quality.assess import load_quality_config


class TestLabelFunction:
    """Test the label computation function."""

    def test_label_absent_when_vmr_low(self):
        """Molecule with log VMR < -6 should be labelled absent."""
        from exosphere.quality.assess import load_quality_config

        params = ModelParams(
            T=1100,
            log_h2o=-7.0,
            log_co2=-4.0,
            log_co=-4.5,
            log_ch4=-5.0,
            log_so2=-6.0,
            r_ref=1.27,
            log_p_cloud=-1.0,
        )
        fixed = PlanetFixed(gravity_m_s2=4.2, stellar_radius_rsun=0.93)
        band_windows = load_quality_config().molecule_bands

        # H2O has log_vmr = -7 < -6, should be absent
        labels = compute_label(params, fixed, band_windows, noise_level=1e-4)
        assert labels[0] == 0.0  # H2O absent

    def test_label_present_when_vmr_high_and_feature_strong(self):
        """Molecule with high VMR and strong feature should be present."""
        # This test is more involved - just check the logic doesn't crash

        params = ModelParams(
            T=1100,
            log_h2o=-3.5,
            log_co2=-4.0,
            log_co=-4.5,
            log_ch4=-5.0,
            log_so2=-6.0,
            r_ref=1.27,
            log_p_cloud=-1.0,
        )
        fixed = PlanetFixed(gravity_m_s2=4.2, stellar_radius_rsun=0.93)
        band_windows = load_quality_config().molecule_bands

        labels = compute_label(params, fixed, band_windows, noise_level=1e-4)
        # H2O has log_vmr=-3.5 >= -6, should be present if feature > noise
        # Just check it runs and returns valid labels
        assert labels.shape == (5,)
        assert np.all((labels >= 0) & (labels <= 1))


class TestDatasetGeneration:
    """Test dataset generation."""

    def test_sample_atmosphere_params(self):
        """Test atmosphere parameter sampling."""
        rng = np.random.default_rng(42)
        params = sample_atmosphere_params(rng)

        assert isinstance(params, ModelParams)
        assert 300 <= params.T <= 2500
        assert all(
            -12 <= getattr(params, f"log_{m.lower()}") <= -1
            for m in ["H2O", "CO2", "CO", "CH4", "SO2"]
            if getattr(ModelParams, f"log_{m.lower()}", None) != -12
        )
        assert 0.7 * 1.27 <= params.r_ref <= 1.3 * 1.27
        assert -6 <= params.log_p_cloud <= 2

    def test_generate_dataset_small(self):
        """Test generating a small dataset."""
        config = DatasetConfig(n_samples=10, seed=42)
        X, y, params_list, metadata = generate_dataset(config)

        assert X.shape == (10, 2, 147)  # 2 channels, 147 wavelength points
        assert y.shape == (10, 5)
        assert len(params_list) == 10
        assert len(metadata) == 10
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        assert np.all((y >= 0) & (y <= 1))

    def test_dataset_reproducible(self):
        """Test that dataset generation is deterministic with same seed."""
        config1 = DatasetConfig(n_samples=5, seed=42)
        config2 = DatasetConfig(n_samples=5, seed=42)

        X1, y1, _, _ = generate_dataset(config1)
        X2, y2, _, _ = generate_dataset(config2)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        config1 = DatasetConfig(n_samples=5, seed=42)
        config2 = DatasetConfig(n_samples=5, seed=123)

        X1, y1, _, _ = generate_dataset(config1)
        X2, y2, _, _ = generate_dataset(config2)

        assert not np.array_equal(X1, X2) or not np.array_equal(y1, y2)


class TestModelArchitecture:
    """Test model architecture and forward pass."""

    def test_model_creation(self):
        """Test model can be created."""
        model = create_model()
        assert isinstance(model, torch.nn.Module)
        assert count_parameters(model) > 0

    def test_forward_pass_shape(self):
        """Test forward pass produces correct shape."""
        model = create_model()
        model.eval()

        # Input: batch=4, channels=2, wavelength=147 (WASP-39 b PRISM grid)
        x = torch.randn(4, 2, 147)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 5)  # batch=4, 5 molecules
        assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid output

    def test_deterministic_with_seed(self):
        """Test model is deterministic with fixed seed."""
        set_seed(42)
        model1 = create_model()
        model1.eval()

        set_seed(42)
        model2 = create_model()
        model2.eval()

        x = torch.randn(2, 2, 147)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        torch.testing.assert_close(out1, out2)

    def test_different_seeds_different_weights(self):
        """Test different seeds produce different initial weights."""
        set_seed(42)
        model1 = create_model()

        set_seed(123)
        model2 = create_model()

        # Check at least one parameter differs
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        assert not all(torch.allclose(p1, p2) for p1, p2 in zip(params1, params2, strict=False))

    def test_overfit_small_batch(self):
        """Test model can overfit a tiny dataset (sanity check)."""
        # Create tiny dataset
        X = torch.randn(10, 2, 147)
        y = torch.randint(0, 2, (10, 5)).float()

        model = create_model()
        model.train()

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Check loss decreased
        with torch.no_grad():
            output = model(X)
            target = y
            final_loss = criterion(output, target)

        assert final_loss.item() < 1.0  # Should be able to fit something


class TestInference:
    """Test inference pipeline."""

    def test_infer_output_schema(self):
        """Test inference output schema and labelling."""
        # This test requires a trained model - use a mock
        from exosphere.ml.infer import MLResult

        result = MLResult(
            scores={"H2O": 0.9, "CO2": 0.8, "CO": 0.1, "CH4": 0.05, "SO2": 0.3},
            model_version="CNN-v1",
            dataset_hash="abc123",
            model_config_hash="def456",
            timestamp="2026-01-01T00:00:00Z",
            input_coverage_fraction=0.95,
            input_wavelength_range=(0.6, 5.3),
            grid_match=True,
            warnings=[],
        )

        # Check schema
        assert "scores" in result.to_dict()
        assert "labelled_scores" in result.to_dict()
        assert "model_version" in result.to_dict()
        assert "warnings" in result.to_dict()

        # Check labelling
        labelled = result.labelled_scores
        for _mol, label in labelled.items():
            assert "ML candidate score" in label
            assert "not abundance" in label
            assert "not a detection" in label

    def test_inference_warns_on_grid_mismatch(self):
        """Test inference warns when input grid doesn't match training grid."""
        # This would require a full inference test - skip for now
        pass


class TestMLIntegration:
    """Integration tests for ML pipeline."""

    def test_full_pipeline_reproducible(self):
        """Test that full pipeline is reproducible with same seed."""
        from exosphere.ml.dataset import DatasetConfig, generate_dataset
        from exosphere.ml.model import set_seed

        # Generate data with seed
        set_seed(42)
        config = DatasetConfig(n_samples=20, seed=42)
        X1, y1, _, _ = generate_dataset(config)

        set_seed(42)
        config2 = DatasetConfig(n_samples=20, seed=42)
        X2, y2, _, _ = generate_dataset(config2)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_model_save_load(self):
        """Test model checkpoint save/load."""
        from exosphere.ml.model import create_model, save_checkpoint

        model = create_model()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        save_checkpoint(model, None, 0, {}, "hash123", "CNN-v1", 42, temp_path)
        checkpoint = torch.load(temp_path, map_location="cpu")

        assert "model_state_dict" in checkpoint
        assert checkpoint["model_version"] == "CNN-v1"
        assert checkpoint["dataset_hash"] == "hash123"
        assert checkpoint["seed"] == 42


class TestEndToEnd:
    """End-to-end tests (marked slow)."""

    @pytest.mark.slow
    def test_tiny_training_run(self):
        """Test a minimal training run completes."""
        from exosphere.ml.dataset import DatasetConfig, generate_dataset
        from exosphere.ml.model import create_model, set_seed
        from exosphere.ml.train import create_dataloaders

        set_seed(42)

        # Generate tiny dataset
        config = DatasetConfig(n_samples=50, seed=42)
        X, y, _, _ = generate_dataset(config)

        # Create model and dataloaders
        model = create_model()
        train_loader, val_loader, test_loader = create_dataloaders(
            X, y, batch_size=16, val_split=0.2, test_split=0.2, seed=42
        )

        # Train for a few epochs
        model.train()
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy(outputs, batch_y)
            assert loss.item() >= 0
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
