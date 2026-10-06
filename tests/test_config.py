"""Tests for project path configuration."""

from exosphere.core import config


def test_data_cache_default():
    cfg = config.load_config()
    assert cfg.data_cache_dir.name == "data_cache"
    assert cfg.data_cache_dir == config.PROJECT_ROOT / "data_cache"


def test_prt_path_unset_by_default(monkeypatch):
    monkeypatch.delenv(config.PRT_INPUT_DATA_PATH_ENV_VAR, raising=False)
    assert config.load_config().prt_input_data_path is None


def test_prt_path_read_from_env_var(monkeypatch):
    monkeypatch.setenv(config.PRT_INPUT_DATA_PATH_ENV_VAR, "D:/pRT/input")
    cfg = config.load_config()
    assert cfg.prt_input_data_path is not None
    assert cfg.prt_input_data_path.parts[-2:] == ("pRT", "input")
