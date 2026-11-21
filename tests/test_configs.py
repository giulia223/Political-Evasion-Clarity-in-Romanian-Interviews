"""Every experiment config loads, and its backbone has a measured normalization mode (T1.2)."""
import pytest

from polero import CONFIGS
from polero.chunking import ChunkConfig
from polero.model import LossConfig
from polero.text import BACKBONE_MODE
from polero.train import TrainConfig

CONFIG_FILES = sorted(CONFIGS.glob("*.json"))


def test_configs_exist():
    assert {p.stem for p in CONFIG_FILES} >= {"robert-large_mh", "xlmr-large_mh", "ro-bert-base_mh"}


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_valid(path):
    cfg = TrainConfig.from_json(path)
    assert cfg.backbone in BACKBONE_MODE
    ChunkConfig(**cfg.chunk)
    LossConfig(**{k: v for k, v in cfg.loss.items() if k != "class_weight_power"})
    assert cfg.seed == 42 and 0 < cfg.lr < 1e-3 and cfg.n_train_rows is None and cfg.max_steps is None
