"""T1.1: EDA functions run on a few rows."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _eda():
    spec = importlib.util.spec_from_file_location("eda", ROOT / "scripts" / "eda.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_structure(train, test_df):
    out = _eda().structure_section(train.head(50), test_df.head(50))
    assert out["train_dup_pairs"] >= 0 and out["interviews"] > 0


def test_tokenizer_section_if_cached(train):
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache

    name = "readerbench/RoBERT-large"
    if not isinstance(try_to_load_from_cache(name, "tokenizer_config.json"), str):
        pytest.skip("tokenizer not cached")
    r = _eda().tokenizer_section(name, train, 50)
    assert r["fast"] and r["best_mode"] in ("basic", "comma", "cedilla")
