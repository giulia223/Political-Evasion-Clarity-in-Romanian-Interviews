"""T1.1: EDA functions run on 50 rows (no network: tokenizers must already be cached)."""
import pytest

from polero.data import load_test, load_train
from polero.eda import TOKENIZERS, build_report, n_chunks


def test_n_chunks():
    # budget = 512 - 3 - 100 = 409; window step = 409 - 128 = 281
    assert n_chunks([100], [409]).tolist() == [1]
    assert n_chunks([100], [410]).tolist() == [2]
    assert n_chunks([100], [409 + 281 + 1]).tolist() == [3]
    assert n_chunks([500], [10]).tolist() == [1]  # question capped at 128


def test_report_on_50_rows():
    try:
        from transformers import AutoTokenizer
        toks = {k: AutoTokenizer.from_pretrained(v, local_files_only=True) for k, v in TOKENIZERS.items()}
    except OSError:
        pytest.skip("tokenizers not cached (run scripts/eda.py once with network)")
    report = build_report(load_train().head(50), load_test().head(50), toks)
    for section in ("## 1. Tokenizers", "## 2. Token lengths", "## 4. Duplicates", "## 5. Interview structure", "## 6. Clarity rates"):
        assert section in report
