"""T1.2: text normalization."""
import pandas as pd
import pytest

from polero.data import load_test, load_train
from polero.text import BACKBONE_MODE, MODES, normalize


@pytest.fixture(scope="module")
def texts():
    df = pd.concat([load_train(), load_test()])
    return pd.concat([df.interview_question, df.interview_answer]).tolist()


def test_exact_mappings():
    assert normalize("a – b ‑ c") == "a - b - c"
    assert normalize("Ťara ťării") == "Țara țării"
    assert normalize("  a b\tc\n\nd   e  ") == "a b c d e"
    assert normalize("„x” “y” «z» ’w‘") == "„x” “y” «z» ’w‘"  # quotes untouched in 'clean'
    assert normalize("„x” “y” «z» ’w‘", "clean+quotes") == "\"x\" \"y\" \"z\" 'w'"
    assert normalize("  a – b ", "raw") == "  a – b "


def test_diacritics_preserved():
    s = "ăâîșțĂÂÎȘȚ şţ ê ğ"
    for mode in MODES:
        assert normalize(s, mode) == s


def test_empty_and_unknown_mode():
    for mode in MODES:
        assert normalize("", mode) == ""
    assert normalize("   \n", "clean") == ""
    with pytest.raises(ValueError):
        normalize("x", "lowercase")


def test_idempotent_on_all_data(texts):
    for mode in MODES:
        once = [normalize(t, mode) for t in texts]
        assert [normalize(t, mode) for t in once] == once


def test_chosen_mode_no_worse_than_raw(texts):
    from transformers import AutoTokenizer
    from polero.eda import unk_fertility

    checked = 0
    for name, mode in BACKBONE_MODE.items():
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except OSError:
            continue
        raw = unk_fertility(texts, tok)
        norm = unk_fertility([normalize(t, mode) for t in texts], tok)
        assert norm["unk_rate"] <= raw["unk_rate"], name
        assert norm["fertility"] <= raw["fertility"] * 1.001, name  # 0.1% tolerance (D-022)
        checked += 1
    if not checked:
        pytest.skip("no tokenizer cached (run scripts/eda.py once with network)")
