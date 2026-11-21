"""T1.2: normalization."""
import pytest

from polero.text import MODES, normalize

S = "  „Ştiinţa”  —  şi   ţara – ‘noastră’ \n Ș Ț "


@pytest.mark.parametrize("mode", MODES)
def test_idempotent(mode):
    assert normalize(normalize(S, mode), mode) == normalize(S, mode)


def test_tables():
    assert normalize("ş ţ Ş Ţ", "comma") == "ș ț Ș Ț"
    assert normalize("ș ț Ș Ț", "cedilla") == "ş ţ Ş Ţ"
    assert normalize('„a” – b', "basic") == '"a" - b'
    assert normalize("ă â î ș", "basic") == "ă â î ș"  # diacritics never stripped


def test_empty():
    for m in MODES:
        assert normalize("", m) == ""
    assert normalize(None, "basic") == ""
