"""T0.2: the package imports and the data CSVs are unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_import():
    import polero

    assert polero.__version__


def test_data_checksums():
    # line endings are normalised so the check holds on Windows (autocrlf) and Linux
    want = json.loads((ROOT / "tests" / "data_checksums.json").read_text())
    for name, digest in want.items():
        got = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert got == digest, f"{name} was modified"
