"""T0.2: package imports and the read-only data files are unchanged."""
import hashlib
import json
from pathlib import Path

import polero

CHECKSUMS = json.loads((Path(__file__).parent / "data_checksums.json").read_text())


def test_import_and_paths():
    assert polero.SEED == 42
    for p in (polero.TRAIN_CSV, polero.TEST_CSV, polero.SAMPLE_CSV):
        assert p.is_file(), f"missing data file {p}"


def test_data_files_unchanged():
    for name, digest in CHECKSUMS.items():
        actual = hashlib.sha256((polero.REPO_ROOT / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} was modified (CLAUDE.md §2.4)"
