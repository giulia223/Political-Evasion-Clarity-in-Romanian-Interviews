"""T0.1: installed packages match the pins in requirements.txt."""
import importlib.metadata as md
import re
from pathlib import Path

REQ = Path(__file__).resolve().parents[1] / "requirements.txt"


def pins():
    for line in REQ.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
        if m:
            yield m.groups()


def test_requirements_pinned_and_installed():
    found = list(pins())
    assert found, "requirements.txt has no pinned packages"
    for name, version in found:
        installed = md.version(name).split("+")[0]  # ignore local tags like +cpu
        assert installed == version, f"{name}: installed {installed}, pinned {version}"


def test_core_imports():
    import torch, transformers  # noqa: F401
    assert torch.tensor([1.0]).sum().item() == 1.0
