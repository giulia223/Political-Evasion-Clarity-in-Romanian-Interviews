"""Text normalization (T1.2, D-021). Diacritics are never stripped or converted.

Modes:
  raw           unchanged
  clean         dashes -> '-', t-caron typo -> ț, all whitespace runs (incl. nbsp, tab, newline) -> one space, strip
  clean+quotes  clean, plus curly/low/angle quotes -> ASCII ' and "
"""
import re

DASHES = {"–": "-", "‑": "-", "‒": "-", "−": "-"}  # en dash, non-breaking hyphen, figure dash, minus
TYPOS = {"ť": "ț", "Ť": "Ț"}  # Slovak t-caron typed for Romanian ț (3 occurrences)
QUOTES = {"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "’": "'", "‘": "'"}

_CLEAN = str.maketrans({**DASHES, **TYPOS})
_QUOTES = str.maketrans(QUOTES)
_WS = re.compile(r"\s+")  # unicode-aware: covers nbsp, tab, newline

MODES = ("raw", "clean", "clean+quotes")

# Chosen mode per backbone (measured in docs/EDA.md §8, D-022).
BACKBONE_MODE = {
    "readerbench/RoBERT-large": "clean",
    "readerbench/RoBERT-base": "clean",
    "dumitrescustefan/bert-base-romanian-cased-v1": "clean",
    "FacebookAI/xlm-roberta-large": "clean",
}


def normalize(text, mode="clean"):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    if mode == "raw":
        return text
    text = text.translate(_CLEAN)
    if mode == "clean+quotes":
        text = text.translate(_QUOTES)
    return _WS.sub(" ", text).strip()


def mode_for(backbone):
    """Normalization mode for a backbone; 'clean' for anything not measured yet."""
    return BACKBONE_MODE.get(backbone, "clean")
