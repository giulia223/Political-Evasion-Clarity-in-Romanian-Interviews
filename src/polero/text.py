"""Text normalization (T1.2).

normalize(text, mode):
  "none"     : unchanged
  "basic"    : unicode NFC, whitespace, quotes and dashes unified; diacritics untouched
  "comma"    : basic + cedilla ş ţ Ş Ţ -> comma-below ș ț Ș Ț
  "cedilla"  : basic + comma-below -> cedilla (for vocabularies built on cedilla text)
Diacritics are never stripped. The mode per backbone is set in the config (D-019).
"""
import re
import unicodedata

TO_COMMA = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})
TO_CEDILLA = str.maketrans({"ș": "ş", "ț": "ţ", "Ș": "Ş", "Ț": "Ţ"})
QUOTES = str.maketrans({"„": '"', "”": '"', "“": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‚": "'"})
DASHES = str.maketrans({"–": "-", "—": "-", "‒": "-", "―": "-", "­": ""})
_WS = re.compile(r"\s+")
MODES = ("none", "basic", "comma", "cedilla")


def normalize(text, mode="basic"):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    if mode == "none" or not text:
        return text or ""
    t = unicodedata.normalize("NFC", text).translate(QUOTES).translate(DASHES)
    t = _WS.sub(" ", t).strip()
    if mode == "comma":
        t = t.translate(TO_COMMA)
    elif mode == "cedilla":
        t = t.translate(TO_CEDILLA)
    return t
