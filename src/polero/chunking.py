"""Sliding-window encoding (T2.1): the (capped) question is repeated in every chunk; the answer is windowed with overlap."""
import numpy as np

COMMA = {"ș": "ş", "ț": "ţ", "Ș": "Ş", "Ț": "Ţ"}
CEDILLA = {v: k for k, v in COMMA.items()}


def normalize(text, mode="comma"):
    """Whitespace/quote cleanup and diacritic form selection. mode: 'comma' | 'cedilla' | 'raw'."""
    if mode == "raw":
        return text
    text = " ".join(text.replace(" ", " ").split())
    text = text.replace("„", '"').replace("”", '"').replace("“", '"').replace("«", '"').replace("»", '"')
    table = CEDILLA if mode == "comma" else COMMA
    return text.translate(str.maketrans(table))


def pick_diacritic_mode(tokenizer, sample_texts):
    """Choose comma vs cedilla by the lower UNK rate (ties -> fewer tokens)."""
    best = None
    for mode in ("comma", "cedilla"):
        ids = [i for t in sample_texts for i in tokenizer(normalize(t, mode), add_special_tokens=False)["input_ids"]]
        unk = sum(i == tokenizer.unk_token_id for i in ids) / max(1, len(ids))
        key = (unk, len(ids))
        if best is None or key < best[0]:
            best = (key, mode)
    return best[1]


def select_chunks(n, max_chunks):
    """Indices of chunks to keep: first, last, evenly spaced in between."""
    if max_chunks is None or n <= max_chunks:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_chunks).round().astype(int).tolist()))


_TEMPLATES = {}


def pair_template(tokenizer):
    """(prefix, middle, suffix) special-token ids around a pair, learned by encoding a known pair.

    transformers 5 dropped build_inputs_with_special_tokens on fast tokenizers, so we infer the layout once.
    """
    key = id(tokenizer)
    if key not in _TEMPLATES:
        a = tokenizer("casa", add_special_tokens=False)["input_ids"]
        b = tokenizer("masa", add_special_tokens=False)["input_ids"]
        full = tokenizer("casa", "masa")["input_ids"]
        i = next(i for i in range(len(full)) if full[i : i + len(a)] == a)
        j = next(j for j in range(i + len(a), len(full)) if full[j : j + len(b)] == b)
        _TEMPLATES[key] = (full[:i], full[i + len(a) : j], full[j + len(b) :])
    return _TEMPLATES[key]


def encode_example(tokenizer, question, answer, max_len=512, q_max=128, stride=128, max_chunks=8):
    """Return a list of input_ids lists (one per chunk) for one Q/A pair."""
    q = tokenizer(question, add_special_tokens=False)["input_ids"][-q_max:]  # keep the end of the question
    a = tokenizer(answer, add_special_tokens=False)["input_ids"]
    pre, mid, suf = pair_template(tokenizer)
    window = max_len - len(q) - len(pre) - len(mid) - len(suf)
    assert window > stride, "window must exceed stride"
    step = window - stride
    starts = [0]
    while starts[-1] + window < len(a):
        starts.append(starts[-1] + step)
    chunks = [pre + q + mid + a[s : s + window] + suf for s in starts]
    return [chunks[i] for i in select_chunks(len(chunks), max_chunks)]


def encode_dataset(tokenizer, questions, answers, **kw):
    """List of per-example chunk lists."""
    return [encode_example(tokenizer, q, a, **kw) for q, a in zip(questions, answers)]
