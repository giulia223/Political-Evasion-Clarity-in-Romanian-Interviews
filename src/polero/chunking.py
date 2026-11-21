"""Sliding-window encoding of question + answer pairs (T2.1, D-021, D-023).

Each example becomes one or more chunks laid out as the tokenizer's own pair template:
    [prefix] question [middle] answer-window [suffix]      e.g. [CLS] q [SEP] a [SEP]
- the question (capped at q_cap tokens, keeping its END) is repeated in every chunk;
- the answer is split into windows that overlap by `overlap` tokens, the last window
  aligned to the end of the answer;
- at most max_chunks windows are kept (first, last and evenly spaced ones in between).
No padding happens here; `collate_chunks` pads per batch.
"""
from dataclasses import dataclass

import numpy as np
import torch

from polero.text import mode_for, normalize


@dataclass
class ChunkConfig:
    max_length: int = 512
    q_cap: int = 192
    overlap: int = 128
    max_chunks: int = 8
    norm_mode: str | None = None  # None -> polero.text.mode_for(backbone)


class PairTemplate:
    """Special tokens and token-type ids of a tokenizer's pair encoding, learned from one dummy pair."""

    def __init__(self, tok):
        q = tok("a", add_special_tokens=False)["input_ids"]
        a = tok("b", add_special_tokens=False)["input_ids"]
        enc = tok("a", "b", return_special_tokens_mask=True)
        ids, special = enc["input_ids"], enc["special_tokens_mask"]
        types = enc.get("token_type_ids") or [0] * len(ids)
        content = [i for i, s in enumerate(special) if not s]
        if [ids[i] for i in content] != q + a:
            raise ValueError("could not infer the pair template from this tokenizer")
        q_end, a_start = content[len(q) - 1] + 1, content[len(q)]
        self.prefix, self.middle, self.suffix = ids[: content[0]], ids[q_end:a_start], ids[content[-1] + 1:]
        self.prefix_t, self.middle_t, self.suffix_t = types[: content[0]], types[q_end:a_start], types[content[-1] + 1:]
        self.q_type, self.a_type = types[content[0]], types[a_start]
        self.uses_token_types = "token_type_ids" in tok.model_input_names
        self.n_special = len(self.prefix) + len(self.middle) + len(self.suffix)

    def build(self, q_ids, a_ids):
        ids = self.prefix + q_ids + self.middle + a_ids + self.suffix
        types = self.prefix_t + [self.q_type] * len(q_ids) + self.middle_t + [self.a_type] * len(a_ids) + self.suffix_t
        return ids, types


def window_starts(n_answer, budget, overlap):
    """Start offsets of answer windows of size `budget`; the last one ends exactly at n_answer."""
    if n_answer <= budget:
        return [0]
    step = budget - overlap
    if step <= 0:
        raise ValueError(f"overlap {overlap} must be smaller than the answer budget {budget}")
    n = 1 + int(np.ceil((n_answer - budget) / step))
    return [k * step for k in range(n - 1)] + [n_answer - budget]


def select_chunks(n, max_chunks):
    """Indices of the chunks kept when there are more than max_chunks: first, last, evenly spaced."""
    if n <= max_chunks:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_chunks).round().astype(int).tolist()))


def encode_pair(question, answer, tok, cfg=ChunkConfig(), template=None):
    """Returns a dict: chunks (list of {input_ids, token_type_ids}), n_windows, n_kept, q_truncated."""
    template = template or PairTemplate(tok)
    q_ids = tok(question, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    q_truncated = len(q_ids) > cfg.q_cap
    q_ids = q_ids[-cfg.q_cap:] if q_truncated else q_ids
    budget = cfg.max_length - template.n_special - len(q_ids)
    starts = window_starts(len(a_ids), budget, cfg.overlap)
    keep = select_chunks(len(starts), cfg.max_chunks)
    chunks = []
    for k in keep:
        ids, types = template.build(q_ids, a_ids[starts[k]: starts[k] + budget])
        chunks.append({"input_ids": ids, "token_type_ids": types})
    return {"chunks": chunks, "n_windows": len(starts), "n_kept": len(keep), "q_truncated": q_truncated,
            "starts": [starts[k] for k in keep], "budget": budget, "n_answer": len(a_ids)}


def encode_examples(questions, answers, tok, cfg=ChunkConfig(), backbone=None):
    """Normalize (T1.2) and encode many pairs. Returns a list of encode_pair dicts."""
    mode = cfg.norm_mode or mode_for(backbone or getattr(tok, "name_or_path", ""))
    template = PairTemplate(tok)
    return [encode_pair(normalize(q, mode), normalize(a, mode), tok, cfg, template) for q, a in zip(questions, answers)]


def collate_chunks(examples, pad_id, use_token_types=True):
    """Flatten the chunks of a batch of encoded examples and pad them to the longest chunk.

    Returns tensors input_ids, attention_mask, (token_type_ids), and example_index
    (the position in `examples` of each chunk), ready for MultiHeadChunkClassifier.
    """
    chunks = [(i, c) for i, ex in enumerate(examples) for c in ex["chunks"]]
    width = max(len(c["input_ids"]) for _, c in chunks)
    ids = torch.full((len(chunks), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(chunks), width), dtype=torch.long)
    types = torch.zeros((len(chunks), width), dtype=torch.long)
    for row, (_, c) in enumerate(chunks):
        n = len(c["input_ids"])
        ids[row, :n] = torch.tensor(c["input_ids"])
        mask[row, :n] = 1
        types[row, :n] = torch.tensor(c["token_type_ids"])
    out = {"input_ids": ids, "attention_mask": mask, "example_index": torch.tensor([i for i, _ in chunks])}
    if use_token_types:
        out["token_type_ids"] = types
    return out
