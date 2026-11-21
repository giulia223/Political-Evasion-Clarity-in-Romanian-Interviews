"""Sliding-window encoding of question + answer (T2.1, D-018).

Each example becomes 1..max_chunks chunks of the form
    <special> question <special> answer-window <special>
The question (capped at q_max tokens, truncated from the start so its end
survives) is repeated in every chunk. The answer is split into windows that
overlap by `stride` tokens. The special-token template is inferred from the
tokenizer itself, so BERT-, RoBERTa- and XLM-R-style tokenizers all work.
"""
import numpy as np


class PairTemplate:
    """Special-token layout of a pair encoding, inferred from the tokenizer."""

    def __init__(self, tok):
        a, b = "alfa", "beta"
        enc = tok(a, b, add_special_tokens=True)
        ids = list(enc["input_ids"])
        tts = list(enc.get("token_type_ids") or [0] * len(ids))
        ai = list(tok(a, add_special_tokens=False)["input_ids"])
        bi = list(tok(b, add_special_tokens=False)["input_ids"])
        ia = _find(ids, ai, 0)
        ib = _find(ids, bi, ia + len(ai))
        self.prefix, self.mid, self.suffix = ids[:ia], ids[ia + len(ai):ib], ids[ib + len(bi):]
        self.tt_prefix, self.tt_a = tts[:ia], tts[ia]
        self.tt_mid, self.tt_b = tts[ia + len(ai):ib], tts[ib]
        self.tt_suffix = tts[ib + len(bi):]
        self.n_special = len(self.prefix) + len(self.mid) + len(self.suffix)
        self.has_tt = "token_type_ids" in enc

    def build(self, q, a):
        ids = self.prefix + q + self.mid + a + self.suffix
        tt = self.tt_prefix + [self.tt_a] * len(q) + self.tt_mid + [self.tt_b] * len(a) + self.tt_suffix
        return ids, tt


def _find(seq, sub, start):
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    raise ValueError("could not infer the tokenizer's pair template")


def select_chunks(n, max_chunks):
    """Indices of chunks to keep: first, last and evenly spaced in between."""
    if max_chunks is None or n <= max_chunks:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_chunks).round().astype(int).tolist()))


def answer_windows(n_tokens, room, stride):
    """Start offsets of answer windows of size `room` overlapping by `stride`."""
    if n_tokens <= room:
        return [0]
    step = max(1, room - stride)
    starts = list(range(0, n_tokens - room + 1, step))
    if starts[-1] + room < n_tokens:
        starts.append(n_tokens - room)
    return starts


def encode_examples(tok, questions, answers, max_length=512, stride=128, q_max=128,
                    max_chunks=8, template=None):
    """Return (chunks, stats). chunks[i] = list of (input_ids, token_type_ids) for example i."""
    tpl = template or PairTemplate(tok)
    qs = tok(list(questions), add_special_tokens=False)["input_ids"]
    ans = tok(list(answers), add_special_tokens=False)["input_ids"]
    out, truncated = [], 0
    for q, a in zip(qs, ans):
        q = list(q)[-q_max:] if q_max else list(q)
        room = max_length - tpl.n_special - len(q)
        if room < 16:  # pathological question length: shrink it
            q = q[-(max_length - tpl.n_special - 16):]
            room = 16
        a = list(a)
        starts = answer_windows(len(a), room, stride)
        keep = select_chunks(len(starts), max_chunks)
        if len(keep) < len(starts):
            truncated += 1
        out.append([tpl.build(q, a[s:s + room]) for s in (starts[i] for i in keep)])
    stats = {"examples": len(out), "chunks": sum(len(c) for c in out),
             "multi_chunk": sum(len(c) > 1 for c in out), "max_chunks_truncated": truncated}
    return out, stats


def collate(batch_chunks, pad_id, use_tt=True):
    """Flatten chunks of several examples into padded tensors.

    batch_chunks: list (examples) of lists (chunks) of (ids, tt).
    Returns dict with input_ids, attention_mask, [token_type_ids], example_index.
    """
    import torch

    flat, ex = [], []
    for i, chunks in enumerate(batch_chunks):
        for c in chunks:
            flat.append(c)
            ex.append(i)
    L = max(len(c[0]) for c in flat)
    ids = torch.full((len(flat), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(flat), L), dtype=torch.long)
    tt = torch.zeros((len(flat), L), dtype=torch.long)
    for j, (c_ids, c_tt) in enumerate(flat):
        ids[j, :len(c_ids)] = torch.tensor(c_ids)
        mask[j, :len(c_ids)] = 1
        tt[j, :len(c_tt)] = torch.tensor(c_tt)
    out = {"input_ids": ids, "attention_mask": mask, "example_index": torch.tensor(ex, dtype=torch.long)}
    if use_tt:
        out["token_type_ids"] = tt
    return out
