"""Multi-head chunk classifier (T2.2, D-003, D-004, D-005).

encoder(chunk) -> position-0 hidden state per chunk -> pool over the chunks of each example
(element-wise max by default) -> dropout -> clarity head (3) and evasion head (9).
Loss = w_clarity * L(clarity) + w_evasion * L(evasion), with L = cross-entropy by default.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from polero.labels import CLARITY, EVASION

POOLINGS = ("max", "mean", "attention")
LOSSES = ("ce", "focal")


@dataclass
class LossConfig:
    kind: str = "ce"                 # "ce" | "focal"
    gamma: float = 2.0               # focal only
    label_smoothing: float = 0.0     # ce only
    w_clarity: float = 1.0
    w_evasion: float = 1.0
    clarity_weights: list | None = None   # per-class weights (len 3), e.g. from class_weights()
    evasion_weights: list | None = None   # per-class weights (len 9)


def class_weights(labels, classes, power=0.5):
    """Inverse-frequency weights ** power (power 0.5 = sqrt-inverse), normalized to mean 1."""
    counts = np.array([max(sum(1 for l in labels if l == c), 1) for c in classes], dtype=float)
    w = (counts.sum() / counts) ** power
    return (w / w.mean()).tolist()


def focal_loss(logits, target, gamma, weight=None):
    logp = F.log_softmax(logits, -1)
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    loss = -((1 - logp_t.exp()) ** gamma) * logp_t
    if weight is not None:
        w = weight[target]
        return (loss * w).sum() / w.sum()
    return loss.mean()


def pool_chunks(h, example_index, n_examples, how="max", scores=None):
    """Pool chunk vectors h [n_chunks, d] into [n_examples, d] by example_index."""
    idx = example_index[:, None].expand_as(h)
    if how == "max":
        out = h.new_full((n_examples, h.shape[1]), float("-inf"))
        return out.scatter_reduce(0, idx, h, reduce="amax", include_self=True)
    if how == "mean":
        return h.new_zeros((n_examples, h.shape[1])).scatter_reduce(0, idx, h, reduce="mean", include_self=False)
    if how == "attention":  # softmax over each example's chunks
        s_max = scores.new_full((n_examples,), float("-inf")).scatter_reduce(0, example_index, scores, reduce="amax")
        e = (scores - s_max[example_index]).exp()
        denom = e.new_zeros(n_examples).index_add(0, example_index, e)
        a = e / denom[example_index]
        return h.new_zeros((n_examples, h.shape[1])).index_add(0, example_index, h * a[:, None])
    raise ValueError(f"unknown pooling {how!r}; choose from {POOLINGS}")


def enable_checkpointing(encoder):
    """Activation (gradient) checkpointing; transformers 5 renamed the method."""
    fn = getattr(encoder, "activation_checkpointing_enable", None) or encoder.gradient_checkpointing_enable
    fn()


class MultiHeadChunkClassifier(nn.Module):
    def __init__(self, encoder, pooling="max", dropout=0.1, loss=LossConfig()):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}")
        if loss.kind not in LOSSES:
            raise ValueError(f"unknown loss {loss.kind!r}")
        self.encoder, self.pooling, self.loss_cfg = encoder, pooling, loss
        d = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.clarity_head = nn.Linear(d, len(CLARITY))
        self.evasion_head = nn.Linear(d, len(EVASION))
        self.attn = nn.Linear(d, 1) if pooling == "attention" else None
        for name, w in (("clarity_w", loss.clarity_weights), ("evasion_w", loss.evasion_weights)):
            self.register_buffer(name, torch.tensor(w, dtype=torch.float) if w is not None else None)

    @classmethod
    def from_backbone(cls, name, gradient_checkpointing=False, **kw):
        from transformers import AutoModel
        try:
            encoder = AutoModel.from_pretrained(name, add_pooling_layer=False)
        except TypeError:  # architectures without a pooler argument
            encoder = AutoModel.from_pretrained(name)
        model = cls(encoder, **kw)
        if gradient_checkpointing:
            enable_checkpointing(model.encoder)
        return model

    def forward(self, input_ids, attention_mask, example_index, token_type_ids=None,
                clarity_labels=None, evasion_labels=None, n_examples=None):
        kw = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state[:, 0]
        n = int(example_index.max()) + 1 if n_examples is None else n_examples
        scores = self.attn(h).squeeze(-1) if self.attn is not None else None
        v = self.dropout(pool_chunks(h, example_index, n, self.pooling, scores))
        out = {"clarity_logits": self.clarity_head(v), "evasion_logits": self.evasion_head(v)}
        if clarity_labels is not None and evasion_labels is not None:
            lc = self._loss(out["clarity_logits"], clarity_labels, self.clarity_w)
            le = self._loss(out["evasion_logits"], evasion_labels, self.evasion_w)
            out["loss"] = self.loss_cfg.w_clarity * lc + self.loss_cfg.w_evasion * le
            out["clarity_loss"], out["evasion_loss"] = lc.detach(), le.detach()
        return out

    def _loss(self, logits, target, weight):
        if self.loss_cfg.kind == "focal":
            return focal_loss(logits, target, self.loss_cfg.gamma, weight)
        return F.cross_entropy(logits, target, weight=weight, label_smoothing=self.loss_cfg.label_smoothing)
