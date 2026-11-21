"""Multi-head chunked encoder (T2.2, D-005).

encoder -> position-0 hidden state per chunk -> pool over the chunks of each
example (max by default; mean/attention for ablation) -> dropout -> two heads
(clarity 3, evasion 9).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import CLARITY, EVASION


def pool_chunks(h, example_index, n_examples, mode="max", attn=None):
    """Pool chunk vectors h [n_chunks, H] into [n_examples, H]."""
    H = h.size(1)
    idx = example_index.unsqueeze(1).expand(-1, H)
    if mode == "max":
        out = h.new_full((n_examples, H), float("-inf"))
        return out.scatter_reduce(0, idx, h, reduce="amax", include_self=True)
    if mode == "mean":
        s = h.new_zeros((n_examples, H)).index_add(0, example_index, h)
        cnt = torch.bincount(example_index, minlength=n_examples).clamp(min=1).to(h.dtype)
        return s / cnt.unsqueeze(1)
    if mode == "attn":
        score = attn(h).squeeze(-1)  # [n_chunks]
        mx = score.new_full((n_examples,), float("-inf")).scatter_reduce(
            0, example_index, score, reduce="amax", include_self=True)
        w = torch.exp(score - mx[example_index])
        den = w.new_zeros(n_examples).index_add(0, example_index, w)
        w = (w / den[example_index]).unsqueeze(1)
        return h.new_zeros((n_examples, H)).index_add(0, example_index, w * h)
    raise ValueError(f"unknown pooling {mode}")


class MultiHeadChunkClassifier(nn.Module):
    def __init__(self, backbone_name=None, config=None, pooling="max", dropout=0.1,
                 n_clarity=len(CLARITY), n_evasion=len(EVASION)):
        super().__init__()
        from transformers import AutoModel

        if config is not None:
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(backbone_name)
        H = self.encoder.config.hidden_size
        self.pooling = pooling
        self.attn = nn.Linear(H, 1) if pooling == "attn" else None
        self.dropout = nn.Dropout(dropout)
        self.head_cl = nn.Linear(H, n_clarity)
        self.head_ev = nn.Linear(H, n_evasion)

    def gradient_checkpointing(self, on=True):
        if on:
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, example_index, token_type_ids=None, n_examples=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        h = self.encoder(**kw).last_hidden_state[:, 0]
        n = int(n_examples if n_examples is not None else example_index.max().item() + 1)
        z = self.dropout(pool_chunks(h, example_index, n, self.pooling, self.attn))
        return self.head_cl(z), self.head_ev(z)


def ce_loss(logits, target, weight=None, focal_gamma=0.0, label_smoothing=0.0):
    if focal_gamma and focal_gamma > 0:
        logp = F.log_softmax(logits.float(), -1)
        lp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - lp.exp()) ** focal_gamma) * lp
        if weight is not None:
            loss = loss * weight[target]
            return loss.sum() / weight[target].sum()
        return loss.mean()
    return F.cross_entropy(logits.float(), target, weight=weight, label_smoothing=label_smoothing)


def multitask_loss(logits_cl, logits_ev, y_cl, y_ev, w_c=1.0, w_e=1.0, weight_cl=None, weight_ev=None,
                   focal_gamma=0.0, label_smoothing=0.0):
    lc = ce_loss(logits_cl, y_cl, weight_cl, focal_gamma, label_smoothing)
    le = ce_loss(logits_ev, y_ev, weight_ev, focal_gamma, label_smoothing)
    return w_c * lc + w_e * le
