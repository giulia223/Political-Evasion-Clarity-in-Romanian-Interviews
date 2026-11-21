"""Multi-head chunk classifier (T2.2): shared encoder, position-0 state per chunk, pooled over chunks, clarity + evasion heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MultiHeadChunkClassifier(nn.Module):
    def __init__(self, backbone=None, encoder=None, pooling="max", dropout=0.1, grad_ckpt=False):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(backbone)
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        h = self.encoder.config.hidden_size
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.head_cl = nn.Linear(h, 3)
        self.head_ev = nn.Linear(h, 9)
        if pooling == "attn":
            self.attn = nn.Linear(h, 1)

    def pool(self, H, example_index, n_examples):
        if self.pooling == "max":
            out = H.new_full((n_examples, H.size(1)), float("-inf"))
            return out.index_reduce(0, example_index, H, "amax", include_self=True)
        if self.pooling == "mean":
            out = H.new_zeros((n_examples, H.size(1)))
            out = out.index_add(0, example_index, H)
            cnt = torch.bincount(example_index, minlength=n_examples).clamp(min=1).to(H.dtype)
            return out / cnt[:, None]
        if self.pooling == "attn":
            s = self.attn(H).squeeze(-1).float()
            s = s - s.max()
            w = s.exp()
            den = torch.zeros(n_examples, device=H.device).index_add(0, example_index, w)
            w = (w / den[example_index]).to(H.dtype)
            return H.new_zeros((n_examples, H.size(1))).index_add(0, example_index, H * w[:, None])
        if self.pooling == "first":
            first = torch.zeros(n_examples, dtype=torch.long, device=H.device)
            idx = torch.arange(len(example_index), device=H.device)
            first = first.scatter_reduce(0, example_index, idx, "amin", include_self=False)
            return H[first]
        raise ValueError(self.pooling)

    def forward(self, input_ids, attention_mask, example_index, n_examples):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        z = self.drop(self.pool(H, example_index, n_examples))
        return self.head_cl(z), self.head_ev(z)


def multitask_loss(logits_cl, logits_ev, y_cl, y_ev, w_c=1.0, w_e=1.0, ls=0.0, class_w_ev=None, focal_gamma=0.0):
    def ce(logits, y, w):
        if focal_gamma > 0:
            logp = F.log_softmax(logits.float(), -1)
            lp = logp.gather(1, y[:, None]).squeeze(1)
            loss = -((1 - lp.exp()) ** focal_gamma) * lp
            if w is not None:
                loss = loss * w[y]
            return loss.mean()
        return F.cross_entropy(logits.float(), y, weight=w, label_smoothing=ls)

    return w_c * ce(logits_cl, y_cl, None) + w_e * ce(logits_ev, y_ev, class_w_ev)
