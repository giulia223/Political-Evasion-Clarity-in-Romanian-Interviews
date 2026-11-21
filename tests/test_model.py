"""T2.2 / T3.1: multi-head model and losses (tiny random BERT, CPU)."""
import pytest
import torch

from polero.model import MultiHeadChunkClassifier, ce_loss, multitask_loss, pool_chunks
from polero.tiny import tiny_config


def _model(pooling="max"):
    torch.manual_seed(0)
    return MultiHeadChunkClassifier(config=tiny_config(100, 64), pooling=pooling, dropout=0.0)


def _batch(chunks_per_ex=(1, 3, 1), L=12):
    g = torch.Generator().manual_seed(0)
    n = sum(chunks_per_ex)
    ids = torch.randint(5, 100, (n, L), generator=g)
    ex = torch.tensor([i for i, c in enumerate(chunks_per_ex) for _ in range(c)])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "example_index": ex}


@pytest.mark.parametrize("pooling", ["max", "mean", "attn"])
def test_shapes(pooling):
    m = _model(pooling)
    lc, le = m(**_batch())
    assert lc.shape == (3, 3) and le.shape == (3, 9)


def test_maxpool_manual():
    h = torch.randn(5, 4)
    ex = torch.tensor([0, 1, 1, 1, 2])
    out = pool_chunks(h, ex, 3, "max")
    assert torch.allclose(out[0], h[0]) and torch.allclose(out[1], h[1:4].max(0).values)
    assert torch.allclose(out[2], h[4])
    mean = pool_chunks(h, ex, 3, "mean")
    assert torch.allclose(mean[1], h[1:4].mean(0))


def test_overfit_and_grads():
    m = _model()
    m.train()
    b = _batch((1, 2, 1, 1, 3, 1, 1, 2))
    yc = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    ye = torch.tensor([0, 1, 6, 0, 4, 8, 0, 2])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for step in range(100):
        opt.zero_grad()
        lc, le = m(**b)
        loss = multitask_loss(lc, le, yc, ye)
        loss.backward()
        if step == 0:
            g = m.encoder.embeddings.word_embeddings.weight.grad
            assert g is not None and g.abs().sum() > 0
        opt.step()
    assert loss.item() < 0.1


@pytest.mark.parametrize("kw", [{}, {"focal_gamma": 2.0}, {"label_smoothing": 0.1},
                                {"weight": torch.tensor([1.0, 2.0, 3.0])},
                                {"weight": torch.tensor([1.0, 2.0, 3.0]), "focal_gamma": 2.0}])
def test_loss_variants(kw):
    logits = torch.randn(6, 3, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    loss = ce_loss(logits, y, **kw)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_zero_equals_ce():
    logits = torch.randn(6, 3)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    assert torch.allclose(ce_loss(logits, y), ce_loss(logits, y, focal_gamma=0.0))
