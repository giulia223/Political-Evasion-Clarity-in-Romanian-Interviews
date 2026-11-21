"""T2.2: multi-head chunk classifier, using a tiny randomly initialised BERT (no download)."""
import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from polero.labels import CLARITY, EVASION
from polero.model import (LossConfig, MultiHeadChunkClassifier, class_weights, enable_checkpointing,
                          focal_loss, pool_chunks)

torch.manual_seed(42)


def tiny_encoder():
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    return BertModel(cfg, add_pooling_layer=False)


def batch(chunks_per_example, length=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(chunks_per_example)
    return {"input_ids": torch.randint(5, 100, (n, length), generator=g),
            "attention_mask": torch.ones(n, length, dtype=torch.long),
            "example_index": torch.tensor([i for i, k in enumerate(chunks_per_example) for _ in range(k)])}


@pytest.mark.parametrize("pooling", ["max", "mean", "attention"])
def test_shapes_with_variable_chunks(pooling):
    m = MultiHeadChunkClassifier(tiny_encoder(), pooling=pooling).eval()
    out = m(**batch([1, 3, 1]))
    assert out["clarity_logits"].shape == (3, len(CLARITY))
    assert out["evasion_logits"].shape == (3, len(EVASION))
    assert torch.isfinite(out["evasion_logits"]).all()


def test_max_pool_matches_manual():
    h = torch.randn(5, 4)
    idx = torch.tensor([0, 1, 1, 1, 2])
    pooled = pool_chunks(h, idx, 3, "max")
    assert torch.equal(pooled[0], h[0]) and torch.equal(pooled[2], h[4])
    assert torch.equal(pooled[1], h[1:4].max(0).values)
    assert torch.allclose(pool_chunks(h, idx, 3, "mean")[1], h[1:4].mean(0))
    att = pool_chunks(h, idx, 3, "attention", scores=torch.zeros(5))  # equal scores -> mean
    assert torch.allclose(att[1], h[1:4].mean(0), atol=1e-6) and torch.allclose(att[0], h[0])


def test_model_pool_equals_manual_cls_max():
    m = MultiHeadChunkClassifier(tiny_encoder(), dropout=0.0).eval()
    b = batch([2, 1])
    with torch.no_grad():
        h = m.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state[:, 0]
        expected = m.evasion_head(torch.stack([h[:2].max(0).values, h[2]]))
        assert torch.allclose(m(**b)["evasion_logits"], expected, atol=1e-6)


def test_overfits_8_examples():
    torch.manual_seed(0)
    m = MultiHeadChunkClassifier(tiny_encoder(), dropout=0.0)
    b = batch([1, 2, 1, 3, 1, 1, 2, 1], seed=1)
    yc, ye = torch.tensor([0, 1, 2, 1, 0, 1, 2, 0]), torch.tensor([0, 1, 7, 4, 0, 2, 8, 0])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(100):
        loss = m(**b, clarity_labels=yc, evasion_labels=ye)["loss"]
        opt.zero_grad(), loss.backward(), opt.step()
    assert loss.item() < 0.1


def test_gradients_reach_encoder():
    m = MultiHeadChunkClassifier(tiny_encoder())
    out = m(**batch([2, 1]), clarity_labels=torch.tensor([0, 1]), evasion_labels=torch.tensor([0, 3]))
    out["loss"].backward()
    g = m.encoder.embeddings.word_embeddings.weight.grad
    assert g is not None and g.abs().sum() > 0


@pytest.mark.parametrize("loss", [
    LossConfig(), LossConfig(kind="focal", gamma=2.0), LossConfig(label_smoothing=0.1),
    LossConfig(w_clarity=0.5, evasion_weights=[1.0] * 8 + [3.0], clarity_weights=[1.0, 1.0, 2.0]),
    LossConfig(kind="focal", evasion_weights=[0.5] * 9),
])
def test_loss_variants_finite(loss):
    m = MultiHeadChunkClassifier(tiny_encoder(), loss=loss)
    out = m(**batch([1, 2, 1]), clarity_labels=torch.tensor([0, 1, 2]), evasion_labels=torch.tensor([0, 3, 8]))
    assert torch.isfinite(out["loss"]) and out["loss"].ndim == 0


def test_focal_reduces_to_ce_at_gamma_zero():
    logits, y = torch.randn(6, 9), torch.randint(0, 9, (6,))
    assert torch.allclose(focal_loss(logits, y, 0.0), F.cross_entropy(logits, y))
    w = torch.rand(9) + 0.5
    assert torch.allclose(focal_loss(logits, y, 0.0, w), F.cross_entropy(logits, y, weight=w))


def test_class_weights():
    w = class_weights(["a"] * 90 + ["b"] * 10, ["a", "b"], power=1.0)
    assert w[1] / w[0] == pytest.approx(9.0) and sum(w) / 2 == pytest.approx(1.0)
    w = class_weights(["a"] * 90 + ["b"] * 10, ["a", "b"], power=0.5)
    assert w[1] / w[0] == pytest.approx(3.0)


def test_invalid_options():
    with pytest.raises(ValueError):
        MultiHeadChunkClassifier(tiny_encoder(), pooling="sum")
    with pytest.raises(ValueError):
        MultiHeadChunkClassifier(tiny_encoder(), loss=LossConfig(kind="hinge"))


def test_gradient_checkpointing_runs():
    m = MultiHeadChunkClassifier(tiny_encoder())
    enable_checkpointing(m.encoder)
    m.train()
    out = m(**batch([2, 1]), clarity_labels=torch.tensor([0, 1]), evasion_labels=torch.tensor([0, 3]))
    out["loss"].backward()
    assert m.encoder.embeddings.word_embeddings.weight.grad is not None


def test_end_to_end_with_chunking(tmp_path):
    from transformers import BertTokenizerFast
    from polero.chunking import ChunkConfig, collate_chunks, encode_pair
    words = [f"w{i}" for i in range(90)]
    (tmp_path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    cfg = ChunkConfig(max_length=32, q_cap=8, overlap=4, max_chunks=4)
    exs = [encode_pair("w1 w2", " ".join(words[:n]), tok, cfg) for n in (3, 60, 1)]
    b = collate_chunks(exs, tok.pad_token_id)
    assert b["example_index"].tolist().count(1) > 1  # middle example spans several chunks
    out = MultiHeadChunkClassifier(tiny_encoder())(**b, clarity_labels=torch.tensor([0, 1, 2]),
                                                   evasion_labels=torch.tensor([0, 4, 8]))
    assert out["evasion_logits"].shape == (3, 9) and torch.isfinite(out["loss"])
