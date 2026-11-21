"""Chunking, model pooling/overfit, train smoke run, LLM parsing with a fake backend."""
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertModel

from polero.chunking import encode_example, normalize, select_chunks
from polero.data import load_all
from polero.labels import EVASION
from polero.llm import FakeBackend, classify, fewshot_block, parse_label, select_fewshot
from polero.model import MultiHeadChunkClassifier, multitask_loss
from polero.train import EarlyStopper, smoke_setup


def _tok():
    df = load_all()
    return smoke_setup(df)[0]


def test_chunking_coverage_and_question():
    tok = _tok()
    q = "Când va fi gata autostrada?"
    a = " ".join(["Lucrările vor fi finalizate în martie"] * 400)
    chunks = encode_example(tok, q, a, max_len=64, q_max=16, stride=8, max_chunks=None)
    qids = tok(q, add_special_tokens=False)["input_ids"]
    aids = tok(a, add_special_tokens=False)["input_ids"]
    assert len(chunks) > 1 and all(len(c) <= 64 for c in chunks)
    assert all(c[1 : 1 + len(qids)] == qids for c in chunks)
    covered = sum(len(c) - len(qids) - 3 for c in chunks)
    assert covered >= len(aids)
    assert len(encode_example(tok, q, "Da.", max_len=64, q_max=16, stride=8)) == 1
    assert select_chunks(20, 5) == [0, 5, 10, 14, 19] or len(select_chunks(20, 5)) == 5
    assert select_chunks(20, 5)[0] == 0 and select_chunks(20, 5)[-1] == 19


def test_normalize_idempotent():
    s = "Ştiu  că ţara „merge”"
    for mode in ("comma", "cedilla"):
        assert normalize(normalize(s, mode), mode) == normalize(s, mode)
    assert normalize("", "comma") == ""
    assert normalize("ş ţ", "comma") == "ș ț"


def _tiny_model(pooling="max"):
    conf = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                      max_position_embeddings=32)
    torch.manual_seed(0)
    return MultiHeadChunkClassifier(encoder=BertModel(conf, add_pooling_layer=False), pooling=pooling, dropout=0.0)


def test_pooling_shapes_and_max():
    m = _tiny_model()
    ids = torch.randint(5, 50, (5, 10))
    mask = torch.ones_like(ids)
    ex = torch.tensor([0, 1, 1, 1, 2])
    lc, le = m(ids, mask, ex, 3)
    assert lc.shape == (3, 3) and le.shape == (3, 9)
    H = m.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
    manual = torch.stack([H[0], H[1:4].max(0).values, H[4]])
    assert torch.allclose(m.pool(H, ex, 3), manual)
    for pooling in ("mean", "attn", "first"):
        assert _tiny_model(pooling)(ids, mask, ex, 3)[1].shape == (3, 9)


def test_overfit_and_grads():
    m = _tiny_model()
    m.train()
    ids = torch.randint(5, 50, (8, 12))
    mask = torch.ones_like(ids)
    ex = torch.arange(8)
    y_ev = torch.arange(8) % 9
    y_cl = y_ev % 3
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = multitask_loss(*m(ids, mask, ex, 8), y_cl, y_ev)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
    assert torch.isfinite(multitask_loss(*m(ids, mask, ex, 8), y_cl, y_ev, ls=0.1, focal_gamma=2.0))


def test_early_stopper():
    s = EarlyStopper(2)
    assert s.step(0.5) and not s.step(0.4) and not s.stop and not s.step(0.5) and s.stop


def test_train_smoke(tmp_path, monkeypatch):
    import polero.train as T

    monkeypatch.setattr(T, "ARTIFACTS", tmp_path)
    cfg = dict(T.DEFAULTS, smoke=True)
    T.run(cfg, [0], torch.device("cpu"), "smoke")
    oof = pd.read_csv(tmp_path / "smoke" / "oof.csv")
    te = pd.read_csv(tmp_path / "smoke" / "test_fold0.csv")
    assert oof.shape[1] == 13 and te.shape == (8, 13)
    assert np.allclose(oof.iloc[:, 1:10].sum(1), 1, atol=1e-4)


def test_llm_parse_and_fake(tmp_path):
    assert parse_label("blah\nFINAL: F") == "Deflection"
    assert parse_label("FINAL: **C**") == "General"
    assert parse_label("FINAL: Partial/half-answer") == "Partial/half-answer"
    assert parse_label("FINAL: declining to answer") == "Declining to answer"
    assert parse_label("<think>FINAL: A</think>no answer") is None
    assert parse_label("FINAL: Z") is None
    df = load_all()
    tr = df[~df["is_test"]]
    shots = select_fewshot(tr.iloc[:600])
    assert set(shots["evasion_label"]) <= set(EVASION)
    rows = tr.iloc[1000:1004]
    assert not set(rows["datapointID"]) & set(shots["datapointID"])
    fb = FakeBackend("reasoning...\nFINAL: B")
    P, info = classify(rows, fb, fewshot_block(shots), n=2, cache_path=tmp_path / "c.jsonl", workers=2)
    assert P.shape == (4, 9) and np.allclose(P[:, 1], 1) and fb.calls == 8
    classify(rows, fb, fewshot_block(shots), n=2, cache_path=tmp_path / "c.jsonl", workers=2)
    assert fb.calls == 8  # all cache hits
