"""T2.1: sliding-window encoding (local tiny tokenizer, no downloads)."""
from polero.chunking import PairTemplate, answer_windows, collate, encode_examples, select_chunks

Q = "ce părere aveți despre guvern"


def _answer_ids(tok, a):
    return tok(a, add_special_tokens=False)["input_ids"]


def test_template(tiny_tok):
    tpl = PairTemplate(tiny_tok)
    assert tpl.prefix == [tiny_tok.cls_token_id] and tpl.mid == [tiny_tok.sep_token_id]
    assert tpl.n_special == 3


def test_short_one_chunk(tiny_tok):
    ch, _ = encode_examples(tiny_tok, [Q], ["da"], max_length=64, stride=16)
    assert len(ch[0]) == 1
    ids, tt = ch[0][0]
    assert len(ids) == len(tt) and ids[0] == tiny_tok.cls_token_id and ids[-1] == tiny_tok.sep_token_id


def test_long_coverage_and_question_everywhere(tiny_tok):
    a = " ".join(["guvernul nu are dreptate în această privință"] * 400)  # ~3000 words
    L, S = 128, 32
    ch, st = encode_examples(tiny_tok, [Q], [a], max_length=L, stride=S, q_max=32, max_chunks=None)
    q_ids = tiny_tok(Q, add_special_tokens=False)["input_ids"]
    a_ids = _answer_ids(tiny_tok, a)
    room = L - 3 - len(q_ids)
    assert len(ch[0]) == len(answer_windows(len(a_ids), room, S))
    covered = []
    for ids, _ in ch[0]:
        assert len(ids) <= L
        assert ids[1:1 + len(q_ids)] == q_ids  # question in every chunk
        covered += ids[2 + len(q_ids):-1]
    assert len(covered) >= len(a_ids)
    assert set(covered) == set(a_ids)


def test_windows_cover_all():
    for n in (1, 50, 100, 101, 999):
        s = answer_windows(n, 40, 10)
        cov = set()
        for x in s:
            cov |= set(range(x, min(n, x + 40)))
        assert cov == set(range(n))


def test_max_chunks():
    assert select_chunks(3, 8) == [0, 1, 2]
    k = select_chunks(20, 5)
    assert k[0] == 0 and k[-1] == 19 and len(k) == 5


def test_max_chunks_reported(tiny_tok):
    a = " ".join(["cuvânt"] * 2000)
    ch, st = encode_examples(tiny_tok, [Q], [a], max_length=64, stride=16, max_chunks=4)
    assert len(ch[0]) == 4 and st["max_chunks_truncated"] == 1


def test_question_truncated_from_start(tiny_tok):
    q = "început " * 100 + "final"
    ch, _ = encode_examples(tiny_tok, [q], ["da"], max_length=64, q_max=10)
    ids = ch[0][0][0]
    q_ids = tiny_tok(q, add_special_tokens=False)["input_ids"]
    assert ids[1:11] == q_ids[-10:] and ids[11] == tiny_tok.sep_token_id


def test_collate(tiny_tok):
    ch, _ = encode_examples(tiny_tok, [Q, Q], ["da", " ".join(["nu"] * 300)], max_length=64, stride=16)
    b = collate(ch, tiny_tok.pad_token_id)
    assert b["input_ids"].shape[0] == len(ch[0]) + len(ch[1])
    assert b["example_index"].tolist() == [0] * len(ch[0]) + [1] * len(ch[1])
    assert b["attention_mask"].sum().item() == sum(len(c[0]) for e in ch for c in e)
