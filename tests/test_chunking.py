"""T2.1: sliding-window encoding. Offline tests use a tiny local WordPiece tokenizer;
real-tokenizer tests skip when the tokenizer isn't cached."""
import pytest
from transformers import AutoTokenizer, BertTokenizerFast

from polero.chunking import (ChunkConfig, PairTemplate, collate_chunks, encode_examples, encode_pair,
                             select_chunks, window_starts)
from polero.data import load_train
from polero.eda import n_chunks

WORDS = [f"w{i}" for i in range(200)]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "?", ".", *WORDS]))
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)


def text(n, offset=0):
    return " ".join(WORDS[(offset + i) % 200] for i in range(n))  # each word = exactly 1 token


CFG = ChunkConfig(max_length=64, q_cap=16, overlap=8, max_chunks=8)


def test_template(tok):
    t = PairTemplate(tok)
    assert t.n_special == 3 and t.prefix == [tok.cls_token_id] and t.q_type == 0 and t.a_type == 1


def test_short_pair_is_one_chunk_identical_to_tokenizer(tok):
    ex = encode_pair(text(5), text(7, 50), tok, CFG)
    assert ex["n_windows"] == 1 and len(ex["chunks"]) == 1
    ref = tok(text(5), text(7, 50))
    assert ex["chunks"][0]["input_ids"] == ref["input_ids"]
    assert ex["chunks"][0]["token_type_ids"] == ref["token_type_ids"]


def test_one_word_answer(tok):
    ex = encode_pair(text(3), "w9", tok, CFG)
    assert len(ex["chunks"]) == 1 and len(ex["chunks"][0]["input_ids"]) == 3 + 1 + 3  # no padding


def test_answer_coverage_and_question_in_every_chunk(tok):
    q, a = text(10), text(300, 20)
    ex = encode_pair(q, a, tok, ChunkConfig(max_length=64, q_cap=16, overlap=8, max_chunks=100))
    q_ids = tok(q, add_special_tokens=False)["input_ids"]
    a_ids = tok(a, add_special_tokens=False)["input_ids"]
    covered = set()
    for c, s in zip(ex["chunks"], ex["starts"]):
        ids = c["input_ids"]
        assert len(ids) <= 64
        assert ids[1: 1 + len(q_ids)] == q_ids  # question right after [CLS]
        window = ids[2 + len(q_ids): -1]
        assert window == a_ids[s: s + len(window)]
        covered.update(range(s, s + len(window)))
    assert covered == set(range(len(a_ids)))
    assert ex["starts"][-1] + ex["budget"] == len(a_ids)  # last window aligned to the end


def test_long_answer_chunk_count_matches_formula(tok):
    cfg = ChunkConfig(max_length=512, q_cap=192, overlap=128, max_chunks=100)
    long_answer = " ".join(WORDS[i % 200] for i in range(3000))
    ex = encode_pair(text(20), long_answer, tok, cfg)
    # budget 512-3-20 = 489, step 361: 1 + ceil((3000-489)/361) = 8
    assert ex["n_windows"] == 8 == n_chunks([20], [3000], 512, 128, 192, 3)[0]


def test_question_cap_keeps_end(tok):
    q = text(40)
    ex = encode_pair(q, text(5), tok, CFG)
    q_ids = tok(q, add_special_tokens=False)["input_ids"]
    assert ex["q_truncated"] and ex["chunks"][0]["input_ids"][1:17] == q_ids[-16:]


def test_max_chunks(tok):
    ex = encode_pair(text(5), text(2000), tok, CFG)
    assert ex["n_windows"] > 8 and ex["n_kept"] == len(ex["chunks"]) == 8
    all_starts = window_starts(ex["n_answer"], ex["budget"], 8)
    assert ex["starts"][0] == 0 and ex["starts"][-1] == all_starts[-1]
    assert select_chunks(3, 8) == [0, 1, 2] and select_chunks(20, 4) == [0, 6, 13, 19]


def test_collate(tok):
    exs = [encode_pair(text(3), text(2), tok, CFG), encode_pair(text(3), text(200), tok, CFG), encode_pair(text(3), "w1", tok, CFG)]
    b = collate_chunks(exs, tok.pad_token_id)
    n = [len(e["chunks"]) for e in exs]
    assert b["example_index"].tolist() == [0] * n[0] + [1] * n[1] + [2] * n[2]
    assert b["input_ids"].shape == b["attention_mask"].shape == b["token_type_ids"].shape
    assert b["attention_mask"].sum(1).tolist() == [len(c["input_ids"]) for e in exs for c in e["chunks"]]
    assert "token_type_ids" not in collate_chunks(exs, 0, use_token_types=False)


@pytest.mark.parametrize("name", ["readerbench/RoBERT-large", "dumitrescustefan/bert-base-romanian-cased-v1",
                                  "FacebookAI/xlm-roberta-large"])
def test_real_tokenizers_match_pair_encoding(name):
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"{name} not cached")
    tr = load_train().head(300)
    exs = encode_examples(tr.interview_question, tr.interview_answer, tok, ChunkConfig(norm_mode="raw"))
    for ex, q, a in zip(exs, tr.interview_question, tr.interview_answer):
        if ex["n_windows"] == 1 and not ex["q_truncated"]:
            ref = tok(q, a)
            assert ex["chunks"][0]["input_ids"] == ref["input_ids"]
        assert all(len(c["input_ids"]) <= 512 for c in ex["chunks"])
