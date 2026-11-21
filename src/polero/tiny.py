"""Tiny local tokenizer and model config for tests and --smoke (no downloads)."""
import re
from collections import Counter
from pathlib import Path

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def make_tiny_tokenizer(texts, out_dir, vocab_size=800):
    """Word-level-ish WordPiece vocab (top words + single characters), BERT-style."""
    from transformers import BertTokenizer

    words = Counter(w for t in texts for w in re.findall(r"\w+|[^\w\s]", t.lower()))
    chars = sorted({c for t in texts for c in t.lower() if not c.isspace()})
    vocab = list(SPECIAL) + chars + ["##" + c for c in chars]
    for w, _ in words.most_common():
        if len(vocab) >= vocab_size:
            break
        if w not in vocab:
            vocab.append(w)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    # transformers v5: BertTokenizer takes the vocab as a dict (or a path) in `vocab`
    tok = BertTokenizer(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True, strip_accents=False)
    tok.save_pretrained(str(out))
    return tok


def tiny_config(vocab_size, max_len=512):
    from transformers import BertConfig

    return BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=max_len, type_vocab_size=2)
