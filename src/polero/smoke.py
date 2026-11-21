"""Offline tiny backbone for --smoke runs and tests: a deterministic WordPiece tokenizer built
from the given texts plus a randomly initialised 2-layer BERT. Nothing is downloaded."""
from collections import Counter

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def smoke_tokenizer(texts, vocab_size=600):
    """Deterministic WordPiece vocab: specials, every character (plain and ## form), then the most
    frequent words ranked by (-count, word). WordPiece *training* breaks ties non-deterministically."""
    norm = normalizers.BertNormalizer(lowercase=True, strip_accents=False)
    pre = pre_tokenizers.BertPreTokenizer()
    counts = Counter(w for t in texts for w, _ in pre.pre_tokenize_str(norm.normalize_str(t)))
    chars = sorted({c for w in counts for c in w})
    words = [w for w, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if len(w) > 1]
    tokens = list(dict.fromkeys(SPECIALS + chars + ["##" + c for c in chars] + words))[:max(vocab_size, len(SPECIALS) + 2 * len(chars))]
    tk = Tokenizer(models.WordPiece(vocab={t: i for i, t in enumerate(tokens)}, unk_token="[UNK]"))
    tk.normalizer, tk.pre_tokenizer = norm, pre
    cls, sep = tk.token_to_id("[CLS]"), tk.token_to_id("[SEP]")
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]",
                                   model_input_names=["input_ids", "token_type_ids", "attention_mask"])


def smoke_encoder(vocab_size, max_positions=512):
    cfg = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=max_positions)
    return BertModel(cfg, add_pooling_layer=False)
