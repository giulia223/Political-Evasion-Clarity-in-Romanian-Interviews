## 7. Recommendations (inputs to T1.2, T2.1 and T3.2)

Notes on the numbers above:
- In §2 the RoBERT and XLM-R token counts are almost identical. Romanian BERT (cased, 50k vocab) uses about 17% fewer tokens.
- The extra settings comparison below was computed with the RoBERT-large tokenizer on train + test.

### Sequence length and chunking (T2.1)
- **`max_length` 512.** 82% of Q+A pairs fit in 256 tokens and 96% fit in 512, so dynamic padding matters more than length. Batch by chunk budget (T2.3).
- **Question cap 192 tokens**, keeping the end of the question. A cap of 128 truncates 199 questions (5.6%); 192 truncates 59 (1.7%) and adds only 16 chunks in total (3,760 → 3,776). A cap of 256 truncates 19, but leaves too little room for the answer in each chunk.
- **Answer window overlap 128 tokens.** An overlap of 256 (the paper's 50%) needs more chunks (3,931 vs 3,776) and creates long chunk lists for the longest answers (max 29 vs 16).
- **`max_chunks` 8** (first, last and evenly spaced chunks). Only 2 rows need more than 8 chunks at cap 192 and overlap 128. The longest answer (4,657 tokens) needs 16.
- **About 96.5% of examples are a single chunk**, so sliding-window pooling only affects about 120 rows. It is cheap to keep, but don't expect a big gain from it.

### Normalization (T1.2)
- **Keep the comma-below diacritics as they are.** No tokenizer has UNKs on raw text apart from the characters listed below. Converting to cedilla raises Romanian BERT's UNK rate to 9.3%, and the fertility difference for RoBERT and XLM-R is negligible (≤ 0.005).
- **Map `–` (en dash, 183) to `-`.** It is the main UNK source for Romanian BERT. Also map `‑` (non-breaking hyphen) to `-`.
- **Map `ť Ť` (t-caron, 3 occurrences) to `ț Ț`.** These are typos for Romanian ț. `ê` and `ğ` appear in foreign names; leave them.
- **Whitespace:** map nbsp (51) to a space, a tab (1) to a space, and collapse runs of spaces (318). Keep or collapse newlines (930); BERT-style tokenizers treat them as whitespace anyway.
- **Quotes:** `“ ” „ « » ’ ‘` (about 370 occurrences) can be mapped to ASCII `"` and `'` for consistency. This is optional, since none of them cause UNKs.
- **RoBERT-large lowercases inside its tokenizer.** Don't lowercase ourselves, because the cased models need the case.

### Data facts to remember
- **Mixed date formats:** `date` mixes `YYYY-MM-DD` and `DD.MM.YYYY` (1,402 train and 141 test rows from presidency.ro 2015+). Always use `polero.data.event_dates`.
- **Diacritics in titles:** titles contain both comma and cedilla forms. This matters only if titles are used as features.
- **Train/test overlap:** 5 test answers appear verbatim in train, 4 of them one-word replies ("Da.", "Nu."). There is no exact Q+A overlap, so there is nothing to exploit and no leakage risk.

### Context and metadata (T3.2)
- **Neighbouring turns are available:** 232 of 296 test rows have a labeled previous turn, and 237 have a labeled next turn. Only 11 test rows are alone in their transcript. Turns are often not adjacent (41% of gaps are 1; 17% are 5 or more).
- **The previous turn's label is a weak signal:** P(Clear Reply) is .577 after a Clear Reply versus .500–.506 otherwise. This supports the roadmap's advice to avoid using neighbour labels.
- **Metadata is a strong prior.** Interviews are 63% Clear Reply, against 46% for declarations and press conferences. The 2015–2026 period has about 3× the Clear Non-Reply rate of 2001–2007 (11.6% vs 3.9%). The president follows the same split: Iliescu and Băsescu are about 62% Clear Reply, Iohannis and Dan about 46%. A short metadata prefix (context type plus period) is the most promising T3.2 variant.

## 8. Normalization check (T1.2)

Measured on all questions and answers (train + test) with `polero.text.normalize`:

| tokenizer | mode | UNK rate | fertility |
|---|---|---|---|
| RoBERT-large | raw | 0.00000 | 1.48852 |
| RoBERT-large | clean | 0.00000 | 1.48852 |
| RoBERT-large | clean+quotes | 0.00000 | 1.48852 |
| ro-BERT-base-cased | raw | 0.00041 | 1.23279 |
| ro-BERT-base-cased | clean | 0.00001 | 1.23280 |
| ro-BERT-base-cased | clean+quotes | 0.00001 | 1.23280 |
| XLM-R-large | raw | 0.00000 | 1.49799 |
| XLM-R-large | clean | 0.00000 | 1.49798 |
| XLM-R-large | clean+quotes | 0.00000 | 1.49793 |

**Chosen mode: `clean` for every backbone** (`polero.text.BACKBONE_MODE`).
- It removes 98% of Romanian BERT's UNKs; the rest are `ê` and `ğ` in foreign names.
- It leaves fertility unchanged within 0.001%.
- Mapping quotes as well (`clean+quotes`) makes no measurable difference, so it isn't used.
