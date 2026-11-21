"""T1.1 exploratory analysis. Writes docs/EDA.md (run: python scripts/eda.py).

Only tokenizers are downloaded (public, small); no model weights.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from polero import REPO_ROOT
from polero.data import event_dates, load_test, load_train
from polero.labels import CLARITY, EVASION

TOKENIZERS = {
    "RoBERT-large": "readerbench/RoBERT-large",
    "ro-BERT-base-cased": "dumitrescustefan/bert-base-romanian-cased-v1",
    "XLM-R-large": "FacebookAI/xlm-roberta-large",
}
MAX_LEN, STRIDE, Q_CAP = 512, 128, 128
TO_CEDILLA = str.maketrans("șțȘȚ", "şţŞŢ")
ODD_CHARS = {"nbsp \\u00a0": " ", "“ ” (curly)": "“”", "„ (low)": "„", "« »": "«»", "– en dash": "–",
             "— em dash": "—", "… ellipsis": "…", "’ ‘": "’‘", "tab": "\t", "newline": "\n"}


# ---------- small helpers ----------
def md_table(df, floatfmt="{:.3f}", index=True):
    d = df.reset_index() if index else df
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def describe(x):
    x = np.asarray(x)
    return {"mean": x.mean(), "median": np.median(x), "p90": np.percentile(x, 90), "p95": np.percentile(x, 95),
            "p99": np.percentile(x, 99), "max": int(x.max())}


def n_chunks(q_len, a_len, max_len=MAX_LEN, stride=STRIDE, q_cap=Q_CAP, n_special=3):
    """Chunks needed when the (capped) question is repeated in every chunk and the answer
    is split into windows overlapping by `stride` tokens (T2.1 design)."""
    q = np.minimum(np.asarray(q_len), q_cap)
    budget = max_len - n_special - q
    extra = np.maximum(np.asarray(a_len) - budget, 0)
    return 1 + np.ceil(extra / np.maximum(budget - stride, 1)).astype(int)


# ---------- analyses ----------
def token_stats(df, tok):
    """Per-row token counts (no special tokens) for question and answer, plus pair length."""
    q = [len(x) for x in tok(df.interview_question.tolist(), add_special_tokens=False)["input_ids"]]
    a = [len(x) for x in tok(df.interview_answer.tolist(), add_special_tokens=False)["input_ids"]]
    pair_special = len(tok("a", "b")["input_ids"]) - len(tok("a", add_special_tokens=False)["input_ids"]) * 2
    out = pd.DataFrame({"q": q, "a": a}, index=df.index)
    out["pair"] = out.q + out.a + pair_special
    out["chunks"] = n_chunks(out.q, out.a, n_special=pair_special)
    return out


def unk_fertility(texts, tok):
    """UNK rate (share of tokens) and fertility (tokens per whitespace word)."""
    ids = tok(list(texts), add_special_tokens=False)["input_ids"]
    n_tok = sum(map(len, ids))
    n_unk = sum(i == tok.unk_token_id for x in ids for i in x)
    n_words = sum(len(t.split()) for t in texts)
    return {"unk_rate": n_unk / max(n_tok, 1), "fertility": n_tok / max(n_words, 1)}


def odd_char_counts(df):
    text = pd.concat([df.interview_question, df.interview_answer])
    rows = {name: int(text.str.count(re.escape(ch) if len(ch) == 1 else f"[{re.escape(ch)}]").sum()) for name, ch in ODD_CHARS.items()}
    rows["double spaces"] = int(text.str.count("  ").sum())
    rows["leading/trailing ws"] = int((text != text.str.strip()).sum())
    return pd.Series(rows, name="count")


def train_test_overlap(train, test):
    norm = lambda s: s.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    rows = []
    for name, f in [("exact answer", lambda d: d.interview_answer),
                    ("exact Q+A pair", lambda d: d.interview_question + "\x00" + d.interview_answer),
                    ("normalized answer", lambda d: norm(d.interview_answer))]:
        tr_keys, te_keys = f(train), f(test)
        hit = te_keys.isin(set(tr_keys))
        labels = train.assign(k=tr_keys)[tr_keys.isin(set(te_keys[hit]))].evasion_label.value_counts().to_dict()
        rows.append({"match": name, "test rows": int(hit.sum()), "train labels of matches": labels})
    short = test[test.interview_answer.isin(set(train.interview_answer))].interview_answer.str.split().str.len()
    return pd.DataFrame(rows), short.tolist()


def neighbour_structure(train, test):
    """For each test row: is there an earlier/later turn in the same transcript (url), and is it labeled?"""
    both = pd.concat([train.assign(split="train"), test.assign(split="test")]).sort_values(["url", "question_order"])
    both["prev_split"] = both.groupby("url").split.shift(1)
    both["next_split"] = both.groupby("url").split.shift(-1)
    both["gap_prev"] = both.groupby("url").question_order.diff()
    t = both[both.split == "test"]
    stats = {
        "transcripts (urls)": both.url.nunique(),
        "test transcripts": t.url.nunique(),
        "test rows with a previous turn": int(t.prev_split.notna().sum()),
        "… previous turn is labeled (train)": int((t.prev_split == "train").sum()),
        "test rows with a next turn": int(t.next_split.notna().sum()),
        "… next turn is labeled (train)": int((t.next_split == "train").sum()),
        "test rows alone in their transcript": int((t.prev_split.isna() & t.next_split.isna()).sum()),
        "median rows per transcript": float(both.groupby("url").size().median()),
        "turn gap = 1 (adjacent)": f"{(both.gap_prev == 1).mean():.1%} of rows with a previous turn" if both.gap_prev.notna().any() else "–",
    }
    gaps = both.gap_prev.dropna()
    stats["turn gap = 1 (adjacent)"] = f"{(gaps == 1).mean():.1%} of rows with a previous turn"
    stats["turn gap ≥ 5"] = f"{(gaps >= 5).mean():.1%}"
    return pd.Series(stats, name="value")


def label_rates(train, by):
    ct = pd.crosstab(train[by], train.clarity_label, normalize="index")[list(CLARITY)]
    ct.insert(0, "n", train[by].value_counts())
    return ct


def label_transition(train):
    """P(evasion label of a turn | clarity of the previous labeled turn in the same transcript)."""
    d = train.sort_values(["url", "question_order"]).copy()
    d["prev_clarity"] = d.groupby("url").clarity_label.shift(1)
    d = d.dropna(subset=["prev_clarity"])
    return pd.crosstab(d.prev_clarity, d.clarity_label, normalize="index")[list(CLARITY)].reindex(list(CLARITY))


# ---------- report ----------
def build_report(train, test, tokenizers):
    """tokenizers: {display name: tokenizer}. Returns the markdown report as a string."""
    out = ["# EDA: PolERo (T1.1)", "",
           f"Generated by `python scripts/eda.py`. Train {len(train)} rows, test {len(test)} rows. "
           "All token counts are without special tokens unless stated; `pair` includes them.", ""]

    # tokenizers
    info, norm_rows, len_rows, label_rows, chunk_rows = [], [], [], [], []
    answers = pd.concat([train.interview_answer, test.interview_answer])
    for name, tok in tokenizers.items():
        nz = tok.backend_tokenizer.normalizer if tok.is_fast else None
        info.append({"tokenizer": name, "fast": tok.is_fast, "vocab": len(tok), "lowercases": "lowercase=True" in str(nz),
                     "comma ș/ț in vocab": sum(("ș" in v) or ("ț" in v) for v in tok.get_vocab()),
                     "cedilla ş/ţ in vocab": sum(("ş" in v) or ("ţ" in v) for v in tok.get_vocab())})
        for mode, texts in [("raw (comma)", answers), ("→ cedilla", answers.str.translate(TO_CEDILLA))]:
            norm_rows.append({"tokenizer": name, "text": mode, **unk_fertility(texts, tok)})
        st = token_stats(pd.concat([train, test]), tok)
        for part in ("q", "a", "pair"):
            len_rows.append({"tokenizer": name, "part": part, **describe(st[part])})
        len_rows[-1].update({">256": (st.pair > 256).mean(), ">512": (st.pair > 512).mean()})
        trs = st.iloc[: len(train)].assign(label=train.evasion_label.to_numpy())
        g = trs.groupby("label").pair
        label_rows.append(pd.DataFrame({f"{name} median": g.median(), f"{name} >512": g.apply(lambda x: (x > 512).mean())}))
        c = st.chunks.value_counts().sort_index()
        chunk_rows.append({"tokenizer": name, **{f"{k} chunk{'s' if k > 1 else ''}": int(v) for k, v in c.items() if k <= 4},
                           ">4 chunks": int(c[c.index > 4].sum()), "max": int(st.chunks.max()),
                           "q > cap": int((st.q > Q_CAP).sum())})

    out += ["## 1. Tokenizers", "", md_table(pd.DataFrame(info), index=False), "",
            "### Diacritics: UNK rate and fertility (tokens per word) on all answers", "",
            "The data uses comma-below ș ț only (no cedilla). Converting to cedilla is the only candidate normalization.", "",
            md_table(pd.DataFrame(norm_rows), floatfmt="{:.4f}", index=False), "",
            "## 2. Token lengths (train + test)", "", md_table(pd.DataFrame(len_rows).astype(object).fillna(""), floatfmt="{:.2f}", index=False), "",
            "### Pair length by evasion label (train)", "", md_table(pd.concat(label_rows, axis=1).reindex(list(EVASION)), floatfmt="{:.2f}"), "",
            f"### Chunks needed (max_len {MAX_LEN}, stride {STRIDE}, question repeated per chunk, capped at {Q_CAP} tokens)", "",
            md_table(pd.DataFrame(chunk_rows).fillna(0), index=False), ""]

    # text oddities
    out += ["## 3. Characters relevant to normalization (T1.2)", "", md_table(odd_char_counts(pd.concat([train, test]))), ""]

    # duplicates
    ov, short = train_test_overlap(train, test)
    dup_train = int(train.duplicated(["interview_question", "interview_answer"]).sum())
    out += ["## 4. Duplicates", "", f"- Exact duplicate Q+A pairs inside train: {dup_train} (kept in one fold, D-019).", "",
            md_table(ov, index=False), "",
            f"Word counts of test answers that also appear in train: {sorted(short)}. They are formulaic replies, "
            "so copying train labels is not reliable evidence.", ""]

    # neighbours
    out += ["## 5. Interview structure (neighbouring turns, train + test together)", "",
            md_table(neighbour_structure(train, test)), "",
            "### Clarity of a turn given the clarity of the previous labeled turn (train)", "",
            md_table(label_transition(train)), ""]

    # metadata
    out += ["## 6. Clarity rates by metadata (train)", "", "### President", "", md_table(label_rates(train, "president")), "",
            "### Context", "", md_table(label_rates(train, "context")), "",
            "### Year bucket", "", md_table(label_rates(train.assign(period=pd.cut(event_dates(train).dt.year, [2000, 2007, 2014, 2026],
                                                                                    labels=["2001–2007", "2008–2014", "2015–2026"]).astype(str)), "period")), ""]
    return "\n".join(out)


def load_tokenizers(names=TOKENIZERS, local_only=False):
    from transformers import AutoTokenizer
    return {k: AutoTokenizer.from_pretrained(v, local_files_only=local_only) for k, v in names.items()}


def main(out=REPO_ROOT / "docs" / "EDA.md", recommendations=""):
    report = build_report(load_train(), load_test(), load_tokenizers())
    Path(out).write_text(report + ("\n" + recommendations if recommendations else "") + "\n")
    return report
