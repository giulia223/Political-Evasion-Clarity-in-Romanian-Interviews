"""Model inputs with optional neighbour-turn context and metadata (T3.2).

Neighbours are reconstructed per `url`, ordered by `question_order`, over train
and test rows together (both are inputs). Labels are never read here.
"""
import pandas as pd

from .text import normalize


def neighbour_questions(frames):
    """Return {datapointID: (prev_question, next_question)} across all frames."""
    cols = ["datapointID", "url", "question_order", "interview_question"]
    allq = pd.concat([f[cols] for f in frames], ignore_index=True).drop_duplicates("datapointID")
    allq = allq.sort_values(["url", "question_order", "datapointID"])
    out = {}
    for _, g in allq.groupby("url", sort=False):
        ids, qs = g["datapointID"].tolist(), g["interview_question"].tolist()
        for i, d in enumerate(ids):
            out[d] = (qs[i - 1] if i > 0 else "", qs[i + 1] if i + 1 < len(ids) else "")
    return out


def _clip_words(text, n_words, keep_end=True):
    w = text.split()
    if len(w) <= n_words:
        return text
    return " ".join(w[-n_words:] if keep_end else w[:n_words])


def build_texts(df, cfg, neighbours=None):
    """Return (questions, answers) lists for the model.

    cfg keys: text_mode, context ("none" | "prev_q" | "prev_next_q"),
    context_words (cap per neighbour, default 48 ~ 64 tokens), meta_prefix (bool).
    """
    mode = cfg.get("text_mode", "basic")
    ctx = cfg.get("context", "none")
    cw = cfg.get("context_words", 48)
    qs, ans = [], []
    for r in df.itertuples(index=False):
        q = normalize(r.interview_question, mode)
        parts = []
        if cfg.get("meta_prefix"):
            year = str(r.date)[:4] if str(r.date)[:4].isdigit() else ""
            parts.append(f"[{r.president} | {r.context}{' | ' + year if year else ''}]")
        if ctx != "none" and neighbours is not None:
            prev_q, next_q = neighbours.get(r.datapointID, ("", ""))
            if prev_q:
                parts.append("Întrebarea anterioară: " + _clip_words(normalize(prev_q, mode), cw))
            if ctx == "prev_next_q" and next_q:
                parts.append("Întrebarea următoare: " + _clip_words(normalize(next_q, mode), cw, keep_end=False))
        parts.append("Întrebare: " + q if parts else q)
        qs.append(" ".join(parts))
        ans.append(normalize(r.interview_answer, mode))
    return qs, ans
