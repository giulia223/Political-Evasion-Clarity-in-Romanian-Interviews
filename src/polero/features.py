"""Hand-crafted features for the CPU baseline: lengths, Q/A overlap, Romanian refusal/ignorance/clarification lexicons,
metadata, and a leakage-safe transcript label prior."""
import re
import unicodedata

import numpy as np
import pandas as pd

from .labels import CLARITY, CLARITY_OF, EVASION


def fold_text(s):
    """Lowercase and strip diacritics (both comma-below and cedilla forms) for lexicon matching."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("`", "'"))


LEXICON = {
    "ignorance": [
        r"\bnu stiu\b", r"\bn-?am de unde sa stiu", r"\bnu cunosc\b", r"\bnu am (nicio |nici o )?informati",
        r"\bn-?am (nicio |nici o )?informati", r"\bnu am date\b", r"\bnu am vazut\b", r"\bn-?am vazut\b", r"\bnu am citit\b",
        r"\bn-?am citit\b", r"\bnu sunt la curent\b", r"\bnu am auzit\b", r"\bn-?am auzit\b", r"\bhabar\b",
        r"\bvoi verifica\b", r"\bo sa (ma )?(informez|verific)", r"\bnu am detalii\b", r"\bnu stiu exact\b", r"\bnu am cunostint",
    ],
    "decline": [
        r"\bnu (vreau|doresc|o sa|voi|pot) (sa )?(comentez|comenta|fac comentarii)", r"\bnu comentez\b", r"\bfara comentarii\b",
        r"\bnu (va )?pot (sa )?(va )?(spun|raspund|dezvalui)", r"\bnu va pot spune\b", r"\bnu (vreau|doresc) sa (va )?(spun|raspund|discut|anticip|intru)",
        r"\bnu (ma )?pronunt", r"\bnu e momentul\b", r"\bnu este momentul\b", r"\bla momentul potrivit\b", r"\bveti afla\b",
        r"\bo sa vedeti\b", r"\bvom vedea\b", r"\bnu anticipez\b", r"\bnu discut\b", r"\bnu intru in\b", r"\bnu raspund\b",
        r"\bnu fac (niciun )?comentari", r"\bnu am ce (sa )?(comenta|spune)", r"\bdupa ce\b.*\bvoi\b", r"\bprematur\b",
        r"\bnu este cazul\b", r"\bnu e cazul\b", r"\bpermiteti-mi sa nu\b",
    ],
    "clarify": [
        r"\bla ce va referiti\b", r"\bva referiti\b", r"\bce anume\b", r"\bnu am inteles\b", r"\bn-?am inteles\b", r"\bpoftim\b",
        r"\brepetati\b", r"\bin ce sens\b", r"\badica\?", r"\bcare\?", r"\bce\?", r"\bcum adica\b", r"\bdespre ce\b",
    ],
    "pivot": [
        r"\binsa\b", r"\bdar\b", r"\bpe de alta parte\b", r"\bimportant (este|e) (ca|sa)\b", r"\besential\b",
        r"\bproblema (reala|adevarata)\b", r"\bcred ca (ar trebui|e mai important)\b", r"\bde fapt\b", r"\bin schimb\b",
    ],
    "general": [r"\bin general\b", r"\bprioritat", r"\bimportant\b", r"\bfoarte\b", r"\btoti\b", r"\bromania\b", r"\bnoi\b"],
    "direct": [r"^\s*da\b", r"^\s*nu\b", r"^\s*sigur\b", r"^\s*evident\b", r"^\s*bineinteles\b", r"^\s*desigur\b", r"^\s*absolut\b"],
    "press": [r"\bpresa\b", r"\bjurnalist", r"\bintrebare", r"\bdumneavoastra\b"],
}
_LEX = {k: [re.compile(p) for p in v] for k, v in LEXICON.items()}
_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = set(
    "si sa de la in pe cu ca nu un o este e a ai am au ce care cum mai din pentru se va vor ne le ma te il lor sunt fi fost "
    "acest aceasta acum asa dar sau daca cand dupa doar foarte prin catre despre intre spre noi voi eu tu el ea ei ele "
    "dumneavoastra domnule doamna presedinte credeti ati".split()
)


def _content_words(s):
    return {w for w in _WORD.findall(s) if len(w) > 2 and w not in _STOP}


def text_features(df):
    """Numeric features from Q and A text. Returns a DataFrame aligned with df."""
    rows = []
    for q, a in zip(df["interview_question"], df["interview_answer"]):
        fq, fa = fold_text(q), fold_text(a)
        nq, na = len(_WORD.findall(fq)), len(_WORD.findall(fa))
        head = " ".join(fa.split()[:40])  # first ~40 words: refusals usually come early
        cq, ca = _content_words(fq), _content_words(fa)
        r = {
            "log_len_a": np.log1p(na),
            "log_len_q": np.log1p(nq),
            "len_ratio": np.log1p(na) - np.log1p(nq),
            "a_short5": float(na <= 5),
            "a_short15": float(na <= 15),
            "a_qmarks": fa.count("?"),
            "a_ends_q": float(fa.strip().endswith("?")),
            "a_digits": len(re.findall(r"\d+", fa)) / (1 + na) * 10,
            "q_qmarks": fq.count("?"),
            "overlap_jacc": len(cq & ca) / (1 + len(cq | ca)),
            "overlap_q_cov": len(cq & ca) / (1 + len(cq)),
            "q_starts_yesno": float(bool(re.match(r"^\s*(credeti|considerati|va|veti|ati|este|e |exista|aveti|sunteti|o sa|intentionati)", fq))),
            "q_wh": float(bool(re.search(r"\b(cand|cat|cati|cate|cine|unde|ce|care|cum|de ce)\b", fq))),
        }
        for k, pats in _LEX.items():
            r[f"lex_{k}"] = sum(bool(p.search(fa)) for p in pats)
            r[f"lexh_{k}"] = sum(bool(p.search(head)) for p in pats)
        rows.append(r)
    return pd.DataFrame(rows, index=df.index)


def meta_features(df):
    year = pd.to_numeric(df["date"].str[:4], errors="coerce").fillna(2015)
    out = pd.get_dummies(df[["president", "context"]].astype(str), dtype=float)
    out["year"] = (year - 2013) / 8
    out["log_qorder"] = np.log1p(df["question_order"].astype(float))
    return out


def transcript_prior(all_df, labeled_mask, alpha=2.0):
    """For every row, the smoothed evasion/clarity distribution of *other* labeled rows in the same transcript.

    `labeled_mask` marks the rows whose labels may be used (training-fold rows only). A row never sees its own label.
    """
    glob9 = all_df.loc[labeled_mask, "evasion_label"].value_counts(normalize=True).reindex(EVASION, fill_value=0).to_numpy()
    Y = np.zeros((len(all_df), 9))
    lab = all_df["evasion_label"].where(labeled_mask, None)
    for i, v in enumerate(lab):
        if isinstance(v, str) and v in EVASION:
            Y[i, EVASION.index(v)] = 1
    url = all_df["url"].to_numpy()
    sums = pd.DataFrame(Y).groupby(url).transform("sum").to_numpy() - Y  # exclude self
    cnt = sums.sum(1, keepdims=True)
    P9 = (sums + alpha * glob9) / (cnt + alpha)
    cl_idx = np.array([CLARITY.index(CLARITY_OF[e]) for e in EVASION])
    P3 = np.stack([P9[:, cl_idx == c].sum(1) for c in range(3)], 1)
    cols = {f"tp_ev_{j}": P9[:, j] for j in range(9)}
    cols.update({f"tp_cl_{c}": P3[:, c] for c in range(3)})
    cols["tp_logn"] = np.log1p(cnt[:, 0])
    return pd.DataFrame(cols, index=all_df.index)
