"""CPU baseline: TF-IDF + hand-crafted features + optional frozen embeddings, logistic regression, 7-fold CV.

python -m polero.baseline --exp lr_tfidf_feats [--no-prior] [--emb artifacts/emb/<name>.npy] [--C 2.0]
Writes artifacts/<exp>/{config.json, oof.csv, test_fold{k}.csv} in the standard format.
"""
import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import ARTIFACTS, SEED
from .data import load_all
from .features import fold_text, meta_features, text_features, transcript_prior
from .folds import get_folds
from .labels import CL_COLS, EV_COLS, EVASION
from .metrics import append_result, evaluate_oof, print_report


def build(args):
    df = load_all()
    folds = get_folds()
    df = df.merge(folds, on="datapointID", how="left")
    df["fold"] = df["fold"].fillna(-1).astype(int)
    qa = df["interview_question"].map(fold_text)
    aa = df["interview_answer"].map(fold_text)
    dense = pd.concat([text_features(df), meta_features(df)], axis=1).astype(float)
    emb = None
    if args.emb:
        emb = np.hstack([np.load(p) for p in args.emb])
        assert len(emb) == len(df), "embedding rows must follow load_all() order"
    return df, qa, aa, dense, emb


def run(args):
    df, qa, aa, dense, emb = build(args)
    is_test = df["is_test"].to_numpy()
    tr_idx = np.where(~is_test)[0]
    y = df["evasion_label"].to_numpy()
    out_dir = ARTIFACTS / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    oof = np.zeros((len(df), 9))
    for k in range(7):
        trn = tr_idx[df["fold"].to_numpy()[tr_idx] != k]
        val = tr_idx[df["fold"].to_numpy()[tr_idx] == k]
        labeled = np.zeros(len(df), bool)
        labeled[trn] = True
        parts_dense = [dense]
        if args.prior:
            parts_dense.append(transcript_prior(df, labeled))
        D = pd.concat(parts_dense, axis=1).to_numpy()
        sc = StandardScaler().fit(D[trn])
        Dz = sp.csr_matrix(np.clip(sc.transform(D), -5, 5) * args.dense_w)
        blocks = [Dz]
        if args.tfidf:
            vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True).fit(aa.iloc[trn])
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True).fit(aa.iloc[trn])
            vq = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True).fit(qa.iloc[trn])
            blocks += [vw.transform(aa), vc.transform(aa), vq.transform(qa) * 0.5]
        if emb is not None:
            esc = StandardScaler().fit(emb[trn])
            blocks.append(sp.csr_matrix(esc.transform(emb) * args.emb_w))
        X = sp.hstack(blocks).tocsr()
        clf = LogisticRegression(C=args.C, max_iter=3000, class_weight=args.cw, random_state=SEED)
        clf.fit(X[trn], y[trn])
        order = [list(clf.classes_).index(l) for l in EVASION]
        P = clf.predict_proba(X)[:, order]
        oof[val] = P[val]
        te = pd.DataFrame(P[is_test], columns=EV_COLS)
        te.insert(0, "datapointID", df.loc[is_test, "datapointID"].to_numpy())
        te.to_csv(out_dir / f"test_fold{k}.csv", index=False)
        print(f"fold {k} done", flush=True)
    o = pd.DataFrame(oof[tr_idx], columns=EV_COLS)
    o.insert(0, "datapointID", df.loc[tr_idx, "datapointID"].to_numpy())
    o.to_csv(out_dir / "oof.csv", index=False)
    m = evaluate_oof(o)
    print_report(m)
    append_result(args.exp, m, args.note)
    return m


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--no-prior", dest="prior", action="store_false")
    p.add_argument("--no-tfidf", dest="tfidf", action="store_false")
    p.add_argument("--emb", nargs="*", default=[])
    p.add_argument("--emb-w", type=float, default=0.3)
    p.add_argument("--dense-w", type=float, default=1.0)
    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--cw", default=None, help="None or 'balanced'")
    p.add_argument("--note", default="")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
