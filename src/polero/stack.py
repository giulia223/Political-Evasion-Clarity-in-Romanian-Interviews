"""Stacking (T3.6): multinomial LR on the concatenated log-probabilities of several experiments.

python -m polero.stack --exp robert-large_mh --exp llm_qwen --exp lr_noprior --out stack_v1 [--C 0.5]

OOF of the stack is produced fold by fold (fit on the other folds' OOF rows only), so it is an honest estimate.
The test stack model is fit on all OOF rows. Output: artifacts/<out>/{oof.csv,test.csv}, decodable by polero.predict.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import ARTIFACTS, SEED
from .data import load_test, load_train
from .folds import get_folds
from .labels import EV_COLS, EVASION
from .metrics import append_result, evaluate_oof, print_report
from .predict import load_exp


def features(frames, ids):
    return np.hstack([np.log(np.clip(f.loc[ids].to_numpy(), 1e-4, 1)) for f in frames])


def stack(exps, C=0.5, out=None):
    loaded = [load_exp(e) for e in exps]
    ids = loaded[0][0].index
    for o, _ in loaded[1:]:
        ids = ids.intersection(o.index)
    ids = ids.sort_values()
    X = features([o for o, _ in loaded], ids)
    y = load_train().set_index("datapointID").loc[ids, "evasion_label"].to_numpy()
    folds = get_folds().set_index("datapointID").loc[ids, "fold"].to_numpy()
    te_ids = load_test()["datapointID"].to_numpy()
    X_te = features([t for _, t in loaded], te_ids)

    def fit(Xa, ya):
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED).fit(Xa, ya)
        order = [list(clf.classes_).index(l) for l in EVASION]
        return lambda Z: clf.predict_proba(Z)[:, order]

    P = np.zeros((len(ids), 9))
    for k in np.unique(folds):
        m = folds == k
        P[m] = fit(X[~m], y[~m])(X[m])
    P_te = fit(X, y)(X_te)
    oof = pd.DataFrame(P, columns=EV_COLS)
    oof.insert(0, "datapointID", ids)
    te = pd.DataFrame(P_te, columns=EV_COLS)
    te.insert(0, "datapointID", te_ids)
    if out:
        d = ARTIFACTS / out
        d.mkdir(parents=True, exist_ok=True)
        oof.to_csv(d / "oof.csv", index=False)
        te.to_csv(d / "test.csv", index=False)
    return oof, te


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", action="append", required=True)
    p.add_argument("--C", type=float, default=0.5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    oof, _ = stack(a.exp, a.C, a.out)
    m = evaluate_oof(oof)
    print_report(m)
    append_result(a.out, m, "stack of " + "+".join(a.exp) + f" C={a.C}")


if __name__ == "__main__":
    main()
