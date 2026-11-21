"""T2.4: averaging, consistent decoding, bias tuning."""
import numpy as np
import pandas as pd

from polero.decode import decode, decode_idx, fit_bias, score
from polero.ensemble import average
from polero.labels import CL_COLS, CLARITY, CLARITY_OF, EV_COLS, EVASION
from polero.metrics import evasion_f1


def _frame(ids, p):
    df = pd.DataFrame(p, columns=EV_COLS)
    df.insert(0, "datapointID", ids)
    return df


def test_average():
    rng = np.random.default_rng(0)
    p1, p2 = rng.dirichlet(np.ones(9), 5), rng.dirichlet(np.ones(9), 5)
    out = average([_frame([1, 2, 3, 4, 5], p1), _frame([5, 4, 3, 2, 1], p2[::-1])], weights=[3, 1])
    assert np.allclose(out[EV_COLS].to_numpy(), 0.75 * p1 + 0.25 * p2)


def test_decoders_consistent():
    rng = np.random.default_rng(1)
    pe, pc = rng.dirichlet(np.ones(9), 500), rng.dirichlet(np.ones(3), 500)
    for d in "abc":
        ev, cl = decode(pe, pc, d)
        assert all(CLARITY_OF[e] == c for e, c in zip(ev, cl))


def test_decoder_b_groups():
    # Explicit alone 0.4 but the Ambivalent group sums to 0.6 -> b picks Ambivalent
    p = np.array([[0.4, 0.15, 0.15, 0.15, 0.15, 0, 0, 0, 0]])
    e, c = decode_idx(p, decoder="b")
    assert CLARITY[c[0]] == "Ambivalent" and EVASION[e[0]] in ("Implicit", "General", "Partial/half-answer", "Dodging")
    e, c = decode_idx(p, decoder="a")
    assert EVASION[e[0]] == "Explicit"


def test_bias_tuning_helps_imbalanced():
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.choice(9, n, p=[0.6, 0.1, 0.1, 0.04, 0.06, 0.04, 0.03, 0.02, 0.01])
    logits = rng.normal(0, 1, (n, 9))
    logits[np.arange(n), y] += 1.2
    logits[:, 0] += 1.5  # model biased toward the majority class
    pe = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y_e = [EVASION[i] for i in y]
    y_c = [CLARITY_OF[e] for e in y_e]
    b = fit_bias(y_e, y_c, pe, decoder="a", rounds=2)
    assert score(y_e, y_c, pe, decoder="a", bias=b) > score(y_e, y_c, pe, decoder="a") + 0.05
    ev, _ = decode(pe, decoder="a", bias=b)
    assert evasion_f1(y_e, ev) > evasion_f1(y_e, decode(pe, decoder="a")[0])
