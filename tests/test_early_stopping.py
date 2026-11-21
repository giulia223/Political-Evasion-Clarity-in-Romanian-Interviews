"""T2.3: early stopping, batching and precision helpers."""
import random

import torch

from polero.train import EarlyStopping, make_batches, resolve_precision


def test_patience_logic():
    s = EarlyStopping(patience=2)
    assert s.step(0.5, 0) and not s.should_stop
    assert not s.step(0.4, 1) and not s.should_stop
    assert s.step(0.6, 2) and s.bad == 0          # improvement resets the counter
    assert not s.step(0.6, 3)                     # a tie is not an improvement
    assert not s.step(0.1, 4) and s.should_stop
    assert s.best == 0.6 and s.best_epoch == 2


def _examples(chunk_counts, widths):
    return [{"n_kept": k, "chunks": [{"input_ids": [0] * w}] * k} for k, w in zip(chunk_counts, widths)]


def test_make_batches_respects_budget_and_covers_all():
    rng = random.Random(0)
    counts = [rng.choice([1, 1, 1, 2, 3, 9]) for _ in range(200)]
    ex = _examples(counts, [rng.randint(5, 500) for _ in counts])
    for r in (None, random.Random(1)):
        batches = make_batches(ex, range(200), budget=8, rng=r)
        assert sorted(i for b in batches for i in b) == list(range(200))
        for b in batches:
            n = sum(counts[i] for i in b)
            assert n <= 8 or len(b) == 1  # an oversized example (9 chunks) goes alone


def test_make_batches_deterministic_given_seed():
    ex = _examples([1] * 50, list(range(50)))
    assert make_batches(ex, range(50), 4, random.Random(7)) == make_batches(ex, range(50), 4, random.Random(7))


def test_resolve_precision():
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) == "fp32"
    assert resolve_precision("fp16", cpu) == "fp16"
