# Results

Append-only log of OOF evaluations (written by `python -m polero.evaluate`). All scores are macro-F1 on out-of-fold predictions over the fixed folds (D-006, D-019).
- evasion: 9-class argmax · mapped: clarity from evasion argmax (decoder a) · grouped: clarity from summed group probabilities (decoder b) · head: direct clarity head

| date | experiment | n | coverage | evasion | mapped clarity | grouped clarity | clarity head | note |
|---|---|---|---|---|---|---|---|---|
