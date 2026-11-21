# Results (OOF, 7-fold CV on train_data.csv)

E = evasion macro-F1, M = clarity macro-F1 mapped from evasion. All numbers computed by code.

E nested-bias / M nested-bias: per-class logit biases fitted on the other 6 folds' OOF, applied to the held-out fold.

| date | exp | n | E argmax | M argmax | E groupdec | M groupdec | C head | E nested-bias | M nested-bias | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-21 | lr_tfidf_feats_prior | 3278 | 0.2753 | 0.4857 | 0.2987 | 0.5380 | – | 0.3232 | 0.5425 | tfidf+feats+transcript prior, C=2 |
