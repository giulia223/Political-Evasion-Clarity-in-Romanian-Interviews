# Results (OOF, 7-fold CV on train_data.csv)

E = evasion macro-F1, M = clarity macro-F1 mapped from evasion. All numbers computed by code.

E nested-bias / M nested-bias: per-class logit biases fitted on the other 6 folds' OOF, applied to the held-out fold.

| date | exp | n | E argmax | M argmax | E groupdec | M groupdec | C head | E nested-bias | M nested-bias | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-21 | lr_tfidf_feats_prior | 3278 | 0.2753 | 0.4857 | 0.2987 | 0.5380 | – | 0.3232 | 0.5425 | tfidf+feats+transcript prior, C=2 |
| 2026-11-21 | lr_noprior | 3278 | 0.2880 | 0.4932 | 0.3078 | 0.5390 | – | 0.3356 | 0.5661 | no-prior |
| 2026-11-21 | lr_balanced | 3278 | 0.3111 | 0.5423 | 0.3116 | 0.5153 | – | 0.3041 | 0.5583 | class_weight=balanced |
| 2026-11-21 | lr_C05 | 3278 | 0.2663 | 0.4619 | 0.3022 | 0.5360 | – | 0.3159 | 0.5432 | C=0.5 |
| 2026-11-21 | lr_feats_only | 3278 | 0.2429 | 0.4244 | 0.2852 | 0.5203 | – | 0.2920 | 0.5366 | dense-feats-only |
