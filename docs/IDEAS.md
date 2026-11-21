# Parked Ideas (not to be implemented)

Ideas we considered and deliberately put aside. **Agents must not implement anything here** unless the user moves it into `ROADMAP.md` through a DECISIONS.md entry.

## I-001: Cross-lingual joint training with English QEvasion (formerly T3.3)
- **Status:** parked by the user on 2026-11-21 (D-012). The user judged external English data unnecessary.
- **Idea:** train XLM-R-large multi-head on English QEvasion train data plus Romanian PolERo train folds, and validate on the Romanian folds only. QEvasion uses the same 9 evasion labels and the same 3 clarity labels.
- **Why it might help:** the PolERo paper reports Romanian evasion macro-F1 rising from .508 to **.570** with joint EN+RO training, the best fine-tuned result in the paper. The English examples reinforce the rare classes (Clarification, Claims ignorance, Declining to answer). See docs/RESEARCH.md §1.
- **Costs and risks:**
  - external data may not be allowed by the hackathon rules;
  - licence CC BY-NC-ND 4.0;
  - XLM-R is weaker than RoBERT-large in-language;
  - machine-translated augmentation (EN→RO with NLLB) was inconclusive in the paper.
- **Data:** `ailsntua/QEvasion` on Hugging Face (public, 3,448 train / 308 test). The sub-question field is `question`.
