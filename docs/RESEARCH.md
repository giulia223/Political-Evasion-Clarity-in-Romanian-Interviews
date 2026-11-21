# Research Notes: PolERo (Romanian political evasion)

Collected on 2026-11-21. These notes are the evidence behind the roadmap. All numbers are macro-F1.
C = direct 3-class clarity, E = 9-class evasion, M = clarity mapped up from the predicted evasion label.

## 1. The PolERo paper (Ștefan & Nisioi, EMNLP 2026, arXiv 2609.02391)

### Test-set results on Romanian (PolERo)
| Model | C | E | M |
|---|---|---|---|
| TF-IDF + LR | .453 | .312 | .472 |
| RoBERTa-large, single head | .646 | .428 | .613 |
| XLM-R-large, single head | .680 | .365 | .686 |
| XLM-R-large, multi-head | .694 | .508 | .706 |
| RoBERT-large, single head | .688 | .504 | .701 |
| **RoBERT-large, multi-head (best fine-tuned in-language)** | .729 | .545 | **.763** |
| XLM-R-large multi-head, **joint EN+RO training** | .723 | **.570** | .717 |
| XLM-R-large MH, RO + machine-translated EN→RO | .718 | .489 | .684 |
| Gemma-4-31B-it, few-shot | .672 | .619 | .738 |
| **Qwen3.6-35B-A3B, few-shot (open weights)** | .746 | **.682** | .760 |
| GPT-5.4, few-shot (paid: not allowed for us) | .763 | .666 | .763 |
| Llama-3.3-70B, few-shot | .555 | .312 | .411 (biased toward Deflection) |

### Key findings
- **Multi-head** encoder: a shared encoder with a 3-class clarity head and a 9-class evasion head, trained with loss `L = CE_clarity + CE_evasion` (unweighted). It beats single-head on every metric.
- **Mapping clarity up from evasion beats predicting clarity directly** for RoBERT-large and almost all LLMs. It fails when the backbone collapses onto the majority class (ELECTRA, ModernBERT, mBERT), so the backbone must put real probability mass on rare classes.
- **Romanian-specific encoders beat multilingual ones** in-language: RoBERT-large > XLM-R-large.
- Joint EN+RO training (adding QEvasion) gives the **best fine-tuned evasion score (.570)**, because English examples reinforce rare classes. Machine-translated augmentation was inconclusive.
- **Cumulative ablation (Romanian, RoBERT-large; 3 seeds, test macro-F1):**

  | Step | C | E |
  |---|---|---|
  | Single-head | .678 | .511 |
  | + multi-head | .701 | .531 |
  | + sliding window | .705 | .515 |
  | + cross-validation | .707 | .516 |
  | + 7-fold ensemble | .721 | .537 |

- Test-set variance across seeds is large: E ±.037 and C ±.023 on Romanian. **Don't trust a gain under ~0.02 from a single run.**
- Few-shot examples help most at separating the confusable pairs **Dodging↔Deflection** and **Implicit↔General**, which are the main error axes. **All LLMs that did well were run with reasoning enabled.**
- Training labels come from one annotator; test labels come from three. For **evasion, a prediction counts as correct if it matches any annotator's label**. For clarity, it must match the exact majority label. So expect test evasion scores to come out higher than our CV estimates.

### Hyperparameters
- **Single-head:** up to 20 epochs, AdamW, batch 64, max length 512. Learning rate: 2e-5 for RoBERT-large and Romanian BERT; 1e-5 for XLM-R-large and RoBERTa-large.
- **Multi-head:** 7-fold CV stratified by clarity, 20 epochs, AdamW, learning rate 1e-5, batch 8. Early stopping on combined validation macro-F1 (patience 5). At inference, average the probabilities of the 7 fold models.
- **Chunking:** windows of L=512 with stride S=256 over the tokenized Q+A. Take the **position-0 hidden state** of each chunk, **max-pool element-wise across chunks**, apply dropout 0.1, then the two linear heads.
- **TF-IDF baseline:** word unigrams, balanced-weight LR, C ∈ {0.25 … 10}.

### Dataset facts from the paper
- Mean answer length is 76.9 words (median 45). Clarification answers are the shortest (median 4 words) and Dodging is next (median 13). General, Deflection and Partial are long (~88 words).
- Justice & Anti-corruption questions have the highest Clear Non-Reply rate. Non-replies rise after 2016.

## 2. SG-UniBuc-NLP at SemEval-2026 Task 6 (arXiv 2604.26375; same authors, English)
- RoBERTa-large, input `Question: {Q}\nAnswer: {A}`, 512-token chunks with stride 256, max-pooling of **hidden states** (not logits), dual heads.
- AdamW, weight decay 0.01, lr 1e-5, 10% warmup, linear schedule, batch 8, max 15 epochs with patience 3, gradient clip 1.0, BF16, gradient checkpointing, seed 42.
- **Loss ablation (7-fold CV, English):**

  | Loss | C | E |
  |---|---|---|
  | Cross-entropy | .70 | .45 |
  | Class-weighted CE | .66 | .41 |
  | Focal (γ=2) | .69 | .44 |

  **Class weights hurt.** Reweighting raises recall on minority classes but costs more precision on the majority classes than it gains.
- **Pooling ablation:**

  | Pooling | C | E |
  |---|---|---|
  | First chunk | .67 | .42 |
  | Mean | .68 | .43 |
  | Max | .70 | .45 |

- **Ensemble size:**

  | Folds | C | E |
  |---|---|---|
  | 3 | .66 | .42 |
  | 5 | .68 | .43 |
  | 7 | .70 | .45 |

- Compute: one RTX 3090 for about 5 hours for the full 7-fold run on English, where 28.8% of inputs exceed 512 tokens. Ours are shorter (see §5).
- Code: https://github.com/gabriel-stefan/political-evasion-detection. **It has no license, so read it for ideas only and do not copy code.**

## 3. SemEval-2026 Task 6 CLARITY overview (arXiv 2603.14027; English)
- Top scores: C 0.89, E 0.68. The best encoder-only systems reached C .81 and E .50. LLM systems dominated.
- **Upward derivation from evasion to clarity was "the single most effective approach"**: the winner, TeleAI, solved evasion first.
- TeleAI's three-stage pipeline:
  1. a coarse classifier with confidence scores;
  2. re-examination of low-confidence items, with label-specific guidelines and few-shot examples;
  3. a fine-grained pass for the rest.
- The 2nd-ranked evasion system (0.61): **fine-tuned RoBERTa-large proposes top-5 candidate labels, then an LLM picks among them using few-shot prompts**. It ran on a Colab L4 in under 1.5 hours.
- **Few-shot + chain-of-thought was near-universal among the top systems.** Prompt quality mattered more than model size.
- **Distillation:** fine-tuning 4B–14B open models on chain-of-thought traces from a larger model was competitive.
- Large DeBERTa ensembles saturated. Weighted or focal loss helped only partially. Structural fixes worked better: priors in prompts, routing easy classes deterministically, and synthetic data.

## 4. QEvasion (Thomas et al., EMNLP Findings 2024): English data for cross-lingual use
- Hugging Face dataset `ailsntua/QEvasion`: 3,448 train and 308 test rows, public, **license CC BY-NC-ND 4.0**.
- It uses the same label strings as PolERo, in columns `clarity_label` and `evasion_label`. The columns `interview_question`, `interview_answer` and `question` are also present. Multi-part questions are split into sub-questions (`question`).
- **Not used:** the user decided against external training data (D-012). The idea is parked in docs/IDEAS.md (I-001).

## 5. Our data (checked locally)
- Train has 3,278 rows and test has 296. No missing values. The evasion→clarity mapping holds exactly.
- **Evasion counts in train:**

  | Label | Count |
  |---|---|
  | Explicit | 1801 |
  | General | 438 |
  | Dodging | 286 |
  | Implicit | 276 |
  | Deflection | 179 |
  | Declining to answer | 148 |
  | Claims ignorance | 58 |
  | Partial/half-answer | 56 |
  | Clarification | 36 |

- **Clarity counts in train:** Clear Reply 1801, Ambivalent 1235, Clear Non-Reply 242.
- **Lengths:** answers have median 45 and p90 181 words (max 3,005). Only **~2.7% of Q+A pairs exceed ~380 words (≈512 tokens)**, and ~15% exceed ~190 words (≈256 tokens). Chunking matters only for a small tail, but that tail has long General, Deflection and Partial answers.
- **Diacritics:** the text uses comma-below letters only (ș ț Ș Ț) and no cedilla forms. Check whether each tokenizer's vocabulary uses comma or cedilla (task T1.1).
- 182 of the 194 test interviews also have turns in train (283 test rows), so neighbouring-turn context is available. 5 test answers appear verbatim in train. Train has 6 duplicated Q/A pairs.
- **Submission format** (from `sample_output.csv`): columns `subtaskID,datapointID,answer`, with 592 rows. Subtask 1 rows (clarity) come first and Subtask 2 rows (evasion) second, 296 each, in the order of `test_data.csv`.

## 6. Free-compute options
- **Local:** 16 CPU cores, 30 GB RAM, **no GPU**, Python 3.14 venv. Fine for development, tests and smoke runs. Fine-tuning a large encoder here is impractical.
- **Kaggle Notebooks:** free T4×2 or P100, roughly 30 GPU-hours a week, 12-hour sessions. **Colab free:** a T4 with session limits. The user runs these manually. Agents never handle credentials.
- **Open-weight LLMs:** a quantized Qwen3.6-35B-A3B (MoE, ~3B active parameters; Q4 GGUF ≈ 20 GB) can run with llama.cpp on 30 GB RAM (slow, but works overnight) or on Kaggle T4×2. Downloads are large, so ask the user first.

## Sources
- PolERo: https://aclanthology.org/2026.emnlp-main.2688/ · https://arxiv.org/abs/2609.02391
- SG-UniBuc-NLP: https://aclanthology.org/2026.semeval-1.133/ · https://arxiv.org/abs/2604.26375
- SemEval-2026 Task 6 overview: https://aclanthology.org/2026.semeval-1.449/ · https://arxiv.org/abs/2603.14027
- QEvasion paper: https://aclanthology.org/2024.findings-emnlp.300/ · dataset https://huggingface.co/datasets/ailsntua/QEvasion
- RoBERT-large: https://huggingface.co/readerbench/RoBERT-large (341M parameters; also base 114M and small 19M)
- Reference code (no license): https://github.com/gabriel-stefan/political-evasion-detection
