"""Cross-validated training of the multi-head chunk classifier (T2.3).

CLI:  python -m polero.train --config configs/<exp>.json [--folds 0,1] [--smoke] [--out-dir DIR]

Outputs in artifacts/<exp_id>/ (exp_id = config file name; smoke runs use <exp_id>_smoke):
  config.json            resolved config
  fold{k}/metrics.json   per-epoch history and best epoch
  fold{k}/oof.csv        validation-fold probabilities of the best epoch
  test_fold{k}.csv       test probabilities of the fold's best epoch
  oof.csv                merge of every fold{k}/oof.csv present (resumed runs add to it)
  train.log
"""
import argparse
import copy
import dataclasses
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from polero import ARTIFACTS, SEED
from polero.chunking import ChunkConfig, collate_chunks, encode_examples
from polero.data import load_test, load_train
from polero.folds import load_folds
from polero.labels import CLARITY_TO_ID, EVASION_TO_ID
from polero.metrics import CL_COLS, EV_COLS, evasion_f1, mapped_clarity_f1
from polero.labels import EVASION
from polero.model import LossConfig, MultiHeadChunkClassifier, class_weights, enable_checkpointing

log = logging.getLogger("polero.train")


@dataclass
class TrainConfig:
    backbone: str = "readerbench/RoBERT-large"
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 15
    patience: int = 3
    batch_examples: int = 16        # examples per optimizer step (gradient accumulation fills it)
    batch_chunks: int = 8           # max chunks per forward pass (memory bound)
    eval_batch_chunks: int = 32
    max_grad_norm: float = 1.0
    precision: str = "auto"         # auto | fp32 | fp16 | bf16
    gradient_checkpointing: bool = False
    pooling: str = "max"
    dropout: float = 0.1
    seed: int = SEED
    chunk: dict = field(default_factory=dict)     # ChunkConfig overrides
    loss: dict = field(default_factory=dict)      # LossConfig overrides; "class_weight_power" adds class weights
    max_steps: int | None = None    # stop after this many optimizer steps per fold (smoke/debug)
    n_train_rows: int | None = None  # subsample train (smoke/debug)
    n_test_rows: int | None = None

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        return cls(**data)


SMOKE = dict(backbone="smoke", epochs=1, max_steps=2, batch_examples=4, batch_chunks=4, n_train_rows=64,
             n_test_rows=16, precision="fp32", gradient_checkpointing=False,
             chunk={"max_length": 64, "q_cap": 16, "overlap": 8, "max_chunks": 4})


class EarlyStopping:
    """Stop when the score hasn't improved for `patience` evaluations (ties don't count as improvement)."""

    def __init__(self, patience):
        self.patience, self.best, self.best_epoch, self.bad = patience, -math.inf, -1, 0

    def step(self, score, epoch):
        if score > self.best:
            self.best, self.best_epoch, self.bad = score, epoch, 0
            return True  # improved
        self.bad += 1
        return False

    @property
    def should_stop(self):
        return self.bad >= self.patience


def set_seed(seed):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_precision(p, device):
    if p != "auto":
        return p
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def make_batches(examples, idx, budget, rng=None):
    """Group example indices into batches of at most `budget` chunks (an oversized example goes alone).
    With rng: shuffle, sort by length within windows of 64 to cut padding, shuffle batch order."""
    idx = list(idx)
    width = lambda i: max(len(c["input_ids"]) for c in examples[i]["chunks"])
    if rng is not None:
        rng.shuffle(idx)
        idx = [i for w in range(0, len(idx), 64) for i in sorted(idx[w:w + 64], key=width)]
    else:
        idx = sorted(idx, key=width)
    batches, cur, n = [], [], 0
    for i in idx:
        k = examples[i]["n_kept"]
        if cur and n + k > budget:
            batches.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += k
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def load_backbone(cfg, texts):
    """(tokenizer, encoder). 'smoke' builds an offline tiny backbone."""
    if cfg.backbone == "smoke":
        from polero.smoke import smoke_encoder, smoke_tokenizer
        tok = smoke_tokenizer(texts)
        return tok, smoke_encoder(len(tok), max_positions=cfg.chunk.get("max_length", 512))
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.backbone)
    try:
        enc = AutoModel.from_pretrained(cfg.backbone, add_pooling_layer=False)
    except TypeError:
        enc = AutoModel.from_pretrained(cfg.backbone)
    return tok, enc


@torch.no_grad()
def predict(model, examples, idx, budget, pad_id, use_types, device, amp):
    """Softmax probabilities (evasion [n,9], clarity [n,3]) for examples[idx], in idx order."""
    model.eval()
    pos = {i: p for p, i in enumerate(idx)}
    pe, pc = np.zeros((len(idx), 9)), np.zeros((len(idx), 3))
    for b in make_batches(examples, idx, budget):
        batch = {k: v.to(device) for k, v in collate_chunks([examples[i] for i in b], pad_id, use_types).items()}
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            out = model(**batch, n_examples=len(b))
        rows = [pos[i] for i in b]
        pe[rows] = torch.softmax(out["evasion_logits"].float(), -1).cpu().numpy()
        pc[rows] = torch.softmax(out["clarity_logits"].float(), -1).cpu().numpy()
    return pe, pc


def probs_frame(ids, pe, pc):
    return pd.concat([pd.DataFrame({"datapointID": ids}), pd.DataFrame(pe, columns=EV_COLS), pd.DataFrame(pc, columns=CL_COLS)], axis=1)


def score(y_ev, pe):
    """Early-stopping score: evasion macro-F1 + mapped clarity macro-F1."""
    p = [EVASION[i] for i in pe.argmax(1)]
    return evasion_f1(y_ev, p) + mapped_clarity_f1(y_ev, p), evasion_f1(y_ev, p), mapped_clarity_f1(y_ev, p)


def train_fold(k, cfg, train, test, folds, examples, test_examples, tok, encoder_init, out_dir, device):
    set_seed(cfg.seed + k)
    tr_idx = np.flatnonzero(folds != k)
    va_idx = np.flatnonzero(folds == k)
    y_cl = torch.tensor(train.clarity_label.map(CLARITY_TO_ID).to_numpy())
    y_ev = torch.tensor(train.evasion_label.map(EVASION_TO_ID).to_numpy())

    loss_kw = dict(cfg.loss)
    power = loss_kw.pop("class_weight_power", None)
    if power is not None:  # weights from this fold's training rows only
        loss_kw["evasion_weights"] = class_weights(train.evasion_label.iloc[tr_idx].tolist(), list(EVASION_TO_ID), power)
        loss_kw["clarity_weights"] = class_weights(train.clarity_label.iloc[tr_idx].tolist(), list(CLARITY_TO_ID), power)
    model = MultiHeadChunkClassifier(copy.deepcopy(encoder_init), cfg.pooling, cfg.dropout, LossConfig(**loss_kw)).to(device)
    if cfg.gradient_checkpointing:
        enable_checkpointing(model.encoder)

    precision = resolve_precision(cfg.precision, device)
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    use_types = "token_type_ids" in tok.model_input_names
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(tr_idx) / cfg.batch_examples)
    total = min(cfg.epochs * steps_per_epoch, cfg.max_steps or 10**9)
    warmup = int(cfg.warmup_ratio * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(warmup, 1), max(0.0, (total - s) / max(total - warmup, 1))))

    rng = random.Random(cfg.seed + k)
    stopper, best_state, history, step = EarlyStopping(cfg.patience), None, [], 0
    y_va = train.evasion_label.iloc[va_idx].tolist()
    log.info(f"fold {k}: train {len(tr_idx)} / val {len(va_idx)} rows, {total} steps, precision {precision}")
    for epoch in range(cfg.epochs):
        model.train()
        t0, seen, run_loss, acc = time.time(), 0, 0.0, 0
        opt.zero_grad()
        for b in make_batches(examples, tr_idx, cfg.batch_chunks, rng):
            batch = {kk: v.to(device) for kk, v in collate_chunks([examples[i] for i in b], tok.pad_token_id, use_types).items()}
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                out = model(**batch, n_examples=len(b), clarity_labels=y_cl[b].to(device), evasion_labels=y_ev[b].to(device))
            scaler.scale(out["loss"] * len(b) / cfg.batch_examples).backward()
            acc += len(b)
            run_loss += out["loss"].item() * len(b)
            if acc >= cfg.batch_examples:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt), scaler.update(), sched.step(), opt.zero_grad()
                acc, step = 0, step + 1
                if step >= total:
                    break
            seen += len(b)
        pe, _ = predict(model, examples, va_idx, cfg.eval_batch_chunks, tok.pad_token_id, use_types, device, amp)
        s, ef, mf = score(y_va, pe)
        history.append({"epoch": epoch, "train_loss": run_loss / max(seen, 1), "val_evasion_f1": ef,
                        "val_mapped_clarity_f1": mf, "val_score": s, "seconds": round(time.time() - t0, 1)})
        log.info(f"fold {k} epoch {epoch}: loss {history[-1]['train_loss']:.4f} evasion {ef:.4f} mapped {mf:.4f} ({history[-1]['seconds']}s)")
        if stopper.step(s, epoch):
            best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        if stopper.should_stop or step >= total:
            break

    model.load_state_dict(best_state)
    pe, pc = predict(model, examples, va_idx, cfg.eval_batch_chunks, tok.pad_token_id, use_types, device, amp)
    te_idx = list(range(len(test)))
    te, tc = predict(model, test_examples, te_idx, cfg.eval_batch_chunks, tok.pad_token_id, use_types, device, amp)

    fdir = out_dir / f"fold{k}"
    fdir.mkdir(parents=True, exist_ok=True)
    probs_frame(train.datapointID.iloc[va_idx].to_numpy(), pe, pc).assign(fold=k).to_csv(fdir / "oof.csv", index=False)
    probs_frame(test.datapointID.to_numpy(), te, tc).to_csv(out_dir / f"test_fold{k}.csv", index=False)
    s, ef, mf = score(y_va, pe)
    metrics = {"fold": k, "best_epoch": stopper.best_epoch, "val_evasion_f1": ef, "val_mapped_clarity_f1": mf,
               "n_train": len(tr_idx), "n_val": len(va_idx), "steps": step, "precision": precision, "history": history}
    (fdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"fold {k} done: best epoch {stopper.best_epoch}, evasion {ef:.4f}, mapped clarity {mf:.4f}")
    return metrics


def merge_oof(out_dir):
    parts = sorted(out_dir.glob("fold*/oof.csv"))
    oof = pd.concat([pd.read_csv(p) for p in parts]).sort_values("datapointID").reset_index(drop=True)
    oof.to_csv(out_dir / "oof.csv", index=False)
    return oof


def run(cfg, exp_id, folds_to_run=None, out_dir=None):
    out_dir = Path(out_dir or ARTIFACTS / exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train = load_train().merge(load_folds(), on="datapointID")
        if cfg.n_train_rows:
            train = train.head(cfg.n_train_rows)
        train = train.reset_index(drop=True)
        test = load_test()
        if cfg.n_test_rows:
            test = test.head(cfg.n_test_rows)
        set_seed(cfg.seed)
        tok, encoder = load_backbone(cfg, pd.concat([train.interview_question, train.interview_answer, test.interview_question, test.interview_answer]))
        ccfg = ChunkConfig(**cfg.chunk)
        backbone = None if cfg.backbone == "smoke" else cfg.backbone
        examples = encode_examples(train.interview_question, train.interview_answer, tok, ccfg, backbone)
        test_examples = encode_examples(test.interview_question, test.interview_answer, tok, ccfg, backbone)
        folds = train.fold.to_numpy()
        ks = [int(k) for k in (folds_to_run if folds_to_run is not None else sorted(set(folds)))]
        log.info(f"{exp_id}: backbone {cfg.backbone}, device {device}, folds {ks}, {sum(e['n_kept'] for e in examples)} train chunks")
        results = [train_fold(k, cfg, train, test, folds, examples, test_examples, tok, encoder, out_dir, device) for k in ks]
        merge_oof(out_dir)
        return results
    finally:
        log.removeHandler(handler)
        handler.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the multi-head chunk classifier with fixed CV folds.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--folds", help="comma-separated fold ids (default: all)")
    ap.add_argument("--smoke", action="store_true", help="tiny offline model, 64 rows, 2 steps (CPU, < 2 min)")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cfg = TrainConfig.from_json(args.config)
    exp_id = args.config.stem
    if args.smoke:
        cfg = dataclasses.replace(cfg, **SMOKE)
        exp_id += "_smoke"
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    run(cfg, exp_id, folds, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
