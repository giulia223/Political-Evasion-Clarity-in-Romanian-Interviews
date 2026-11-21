"""CV training of the chunked multi-head encoder (T2.3).

python -m polero.train --config configs/<exp>.json [--folds 0,1] [--device cuda:0] [--smoke]

Per fold: train on the other 6 folds, early-stop on val (evasion macro-F1 + mapped clarity macro-F1), keep the best
epoch's val and test probabilities. Outputs go to artifacts/<exp_id>/ (oof_fold{k}.csv are merged into oof.csv).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS, SEED
from .chunking import encode_dataset, normalize, pick_diacritic_mode
from .data import add_neighbours, load_all
from .folds import get_folds
from .labels import CL2ID, CL_COLS, CLARITY_OF, EV2ID, EV_COLS, EVASION
from .metrics import evasion_f1, mapped_clarity_f1
from .model import MultiHeadChunkClassifier, multitask_loss

DEFAULTS = dict(
    backbone="readerbench/RoBERT-large",
    max_len=512, q_max=128, stride=128, max_chunks=8,
    lr=1.5e-5, head_lr=1e-4, llrd=1.0, weight_decay=0.01, warmup=0.1, epochs=10, patience=3,
    batch_chunks=16, accum=1, eval_chunks=64, grad_clip=1.0,
    w_c=1.0, w_e=1.0, label_smoothing=0.0, focal_gamma=0.0, class_weights=None,
    pooling="max", dropout=0.1, grad_ckpt=False, context="none", seed=SEED, diacritics="auto",
)


class EarlyStopper:
    def __init__(self, patience):
        self.patience, self.best, self.bad = patience, -math.inf, 0

    def step(self, score):
        """Return True if this score is a new best."""
        if score > self.best + 1e-6:
            self.best, self.bad = score, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self):
        return self.bad >= self.patience


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_text(df, cfg):
    q = df["interview_question"]
    if cfg["context"] == "prev_q":
        q = np.where(df["prev_q"] != "", "Anterior: " + df["prev_q"].str[-300:] + " | Întrebare: " + q, q)
        q = pd.Series(q, index=df.index)
    if cfg["context"] == "meta":
        q = df["president"] + " | " + df["context"] + " | " + df["date"].str[:4] + " | " + q
    return q, df["interview_answer"]


def make_batches(idx, chunks, budget, shuffle, rng):
    """Group example indices so each batch has <= budget chunks (an example longer than budget is alone)."""
    order = list(idx)
    if shuffle:
        rng.shuffle(order)
        # light length bucketing: sort within blocks of 50 to reduce padding
        order = [i for b in range(0, len(order), 50) for i in sorted(order[b : b + 50], key=lambda i: -max(map(len, chunks[i])))]
    batches, cur, n = [], [], 0
    for i in order:
        c = len(chunks[i])
        if cur and n + c > budget:
            batches.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += c
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(batch, chunks, pad_id, device):
    seqs, ex = [], []
    for j, i in enumerate(batch):
        for c in chunks[i]:
            seqs.append(c)
            ex.append(j)
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s)
        mask[r, : len(s)] = 1
    return ids.to(device), mask.to(device), torch.tensor(ex, device=device), len(batch)


def param_groups(model, cfg):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = []
    layers = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    n_layers = len(layers) if layers is not None else 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else cfg["weight_decay"]
        if not name.startswith("encoder."):
            lr = cfg["head_lr"]
        else:
            depth = n_layers  # embeddings
            if ".layer." in name:
                depth = n_layers - 1 - int(name.split(".layer.")[1].split(".")[0])
            elif "embeddings" not in name:
                depth = 0  # pooler etc.
            lr = cfg["lr"] * (cfg["llrd"] ** depth)
        groups.append({"params": [p], "lr": lr, "weight_decay": wd})
    return groups


@torch.no_grad()
def predict(model, idx, chunks, cfg, pad_id, device, amp_dtype):
    model.eval()
    P_cl, P_ev = [], []
    for b in make_batches(idx, chunks, cfg["eval_chunks"], False, None):
        ids, mask, ex, n = collate(b, chunks, pad_id, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            lc, le = model(ids, mask, ex, n)
        P_cl.append(torch.softmax(lc.float(), -1).cpu())
        P_ev.append(torch.softmax(le.float(), -1).cpu())
    return torch.cat(P_cl).numpy(), torch.cat(P_ev).numpy()


def score(y_ev, P_ev):
    pred = [EVASION[i] for i in P_ev.argmax(1)]
    e, m = evasion_f1(y_ev, pred), mapped_clarity_f1(y_ev, pred)
    return e, m


def train_fold(k, df, chunks, cfg, out_dir, device, log, encoder_factory=None, tokenizer=None):
    set_seed(cfg["seed"] + k)
    tr = np.where((~df["is_test"]) & (df["fold"] != k))[0]
    va = np.where((~df["is_test"]) & (df["fold"] == k))[0]
    te = np.where(df["is_test"])[0]
    if cfg.get("smoke"):
        tr, va, te = tr[:32], va[:16], te[:8]
    elif cfg.get("limit"):  # real model on a few rows: end-to-end check before a GPU session
        tr, va, te = tr[: cfg["limit"]], va[: cfg["limit"] // 2], te[: cfg["limit"] // 4]
    y_ev = torch.tensor([EV2ID[v] for v in df["evasion_label"].iloc[tr]])
    y_cl = torch.tensor([CL2ID[CLARITY_OF[v]] for v in df["evasion_label"].iloc[tr]])
    encoder = encoder_factory() if encoder_factory else None
    model = MultiHeadChunkClassifier(cfg["backbone"], encoder=encoder, pooling=cfg["pooling"],
                                     dropout=cfg["dropout"], grad_ckpt=cfg["grad_ckpt"]).to(device)
    pad_id = tokenizer.pad_token_id
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg["lr"])
    rng = random.Random(cfg["seed"] + k)
    steps_per_epoch = math.ceil(len(make_batches(tr, chunks, cfg["batch_chunks"], True, random.Random(0))) / cfg["accum"])
    total = steps_per_epoch * cfg["epochs"] if not cfg.get("smoke") else 2
    warm = int(cfg["warmup"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    cw = None
    if cfg["class_weights"] == "sqrt_inv":
        cnt = np.bincount(y_ev.numpy(), minlength=9).astype(float)
        cw = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float, device=device)
        cw = cw / cw.mean()
    stopper = EarlyStopper(cfg["patience"])
    pos = {i: j for j, i in enumerate(tr)}
    best = None
    step = 0
    y_va = df["evasion_label"].iloc[va].tolist()
    for epoch in range(cfg["epochs"] if not cfg.get("smoke") else 1):
        model.train()
        t0, tot_loss, nb = time.time(), 0.0, 0
        batches = make_batches(tr, chunks, cfg["batch_chunks"], True, rng)
        if cfg.get("smoke"):
            batches = batches[:2]
        opt.zero_grad()
        for bi, b in enumerate(batches):
            ids, mask, ex, n = collate(b, chunks, pad_id, device)
            rows = torch.tensor([pos[i] for i in b])
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                lc, le = model(ids, mask, ex, n)
            loss = multitask_loss(lc, le, y_cl[rows].to(device), y_ev[rows].to(device), cfg["w_c"], cfg["w_e"],
                                  cfg["label_smoothing"], cw, cfg["focal_gamma"]) / cfg["accum"]
            scaler.scale(loss).backward()
            tot_loss += loss.item() * cfg["accum"]
            nb += 1
            if (bi + 1) % cfg["accum"] == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if step < total:
                    sched.step()
                step += 1
        pc_va, pe_va = predict(model, va, chunks, cfg, pad_id, device, amp_dtype)
        e, m = score(y_va, pe_va)
        improved = stopper.step(e + m)
        log(f"fold {k} epoch {epoch} loss {tot_loss / max(1, nb):.4f} val E {e:.4f} M {m:.4f} {'*' if improved else ''} ({time.time() - t0:.0f}s)")
        if improved:
            pc_te, pe_te = predict(model, te, chunks, cfg, pad_id, device, amp_dtype)
            best = dict(epoch=epoch, E=e, M=m, pc_va=pc_va, pe_va=pe_va, pc_te=pc_te, pe_te=pe_te)
        if stopper.stop:
            break
    fd = out_dir / f"fold{k}"
    fd.mkdir(parents=True, exist_ok=True)
    (fd / "metrics.json").write_text(json.dumps({k2: best[k2] for k2 in ("epoch", "E", "M")}, indent=2))
    ids = df["datapointID"].to_numpy()
    for name, idx, pc, pe in (("oof_fold", va, best["pc_va"], best["pe_va"]), ("test_fold", te, best["pc_te"], best["pe_te"])):
        o = pd.DataFrame(np.hstack([pe, pc]), columns=EV_COLS + CL_COLS)
        o.insert(0, "datapointID", ids[idx])
        o.to_csv(out_dir / f"{name}{k}.csv", index=False)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def merge_oof(out_dir):
    parts = sorted(out_dir.glob("oof_fold*.csv"))
    if parts:
        pd.concat([pd.read_csv(p) for p in parts]).sort_values("datapointID").to_csv(out_dir / "oof.csv", index=False)
    return len(parts)


def smoke_setup(df):
    """Tiny offline tokenizer + BERT encoder for CPU smoke runs and tests."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(list(df["interview_question"][:200]) + list(df["interview_answer"][:200]),
                            trainers.WordLevelTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok.post_processor = processors.BertProcessing(("[SEP]", tok.token_to_id("[SEP]")), ("[CLS]", tok.token_to_id("[CLS]")))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
    conf = BertConfig(vocab_size=hf.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=128)
    return hf, (lambda: BertModel(conf, add_pooling_layer=False))


def run(cfg, folds_to_run, device, exp_id, log_file=None):
    out_dir = ARTIFACTS / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    logf = open(out_dir / (log_file or "train.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    df = add_neighbours(load_all())
    df = df.merge(get_folds(), on="datapointID", how="left")
    df["fold"] = df["fold"].fillna(-1).astype(int)
    if cfg.get("smoke"):
        tokenizer, factory = smoke_setup(df)
        cfg.update(max_len=64, q_max=16, stride=8, max_chunks=3, batch_chunks=8, eval_chunks=16)
    else:
        from transformers import AutoTokenizer

        tokenizer, factory = AutoTokenizer.from_pretrained(cfg["backbone"]), None
    mode = cfg["diacritics"]
    if mode == "auto":
        mode = pick_diacritic_mode(tokenizer, df["interview_answer"].iloc[:200].tolist())
    log(f"exp {exp_id} device {device} diacritics={mode} cfg={json.dumps(cfg)}")
    q, a = build_text(df, cfg)
    q, a = q.map(lambda t: normalize(t, mode)), a.map(lambda t: normalize(t, mode))
    t0 = time.time()
    chunks = encode_dataset(tokenizer, q, a, max_len=cfg["max_len"], q_max=cfg["q_max"], stride=cfg["stride"], max_chunks=cfg["max_chunks"])
    nch = np.array([len(c) for c in chunks])
    log(f"encoded {len(chunks)} examples in {time.time() - t0:.0f}s; chunks/example mean {nch.mean():.2f} max {nch.max()} multi {np.mean(nch > 1):.3f}")
    results = {}
    for k in folds_to_run:
        results[k] = train_fold(k, df, chunks, cfg, out_dir, device, log, factory, tokenizer)
        log(f"fold {k} best epoch {results[k]['epoch']} E {results[k]['E']:.4f} M {results[k]['M']:.4f}")
    n = merge_oof(out_dir)
    log(f"merged {n} oof folds into {out_dir / 'oof.csv'}")
    logf.close()
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--folds", default="0,1,2,3,4,5,6")
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--exp", default=None, help="override experiment id (default: config file stem)")
    p.add_argument("--limit", type=int, default=None, help="debug: use only this many training rows")
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args(argv)
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(a.config).read_text()))
    cfg["smoke"] = a.smoke
    if a.limit:
        cfg["limit"] = a.limit
    if a.epochs:
        cfg["epochs"] = a.epochs
    exp_id = a.exp or Path(a.config).stem + ("_smoke" if a.smoke else "")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    folds = [int(x) for x in a.folds.split(",")]
    run(cfg, folds, device, exp_id, log_file=f"train_{a.folds.replace(',', '')}.log")


if __name__ == "__main__":
    main()
