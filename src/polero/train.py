"""Cross-validated training of the multi-head chunked encoder (T2.3).

python -m polero.train --config configs/<exp>.json [--folds 0,1] [--device cuda:0] [--smoke]
python -m polero.train --config configs/<exp>.json --merge-only

Outputs in artifacts/<exp_id>/: config.json, fold{k}/metrics.json, fold{k}/oof_part.csv,
test_fold{k}.csv, oof.csv (merged from all finished folds) and train.log.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ARTIFACTS, load_test, load_train
from .folds import load_folds
from .labels import CL2I, CL_COLS, CLARITY_OF, EV2I, EV_COLS, EVASION
from .metrics import evasion_f1, mapped_clarity_f1

DEFAULTS = {
    "backbone": "readerbench/RoBERT-large",
    "text_mode": "basic",
    "context": "none",
    "context_words": 48,
    "meta_prefix": False,
    "max_length": 512,
    "stride": 128,
    "q_max": 128,
    "max_chunks": 8,
    "lr": 2e-5,
    "head_lr": None,
    "llrd": 1.0,
    "weight_decay": 0.01,
    "warmup": 0.1,
    "epochs": 15,
    "patience": 3,
    "min_epochs": 1,
    "batch_examples": 16,
    "token_budget": 4096,
    "eval_token_budget": 16384,
    "grad_clip": 1.0,
    "w_c": 1.0,
    "w_e": 1.0,
    "class_weights": None,
    "focal_gamma": 0.0,
    "label_smoothing": 0.0,
    "pooling": "max",
    "dropout": 0.1,
    "seed": 42,
    "precision": "auto",
    "grad_ckpt": False,
    "max_steps": None,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    cfg.setdefault("exp_id", Path(path).stem)
    return cfg


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class EarlyStopper:
    """Patience logic on a score to maximise. update() returns (improved, stop)."""

    def __init__(self, patience=3, min_epochs=1, eps=1e-6):
        self.patience, self.min_epochs, self.eps = patience, min_epochs, eps
        self.best, self.best_epoch, self.bad, self.epoch = float("-inf"), 0, 0, 0

    def update(self, score):
        self.epoch += 1
        if score > self.best + self.eps:
            self.best, self.best_epoch, self.bad = score, self.epoch, 0
            return True, False
        self.bad += 1
        return False, self.epoch >= self.min_epochs and self.bad >= self.patience


def make_batches(chunks, token_budget, max_examples, rng=None):
    """Group examples so that (#chunks x padded length) <= token_budget."""
    n = len(chunks)
    lens = np.array([max(len(c[0]) for c in ch) for ch in chunks])
    if rng is not None:
        idx = rng.permutation(n)
        idx = np.concatenate([sorted(idx[i:i + 64], key=lambda j: lens[j]) for i in range(0, n, 64)])
    else:
        idx = np.argsort(lens, kind="stable")
    batches, cur, n_ch, L = [], [], 0, 0
    for j in idx:
        nc, nl = len(chunks[j]), max(L, lens[j])
        if cur and ((n_ch + nc) * nl > token_budget or len(cur) >= max_examples):
            batches.append(cur)
            cur, n_ch, nl = [], 0, lens[j]
        cur.append(int(j))
        n_ch += nc
        L = nl
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def auto_budgets(cfg, device):
    """Resolve "auto" token budgets from GPU memory (T4/P100 16 GB, L4 24 GB, A100 40/80 GB)."""
    import torch

    if cfg["token_budget"] != "auto" and cfg["eval_token_budget"] != "auto":
        return cfg
    gb = 0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    big = cfg.get("params_hint", "large") == "large"
    if gb >= 70:
        tb = 8192
    elif gb >= 36:
        tb = 4096 if big else 8192
    elif gb >= 20:
        tb = 3072 if big else 6144
    else:
        tb = 2048 if big else 4096
    if cfg["token_budget"] == "auto":
        cfg["token_budget"] = tb
    if cfg["eval_token_budget"] == "auto":
        cfg["eval_token_budget"] = tb * 4
    return cfg


def _amp(device, precision):
    import torch

    if not str(device).startswith("cuda") or precision == "fp32":
        return None, False
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()
                               and torch.cuda.get_device_capability(device)[0] >= 8):
        return torch.bfloat16, False
    return torch.float16, True


def predict(model, chunks, device, token_budget, use_tt, pad_id, amp_dtype=None):
    import torch

    from .chunking import collate

    model.eval()
    p_cl = np.zeros((len(chunks), 3), dtype=np.float32)
    p_ev = np.zeros((len(chunks), len(EVASION)), dtype=np.float32)
    with torch.inference_mode():
        for b in make_batches(chunks, token_budget, 10 ** 9):
            x = {k: v.to(device) for k, v in collate([chunks[i] for i in b], pad_id, use_tt).items()}
            with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
                lc, le = model(**x, n_examples=len(b))
            p_cl[b] = torch.softmax(lc.float(), -1).cpu().numpy()
            p_ev[b] = torch.softmax(le.float(), -1).cpu().numpy()
    return p_cl, p_ev


def param_groups(model, cfg):
    """AdamW groups: no decay on bias/LayerNorm; optional layer-wise lr decay and head lr."""
    lr, llrd = cfg["lr"], cfg["llrd"]
    head_lr = cfg["head_lr"] or lr
    n_layers = getattr(model.encoder.config, "num_hidden_layers", 0)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith("encoder."):
            plr = head_lr
        elif llrd < 1.0:
            depth = n_layers + 1  # embeddings
            for i in range(n_layers):
                if f".layer.{i}." in name:
                    depth = n_layers - i
                    break
            else:
                if "embeddings" not in name:
                    depth = 0  # pooler etc.
            plr = lr * (llrd ** depth)
        else:
            plr = lr
        nd = p.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name
        key = (plr, 0.0 if nd else cfg["weight_decay"])
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "lr": k[0], "weight_decay": k[1]} for k, ps in groups.items()]


def class_weight_vec(y, n, mode):
    import torch

    if not mode:
        return None
    cnt = np.bincount(y, minlength=n).astype(float) + 1.0
    w = 1.0 / np.sqrt(cnt) if mode == "sqrt_inv" else 1.0 / cnt
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_fold(k, cfg, model_factory, tok_info, tr_idx, va_idx, chunks, y_ev, y_cl, test_chunks,
               device, log, out_dir):
    import torch

    from .chunking import collate
    from .model import multitask_loss

    pad_id, use_tt = tok_info
    set_seed(cfg["seed"] + k)
    model = model_factory().to(device)
    if cfg.get("freeze_embeddings"):
        # saves ~4 GB of optimizer state for XLM-R (250k-token vocabulary) on 16 GB GPUs
        model.encoder.get_input_embeddings().weight.requires_grad_(False)
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing(True)
    amp_dtype, use_scaler = _amp(device, cfg["precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(tr_idx) / cfg["batch_examples"])
    total = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total = min(total, cfg["max_steps"])
    warm = max(1, int(cfg["warmup"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    w_ev = class_weight_vec(y_ev[tr_idx], len(EVASION), cfg["class_weights"])
    w_cl = class_weight_vec(y_cl[tr_idx], 3, cfg["class_weights"])
    w_ev = None if w_ev is None else w_ev.to(device)
    w_cl = None if w_cl is None else w_cl.to(device)

    rng = np.random.default_rng(cfg["seed"] + k)
    tr_chunks = [chunks[i] for i in tr_idx]
    va_chunks = [chunks[i] for i in va_idx]
    y_va_ev = [EVASION[i] for i in y_ev[va_idx]]
    stopper, step, history = EarlyStopper(cfg["patience"], cfg["min_epochs"]), 0, []
    best_va = best_te = None
    for ep in range(cfg["epochs"]):
        model.train()
        t0, run_loss, n_seen, acc = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for b in make_batches(tr_chunks, cfg["token_budget"], cfg["batch_examples"], rng):
            x = {kk: v.to(device) for kk, v in collate([tr_chunks[i] for i in b], pad_id, use_tt).items()}
            gi = tr_idx[b]
            yc = torch.as_tensor(y_cl[gi], device=device)
            ye = torch.as_tensor(y_ev[gi], device=device)
            with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
                lc, le = model(**x, n_examples=len(b))
            loss = multitask_loss(lc, le, yc, ye, cfg["w_c"], cfg["w_e"], w_cl, w_ev,
                                  cfg["focal_gamma"], cfg["label_smoothing"])
            scaler.scale(loss * len(b) / cfg["batch_examples"]).backward()
            run_loss += loss.item() * len(b)
            n_seen += len(b)
            acc += len(b)
            if acc >= cfg["batch_examples"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                acc, step = 0, step + 1
                if cfg["max_steps"] and step >= cfg["max_steps"]:
                    break
        if acc > 0 and not (cfg["max_steps"] and step >= cfg["max_steps"]):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        pc, pe = predict(model, va_chunks, device, cfg["eval_token_budget"], use_tt, pad_id, amp_dtype)
        pred = [EVASION[i] for i in pe.argmax(1)]
        e_f1, m_f1 = evasion_f1(y_va_ev, pred), mapped_clarity_f1(y_va_ev, pred)
        s = e_f1 + m_f1
        history.append({"epoch": ep + 1, "train_loss": run_loss / max(1, n_seen), "val_evasion_f1": e_f1,
                        "val_mapped_clarity_f1": m_f1, "val_score": s, "seconds": round(time.time() - t0, 1),
                        "steps": step})
        log(f"fold {k} ep {ep + 1}: loss {run_loss / max(1, n_seen):.4f} E {e_f1:.4f} M {m_f1:.4f} "
            f"({time.time() - t0:.0f}s, step {step}/{total})")
        improved, stop = stopper.update(s)
        if improved:
            best_va = (pc, pe)
            best_te = predict(model, test_chunks, device, cfg["eval_token_budget"], use_tt, pad_id, amp_dtype)
        elif stop:
            log(f"fold {k}: early stop at epoch {ep + 1} (best {stopper.best_epoch})")
            break
        if cfg["max_steps"] and step >= cfg["max_steps"]:
            break
    fdir = out_dir / f"fold{k}"
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "metrics.json").write_text(json.dumps(
        {"fold": k, "best_epoch": stopper.best_epoch, "best_val_score": stopper.best, "history": history},
        indent=2), encoding="utf-8")
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return best_va, best_te


def probs_frame(ids, p_ev, p_cl):
    df = pd.DataFrame(p_ev, columns=EV_COLS)
    df[CL_COLS] = p_cl
    df.insert(0, "datapointID", np.asarray(ids))
    return df


def merge_oof(out_dir):
    parts = sorted(Path(out_dir).glob("fold*/oof_part.csv"))
    if not parts:
        return None
    oof = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).sort_values("datapointID")
    tmp = Path(out_dir) / f"oof.csv.tmp{os.getpid()}"
    oof.to_csv(tmp, index=False)
    os.replace(tmp, Path(out_dir) / "oof.csv")
    return oof


def smoke_subset(train, n=32, seed=42):
    """A small subset that still contains every evasion class when possible."""
    rng = np.random.default_rng(seed)
    first = train.groupby("evasion_label").head(2)
    rest = train.drop(first.index)
    extra = rest.iloc[rng.permutation(len(rest))[: max(0, n - len(first))]]
    return pd.concat([first, extra]).sort_values("datapointID").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", default=None, help="comma-separated fold ids (default: all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="skip folds whose outputs already exist (resume)")
    ap.add_argument("--out", default=None, help="artifacts root (default: <repo>/artifacts)")
    a = ap.parse_args(argv)

    import torch

    from .chunking import PairTemplate, encode_examples
    from .context import build_texts, neighbour_questions
    from .model import MultiHeadChunkClassifier

    cfg = load_config(a.config)
    if a.smoke:
        cfg.update({"exp_id": cfg["exp_id"] + "_smoke", "epochs": 1, "max_steps": 2, "batch_examples": 4,
                    "token_budget": 2048, "max_length": 128, "stride": 32, "q_max": 32})
    out_dir = Path(a.out or ARTIFACTS) / cfg["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.merge_only:
        oof = merge_oof(out_dir)
        print(f"merged {0 if oof is None else len(oof)} OOF rows into {out_dir / 'oof.csv'}")
        return
    log = Logger(out_dir / "train.log")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = auto_budgets(cfg, device)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if str(device).startswith("cuda"):
        p = torch.cuda.get_device_properties(device)
        log(f"GPU {p.name} {p.total_memory / 1e9:.0f} GB, precision {_amp(device, cfg['precision'])[0]}, "
            f"token_budget {cfg['token_budget']}")
    if os.environ.get("POLERO_THREADS"):
        torch.set_num_threads(int(os.environ["POLERO_THREADS"]))

    train, test = load_train(), load_test()
    folds = load_folds(train)
    if a.smoke:
        train = smoke_subset(train)
    fold_of = train[["datapointID"]].merge(folds, on="datapointID", how="left")["fold"].to_numpy()
    neigh = neighbour_questions([load_train(), test]) if cfg["context"] != "none" else None
    q_tr, a_tr = build_texts(train, cfg, neigh)
    q_te, a_te = build_texts(test, cfg, neigh)

    if a.smoke:
        from .tiny import make_tiny_tokenizer, tiny_config

        tok = make_tiny_tokenizer(q_tr + a_tr, out_dir / "tiny_tok")
        mcfg = tiny_config(len(tok), max_len=cfg["max_length"])

        def factory():
            return MultiHeadChunkClassifier(config=mcfg, pooling=cfg["pooling"], dropout=cfg["dropout"])
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["backbone"])

        def factory():
            return MultiHeadChunkClassifier(cfg["backbone"], pooling=cfg["pooling"], dropout=cfg["dropout"])

    tpl = PairTemplate(tok)
    enc_kw = {k: cfg[k] for k in ("max_length", "stride", "q_max", "max_chunks")}
    chunks, st = encode_examples(tok, q_tr, a_tr, template=tpl, **enc_kw)
    test_chunks, st_te = encode_examples(tok, q_te, a_te, template=tpl, **enc_kw)
    log(f"exp {cfg['exp_id']} device {device} backbone {cfg['backbone'] if not a.smoke else 'tiny'}")
    log(f"train chunks {st} | test chunks {st_te}")
    use_tt = tpl.has_tt
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    y_ev = train["evasion_label"].map(EV2I).to_numpy()
    y_cl = train["clarity_label"].map(CL2I).to_numpy()
    assert all(CLARITY_OF[e] == c for e, c in zip(train["evasion_label"], train["clarity_label"]))

    todo = sorted(int(f) for f in set(fold_of)) if a.folds is None else [int(x) for x in a.folds.split(",")]
    if a.smoke and a.folds is None:
        todo = todo[:2]
    for k in todo:
        tr_idx, va_idx = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        if len(va_idx) == 0:
            continue
        if a.skip_done and (out_dir / f"fold{k}" / "oof_part.csv").exists() and (out_dir / f"test_fold{k}.csv").exists():
            log(f"fold {k} already done, skipping")
            continue
        t0 = time.time()
        (pc, pe), (tc, tev) = train_fold(k, cfg, factory, (pad_id, use_tt), tr_idx, va_idx, chunks, y_ev, y_cl,
                                         test_chunks, device, log, out_dir)
        probs_frame(train["datapointID"].to_numpy()[va_idx], pe, pc).to_csv(out_dir / f"fold{k}" / "oof_part.csv", index=False)
        probs_frame(test["datapointID"].to_numpy(), tev, tc).to_csv(out_dir / f"test_fold{k}.csv", index=False)
        log(f"fold {k} done in {time.time() - t0:.0f}s")
    oof = merge_oof(out_dir)
    log(f"oof rows so far: {0 if oof is None else len(oof)}")


if __name__ == "__main__":
    main()
