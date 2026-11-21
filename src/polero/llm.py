"""Open-weight LLM few-shot classifier (T3.4) against a *localhost* OpenAI-compatible server (vLLM / llama-server / Ollama).

Design:
- one fixed few-shot block (drawn from a single fold, default fold 0) shared by every prompt, so the server's prefix
  cache makes it nearly free; rows of that fold are never evaluated, so LLM outputs on folds 1-6 are valid OOF.
- reasoning allowed, then a strict final line `FINAL: <letter>`.
- n samples per item (self-consistency); the vote shares become 9-class probabilities for ensembling.
- optional encoder candidates (T3.5): the prompt lists the encoder's top-k labels as hints.
- every response is cached in a jsonl keyed by a hash of (model, prompt, sample index).

python -m polero.llm --base-url http://localhost:8000/v1 --model <name> --split test --n 5 --exp llm_qwen
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS, SEED
from .data import add_neighbours, load_all
from .folds import get_folds
from .labels import EV_COLS, EVASION

LETTERS = "ABCDEFGHI"
L2EV = dict(zip(LETTERS, EVASION))

DEFINITIONS = """A. Explicit — the requested information is stated directly, in the exact form requested, without ambiguity or required inference. (Clear Reply)
B. Implicit — the answer is conveyed indirectly; the listener must infer it "between the lines". (Ambivalent)
C. General — stays on the question's topic but is too broad, abstract or vague to give the specific detail requested. (Ambivalent)
D. Partial/half-answer — concretely answers only one component/subset of what was asked and omits the rest. (Ambivalent)
E. Dodging — ignores the substance of the question and talks about something unrelated, without acknowledging the question's premise. (Ambivalent)
F. Deflection — acknowledges/touches the question's topic first, then shifts focus to a different point, achievement or actor instead of answering. (Ambivalent)
G. Declining to answer — explicit, acknowledged refusal to give the information (declines to comment, defers to a future event). (Clear Non-Reply)
H. Claims ignorance — explicitly says they do not know / do not hold the requested information. (Clear Non-Reply)
I. Clarification — does not answer; asks the journalist to clarify what the question means. (Clear Non-Reply)

Key boundaries:
- Explicit vs Implicit: both answer; Explicit states it directly, Implicit requires inference.
- General vs Partial: General is vague/abstract with no concrete specifics; Partial gives a concrete fact for only part of what was asked.
- Dodging vs Deflection: both shift topic; Deflection first acknowledges/touches the question's topic then pivots; Dodging ignores the topic altogether.
- Judge only whether the *specific information requested* was given. Long, talkative answers that never deliver it are not Explicit.
- A president answering a yes/no question with a clear position (even with extra commentary) is Explicit."""

SYSTEM = """You are an expert annotator of political discourse. You classify how a Romanian president responds to a journalist's question, using a 9-label evasion taxonomy (the question and answer are in Romanian).

Labels:
{defs}

Procedure: (1) state in one sentence what exact information the question requests; (2) check whether and how the answer provides it; (3) decide between the closest competing labels. Keep the reasoning brief. End with exactly one line: FINAL: <letter>"""


def fmt_item(q, a, max_words=700, prev_q=None):
    words = a.split()
    if len(words) > max_words:  # keep start and end of very long answers (pivots often come late)
        a = " ".join(words[: max_words // 2]) + " [...] " + " ".join(words[-max_words // 2 :])
    ctx = f"Previous question in the same interview: {prev_q}\n" if prev_q else ""
    return f"{ctx}Question: {q}\nAnswer: {a}"


def select_fewshot(train_fold_df, per_class=2, extra=("Dodging", "Deflection", "Implicit", "General"), max_words=110, seed=SEED):
    """Pick short-ish examples per class from one fold (deterministic)."""
    rng = np.random.RandomState(seed)
    out = []
    for lab in EVASION:
        pool = train_fold_df[(train_fold_df["evasion_label"] == lab)]
        short = pool[pool["interview_answer"].str.split().str.len() <= max_words]
        pool = short if len(short) >= per_class else pool
        n = per_class + (1 if lab in extra else 0)
        out.append(pool.iloc[rng.permutation(len(pool))[:n]])
    shots = pd.concat(out)
    return shots.iloc[rng.permutation(len(shots))]  # mix classes


def fewshot_block(shots):
    parts = []
    for i, (_, r) in enumerate(shots.iterrows(), 1):
        letter = LETTERS[EVASION.index(r["evasion_label"])]
        parts.append(f"### Example {i}\n{fmt_item(r['interview_question'], r['interview_answer'], 250)}\nFINAL: {letter}")
    return "Annotated examples:\n\n" + "\n\n".join(parts)


def build_messages(row, shots_text, candidates=None, use_prev=False):
    user = fmt_item(row["interview_question"], row["interview_answer"], prev_q=row.get("prev_q") if use_prev else None)
    if candidates:
        user += "\n\nA trained classifier considers these labels most likely (it is often but not always right): " + ", ".join(
            f"{LETTERS[EVASION.index(c)]} ({c})" for c in candidates)
    return [
        {"role": "system", "content": SYSTEM.format(defs=DEFINITIONS) + "\n\n" + shots_text},
        {"role": "user", "content": "Classify this response.\n\n" + user},
    ]


_NEAR = {re.sub(r"[^a-z]", "", e.lower()): e for e in EVASION}
_NEAR.update({"partial": "Partial/half-answer", "halfanswer": "Partial/half-answer", "declining": "Declining to answer",
              "decline": "Declining to answer", "ignorance": "Claims ignorance"})


def parse_label(text):
    """Strict parse of the final line; returns an EVASION string or None."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.findall(r"FINAL\s*:\s*\**\s*([A-I])\b(?:[\s.)\]*]|$)", text)
    if m:
        return L2EV[m[-1]]
    m = re.findall(r"FINAL\s*:\s*\**\s*([A-Za-z/ \-]+)", text)
    if m:
        key = re.sub(r"[^a-z]", "", m[-1].lower())
        return _NEAR.get(key)
    return None


class OpenAIBackend:
    def __init__(self, base_url, model, temperature=0.6, max_tokens=1500, top_p=0.95, extra=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        assert "localhost" in self.url or "127.0.0.1" in self.url, "LLM endpoint must be localhost (no paid/remote APIs)"
        self.model, self.temperature, self.max_tokens, self.top_p, self.extra = model, temperature, max_tokens, top_p, extra or {}

    def __call__(self, messages, seed):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": self.max_tokens, "seed": seed, **self.extra}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if parse_label(content) is None and reasoning:  # some servers put everything in the reasoning field
            content = content + "\n" + reasoning
        return content, out.get("usage", {})


class FakeBackend:
    """Deterministic offline backend for tests."""

    def __init__(self, answer="FINAL: A"):
        self.model, self.answer, self.calls = "fake", answer, 0

    def __call__(self, messages, seed):
        self.calls += 1
        return (self.answer(messages) if callable(self.answer) else self.answer), {}


def _key(model, messages, i):
    return hashlib.sha256(json.dumps([model, messages, i], ensure_ascii=False).encode()).hexdigest()


def classify(rows, backend, shots_text, n=1, cache_path=None, workers=16, candidates=None, use_prev=False, retries=1):
    """Return (probs [len(rows), 9], details list). Probs are vote shares; unparsable samples are dropped."""
    cache = {}
    if cache_path and Path(cache_path).exists():
        for line in open(cache_path, encoding="utf-8"):
            d = json.loads(line)
            cache[d["key"]] = d
    fh = open(cache_path, "a", encoding="utf-8") if cache_path else None
    jobs = []
    for r_i, (_, row) in enumerate(rows.iterrows()):
        msgs = build_messages(row, shots_text, candidates[r_i] if candidates else None, use_prev)
        for s in range(n):
            jobs.append((r_i, s, msgs, _key(backend.model, msgs, s)))

    def work(job):
        r_i, s, msgs, key = job
        if key in cache:
            return cache[key]
        for attempt in range(retries + 1):
            try:
                text, usage = backend(msgs, SEED + s + 1000 * attempt)
            except Exception as e:  # network hiccup: retry, then record failure
                text, usage = f"ERROR {e}", {}
            lab = parse_label(text)
            if lab is not None:
                break
        return {"key": key, "row": int(r_i), "sample": s, "label": lab, "text": text, "usage": usage}

    t0, done = time.time(), 0
    results = []
    with ThreadPoolExecutor(workers) as ex:
        for d in ex.map(work, jobs):
            if d["key"] not in cache and fh:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                fh.flush()
            results.append(d)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} samples, {time.time() - t0:.0f}s", flush=True)
    if fh:
        fh.close()
    P = np.zeros((len(rows), 9))
    for d in results:
        if d["label"]:
            P[d["row"], EVASION.index(d["label"])] += 1
    fails = sum(d["label"] is None for d in results)
    empty = P.sum(1) == 0
    P[empty, 0] = 1  # logged fallback: Explicit
    P = P / P.sum(1, keepdims=True)
    return P, {"n_samples": len(results), "parse_fail": fails, "rows_fallback": int(empty.sum()), "seconds": time.time() - t0}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--split", choices=["test", "oof", "oof_sample"], default="test")
    p.add_argument("--fewshot-fold", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--use-prev", action="store_true")
    p.add_argument("--candidates", default=None, help="probs csv (oof/test) of an encoder: add top-k hints")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--extra", default="{}", help="extra JSON body fields, e.g. chat_template_kwargs")
    p.add_argument("--exp", required=True)
    a = p.parse_args(argv)

    df = add_neighbours(load_all()).merge(get_folds(), on="datapointID", how="left")
    df["fold"] = df["fold"].fillna(-1).astype(int)
    shots = select_fewshot(df[(~df["is_test"]) & (df["fold"] == a.fewshot_fold)])
    shots_text = fewshot_block(shots)
    if a.split == "test":
        rows = df[df["is_test"]]
    else:
        rows = df[(~df["is_test"]) & (df["fold"] != a.fewshot_fold)]
        if a.split == "oof_sample":  # stratified 20 per class
            rows = rows.groupby("evasion_label", group_keys=False).apply(lambda g: g.sample(min(20, len(g)), random_state=SEED))
    rows = rows.sort_values("datapointID")
    if a.limit:
        rows = rows.iloc[: a.limit]
    cands = None
    if a.candidates:
        cp = pd.read_csv(a.candidates).set_index("datapointID").loc[rows["datapointID"], EV_COLS].to_numpy()
        cands = [[EVASION[j] for j in np.argsort(-r)[: a.k]] for r in cp]
    out = ARTIFACTS / a.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_example.txt").write_text(json.dumps(build_messages(rows.iloc[0], shots_text), ensure_ascii=False, indent=1), encoding="utf-8")
    backend = OpenAIBackend(a.base_url, a.model, a.temperature, a.max_tokens, extra=json.loads(a.extra))
    P, info = classify(rows, backend, shots_text, a.n, out / "cache.jsonl", a.workers, cands, a.use_prev)
    res = pd.DataFrame(P, columns=EV_COLS)
    res.insert(0, "datapointID", rows["datapointID"].to_numpy())
    res.to_csv(out / f"{a.split}.csv", index=False)
    info.update(vars(a))
    (out / f"info_{a.split}.json").write_text(json.dumps(info, indent=2))
    print(info)
    if a.split != "test":
        from .metrics import evaluate_oof, print_report

        print_report(evaluate_oof(res))


if __name__ == "__main__":
    main()
