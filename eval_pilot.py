import json
import pathlib
import sys
from collections import defaultdict

OUT = pathlib.Path(__file__).parent / "results" / "pilot"

def macro_f1(golds, preds):
    labels = sorted(set(golds))
    f1s = []
    for L in labels:
        tp = sum(g == L and p == L for g, p in zip(golds, preds))
        fp = sum(g != L and p == L for g, p in zip(golds, preds))
        fn = sum(g == L and p != L for g, p in zip(golds, preds))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / len(f1s)

def ece(confs, correct, bins=15):
    buckets = defaultdict(list)
    for c, ok in zip(confs, correct):
        buckets[min(int(c * bins), bins - 1)].append(ok - c)
    n = len(confs)
    return sum(abs(sum(v) / len(v)) * len(v) / n for v in buckets.values())

def cov_acc(confs, correct, threshold):
    kept = [ok for c, ok in zip(confs, correct) if c >= threshold]
    return (len(kept) / len(correct), sum(kept) / len(kept) if kept else float("nan"))

for f in sorted(OUT.glob("*.jsonl")):
    if "__" in f.name:
        continue
    recs = [json.loads(l) for l in f.open()]
    golds = [str(r["gold"]) for r in recs]
    preds = [str(r["choice"]) for r in recs]

    p_top = [r["probabilities"][str(r["choice"])] for r in recs]
    conf = [r["confidence"] for r in recs]
    correct = [g == p for g, p in zip(golds, preds)]
    cost = sum(r["usage"]["cost"] for r in recs if r.get("usage"))
    print(f"\n{f.stem}  (n={len(recs)}, model={recs[0]['model_resolved']})")
    print(f"  acc={sum(correct)/len(correct):.3f}  macroF1={macro_f1(golds, preds):.3f}  cost=${cost:.4f}")
    print(f"  ECE(p_top)={ece(p_top, correct):.3f}  ECE(confidence)={ece(conf, correct):.3f}")
    for t in (0.5, 0.7, 0.9):
        cv, ac = cov_acc(conf, correct, t)
        print(f"  conf>={t}: coverage={cv:.2f} acc={ac:.3f}")
