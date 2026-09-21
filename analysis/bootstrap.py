import csv
import pathlib
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
B = 10_000
SEED = 20260920

def load():
    cells = defaultdict(dict)
    raw = defaultdict(lambda: defaultdict(dict))
    with (HERE / "items.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["conf_elicited"] != "1" or r["kind"] == "local":
                continue
            raw[r["task"]][r["model"]][r["id"]] = (r["gold"], r["pred"])
    return raw

def encode(task_models):
    labels = sorted({g for m in task_models.values() for g, _ in m.values()})
    lab = {L: i for i, L in enumerate(labels)}
    return lab

def macro_f1_boot(g, p, idx, K):
    G, P = g[idx], p[idx]
    f1s = np.zeros((idx.shape[0], K))
    present = np.zeros((idx.shape[0], K), bool)
    for L in range(K):
        gl, pl = G == L, P == L
        tp = (gl & pl).sum(1)
        fp = (~gl & pl).sum(1)
        fn = (gl & ~pl).sum(1)
        denom = 2 * tp + fp + fn
        f1s[:, L] = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        present[:, L] = gl.any(1)
    return (f1s * present).sum(1) / present.sum(1)

def macro_f1_point(g, p, K):
    return macro_f1_boot(g, p, np.arange(len(g))[None, :], K)[0]

def bh_fdr(ps):
    order = np.argsort(ps)
    q = np.empty_like(ps)
    m = len(ps)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * m / rank)
        q[i] = prev
    return q

def main():
    rng = np.random.default_rng(SEED)
    raw = load()
    out = []
    q1 = []
    for task in sorted(raw):
        models = raw[task]
        lab = encode(models)
        K = len(lab)
        jev = models["jev"]
        llms = sorted(m for m in models if m != "jev")

        stats = {}
        for m in llms:
            ids = sorted(set(jev) & set(models[m]))
            g = np.array([lab[jev[i][0]] for i in ids])
            pj = np.array([lab.get(jev[i][1], K) for i in ids])
            pm = np.array([lab.get(models[m][i][1], K) for i in ids])
            idx = rng.integers(0, len(ids), (B, len(ids)))
            d = macro_f1_boot(g, pj, idx, K) - macro_f1_boot(g, pm, idx, K)
            point = macro_f1_point(g, pj, K) - macro_f1_point(g, pm, K)
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            stats[m] = dict(task=task, model=m, n=len(ids), delta_f1=point,
                            ci_lo=lo, ci_hi=hi, p=min(max(p, 1 / B), 1.0))
            out.append(stats[m])

        best = max(llms, key=lambda m: macro_f1_point(
            np.array([lab[models[m][i][0]] for i in sorted(models[m])]),
            np.array([lab.get(models[m][i][1], K) for i in sorted(models[m])]), K))
        s = stats[best]
        q1.append(dict(task=task, best_llm=best, delta_f1=s["delta_f1"],
                       ci_lo=s["ci_lo"], ci_hi=s["ci_hi"]))
        print(f"{task:24s} best_llm={best:34s} dF1={s['delta_f1']:+.3f} "
              f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]")

    ps = np.array([r["p"] for r in out])
    qs = bh_fdr(ps)
    for r, qv in zip(out, qs):
        r["q_bh"] = qv
        r["sig_q05"] = int(qv < 0.05)
    with (HERE / "bootstrap_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)
    with (HERE / "q1_delta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(q1[0]))
        w.writeheader()
        w.writerows(q1)
    print(f"\n{len(out)} cells; significant at BH-FDR q<.05: {int(sum(r['sig_q05'] for r in out))}")

if __name__ == "__main__":
    main()
