import csv
import pathlib
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
B_F1 = 10_000
B_CAL = 2_000
SEED = 20260920
PILOT = {"semeval_stance", "implicit_hate", "discourse"}
CLAUDES = ["anthropic_claude-opus-5", "anthropic_claude-fable-5.1",
           "anthropic_claude-sonnet-5"]

def load():
    raw = defaultdict(lambda: defaultdict(dict))
    with (HERE / "items.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["conf_elicited"] != "1" or r["kind"] == "local":
                continue
            raw[r["task"]][r["model"]][r["id"]] = r
    return raw

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

def ece_vec(conf, corr, idx, bins=15):
    C, OK = conf[idx], corr[idx]
    bin_id = np.minimum((C * bins).astype(int), bins - 1)
    n = idx.shape[1]
    out = np.zeros(idx.shape[0])
    for b in range(bins):
        m = bin_id == b
        cnt = m.sum(1)
        gap = np.abs((np.where(m, OK - C, 0.0)).sum(1))
        out += np.where(cnt > 0, gap / n, 0.0)
    return out

def selection_aware():
    rng = np.random.default_rng(SEED)
    raw = load()
    rows = []
    for task in sorted(raw):
        models = raw[task]
        labels = sorted({r["gold"] for m in models.values() for r in m.values()})
        lab = {L: i for i, L in enumerate(labels)}
        K = len(lab)
        jev = models["jev"]
        llms = sorted(m for m in models if m != "jev")

        ids = sorted(set(jev).intersection(*(set(models[m]) for m in llms)))
        g = np.array([lab[jev[i]["gold"]] for i in ids])
        pj = np.array([lab.get(jev[i]["pred"], K) for i in ids])
        idx = rng.integers(0, len(ids), (B_F1, len(ids)))
        fj = macro_f1_boot(g, pj, idx, K)
        fm = np.stack([macro_f1_boot(
            g, np.array([lab.get(models[m][i]["pred"], K) for i in ids]), idx, K)
            for m in llms])
        d = fj - fm.max(0)
        point = d.mean()
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(task=task, n=len(ids), delta_sel_lo=lo, delta_sel_hi=hi,
                         delta_sel_mean=point))
        print(f"{task:24s} selection-aware dF1 CI [{lo:+.3f},{hi:+.3f}]")
    with (HERE / "q1_selection_aware.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def calibration_cis():
    rng = np.random.default_rng(SEED)
    raw = load()
    per_task = []

    draws = defaultdict(dict)
    points = defaultdict(dict)
    for task in sorted(raw):
        for m in ["jev"] + CLAUDES:
            rs = [r for r in raw[task][m].values()
                  if r["valid"] == "1" and r["conf_status"] == "ok"]
            conf = np.array([float(r["conf"]) for r in rs])
            corr = np.array([float(r["correct"]) for r in rs])
            idx = rng.integers(0, len(rs), (B_CAL, len(rs)))
            e = ece_vec(conf, corr, idx)
            kept = conf[idx] >= 0.9
            cov = kept.mean(1)
            with np.errstate(invalid="ignore"):
                acc = np.where(kept.sum(1) > 0,
                               (corr[idx] * kept).sum(1) / np.maximum(kept.sum(1), 1),
                               np.nan)
            point_kept = conf >= 0.9
            pe = ece_vec(conf, corr, np.arange(len(rs))[None, :])[0]
            pcov = point_kept.mean()
            pacc = corr[point_kept].mean() if point_kept.any() else float("nan")
            draws[m][task] = dict(ece=e, cov=cov, acc=acc)
            points[m][task] = dict(ece=pe, cov=pcov, acc=pacc)
            if m == "jev":
                per_task.append(dict(
                    task=task, n=len(rs), ece=pe,
                    ece_lo=np.percentile(e, 2.5), ece_hi=np.percentile(e, 97.5),
                    cov09=pcov,
                    cov_lo=np.percentile(cov, 2.5), cov_hi=np.percentile(cov, 97.5),
                    acc09=pacc,
                    acc_lo=np.nanpercentile(acc, 2.5), acc_hi=np.nanpercentile(acc, 97.5)))
    with (HERE / "calibration_cis.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(per_task[0]))
        w.writeheader()
        w.writerows(per_task)

    ev = [t for t in sorted(raw) if t not in PILOT]
    tidx = rng.integers(0, len(ev), (B_CAL, len(ev)))
    med_rows = []
    for m in ["jev"] + CLAUDES:
        for metric in ["ece", "acc", "cov"]:
            v = np.array([points[m][t][metric] for t in ev])
            med = np.nanmedian(v[tidx], 1)
            med_rows.append(dict(model=m, metric=metric,
                                 point=np.nanmedian(v),
                                 lo=np.nanpercentile(med, 2.5),
                                 hi=np.nanpercentile(med, 97.5)))
            print(f"{m:34s} median {metric}: {np.nanmedian(v):.3f} "
                  f"[{np.nanpercentile(med, 2.5):.3f},{np.nanpercentile(med, 97.5):.3f}]")
    with (HERE / "calibration_median_cis.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(med_rows[0]))
        w.writeheader()
        w.writerows(med_rows)

if __name__ == "__main__":
    selection_aware()
    calibration_cis()
