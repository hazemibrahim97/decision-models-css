import csv
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent

VENDOR = lambda m: ("decision" if m in ("jev", "local_rlcd-0.6b", "local_qwen3-base")
                    else m.split("_")[0])

rows = [r for r in csv.DictReader((HERE / "cell_metrics.csv").open())
        if r["split"] == "confirmatory"]
models = sorted({r["model"] for r in rows})
out = []
for m in models:
    sub = [r for r in rows if r["model"] == m]
    n_items = sum(int(r["n"]) for r in sub)
    cost = sum(float(r["cost"]) for r in sub)
    out.append(dict(
        model=m, kind=sub[0]["kind"], vendor=VENDOR(m), n_tasks=len(sub),
        cost_per_1k=cost / n_items * 1000,
        median_f1=statistics.median(float(r["macro_f1"]) for r in sub),
        mean_f1=statistics.mean(float(r["macro_f1"]) for r in sub),
        median_acc=statistics.median(float(r["acc"]) for r in sub),
        median_ece=statistics.median(float(r["ece"]) for r in sub),
    ))

by_vendor = defaultdict(list)
for r in out:
    by_vendor[r["vendor"]].append(r)
for vendor, group in by_vendor.items():
    for tier, r in enumerate(sorted(group, key=lambda x: x["cost_per_1k"]), 1):
        r["tier"] = tier if len(group) > 1 else ""

with (HERE / "frontier.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0]))
    w.writeheader()
    w.writerows(out)

print(f"{'model':42s} {'vendor':10s} tier  $/1k    medF1   medECE")
for r in sorted(out, key=lambda x: (x["vendor"], x["cost_per_1k"])):
    print(f"{r['model']:42s} {r['vendor']:10s} {str(r['tier']):>4s}  "
          f"{r['cost_per_1k']:7.3f} {r['median_f1']:.3f}  {r['median_ece']:.4f}")
