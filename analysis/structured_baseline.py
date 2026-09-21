import csv
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pilot_jev import OPTION_TO_GOLD, BINARY_CRITERIA
from scores import macro_f1

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results" / "structured"
MODEL = "google_gemini-3.8-flash"

def gold_map(task):
    if task in OPTION_TO_GOLD:
        return OPTION_TO_GOLD[task]
    return {k: k for k in BINARY_CRITERIA[task]}

def main():
    free = defaultdict(dict)
    with (HERE / "items.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["model"] == MODEL and r["conf_elicited"] == "1":
                free[r["task"]][r["id"]] = (r["gold"], r["pred"] if r["valid"] == "1" else None)

    out = []
    for path in sorted(RES.glob(f"*__{MODEL}__structured.jsonl")):
        task = path.name.split("__")[0]
        gmap = gold_map(task)
        golds, s_preds, f_preds, n_bad = [], [], [], 0
        for line in path.open():
            rec = json.loads(line)
            if rec["id"] not in free[task]:
                continue
            try:
                pred = str(gmap[str(json.loads(rec["text"])["answer"])])
            except Exception:
                pred, n_bad = None, n_bad + 1
            golds.append(str(rec["gold"]))
            s_preds.append(pred)
            f_preds.append(free[task][rec["id"]][1])
        n = len(golds)
        out.append(dict(
            task=task, n=n, invalid_structured=n_bad,
            invalid_free=sum(p is None for p in f_preds),
            acc_structured=sum(g == p for g, p in zip(golds, s_preds)) / n,
            acc_free=sum(g == p for g, p in zip(golds, f_preds)) / n,
            f1_structured=macro_f1(golds, [p if p is not None else "<inv>" for p in s_preds]),
            f1_free=macro_f1(golds, [p if p is not None else "<inv>" for p in f_preds])))
        r = out[-1]
        print(f"{task:24s} n={n:4d} acc {r['acc_free']:.3f}->{r['acc_structured']:.3f} "
              f"F1 {r['f1_free']:.3f}->{r['f1_structured']:.3f} bad={n_bad}")
    with (HERE / "structured_baseline.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)

if __name__ == "__main__":
    main()
