import csv
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
MODELS = ["jev", "google_gemini-3.8-flash", "local_rlcd-0.6b"]

data = defaultdict(list)
for r in csv.DictReader((HERE / "items.csv").open()):
    if (r["model"] in MODELS and r["conf_elicited"] == "1"
            and r["valid"] == "1" and r["conf_status"] == "ok"):
        data[(r["task"], r["model"])].append((float(r["conf"]), int(r["correct"])))

with (HERE / "routing_curves.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "model", "coverage", "accuracy"])
    for (task, model), rows in sorted(data.items()):
        rows.sort(key=lambda x: -x[0])
        run = 0
        n = len(rows)
        for k, (_, ok) in enumerate(rows, 1):
            run += ok
            if k % max(1, n // 100) == 0 or k == n:
                w.writerow([task, model, k / n, run / k])
print("wrote routing_curves.csv")
