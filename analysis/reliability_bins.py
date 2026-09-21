import csv
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
MODELS = ["jev", "google_gemini-3.8-flash", "local_rlcd-0.6b"]
BINS = 15

acc = defaultdict(lambda: [0, 0, 0.0])
for r in csv.DictReader((HERE / "items.csv").open()):
    if (r["model"] not in MODELS or r["conf_elicited"] != "1"
            or r["valid"] != "1" or r["conf_status"] != "ok"):
        continue
    c = float(r["conf"])
    b = min(int(c * BINS), BINS - 1)
    cell = acc[(r["task"], r["model"], b)]
    cell[0] += 1
    cell[1] += int(r["correct"])
    cell[2] += c

with (HERE / "reliability_bins.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "model", "bin", "n", "acc", "mean_conf"])
    for (task, model, b), (n, ok, sc) in sorted(acc.items()):
        w.writerow([task, model, b, n, ok / n, sc / n])
print("wrote", HERE / "reliability_bins.csv")
