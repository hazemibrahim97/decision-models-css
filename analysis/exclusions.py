import csv
import pathlib
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent

TASK_N = {"semeval_stance": 435, "implicit_hate": 498, "discourse": 497,
          "emotion": 498, "ibc": 498, "media_ideology": 498,
          "indian_english_dialect": 266, "raop": 399, "talklife": 498,
          "wiki_politeness": 498, "tempowic": 344, "tropes": 114,
          "flute": 500, "mrf": 500, "conv_go_awry": 500, "persuasion": 434,
          "reddit_humor": 500, "wiki_corpus": 500}

counts = defaultdict(Counter)
for r in csv.DictReader((HERE / "items.csv").open()):
    if r["conf_elicited"] != "1":
        continue
    c = counts[(r["task"], r["model"])]
    c["n"] += 1
    if r["conf_status"] == "empty_text":
        c["empty_text"] += 1
    elif r["valid"] == "0":
        c["invalid"] += 1
    if r["valid"] == "1" and r["conf_status"] in ("absent", "negative", "oob"):
        c["conf_" + r["conf_status"]] += 1

rows = []
for (task, model), c in sorted(counts.items()):
    api_gap = TASK_N[task] - c["n"]
    rows.append(dict(task=task, model=model, n=c["n"], api_gap=api_gap,
                     empty_text=c["empty_text"], invalid=c["invalid"],
                     conf_absent=c["conf_absent"], conf_negative=c["conf_negative"],
                     conf_oob=c["conf_oob"]))

with (HERE / "exclusions.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0]))
    w.writeheader()
    w.writerows(rows)

tot = Counter()
for r in rows:
    for k in ("api_gap", "empty_text", "invalid", "conf_absent",
              "conf_negative", "conf_oob"):
        tot[k] += r[k]
print("grid totals:", dict(tot))
print("\ncells with api_gap or empty_text:")
for r in rows:
    if r["api_gap"] or r["empty_text"]:
        print(f"  {r['task']:24s} {r['model']:42s} gap={r['api_gap']} empty={r['empty_text']}")
