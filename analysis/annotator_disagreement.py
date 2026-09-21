import csv
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
MODELS = ["jev", "google_gemini-3.8-flash", "anthropic_claude-opus-5",
          "anthropic_claude-fable-5.1", "anthropic_claude-sonnet-5",
          "google_gemma-4-31b-it"]

def main(corpus_csv):
    sd_by_text = {}
    for r in csv.DictReader(open(corpus_csv)):
        scores = [float(r[f"Score{i}"]) for i in range(1, 6)]
        sd_by_text[" ".join(r["Request"].split())] = float(np.std(scores))

    ctx = json.load(open(HERE.parent / "LLMs_for_CSS" / "css_data" /
                         "wiki_politeness" / "test.json"))["context"]
    sd_by_id = {}
    for i, text in ctx.items():
        key = " ".join(text.removeprefix("user: ").split())
        if key in sd_by_text:
            sd_by_id[i] = sd_by_text[key]
    print(f"matched {len(sd_by_id)}/{len(ctx)} test items to annotator scores")

    conf = defaultdict(dict)
    with (HERE / "items.csv").open() as fh:
        for r in csv.DictReader(fh):
            if (r["task"] == "wiki_politeness" and r["conf_elicited"] == "1"
                    and r["valid"] == "1" and r["conf_status"] == "ok"):
                conf[r["model"]][r["id"]] = float(r["conf"])

    out = []
    for m in MODELS:
        ids = sorted(set(conf[m]) & set(sd_by_id))
        rho, p = spearmanr([conf[m][i] for i in ids], [sd_by_id[i] for i in ids])
        out.append(dict(model=m, n=len(ids), rho=rho, p=p))
        print(f"{m:34s} n={len(ids)} rho={rho:+.3f} p={p:.4f}")
    with (HERE / "annotator_disagreement.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)

if __name__ == "__main__":
    main(sys.argv[1])
