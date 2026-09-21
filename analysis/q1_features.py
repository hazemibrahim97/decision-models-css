import csv
import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task
from scores import CONFIRMATORY, FAMILY

ZIEMS_FINETUNE_F1 = {
    "indian_english_dialect": 3.0, "emotion": 71.6, "flute": 99.2,
    "reddit_humor": 73.1, "ibc": 64.8, "implicit_hate": 62.5, "mrf": 81.6,
    "raop": 52.0, "tempowic": 62.3, "semeval_stance": 36.1,
    "discourse": 49.6, "talklife": 71.6, "persuasion": 33.3,
    "wiki_politeness": 75.8, "wiki_corpus": 72.7, "conv_go_awry": 64.6,
    "media_ideology": 85.1, "tropes": None,
}

def features(task):
    j = load_task(task)
    K = len(BINARY_CRITERIA.get(task) or OPTION_TO_GOLD[task])
    golds = Counter(str(g) for g in j["labels"].values())
    n = sum(golds.values())
    ent = -sum(c / n * math.log(c / n) for c in golds.values())
    ctx_len = sorted(len(j["context"][i].split()) for i in j["prompts"])
    return dict(K=K, entropy_norm=ent / math.log(K),
                median_ctx_tokens=ctx_len[len(ctx_len) // 2],
                family=FAMILY[task], ziems_finetune_f1=ZIEMS_FINETUNE_F1[task])

def main():
    delta = {r["task"]: float(r["delta_f1"])
             for r in csv.DictReader((HERE / "q1_delta.csv").open())}
    rows = []
    for t in CONFIRMATORY:
        f = features(t)
        f.update(task=t, delta_f1=delta[t])
        rows.append(f)
    fields = ["task", "delta_f1", "K", "entropy_norm", "median_ctx_tokens",
              "family", "ziems_finetune_f1"]
    with (HERE / "q1_features.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)

    rng_seed = 20260920
    print(f"Q1 feature correlations with dF1, n={len(rows)} confirmatory tasks "
          "(Spearman, permutation p, 10^5 resamples):")
    d = np.array([r["delta_f1"] for r in rows])
    for feat in ["K", "entropy_norm", "median_ctx_tokens", "ziems_finetune_f1"]:
        mask = np.array([r[feat] is not None for r in rows])
        x = np.array([r[feat] for r, m in zip(rows, mask) if m], float)
        y = d[mask]
        rho = stats.spearmanr(x, y).statistic

        def statfn(xx):
            return stats.spearmanr(xx, y).statistic

        p = stats.permutation_test(
            (x,), statfn, permutation_type="pairings",
            n_resamples=100_000, alternative="two-sided",
            rng=np.random.default_rng(rng_seed)).pvalue
        print(f"  {feat:20s} n={mask.sum():2d} rho={rho:+.3f} p={p:.4f}")
    print("  family (categorical) — median dF1:")
    for fam in ["utterance", "conversation", "document"]:
        v = [r["delta_f1"] for r in rows if r["family"] == fam]
        print(f"    {fam:13s} n={len(v)} median={np.median(v):+.3f}")

if __name__ == "__main__":
    main()
