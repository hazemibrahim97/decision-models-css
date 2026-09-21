import csv
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pilot_jev import OPTION_TO_GOLD, load_task

A = ROOT / "analysis"
DISCOVERY = {"semeval_stance", "implicit_hate", "discourse"}
CONFIRMATORY = {"conv_go_awry", "emotion", "flute", "ibc", "indian_english_dialect",
                "media_ideology", "mrf", "persuasion", "raop", "reddit_humor",
                "talklife", "tempowic", "tropes", "wiki_corpus", "wiki_politeness"}

def load_items():
    rows = list(csv.DictReader(open(A / "items.csv")))
    assert len(rows) == 198143, len(rows)
    return rows

def conf_items(rows, model, tasks):
    out = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["task"] in tasks and r["conf_status"] == "ok" and r["valid"] == "1":
            out[r["task"]].append((float(r["conf"]), int(r["correct"])))
    return out

def ece(pairs, n_bins=15, equal_mass=False):
    if not pairs:
        return None
    if equal_mass:
        s = sorted(pairs, key=lambda p: p[0])
        bins = [s[i * len(s) // n_bins:(i + 1) * len(s) // n_bins] for i in range(n_bins)]
    else:
        bins = [[] for _ in range(n_bins)]
        for c, y in pairs:
            bins[min(int(c * n_bins), n_bins - 1)].append((c, y))
    n = len(pairs)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(y for _, y in b) / len(b))
               for b in bins if b)

def median(xs):
    xs = sorted(x for x in xs if x is not None)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

def b1(rows, models):
    with open(A / "b1_binning.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ece_w10", "ece_w15", "ece_w20", "ece_m15"])
        for m in models:
            per = conf_items(rows, m, CONFIRMATORY)
            vals = [median(ece(per[t], nb, em) for t in per)
                    for nb, em in [(10, False), (15, False), (20, False), (15, True)]]
            w.writerow([m] + [f"{v:.4f}" for v in vals])

def b2(rows, models):
    eps = 1e-3
    with open(A / "b2_temp_scaling.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "T", "median_ece_before", "median_ece_after"])
        for m in models:
            disc = [p for ps in conf_items(rows, m, DISCOVERY).values() for p in ps]
            if len(disc) < 100:
                continue
            zs = [math.log(min(max(c, eps), 1 - eps) / (1 - min(max(c, eps), 1 - eps))) for c, _ in disc]
            ys = [y for _, y in disc]

            def nll(T):
                tot = 0.0
                for z, y in zip(zs, ys):
                    p = 1 / (1 + math.exp(-z / T))
                    p = min(max(p, eps), 1 - eps)
                    tot -= math.log(p if y else 1 - p)
                return tot
            grid = [10 ** (k / 40 - 1) for k in range(81)]
            T = min(grid, key=nll)
            per = conf_items(rows, m, CONFIRMATORY)
            before = median(ece(ps) for ps in per.values())
            after = median(
                ece([(1 / (1 + math.exp(-math.log(min(max(c, eps), 1 - eps) / (1 - min(max(c, eps), 1 - eps))) / T)), y)
                     for c, y in ps]) for ps in per.values())
            w.writerow([m, f"{T:.3f}", f"{before:.4f}", f"{after:.4f}"])

def b3(rows, models):
    g2l = {}
    for task, l2g in OPTION_TO_GOLD.items():
        if task in CONFIRMATORY:
            g2l[task] = {g: i for i, (_, g) in enumerate(sorted(l2g.items()))}
    with open(A / "b3_position_bias.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "share_first_pred", "share_first_gold", "tv_distance"])
        for m in models:
            pred_pos, gold_pos = Counter(), Counter()
            n = 0
            for r in rows:
                t = r["task"]
                if r["model"] != m or t not in g2l or r["valid"] != "1":
                    continue
                if r["pred"] not in g2l[t] or r["gold"] not in g2l[t]:
                    continue
                pred_pos[g2l[t][r["pred"]]] += 1
                gold_pos[g2l[t][r["gold"]]] += 1
                n += 1
            if not n:
                continue
            ks = set(pred_pos) | set(gold_pos)
            tv = 0.5 * sum(abs(pred_pos[k] / n - gold_pos[k] / n) for k in ks)
            w.writerow([m, n, f"{pred_pos[0] / n:.4f}", f"{gold_pos[0] / n:.4f}", f"{tv:.4f}"])

def b4():
    cells = list(csv.DictReader(open(A / "cell_metrics.csv")))
    with open(A / "b4_conf_vs_ptop.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "ece_conf", "ece_ptop"])
        es, ps = [], []
        for r in cells:
            if r["model"] == "jev" and r["split"] == "confirmatory":
                w.writerow([r["task"], f"{float(r['ece']):.4f}", f"{float(r['ece_ptop']):.4f}"])
                es.append(float(r["ece"])); ps.append(float(r["ece_ptop"]))
        w.writerow(["median", f"{median(es):.4f}", f"{median(ps):.4f}"])

def b5(rows):
    for task in ["talklife", "conv_go_awry", "implicit_hate"]:
        cm = Counter((r["gold"], r["pred"]) for r in rows
                     if r["model"] == "jev" and r["task"] == task and r["valid"] == "1")
        labels = sorted({g for g, _ in cm} | {p for _, p in cm})
        with open(A / f"b5_confusion_{task}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gold\\pred"] + labels)
            for g in labels:
                w.writerow([g] + [cm[(g, p)] for p in labels])
    cm = Counter((r["gold"], r["pred"]) for r in rows
                 if r["model"] == "jev" and r["task"] == "tropes" and r["valid"] == "1" and r["gold"] != r["pred"])
    with open(A / "b5_confusion_tropes_top.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gold", "pred", "count"])
        for (g, p), c in cm.most_common(10):
            w.writerow([g, p, c])

def b6(rows):
    ours = {(r["task"], r["id"]): int(r["correct"]) for r in rows
            if r["model"] in ("jev", "google_gemini-3.8-flash") and r["valid"] == "1"
            for _ in [0] if True}
    by_model = defaultdict(dict)
    for r in rows:
        if r["model"] in ("jev", "google_gemini-3.8-flash") and r["valid"] == "1":
            by_model[r["model"]][(r["task"], r["id"])] = int(r["correct"])
    out = []
    for task in sorted(CONFIRMATORY | DISCOVERY):
        path = ROOT / "LLMs_for_CSS" / "css_data" / task / "answer-chatgpt"
        if not path.exists():
            continue
        j = load_task(task)
        ids = list(j["prompts"])
        ans = list(csv.reader(open(path), delimiter="\t"))

        if len(ans) != len(ids) or not all(str(j["labels"][ids[k]]) == a[1] for k, a in enumerate(ans)):
            print(f"b6: {task} answer file does not align with prompt order, skipped")
            continue

        l2g = OPTION_TO_GOLD.get(task, {})
        gold_space = {a[1] for a in ans}
        def to_gold(a):
            a = a.strip().lstrip("&").strip().rstrip(".")
            for cand in (a, str(l2g.get(a, a)), {"True": "1.0", "False": "0.0"}.get(a)):
                if cand in gold_space:
                    return cand
            return a
        c2023 = {ids[k]: int(a[1] == to_gold(a[2])) for k, a in enumerate(ans)}
        for m in ("jev", "google_gemini-3.8-flash"):
            cells = {i: (by_model[m].get((task, i)), c2023[i]) for i in ids
                     if (task, i) in by_model[m]}
            n = len(cells)
            both = sum(1 for a, b in cells.values() if a and b)
            only_new = sum(1 for a, b in cells.values() if a and not b)
            only_old = sum(1 for a, b in cells.values() if b and not a)
            out.append([task, m, n, both, only_new, only_old, n - both - only_new - only_old])
    with open(A / "b6_vs_2023.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "n", "both_correct", "only_2026", "only_2023", "neither"])
        w.writerows(out)

def b8():
    casc = list(csv.DictReader(open(A / "cascade.csv")))
    casc = [r for r in casc if r["task"] in CONFIRMATORY]
    with open(A / "b8_cascade_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["llm", "t", "cost_scenario", "median_cost_fraction"])
        for llm in sorted({r["llm"] for r in casc}):
            for t in ["0.5", "0.8", "0.95"]:
                sub = [r for r in casc if r["llm"] == llm and r["t"] == t]
                for name, mj, ml in [("baseline", 1, 1), ("jev_x2", 2, 1), ("jev_x0.5", 0.5, 1),
                                     ("llm_x2", 1, 2), ("llm_x0.5", 1, 0.5)]:
                    fr = median((mj * float(r["cost_jev_alone"]) +
                                 ml * (float(r["cost_cascade"]) - float(r["cost_jev_alone"]))) /
                                (ml * float(r["cost_llm_alone"])) for r in sub)
                    w.writerow([llm, t, name, f"{fr:.4f}"])
    cells = [r for r in csv.DictReader(open(A / "cell_metrics.csv"))
             if r["model"] == "jev" and r["split"] == "confirmatory"]
    with open(A / "b8_human_review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "usd_per_item_human", "median_accuracy", "median_cost_per_1k"])
        for t in ["0.9", "0.95"]:
            for h in [0.05, 0.25, 1.0]:
                accs, costs = [], []
                for r in cells:
                    cov, acc = r[f"cov{t}"], r[f"acc{t}"]
                    if not cov or not acc:
                        continue
                    cov, acc = float(cov), float(acc)
                    accs.append(acc * cov + 1.0 * (1 - cov))
                    costs.append(float(r["cost"]) / float(r["n"]) * 1000 + (1 - cov) * h * 1000)
                w.writerow([t, h, f"{median(accs):.4f}", f"{median(costs):.2f}"])

def b9(rows):
    cells = {r["task"]: r for r in csv.DictReader(open(A / "cell_metrics.csv"))
             if r["model"] == "jev" and r["split"] == "confirmatory"}
    tasks = [t for t, r in cells.items()
             if r["macro_f1"] and float(r["acc"]) - float(r["macro_f1"]) >= 0.10 and t != "tropes"]
    with open(A / "b9_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "class", "support", "precision", "recall", "f1"])
        for task in sorted(tasks):
            for m in ("jev", "google_gemini-3.8-flash"):
                sub = [r for r in rows if r["task"] == task and r["model"] == m and r["valid"] == "1"]
                for c in sorted({r["gold"] for r in sub}):
                    tp = sum(1 for r in sub if r["gold"] == c and r["pred"] == c)
                    fp = sum(1 for r in sub if r["gold"] != c and r["pred"] == c)
                    fn = sum(1 for r in sub if r["gold"] == c and r["pred"] != c)
                    p = tp / (tp + fp) if tp + fp else 0.0
                    rc = tp / (tp + fn) if tp + fn else 0.0
                    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
                    w.writerow([task, m, c, tp + fn, f"{p:.3f}", f"{rc:.3f}", f"{f1:.3f}"])

def main():
    rows = load_items()
    models = sorted({r["model"] for r in rows})
    b1(rows, models)
    b2(rows, models)
    b3(rows, models)
    b4()
    b5(rows)
    b6(rows)
    b8()
    b9(rows)

    reg = {r["model"]: float(r["ece"]) for r in csv.DictReader(open(A / "cell_metrics.csv"))
           if r["split"] == "confirmatory" and r["task"] == "ibc" and r["ece"]}
    per = conf_items(rows, "jev", {"ibc"})
    assert abs(ece(per["ibc"]) - reg["jev"]) < 1e-6, (ece(per["ibc"]), reg["jev"])
    print("appendix_extras: all outputs written, self-check passed")

if __name__ == "__main__":
    main()
