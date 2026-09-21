import csv
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DISCOVERY = {"semeval_stance", "implicit_hate", "discourse"}

FAMILY = {
    "indian_english_dialect": "utterance", "implicit_hate": "utterance",
    "flute": "utterance", "emotion": "utterance", "reddit_humor": "utterance",
    "mrf": "utterance", "ibc": "utterance", "raop": "utterance",
    "tempowic": "utterance", "semeval_stance": "utterance",
    "discourse": "conversation", "talklife": "conversation",
    "persuasion": "conversation", "wiki_politeness": "conversation",
    "conv_go_awry": "conversation", "wiki_corpus": "conversation",
    "media_ideology": "document", "tropes": "document",
}
CONFIRMATORY = sorted(set(FAMILY) - DISCOVERY)

def macro_f1(golds, preds):
    labels = sorted(set(golds))
    f1s = []
    for L in labels:
        tp = sum(g == L and p == L for g, p in zip(golds, preds))
        fp = sum(g != L and p == L for g, p in zip(golds, preds))
        fn = sum(g == L and p != L for g, p in zip(golds, preds))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / len(f1s)

def ece(confs, correct, bins=15):
    buckets = defaultdict(list)
    for c, ok in zip(confs, correct):
        buckets[min(int(c * bins), bins - 1)].append(ok - c)
    n = len(confs)
    return sum(abs(sum(v) / len(v)) * len(v) / n for v in buckets.values())

def cov_acc(confs, correct, t):
    kept = [ok for c, ok in zip(confs, correct) if c >= t]
    return (len(kept) / len(correct) if correct else float("nan"),
            sum(kept) / len(kept) if kept else float("nan"))

def curve_auc(confs, correct):
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    run = 0
    auc = 0.0
    for k, i in enumerate(order, 1):
        run += correct[i]
        auc += run / k
    return auc / len(order) if order else float("nan")

def load_cells():
    cells = defaultdict(list)
    with (HERE / "items.csv").open() as fh:
        for r in csv.DictReader(fh):
            cells[(r["task"], r["model"], r["conf_elicited"])].append(r)
    return cells

def cell_metrics(rows):
    golds = [r["gold"] for r in rows]
    preds = [r["pred"] for r in rows]
    correct_all = [int(r["correct"]) for r in rows]
    elig = [r for r in rows if r["valid"] == "1" and r["conf_status"] == "ok"]
    confs = [float(r["conf"]) for r in elig]
    corr = [int(r["correct"]) for r in elig]
    m = {
        "n": len(rows),
        "n_valid": sum(r["valid"] == "1" for r in rows),
        "n_eligible": len(elig),
        "acc": sum(correct_all) / len(rows),
        "macro_f1": macro_f1(golds, preds),
        "cost": sum(float(r["cost"] or 0) for r in rows),
        "ece": ece(confs, corr) if elig else float("nan"),
        "brier": (statistics.mean(float(r["brier"]) for r in elig)
                  if elig else float("nan")),
        "auc": curve_auc(confs, corr),
    }
    ptop_rows = [r for r in elig if r["p_top"]]
    m["ece_ptop"] = (ece([float(r["p_top"]) for r in ptop_rows],
                         [int(r["correct"]) for r in ptop_rows])
                     if ptop_rows else float("nan"))
    for t in THRESHOLDS:
        m[f"cov{t}"], m[f"acc{t}"] = cov_acc(confs, corr, t)
    return m

def main():
    cells = load_cells()
    kind_of = {}
    out_rows = []
    for (task, model, ce), rows in sorted(cells.items()):
        if ce != "1":
            continue
        kind_of[model] = rows[0]["kind"]
        m = cell_metrics(rows)
        m.update(task=task, model=model, kind=rows[0]["kind"],
                 family=FAMILY[task], split="discovery" if task in DISCOVERY else "confirmatory")
        out_rows.append(m)
    fields = ["task", "family", "split", "model", "kind", "n", "n_valid",
              "n_eligible", "acc", "macro_f1", "ece", "ece_ptop", "brier",
              "auc", "cost"] + [f"{p}{t}" for t in THRESHOLDS for p in ("cov", "acc")]
    with (HERE / "cell_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(out_rows)

    rob = []
    for (task, model, ce), rows in sorted(cells.items()):
        if ce != "0":
            continue
        acc_no = sum(int(r["correct"]) for r in rows) / len(rows)
        conf_rows = cells.get((task, model, "1"))
        acc_yes = (sum(int(r["correct"]) for r in conf_rows) / len(conf_rows)
                   if conf_rows else float("nan"))
        rob.append(dict(task=task, model=model, n_noconf=len(rows),
                        acc_noconf=acc_no, acc_conf=acc_yes,
                        delta=acc_yes - acc_no))
    with (HERE / "robustness_conf_line.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["task", "model", "n_noconf", "acc_noconf",
                                "acc_conf", "delta"])
        w.writeheader()
        w.writerows(rob)

    by = {(r["task"], r["model"]): r for r in out_rows}
    lines = []

    def med(model, tasks, key="ece"):
        vals = [by[(t, model)][key] for t in tasks if (t, model) in by]
        vals = [v for v in vals if v == v]
        return statistics.median(vals), len(vals)

    llms = sorted(m for m, k in kind_of.items() if k == "llm")
    jev_med, njev = med("jev", CONFIRMATORY)
    lines.append(f"H2 (calibration): Jev median ECE over {njev} confirmatory tasks = {jev_med:.4f}")
    worst = []
    for m in llms:
        mm, nm = med(m, CONFIRMATORY)
        worst.append((mm, m, nm))
        lines.append(f"  {m:42s} median ECE = {mm:.4f} (n={nm})")
    min_llm = min(worst)
    lines.append(f"  -> min LLM median ECE = {min_llm[0]:.4f} ({min_llm[1]})")
    lines.append(f"  H2 {'SUPPORTED' if jev_med < min_llm[0] else 'UNSUPPORTED'}"
                 f" (Jev {jev_med:.4f} vs every-LLM min {min_llm[0]:.4f})\n")

    cov_med, _ = med("jev", CONFIRMATORY, "cov0.9")
    acc_med, _ = med("jev", CONFIRMATORY, "acc0.9")
    h3 = acc_med >= 0.85 and cov_med >= 0.15
    lines.append(f"H3 (routing): Jev at t=0.9, median coverage = {cov_med:.3f}, "
                 f"median accuracy = {acc_med:.3f} -> {'SUPPORTED' if h3 else 'UNSUPPORTED'}\n")

    other_utt = [t for t in CONFIRMATORY
                 if FAMILY[t] == "utterance" and t != "indian_english_dialect"]
    dial = by[("indian_english_dialect", "jev")]["ece"]
    med_utt, nu = med("jev", other_utt)
    h4 = dial > med_utt
    lines.append(f"H4 (uneven calibration): Jev dialect ECE = {dial:.4f} vs median over "
                 f"{nu} other utterance tasks = {med_utt:.4f} -> "
                 f"{'SUPPORTED' if h4 else 'UNSUPPORTED'}\n")

    lines.append("§5.10 decision-class internal (medians over confirmatory tasks present):")
    for m in ["jev", "local_rlcd-0.6b", "local_qwen3-base"]:
        f1m, nf = med(m, CONFIRMATORY, "macro_f1")
        em, _ = med(m, CONFIRMATORY, "ece")
        am, _ = med(m, CONFIRMATORY, "acc")
        lines.append(f"  {m:22s} n_tasks={nf:2d} median acc={am:.3f} "
                     f"median macro-F1={f1m:.3f} median ECE={em:.4f}")

    text = "\n".join(lines)
    (HERE / "verdicts.txt").write_text(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
