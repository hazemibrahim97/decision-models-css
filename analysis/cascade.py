import csv
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

def main():
    cells = defaultdict(dict)
    with (HERE / "items.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["conf_elicited"] == "1" and r["kind"] != "local":
                cells[(r["task"], r["model"])][r["id"]] = r

    metrics = list(csv.DictReader((HERE / "cell_metrics.csv").open()))
    conf = [m for m in metrics if m["split"] == "confirmatory" and m["kind"] == "llm"]
    tasks = sorted({m["task"] for m in metrics})

    def median_f1(model):
        return statistics.median(float(m["macro_f1"]) for m in conf
                                 if m["model"] == model)

    llms = sorted({m["model"] for m in conf})
    grid_cost = {m: sum(float(r["cost"]) for c in cells for r in cells[c].values()
                        if c[1] == m) for m in llms}
    best_frontier = max(llms, key=median_f1)
    cheap = [m for m in llms if grid_cost[m] / 7977 * 1000 < 0.50]
    best_cheap = max(cheap, key=median_f1)

    best_cheap_closed = "anthropic_claude-haiku-4.5"
    print(f"best frontier LLM = {best_frontier} (median F1 {median_f1(best_frontier):.3f})")
    print(f"best sub-$0.5/1k LLM = {best_cheap} (median F1 {median_f1(best_cheap):.3f}, "
          f"${grid_cost[best_cheap]/7977*1000:.3f}/1k)")

    cas_rows, comp_rows = [], []
    for task in tasks:
        jev = cells[(task, "jev")]
        for L in (best_frontier, best_cheap, best_cheap_closed):
            llm = cells[(task, L)]
            ids = sorted(set(jev) & set(llm))
            jev_cost_all = sum(float(jev[i]["cost"]) for i in ids)
            llm_acc = sum(int(llm[i]["correct"]) for i in ids) / len(ids)
            llm_cost = sum(float(llm[i]["cost"]) for i in ids)
            jev_acc = sum(int(jev[i]["correct"]) for i in ids) / len(ids)
            base = dict(task=task, llm=L, n=len(ids), acc_llm_alone=llm_acc,
                        cost_llm_alone=llm_cost, acc_jev_alone=jev_acc,
                        cost_jev_alone=jev_cost_all)
            for t in THRESHOLDS:
                ok = cost = routed = 0
                for i in ids:
                    if float(jev[i]["conf"]) >= t:
                        ok += int(jev[i]["correct"])
                    else:
                        ok += int(llm[i]["correct"])
                        cost += float(llm[i]["cost"])
                        routed += 1
                cas_rows.append(dict(base, t=t, acc_cascade=ok / len(ids),
                                     cost_cascade=jev_cost_all + cost,
                                     frac_routed=routed / len(ids)))

        llm = cells[(task, best_frontier)]
        ids = sorted(set(jev) & set(llm))
        both = sum(int(jev[i]["correct"]) and int(llm[i]["correct"]) for i in ids)
        only_j = sum(int(jev[i]["correct"]) and not int(llm[i]["correct"]) for i in ids)
        only_l = sum(not int(jev[i]["correct"]) and int(llm[i]["correct"]) for i in ids)
        neither = len(ids) - both - only_j - only_l
        agree = sum(str(jev[i]["pred"]) == str(llm[i]["pred"]) for i in ids) / len(ids)
        comp_rows.append(dict(task=task, llm=best_frontier, n=len(ids),
                              agreement=agree, both=both, only_jev=only_j,
                              only_llm=only_l, neither=neither))

    with (HERE / "cascade.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(cas_rows[0]))
        w.writeheader()
        w.writerows(cas_rows)
    with (HERE / "complementarity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(comp_rows[0]))
        w.writeheader()
        w.writerows(comp_rows)

    confirm = {m["task"] for m in conf}
    print("\ncascade medians over 15 confirmatory tasks:")
    for L in (best_frontier, best_cheap, best_cheap_closed):
        sub = [r for r in cas_rows if r["llm"] == L and r["task"] in confirm]
        alone = statistics.median(r["acc_llm_alone"] for r in sub if r["t"] == 0.9)
        calone = statistics.median(r["cost_llm_alone"] for r in sub if r["t"] == 0.9)
        print(f"  L={L}: alone median acc={alone:.3f} median task cost=${calone:.2f}")
        for t in THRESHOLDS:
            st = [r for r in sub if r["t"] == t]
            print(f"    t={t:<4} acc={statistics.median(r['acc_cascade'] for r in st):.3f} "
                  f"cost=${statistics.median(r['cost_cascade'] for r in st):.2f} "
                  f"routed={statistics.median(r['frac_routed'] for r in st):.2f}")

if __name__ == "__main__":
    main()
