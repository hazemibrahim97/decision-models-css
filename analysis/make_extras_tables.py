import csv
import pathlib

from make_appendix_tables import DISCOVERY, f, name, write
from make_tables import TASK_LABEL

HERE = pathlib.Path(__file__).resolve().parent

def read(fn):
    return list(csv.DictReader((HERE / fn).open()))

def b12():
    q1 = {r["task"]: r for r in read("q1_delta.csv")}
    rows = read("q1_selection_aware.csv")
    lines = [r"\begin{tabular}{lrcc}", r"\toprule",
             r"Task & $\Delta$F1 & Fixed-best 95\% CI & Selection-aware 95\% CI \\",
             r"\midrule"]
    for r in sorted(rows, key=lambda r: TASK_LABEL[r["task"]]):
        t = r["task"]
        lab = TASK_LABEL[t] + ("$^{d}$" if t in DISCOVERY else "")
        lines.append(
            f"{lab} & {float(q1[t]['delta_f1'])*100:+.1f} & "
            f"[{float(q1[t]['ci_lo'])*100:+.1f}, {float(q1[t]['ci_hi'])*100:+.1f}] & "
            f"[{float(r['delta_sel_lo'])*100:+.1f}, {float(r['delta_sel_hi'])*100:+.1f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("b12_selection_aware.tex", "\n".join(lines) + "\n")

def b13():
    rows = read("calibration_cis.csv")
    lines = [r"\begin{tabular}{lrccc}", r"\toprule",
             r"Task & $n$ & ECE [95\% CI] & Cov@0.9 [95\% CI] & Acc@0.9 [95\% CI] \\",
             r"\midrule"]
    for r in sorted(rows, key=lambda r: TASK_LABEL[r["task"]]):
        lab = TASK_LABEL[r["task"]] + ("$^{d}$" if r["task"] in DISCOVERY else "")
        acc = (f"{f(r['acc09'])} [{f(r['acc_lo'])}, {f(r['acc_hi'])}]"
               if r["acc09"] not in ("", "nan") else "--")
        lines.append(
            f"{lab} & {r['n']} & {f(r['ece'])} [{f(r['ece_lo'])}, {f(r['ece_hi'])}] & "
            f"{f(r['cov09'])} [{f(r['cov_lo'])}, {f(r['cov_hi'])}] & {acc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("b13_calibration_cis.tex", "\n".join(lines) + "\n")

def b14():
    rows = read("annotator_disagreement.csv")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model & $n$ & Spearman $\rho$ & $p$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{name(r['model'])} & {r['n']} & "
                     f"{float(r['rho']):+.3f} & {f(r['p'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("b14_annotator_disagreement.tex", "\n".join(lines) + "\n")

def b15():
    rows = read("structured_baseline.csv")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Task & $n$ & \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{Accuracy} & Invalid \\",
             r" & & free text & structured & free text & structured & free/struct.\ \\",
             r"\midrule"]
    for r in sorted(rows, key=lambda r: TASK_LABEL[r["task"]]):
        lab = TASK_LABEL[r["task"]] + ("$^{d}$" if r["task"] in DISCOVERY else "")
        lines.append(
            f"{lab} & {r['n']} & {float(r['f1_free'])*100:.1f} & "
            f"{float(r['f1_structured'])*100:.1f} & {float(r['acc_free'])*100:.1f} & "
            f"{float(r['acc_structured'])*100:.1f} & "
            f"{r['invalid_free']}/{r['invalid_structured']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("b15_structured.tex", "\n".join(lines) + "\n")

if __name__ == "__main__":
    b12()
    b13()
    b14()
    b15()
    print("wrote b12, b13, b14, b15")
