import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task, parse_prompt
from parse_llm_answer import parse_answer

PILOT = ROOT / "results" / "pilot"
OUT = ROOT / "analysis" / "items.csv"

BINARY_GOLD = {
    "conv_go_awry": {"True": True, "False": False},
    "reddit_humor": {"True": True, "False": False},
    "wiki_corpus": {"True": True, "False": False},
    "persuasion": {"True": 1.0, "False": 0.0},
}

CONF_RE = re.compile(
    r"confidence\s*[:=]?\s*\*{0,2}\s*(-)?\s*(\d+(?:\.\d+)?)\s*%?", re.I)

def parse_confidence(text):

    matches = CONF_RE.findall(text)
    if not matches:
        return None, "absent"
    neg, val = matches[-1]
    if neg:
        return None, "negative"
    v = float(val)
    if v > 100:
        return None, "oob"
    return v / 100.0, "ok"

def task_maps(task):

    j = load_task(task)
    if task in BINARY_GOLD:
        l2g = BINARY_GOLD[task]
        return {pid: (l2g, {"True": "True", "False": "False"}) for pid in j["prompts"]}
    l2g = OPTION_TO_GOLD[task]
    maps = {}
    for pid, prompt in j["prompts"].items():
        _, opts = parse_prompt(prompt)
        l2d = {L: re.split(r" \(", t)[0].strip() for L, t in opts.items()}
        maps[pid] = (l2g, l2d)
    return maps

def rows_llm(task, model, path, conf_elicited):
    maps = task_maps(task)
    for line in path.open():
        r = json.loads(line)
        text = r.get("text") or ""
        gold = r["gold"]
        if not text.strip():
            yield dict(task=task, model=model, kind="llm", conf_elicited=conf_elicited,
                       id=r["id"], gold=gold, pred="", valid=0, correct=0,
                       conf="", conf_status="empty_text", p_top="",
                       cost=(r.get("usage") or {}).get("cost", 0))
            continue
        l2g, l2d = maps[r["id"]]
        pred = parse_answer(text, l2g, l2d)
        valid = int(pred is not None)
        correct = int(valid and str(pred) == str(gold))
        conf, status = parse_confidence(text) if conf_elicited else (None, "absent")
        brier = (conf - correct) ** 2 if (valid and conf is not None) else ""
        yield dict(task=task, model=model, kind="llm", conf_elicited=conf_elicited,
                   id=r["id"], gold=gold, pred="" if pred is None else pred,
                   valid=valid, correct=correct,
                   conf="" if conf is None else conf, conf_status=status,
                   p_top="", brier=brier, cost=(r.get("usage") or {}).get("cost", 0))

def rows_decision(task, model, kind, path):
    for line in path.open():
        r = json.loads(line)
        probs = r["probabilities"]
        p_top = probs[str(r["choice"])]
        gold_key = str(r["gold"])
        assert gold_key in probs, (task, model, r["id"], gold_key, list(probs)[:5])
        brier = sum((p - (k == gold_key)) ** 2 for k, p in probs.items())
        yield dict(task=task, model=model, kind=kind, conf_elicited=1,
                   id=r["id"], gold=r["gold"], pred=r["choice"], valid=1,
                   correct=int(str(r["choice"]) == str(r["gold"])),
                   conf=r["confidence"], conf_status="ok", p_top=p_top,
                   brier=brier, cost=(r.get("usage") or {}).get("cost", 0))

def main():
    fields = ["task", "model", "kind", "conf_elicited", "id", "gold", "pred",
              "valid", "correct", "conf", "conf_status", "p_top", "brier", "cost"]
    n = 0
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for f in sorted(PILOT.glob("*.jsonl")):
            parts = f.stem.split("__")
            task = parts[0]
            if len(parts) == 1:
                rows = rows_decision(task, "jev", "jev", f)
            elif parts[1].startswith("local_"):
                rows = rows_decision(task, parts[1], "local", f)
            else:
                rows = rows_llm(task, parts[1], f, int(len(parts) == 3))
            for row in rows:
                w.writerow(row)
                n += 1
    print(f"wrote {n} rows -> {OUT}")

if __name__ == "__main__":
    main()

    import collections
    counts = collections.Counter()
    with OUT.open() as fh:
        for row in csv.DictReader(fh):
            counts[(row["model"], row["conf_elicited"])] += 1
    assert counts[("jev", "1")] == 7977, counts[("jev", "1")]
    assert counts[("local_rlcd-0.6b", "1")] == 7863
    assert counts[("local_qwen3-base", "1")] == 7863

    for m, expect in [("local_laya", 7863), ("local_decider-0.8b", 7863),
                      ("local_decider-2b", 7863), ("local_semif-4b", 7597),
                      ("local_nimble-9b", 7863),

                      ("local_opendecision", 7863), ("local_verdict", 7863),
                      ("local_von", 7863), ("local_kev-0.8b", 7863),
                      ("local_kev-4b", 7863), ("local_kev-9b", 7863)]:
        got = counts[(m, "1")]
        assert got == expect, (m, got, expect)
    llm_conf = {m: c for (m, e), c in counts.items()
                if e == "1" and not m.startswith(("jev", "local_"))}
    assert len(llm_conf) == 19 and all(c in (7976, 7977) for c in llm_conf.values()), llm_conf
    assert sum(c == 7976 for c in llm_conf.values()) == 3
    print("self-check passed:", dict(sorted(llm_conf.items())))
