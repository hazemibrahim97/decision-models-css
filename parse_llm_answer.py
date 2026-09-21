import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pilot_jev import OPTION_TO_GOLD, load_task, parse_prompt

OUT = pathlib.Path(__file__).parent / "results" / "pilot"

def parse_answer(text, letter_to_gold, letter_to_display):
    first = next((l.strip() for l in text.strip().splitlines() if l.strip()), "")
    m = re.match(r"^([A-Z]+)(?:$|[:.)\s])", first)
    if m and m.group(1) in letter_to_gold:
        return letter_to_gold[m.group(1)]
    hits = {L for L, disp in letter_to_display.items()
            if disp and re.search(re.escape(disp), text, re.I)}
    if len(hits) == 1:
        return letter_to_gold[hits.pop()]
    return None

if __name__ == "__main__":
    for f in sorted(OUT.glob("*__*.jsonl")):
        task = f.name.split("__")[0]
        j = load_task(task)
        recs = [json.loads(l) for l in f.open()]
        n_ok = n_correct = 0
        for r in recs:
            _, opts = parse_prompt(j["prompts"][r["id"]])
            l2g = OPTION_TO_GOLD[task]

            l2d = {L: re.split(r" \(", t)[0].strip() for L, t in opts.items()}
            pred = parse_answer(r["text"], l2g, l2d)
            if pred is not None:
                n_ok += 1
                n_correct += str(pred) == str(r["gold"])
        print(f"{f.name:55s} parsed {n_ok}/{len(recs)} ({n_ok/len(recs):.1%})  acc(parsed)={n_correct/max(n_ok,1):.3f}")
