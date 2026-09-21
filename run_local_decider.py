import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task, parse_prompt, task_criteria

from decider.infer import Decider

OUT = pathlib.Path(__file__).parent / "results" / "pilot"
ALL_TASKS = [t for t in list(OPTION_TO_GOLD) + list(BINARY_CRITERIA) if t != "tropes"]

def run_task(d, model, slug, task, limit=None):
    j = load_task(task)
    outfile = OUT / f"{task}__local_{slug}.jsonl"
    done = set()
    if outfile.exists():
        done = {json.loads(l)["id"] for l in outfile.open() if l.strip()}
    todo = [i for i in j["labels"] if i not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{task}: {len(todo)} to do ({len(done)} done)", flush=True)
    with outfile.open("a") as f:
        for k, i in enumerate(todo, 1):
            instr, opts = parse_prompt(j["prompts"][i])
            criteria = task_criteria(task, opts)
            res = d.system_one(j["context"][i], {"label": {
                "type": "choice", "instructions": instr, "criteria": criteria}})
            ans = res["answers"]["label"]
            f.write(json.dumps({
                "id": i,
                "choice": ans["choice"],
                "probabilities": ans["probabilities"],
                "confidence": ans["confidence"],
                "model_resolved": model,
                "usage": None,
                "gold": j["labels"][i],
            }) + "\n")
            if k % 50 == 0:
                print(f"  {task}: {k}/{len(todo)}", flush=True)
    print(f"{task}: done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int)
    ap.add_argument("tasks", nargs="*", default=None)
    a = ap.parse_args()
    d = Decider(a.model, device=a.device)
    for t in (a.tasks or ALL_TASKS):
        run_task(d, a.model, a.slug, t, a.limit)
