import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task, parse_prompt, task_criteria

from laya import Router

OUT = pathlib.Path(__file__).parent / "results" / "pilot"
ALL_TASKS = [t for t in list(OPTION_TO_GOLD) + list(BINARY_CRITERIA) if t != "tropes"]

def run_task(router, task, limit=None):
    j = load_task(task)
    outfile = OUT / f"{task}__local_laya.jsonl"
    done = set()
    if outfile.exists():
        done = {json.loads(l)["id"] for l in outfile.open() if l.strip()}
    todo = [i for i in j["labels"] if i not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{task}: {len(todo)} to do ({len(done)} done)")
    with outfile.open("a") as f:
        for k, i in enumerate(todo, 1):
            instr, opts = parse_prompt(j["prompts"][i])
            criteria = task_criteria(task, opts)
            res = router.predict(j["context"][i], {"label": {
                "type": "choice", "instructions": instr, "criteria": criteria}})
            ans = res["answers"]["label"]
            f.write(json.dumps({
                "id": i,
                "choice": ans["choice"],
                "probabilities": ans["probabilities"],
                "confidence": ans["confidence"],
                "model_resolved": f"laya-0.3.4/{res['routing']['repo']}",
                "usage": res.get("usage"),
                "gold": j["labels"][i],
            }) + "\n")
            if k % 50 == 0:
                print(f"  {task}: {k}/{len(todo)}")
    print(f"{task}: done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("tasks", nargs="*", default=None)
    a = ap.parse_args()
    router = Router(preload=True)
    for t in (a.tasks or ALL_TASKS):
        run_task(router, t, a.limit)
