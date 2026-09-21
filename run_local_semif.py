import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task, parse_prompt, task_criteria

from semif_phase1 import mlx_backend

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_STATE_TOKENS = 1536
OUT = pathlib.Path(__file__).parent / "results" / "pilot"
ALL_TASKS = [t for t in list(OPTION_TO_GOLD) + list(BINARY_CRITERIA)
             if t not in {"tropes", "indian_english_dialect"}]

def truncate_state(tok, state, budget=MAX_STATE_TOKENS):
    ids = tok.encode(state.strip(), add_special_tokens=False)
    if len(ids) <= budget:
        return state.strip()
    return tok.decode(ids[-budget:]).strip()

def run_task(model, tok, metadata, task, limit=None):
    j = load_task(task)
    outfile = OUT / f"{task}__local_semif-4b.jsonl"
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
            row = {"id": i, "state": truncate_state(tok, j["context"][i]), "question": instr,
                   "options": [{"id": str(L), "description": d or str(L)} for L, d in criteria.items()]}
            res = mlx_backend.score(model, tok, row, metadata, 4096)
            probs = dict(zip(res["option_ids"], res["probabilities"]))
            best = max(probs, key=probs.get)
            f.write(json.dumps({
                "id": i,
                "choice": best,
                "probabilities": probs,
                "confidence": probs[best],
                "model_resolved": f"semif-direct-mlx/{MODEL}@{REVISION[:8]}",
                "usage": {"input_tokens": res["input_tokens"]},
                "gold": j["labels"][i],
            }) + "\n")
            if k % 50 == 0:
                print(f"  {task}: {k}/{len(todo)}", flush=True)
    print(f"{task}: done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("tasks", nargs="*", default=None)
    a = ap.parse_args()
    model, tok, metadata = mlx_backend.load_model(MODEL, REVISION)
    for t in (a.tasks or ALL_TASKS):
        run_task(model, tok, metadata, t, a.limit)
