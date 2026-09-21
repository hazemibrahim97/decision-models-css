import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nimble-model"))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task, parse_prompt, task_criteria
from inference import NimbleModel

OUT = HERE / "results"
ALL_TASKS = [t for t in list(OPTION_TO_GOLD) + list(BINARY_CRITERIA) if t != "tropes"]

def score_item(m, context, instr, criteria):
    schema = {"label": {
        "type": "enum",
        "choices": list(criteria),
        "description": instr,
        "choice_descriptions": {k: v for k, v in criteria.items() if v},
    }}
    for budget in (None, 1536, 1024, 512, 256):
        text = context.strip()
        if budget is not None:
            ids = m.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > budget:
                text = m.tokenizer.decode(ids[-budget:]).strip()
        try:
            return m.score(text or ".", schema)
        except ValueError as e:
            err = e
    raise err

def run_task(m, task, limit=None):
    j = load_task(task)
    outfile = OUT / f"{task}__local_nimble-9b.jsonl"
    outfile.parent.mkdir(parents=True, exist_ok=True)
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
            try:
                res = score_item(m, j["context"][i], instr, criteria)
            except Exception as e:
                f.write(json.dumps({"id": i, "error": str(e)[:300], "gold": j["labels"][i]}) + "\n")
                continue
            fld = res["fields"]["label"]
            probs = fld["probabilities"]
            f.write(json.dumps({
                "id": i,
                "choice": fld["prediction"],
                "probabilities": probs,
                "confidence": max(probs.values()),
                "model_resolved": "bespokelabs/Bespoke-Nimble-9B",
                "usage": None,
                "gold": j["labels"][i],
            }) + "\n")
            if k % 50 == 0:
                print(f"  {task}: {k}/{len(todo)}", flush=True)
    print(f"{task}: done", flush=True)

if __name__ == "__main__":
    m = NimbleModel(str(HERE / "nimble-model"))
    for t in (sys.argv[1:] or ALL_TASKS):
        run_task(m, t)
