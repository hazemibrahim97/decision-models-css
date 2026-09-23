import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pilot_jev import BINARY_CRITERIA, OPTION_TO_GOLD, load_task, parse_prompt, task_criteria

OUT = pathlib.Path(__file__).parent / "results" / "pilot"
ALL_TASKS = [t for t in list(OPTION_TO_GOLD) + list(BINARY_CRITERIA) if t != "tropes"]

def backend(name, url):

    if name == "opendecision":
        from importlib.metadata import version
        from opendecision import OpenDecisionEngine
        e = OpenDecisionEngine()

        def ask(state, instr, crit):

            r = e.choice(state=state, instructions=instr.replace("{", "{{").replace("}", "}}"), criteria=crit)
            return r["choice"], r["probabilities"], r["confidence"], None
        return ask, f"opendecision-{version('OpenDecision')}/MoritzLaurer/ModernBERT-large-zeroshot-v2.0"

    if name == "von":
        import von

        def ask(state, instr, crit):
            r = von.system_one(state=state, questions={"label": {
                "type": "choice", "instructions": instr, "criteria": crit}})
            a = (r["answers"] if isinstance(r, dict) else r.answers)["label"]
            a = a if isinstance(a, dict) else a.model_dump()
            return a["choice"], a["probabilities"], a["confidence"], None
        return ask, f"wfzyx/von weights-1.1.0 snapshot-d8bb5e0 sdk-{von.__version__}"

    if name == "verdict":
        from rlcd import Choice, DecisionEngine, Option
        art = pathlib.Path(__file__).parent / "Verdict-open-jev" / "artifacts" / "v2"
        e = DecisionEngine(str(art))
        assert e.ort_session is not None and e.calibrator is not None

        def ask(state, instr, crit):
            q = Choice(id="label", question=instr, options=[Option(id=k, description=v) for k, v in crit.items()])
            r = e.evaluate(context=state, queries=[q]).results[0]

            p = {k: r.probabilities[k] for k in crit}
            z = sum(p.values())
            p = {k: v / z for k, v in p.items()}
            top = max(p, key=p.get)
            return top, p, p[top], {"engine_choice": r.selected_id, "is_abstention": r.is_abstention,
                                    "raw_probabilities": dict(r.probabilities)}
        return ask, "heman10x/rlcd-modernbert-151m@v1.4 (artifacts/v2 onnx)"

    if name.startswith("kev"):
        import urllib.request

        def ask(state, instr, crit):
            body = json.dumps({"model": "kev-latest", "state": state, "questions": {"label": {
                "type": "choice", "instructions": instr, "criteria": crit}}}).encode()
            req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            a = r["answers"]["label"]
            return a["choice"], a["probabilities"], a["confidence"], {"model": r.get("model")}
        return ask, f"jaredpalmer/{name}"

    raise SystemExit(f"unknown backend {name}")

def run_task(ask, resolved, slug, task, limit=None):
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
            choice, probs, conf, extra = ask(j["context"][i], instr, task_criteria(task, opts))
            f.write(json.dumps({
                "id": i, "choice": choice, "probabilities": probs, "confidence": conf,
                "model_resolved": resolved, "usage": None, "extra": extra,
                "gold": j["labels"][i],
            }) + "\n")
            if k % 50 == 0:
                print(f"  {task}: {k}/{len(todo)}", flush=True)
    print(f"{task}: done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("--url", default="http://127.0.0.1:8009")
    ap.add_argument("--limit", type=int)
    ap.add_argument("tasks", nargs="*", default=None)
    a = ap.parse_args()
    ask, resolved = backend(a.backend, a.url)
    for t in (a.tasks or ALL_TASKS):
        run_task(ask, resolved, a.backend, t, a.limit)
