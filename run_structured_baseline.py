import concurrent.futures as cf
import json
import os
import pathlib
import sys
import time
import urllib.request

from pilot_jev import OPTION_TO_GOLD, BINARY_CRITERIA, load_task

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = pathlib.Path("~/.config/openrouter/api_key").expanduser().read_text().strip()
OUT = pathlib.Path(__file__).parent / "results" / "structured"
MODEL = "google/gemini-3.8-flash"
TASKS = list(OPTION_TO_GOLD) + list(BINARY_CRITERIA)

SUFFIX = ('\n\nRespond in JSON with fields "answer" (your chosen option) and '
          '"confidence" (a number from 0 to 100 giving the probability that '
          "your answer is correct).")

def options(task):
    return list(OPTION_TO_GOLD.get(task, BINARY_CRITERIA.get(task)))

def call_llm(task, item_id, context, prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": context + prompt + SUFFIX}],
        "max_tokens": 2000,
        "temperature": 0,
        "reasoning": {"effort": "low"},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "annotation", "strict": True, "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": options(task)},
                               "confidence": {"type": "number"}},
                "required": ["answer", "confidence"],
                "additionalProperties": False}}},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    ch = resp["choices"][0]
    return {
        "id": item_id,
        "text": ch["message"]["content"],
        "model_resolved": resp.get("model", MODEL),
        "provider": resp.get("provider"),
        "usage": resp.get("usage"),
    }

def run(task):
    j = load_task(task)
    outfile = OUT / f"{task}__{MODEL.replace('/', '_')}__structured.jsonl"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outfile.exists():
        done = {json.loads(l)["id"] for l in outfile.open() if l.strip()}
    todo = [i for i in j["labels"] if i not in done]
    print(f"{task}: {len(todo)} to do")
    n_err = 0
    with outfile.open("a") as f, cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 8))) as ex:
        futs = {ex.submit(call_llm, task, i, j["context"][i], j["prompts"][i]): i
                for i in todo}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            try:
                rec = fut.result()
                rec["gold"] = j["labels"][rec["id"]]
                f.write(json.dumps(rec) + "\n")
            except Exception as e:
                n_err += 1
                print(f"  ERR {futs[fut]}: {e}", file=sys.stderr)
            if k % 200 == 0:
                print(f"  {task}: {k}/{len(todo)}")
    print(f"{task}: done, {n_err} errors")

if __name__ == "__main__":
    for t in (sys.argv[1:] or TASKS):
        run(t)
