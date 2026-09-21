import concurrent.futures as cf
import json
import os
import pathlib
import sys
import time
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = pathlib.Path("~/.config/openrouter/api_key").expanduser().read_text().strip()
DATA = pathlib.Path(__file__).parent / "LLMs_for_CSS" / "css_data"
OUT = pathlib.Path(__file__).parent / "results" / "pilot"

MODELS = [

    "openai/gpt-5.6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol",

    "anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5", "anthropic/claude-fable-5.1",

    "google/gemini-3.5-flash-lite", "google/gemini-3.8-flash",
    "google/gemini-3.1-pro-preview",

    "meta-llama/llama-4-scout", "meta-llama/llama-4-maverick",

    "qwen/qwen3-235b-a22b-2507", "deepseek/deepseek-v3.2",
    "openai/gpt-oss-120b", "z-ai/glm-5.3", "moonshotai/kimi-k2.6",
    "mistralai/mistral-medium-3-5", "google/gemma-4-31b-it",
]
TASKS = ["semeval_stance", "implicit_hate", "discourse"]
if os.environ.get("TASKS") == "all":
    from pilot_jev import OPTION_TO_GOLD as _o, BINARY_CRITERIA as _b
    TASKS = list(_o) + list(_b)

CONF_SUFFIX = ("\n\nAfter your answer, on a new line, write 'Confidence: X' where X is a "
               "number from 0 to 100 giving the probability that your answer is correct.")
WITH_CONF = os.environ.get("CONFIDENCE") == "1"

def call_llm(model, item_id, context, prompt):

    body = {
        "model": model,
        "messages": [{"role": "user", "content": context + prompt + (CONF_SUFFIX if WITH_CONF else "")}],
        "max_tokens": 2000,
    }
    if model.startswith("openai/"):
        body["reasoning"] = {"effort": "low"}
    else:
        body["temperature"] = 0
        if model.startswith("google/gemini") or model.startswith("z-ai/"):
            body["reasoning"] = {"effort": "low"}
        elif model.startswith("moonshotai/"):
            body["reasoning"] = {"enabled": False}
    body = json.dumps(body).encode()
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
        "model_resolved": resp.get("model", model),
        "provider": resp.get("provider"),
        "usage": resp.get("usage"),
    }

def run(task, model):
    from pilot_jev import load_task
    j = load_task(task)
    slug = model.replace("/", "_") + ("__conf" if WITH_CONF else "")
    outfile = OUT / f"{task}__{slug}.jsonl"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outfile.exists():
        done = {json.loads(l)["id"] for l in outfile.open() if l.strip()}
    todo = [i for i in j["labels"] if i not in done]
    print(f"{task} / {model}: {len(todo)} to do")
    n_err = 0
    with outfile.open("a") as f, cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 8))) as ex:
        futs = {ex.submit(call_llm, model, i, j["context"][i], j["prompts"][i]): i
                for i in todo}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            try:
                rec = fut.result()
                rec["gold"] = j["labels"][rec["id"]]
                f.write(json.dumps(rec) + "\n")
            except Exception as e:
                n_err += 1
                print(f"  ERR {futs[fut]}: {e}", file=sys.stderr)
            if k % 100 == 0:
                print(f"  {task}/{model}: {k}/{len(todo)}")
    print(f"{task} / {model}: done, {n_err} errors")

if __name__ == "__main__":
    models = sys.argv[1:] or MODELS
    for m in models:
        for t in TASKS:
            run(t, m)
