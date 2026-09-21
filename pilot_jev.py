import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time
import urllib.request

MODEL = "typesafe/jev-1.13"
URL = "https://openrouter.ai/api/alpha/decisions"
KEY = pathlib.Path("~/.config/openrouter/api_key").expanduser().read_text().strip()
DATA = pathlib.Path(__file__).parent / "LLMs_for_CSS" / "css_data"
OUT = pathlib.Path(__file__).parent / "results" / "pilot"

TROPES_LABELS = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA",
    "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL",
    "AM", "AN", "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", "AW",
    "AX", "AY", "AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH",
    "BI", "BJ", "BK", "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS",
    "BT", "F, D", "K, L", "K, T", "Q, T", "S, R", "V, L", "V, N",
    "AK, K", "AN, E", "AO, E", "AV, N", "C, AR", "D, AR", "F, AB",
    "F, BR", "H, BI", "J, AR", "K, BQ", "R, AX", "V, AQ", "V, BS",
    "Y, BR", "AA, BC", "AE, AR", "AG, AR", "AH, AY", "AK, AV", "AU, BI",
    "AW, BS", "BB, AB", "BF, AG", "BI, AB", "BJ, BI", "BJ, BM",
    "AJ, L, K", "AV, N, BQ", "AW, AO, T", "AW, BF, BP", "BI, BR, AB",
    "AJ, O, C, K", "V, N, K, AO, I", "AV, K, N, S, BQ",
)

OPTION_TO_GOLD = {
    "semeval_stance": {"A": "Against", "B": "Favor", "C": "None"},
    "implicit_hate": {
        "A": "white_grievance", "B": "incitement", "C": "inferiority",
        "D": "irony", "E": "stereotypical", "F": "threatening",
    },
    "discourse": {
        "A": "question", "B": "answer", "C": "agreement", "D": "disagreement",
        "E": "appreciation", "F": "elaboration", "G": "humor",
    },
    "emotion": {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F"},
    "ibc": {"A": "Liberal", "B": "Conservative", "C": "Neutral"},
    "media_ideology": {"A": "left", "B": "right", "C": "center"},
    "indian_english_dialect": {
        "A": "Article Omission",
        "B": "Copula Omission",
        "C": "Direct Object Pronoun Drop",
        "D": "Extraneous Article",
        "E": "Focus Itself",
        "F": "Focus Only",
        "G": 'General Extender "and all"',
        "H": "Habitual Progressive",
        "I": 'Invariant Tag "isn’t it, no, na"',
        "J": "Inversion In Embedded Clause",
        "K": "Lack Of Agreement",
        "L": "Lack Of Inversion In Wh-questions",
        "M": "Left Dislocation",
        "N": "Mass Nouns As Count Nouns",
        "O": 'Non-initial Existential "is / are there"',
        "P": "Object Fronting",
        "Q": "Prepositional Phrase Fronting With Reduction",
        "R": "Preposition Omission",
        "S": "Resumptive Object Pronoun",
        "T": "Resumptive Subject Pronoun",
        "U": "Stative Progressive",
        "V": "Topicalized Non-argument Constituent",
        "W": "None of the Above",
    },
    "raop": {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G"},
    "talklife": {"A": "A", "B": "B", "C": "C"},
    "wiki_politeness": {"A": 1, "B": 0, "C": -1},
    "tempowic": {"A": "Same", "B": "Different"},
    "tropes": {label: label for label in TROPES_LABELS},
    "flute": {"A": "Idiom", "B": "Metaphor", "C": "Sarcasm", "D": "Simile"},
    "mrf": {"A": "Misinformation", "B": "Trustworthy"},
}

BINARY_CRITERIA = {
    "conv_go_awry": {
        "True": "True: the previous conversation eventually derails into a personal attack.",
        "False": "False: the previous conversation does not eventually derail into a personal attack.",
    },
    "persuasion": {
        "1.0": "True: this reply would convince the original poster.",
        "0.0": "False: this reply would not convince the original poster.",
    },
    "reddit_humor": {
        "True": "True: the joke is humorous to most people.",
        "False": "False: the joke is not humorous to most people.",
    },
    "wiki_corpus": {
        "True": "True: the named user is in a position of power in the conversation.",
        "False": "False: the named user is not in a position of power in the conversation.",
    },
}

SKIPPED_TASKS = {
    "hippocorpus": "event extraction with variable-length per-item sentence sets, not a closed single-choice classification task",
    "wikievents": "event-argument extraction with per-item free-text JSON labels, not a closed single-choice classification task",
}

ALL_TASKS = list(OPTION_TO_GOLD) + list(BINARY_CRITERIA) + list(SKIPPED_TASKS)

def parse_prompt(prompt):
    lines = prompt.strip().split("\n")
    instr, opts, cur = [], {}, None
    for line in lines:
        m = re.match(r"^([A-Z]+): (.*)$", line)
        if m:
            cur = m.group(1)
            opts[cur] = m.group(2)
        elif line.startswith("Constraint:"):
            cur = None
        elif cur:
            opts[cur] += " " + line.strip()
        else:
            instr.append(line)
    return " ".join(instr).strip(), opts

def load_task(task):
    name = "test-classification.json" if task in {"flute", "mrf"} else "test.json"
    return json.loads((DATA / task / name).read_text())

def coerce_choice(task, choice):
    if task in {"conv_go_awry", "reddit_humor", "wiki_corpus"}:
        return choice == "True"
    if task == "persuasion":
        return float(choice)
    if task == "wiki_politeness":
        return int(choice)
    return choice

def task_criteria(task, opts):
    if task in BINARY_CRITERIA:
        return BINARY_CRITERIA[task]
    if task == "tropes":
        criteria = {}
        for label in OPTION_TO_GOLD[task]:
            parts = [p.strip() for p in label.split(",")]
            criteria[label] = "; ".join(f"{p}: {opts[p]}" for p in parts)
        return criteria
    criteria = {}
    for letter, gold in OPTION_TO_GOLD[task].items():
        text = opts.get(letter)
        if text is None and task == "indian_english_dialect" and gold == "None of the Above":
            text = "None of the listed Indian English dialect features apply."
        criteria[str(gold)] = text
    return criteria

def call_jev(task, item_id, context, prompt):
    instr, opts = parse_prompt(prompt)
    criteria = task_criteria(task, opts)
    body = json.dumps({
        "model": MODEL,
        "state": context,
        "questions": {"label": {"type": "choice", "instructions": instr, "criteria": criteria}},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    ans = resp["answers"]["label"]
    return {
        "id": item_id,
        "choice": coerce_choice(task, ans["choice"]),
        "probabilities": ans["probabilities"],
        "confidence": ans.get("confidence"),
        "model_resolved": resp.get("model", MODEL),
        "usage": resp.get("usage"),
    }

def run_task(task, limit=None):
    j = load_task(task)
    outfile = OUT / f"{task}.jsonl"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outfile.exists():
        done = {json.loads(l)["id"] for l in outfile.open() if l.strip()}
    todo = [(i, j["context"][i], j["prompts"][i]) for i in j["labels"] if i not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{task}: {len(todo)} to do ({len(done)} already done)")
    n_err = 0
    with outfile.open("a") as f, cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(call_jev, task, i, c, p): i for i, c, p in todo}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            try:
                rec = fut.result()
                rec["gold"] = j["labels"][rec["id"]]
                f.write(json.dumps(rec) + "\n")
            except Exception as e:
                n_err += 1
                print(f"  ERR {futs[fut]}: {e}", file=sys.stderr)
            if k % 50 == 0:
                print(f"  {task}: {k}/{len(todo)}")
    print(f"{task}: done, {n_err} errors")

def smoke_task(t):
    j = load_task(t)
    p = next(iter(j["prompts"].values()))
    instr, opts = parse_prompt(p)
    labels = set(j["labels"].values())
    assert instr, t
    if t in OPTION_TO_GOLD:
        mapped = set(OPTION_TO_GOLD[t].values())
        assert mapped == labels, (t, mapped ^ labels)
        if t == "tropes":
            for label in labels:
                for part in [p.strip() for p in label.split(",")]:
                    assert part in opts, (t, label, part)
        else:
            assert set(opts).issubset(OPTION_TO_GOLD[t]), (t, opts.keys())
    elif t in BINARY_CRITERIA:
        mapped = {coerce_choice(t, x) for x in BINARY_CRITERIA[t]}
        assert mapped == labels, (t, mapped ^ labels)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="maximum new items to run per task")
    ap.add_argument("tasks", nargs="*")
    args = ap.parse_args()
    tasks = args.tasks or ALL_TASKS

    for t in tasks:
        if t in SKIPPED_TASKS:
            print(f"{t}: SKIP ({SKIPPED_TASKS[t]})")
            continue
        smoke_task(t)
    for t in tasks:
        if t in SKIPPED_TASKS:
            continue
        run_task(t, args.limit)
