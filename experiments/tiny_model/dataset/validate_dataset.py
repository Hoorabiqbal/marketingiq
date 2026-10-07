"""
Check dataset files before any training run.

    python validate_dataset.py seed_examples.jsonl
    python validate_dataset.py train.jsonl val.jsonl test.jsonl
    python validate_dataset.py --execute eval_frozen.jsonl

1. Every gold plan (and every `previous` plan) goes through production's validate_plan(). A
   not_answerable plan must be rejected as "not answerable (<reason>)"; everything else must pass.
   A label the real validator rejects would teach the model to produce plans that can't run.
2. Duplicates within a file: the same normalized question with the same `previous` is an error;
   near-duplicate questions (word-set Jaccard >= 0.8) are listed for review.
3. Leakage, when several files are given: no normalized question in two files, no template_id
   shared between a test/eval file and any other file, and near-duplicate questions across files
   are listed for review.
4. With --execute: every answerable gold plan runs through production's execute_plan() on the
   local DuckDB data (no LLM) and must return status "ok", so no label validates but can't run.

Exit code 1 on any label error, execution failure or exact leak.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import plan_format as pf

REQUIRED = {"id", "question", "previous", "plan"}
NEAR_DUP = 0.8


def load(path: Path) -> list:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            row["_where"] = f"{path.name}:{n}"
            rows.append(row)
    return rows


def check_plan(plan: dict) -> str:
    """None when the gold plan is right for the validator, else the problem."""
    qp = pf.backend()
    if set(plan) - set(pf.field_order()):
        return f"unknown keys {sorted(set(plan) - set(pf.field_order()))}"
    if pf.to_sparse(plan) != plan:
        return "not in sparse form (default values present or keys out of order)"
    try:
        qp.validate_plan(pf.to_canonical(plan))
    except qp.PlanError as e:
        if plan["intent"] == "not_answerable" and str(e) == f"not answerable ({plan.get('reason')})":
            return None
        return f"rejected by validate_plan: {e}"
    return "not_answerable plan was accepted" if plan["intent"] == "not_answerable" else None


def normalize(q: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


def is_test(name: str) -> bool:
    return "test" in name or "eval" in name


def near_duplicates(rows_a: list, rows_b: list, same_file: bool) -> list:
    words = [(set(normalize(r["question"]).split()), r) for r in rows_b]
    found = []
    for i, ra in enumerate(rows_a):
        wa = set(normalize(ra["question"]).split())
        for j, (wb, rb) in enumerate(words):
            if same_file and j <= i:
                continue
            if wa and wb and wa != wb and len(wa & wb) / len(wa | wb) >= NEAR_DUP:
                found.append((ra, rb))
    return found


def check_execution(row) -> str:
    qp = pf.backend()
    if row["plan"]["intent"] == "not_answerable":
        return None
    plan = pf.postprocess(qp.validate_plan(pf.to_canonical(row["plan"])), row["question"])
    try:
        out = qp.execute_plan(plan, {})
    except Exception as e:  # noqa: BLE001 - a crash is a finding about the label (or the backend), not a stop
        return f"execute_plan raised {type(e).__name__}: {e}"
    return None if out.status == "ok" else f"execute_plan status {out.status}: {out.lines[:1]}"


def check_leakage(files: dict) -> int:
    errors = 0
    names = list(files)
    for a, b in combinations(names, 2):
        qa = {normalize(r["question"]): r for r in files[a]}
        qb = {normalize(r["question"]): r for r in files[b]}
        for q in qa.keys() & qb.keys():
            print(f"LEAK  same question in {qa[q]['_where']} and {qb[q]['_where']}: {q!r}")
            errors += 1
        if is_test(a) or is_test(b):
            ta = {r.get("template_id") for r in files[a]} - {None}
            tb = {r.get("template_id") for r in files[b]} - {None}
            for t in sorted(ta & tb):
                print(f"LEAK  template {t} in both {a} and {b}")
                errors += 1
        for ra, rb in near_duplicates(files[a], files[b], False):
            print(f"REVIEW near-duplicate {ra['_where']} ~ {rb['_where']}: {ra['question']!r} / {rb['question']!r}")
    return errors


def main(paths: list) -> int:
    execute = "--execute" in paths
    files = {Path(p).stem: load(Path(p)) for p in paths if p != "--execute"}
    errors = 0
    for rows in files.values():
        seen, questions = set(), {}
        for r in rows:
            key = (normalize(r.get("question", "")), json.dumps(r.get("previous"), sort_keys=True))
            if key in questions:
                print(f"ERROR {r['_where']}: duplicate of {questions[key]}")
                errors += 1
            questions.setdefault(key, r["_where"])
        for ra, rb in near_duplicates(rows, rows, True):
            if ra.get("previous") == rb.get("previous"):
                print(f"REVIEW near-duplicate {ra['_where']} ~ {rb['_where']}: {ra['question']!r} / {rb['question']!r}")
        for r in rows:
            problems = []
            if set(r) - {"_where"} < REQUIRED or not REQUIRED <= set(r):
                problems.append(f"missing fields {sorted(REQUIRED - set(r))}")
            if r.get("id") in seen:
                problems.append("duplicate id")
            seen.add(r.get("id"))
            if "plan" in r and (p := check_plan(r["plan"])):
                problems.append(f"plan {p}")
            if r.get("previous") is not None:
                if r["previous"]["intent"] == "not_answerable":
                    problems.append("previous plan must be an answerable analysis")
                elif p := check_plan(r["previous"]):
                    problems.append(f"previous {p}")
            if execute and not problems and (p := check_execution(r)):
                problems.append(p)
            for p in problems:
                print(f"ERROR {r['_where']} ({r.get('id')}): {p}")
            errors += len(problems)
    if len(files) > 1:
        errors += check_leakage(files)
    total = sum(len(r) for r in files.values())
    print(f"{total} examples in {len(files)} file(s): {'OK' if not errors else f'{errors} error(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [str(Path(__file__).with_name("seed_examples.jsonl"))]))
