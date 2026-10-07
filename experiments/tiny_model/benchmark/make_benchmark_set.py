"""
Phase 6: build the small local benchmark / smoke-test set from validation.jsonl (never eval_frozen.jsonl).

    python experiments/tiny_model/benchmark/make_benchmark_set.py
    -> experiments/tiny_model/benchmark/benchmark_questions.jsonl

A fixed, seeded, per-category sample of the frozen Phase 3 validation split (30 questions covering
every request type). `smoke: true` marks the 8 rows the export notebook uses as its smoke test
(total, breakdown, comparison, relationship, follow-up, unsupported, time, campaign list).

Validation data is development data: it was only used to report Phase 4 metrics, never to pick a
checkpoint. The frozen set is read for one thing only: refusing to write any question that also
appears there (normalized text), so no runtime setting can ever be chosen on a frozen question.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset"
VALIDATION = DATASET / "validation.jsonl"
FROZEN = DATASET / "eval_frozen.jsonl"
OUT = HERE / "benchmark_questions.jsonl"

VALIDATION_SHA256 = "e247278e7ca147e9e2ea8786784d8409a930c2b2b357afa8d54ddbe672c35e73"
SEED = 20260929
PER_CATEGORY = {"total": 3, "breakdown": 3, "comparison": 3, "relationship": 3, "follow_up": 4, "unsupported": 3,
                "time": 3, "filters": 2, "campaign_list": 2, "visualization": 2, "ambiguous": 1, "performance": 1}
SMOKE = ("total", "breakdown", "comparison", "relationship", "follow_up", "unsupported", "time", "campaign_list")


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def main():
    data = VALIDATION.read_bytes()
    if hashlib.sha256(data).hexdigest() != VALIDATION_SHA256:
        sys.exit("validation.jsonl does not match its frozen Phase 3 SHA-256")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    frozen_questions = {norm(json.loads(line)["question"])
                        for line in FROZEN.read_text(encoding="utf-8").splitlines() if line.strip()}

    rng = random.Random(SEED)
    picked = []
    for category, n in PER_CATEGORY.items():
        pool = [r for r in rows if r["category"] == category and norm(r["question"]) not in frozen_questions]
        if len(pool) < n:
            sys.exit(f"only {len(pool)} validation rows for {category}")
        chosen = rng.sample(pool, n)
        for i, r in enumerate(chosen):
            picked.append({**r, "smoke": category in SMOKE and i == 0})
    if any(norm(r["question"]) in frozen_questions for r in picked):
        sys.exit("a benchmark question appears in eval_frozen.jsonl")

    OUT.write_text("".join(json.dumps(r) + "\n" for r in picked), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(picked)} questions ({sum(r['smoke'] for r in picked)} smoke), "
          f"{sum(r['previous'] is not None for r in picked)} follow-ups, 0 in eval_frozen.jsonl")
    print(f"SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
