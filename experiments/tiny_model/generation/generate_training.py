"""
Phase 3: generate, validate and freeze the SFT training + validation sets.

    python generate_training.py                 # writes ../dataset/train.jsonl, validation.jsonl, train_manifest.json
    python generate_training.py --out DIR       # same run into DIR (reproducibility check; no freeze guard)

Every example: question (+ optional previous plan) -> gold plan in the sparse canonical form. Gold
plans come from the semantic spec that also produced the question (families.py) or were written by
hand (curated.py); no LLM is involved. An example is kept only if:
  - the gold plan (and any previous plan) is right for production's validate_plan(): answerable plans
    pass, not_answerable plans are refused with their reason, no unknown keys;
  - production's post-processing (query_planner.answer) leaves it unchanged;
  - an answerable plan runs through production's execute_plan() on the local DuckDB data with status ok;
  - it doesn't collide with the frozen evaluation set (normalized text, word overlap, or skeleton —
    see leakage.py), with another example (same question + previous), or with a conflicting label.
Train and validation are split by template frame; validation rows that resemble a train row are dropped.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset"
sys.path.insert(0, str(DATASET))

import plan_format as pf  # noqa: E402
import leakage  # noqa: E402
import validate_dataset as vd  # noqa: E402

SEED = 20260929
VERSION = "train-v1"
EVAL_SHA256 = "51966831e505de8ac59fec49dad53cb1c2d4daa8744fa46d75794be831c7f2b1"
MAX_ATTEMPTS_PER_TARGET = 6
SMALL_FAMILY_VAL_KEEP = 0.4   # families with < 8 frames: keep 40% of held-out-frame draws (validation ~10%)
def is_val_frame(i: int, n_frames: int) -> bool:
    """Frames held out for validation: one in eight, or the last frame of a family with fewer than 8."""
    return i == n_frames - 1 if n_frames < 8 and i < n_frames else i % 8 == 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Checker:
    def __init__(self):
        self.qp = pf.backend()
        self.cache = {}

    def reason(self, question: str, plan: dict, previous) -> str:
        """None when the example is a valid gold example, else why not."""
        qp = self.qp
        problem = vd.check_plan(plan)
        if problem:
            return f"validate_plan: {problem}"
        if previous is not None and (previous.get("intent") == "not_answerable" or vd.check_plan(previous)):
            return "previous plan invalid"
        if plan["intent"] == "not_answerable":
            return None
        p = qp.validate_plan(pf.to_canonical(plan))
        post = pf.postprocess(copy.deepcopy(p), question)
        if (post.visualization, post.intent, post.grain) != (p.visualization, p.intent, p.grain):
            return "label changed by production post-processing"
        key = json.dumps(qp.plan_to_raw(post), sort_keys=True)
        if key not in self.cache:
            try:
                status = qp.execute_plan(post, {}).status
                self.cache[key] = None if status == "ok" else f"execute_plan status {status}"
            except Exception as e:  # noqa: BLE001  e.g. the known campaign_list KeyError
                self.cache[key] = f"execute_plan raised {type(e).__name__}"
        return self.cache[key]


def generate():
    import families as fam
    import curated
    qp = pf.backend()
    eval_rows = [json.loads(line) for line in (DATASET / "eval_frozen.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_index = leakage.Index([r["question"] for r in eval_rows])
    eval_templates = {r["template_id"] for r in eval_rows}
    checker = Checker()
    r = random.Random(SEED)
    kept = {}                              # (normalized question, previous) -> row
    banned = set()
    rejected, samples = Counter(), defaultdict(list)
    stats = Counter()

    def reject(why, row):
        rejected[why] += 1
        if len(samples[why]) < 5:
            samples[why].append({"question": row["question"], "previous": row["previous"], "plan": row["plan"]})

    def consider(row, clean_question):
        stats["candidates"] += 1
        why = checker.reason(row["question"], row["plan"], row["previous"])
        if why and clean_question != row["question"]:        # the noise broke a chart/time word: keep it clean
            row["question"], row["style"] = clean_question, row["style_clean"]
            why = checker.reason(row["question"], row["plan"], row["previous"])
        if why:
            reject(why, row)
            return False
        for q in dict.fromkeys([clean_question, row["question"]]):
            hit = eval_index.conflict(q)
            if hit:
                reject(f"eval leakage ({hit[0]})", row)
                return False
        key = (leakage.normalize(row["question"]), json.dumps(row["previous"], sort_keys=True))
        if key in banned:
            reject("conflicting label", row)
            return False
        if key in kept:
            if kept[key]["plan"] != row["plan"]:
                reject("conflicting label", row)
                reject("conflicting label", kept.pop(key))
                banned.add(key)
            else:
                reject("duplicate", row)
            return False
        kept[key] = row
        return True

    for name, family_category, target, fn, n_frames in fam.FAMILIES:
        accepted = attempts = 0
        while accepted < target and attempts < target * MAX_ATTEMPTS_PER_TARGET:
            attempts += 1
            draw = fn(r)
            if draw is None:
                stats["draws_not_applicable"] += 1
                continue
            i, q, plan, prev, style = draw
            noisy, noise_style = (q, None) if style == "adversarial" else fam.noise(r, q)
            split = "validation" if is_val_frame(i, n_frames) else "train"
            if split == "validation" and n_frames < 8 and r.random() > SMALL_FAMILY_VAL_KEEP:
                stats["small_family_validation_draws_skipped"] += 1
                continue
            category = fam.GRAMMAR_CATEGORY[fam.GRAMMAR[i][1]] if name == "gram" else family_category
            row = {"template_id": f"trn-{name}-{i:03d}", "category": category, "source": "generated",
                   "style": noise_style or style, "style_clean": style, "question": noisy, "previous": prev,
                   "plan": plan, "_split": split}
            accepted += consider(row, q)
        stats[f"family_{name}_accepted"] = accepted
    for _ in range(fam.KNOWN_BUG_ATTEMPTS):
        i, q, plan, prev, style = fam.cl_known_bug(r)
        row = {"template_id": f"trn-clbug-{i:03d}", "category": "campaign_list", "source": "generated",
               "style": style, "style_clean": style, "question": q, "previous": prev, "plan": plan,
               "_split": "train"}
        consider(row, q)
    for n, (category, q, plan, prev, style) in enumerate(curated.items()):
        split = "validation" if n % 9 == 4 else "train"
        row = {"template_id": f"trn-curated-{category}" + ("-val" if split == "validation" else ""),
               "category": category, "source": "curated", "style": style, "style_clean": style, "question": q,
               "previous": prev, "plan": plan, "_split": split}
        consider(row, q)

    rows = list(kept.values())
    assert not ({r_["template_id"] for r_ in rows} & eval_templates)
    train = [x for x in rows if x["_split"] == "train"]
    val = []
    train_index = leakage.Index([x["question"] for x in train])
    for x in (x for x in rows if x["_split"] == "validation"):
        hit = train_index.conflict(x["question"])
        if hit and hit[0] != "near_skeleton":   # different frames may share generic words; same text/skeleton may not
            reject(f"validation resembles train ({hit[0]})", x)
        else:
            val.append(x)
    return train, val, rejected, samples, stats, checker


def finish(rows, prefix):
    out = []
    for n, x in enumerate(rows, 1):
        out.append({"id": f"{prefix}-{n:05d}", "template_id": x["template_id"], "category": x["category"],
                    "source": x["source"], "style": x["style"], "question": x["question"], "previous": x["previous"],
                    "plan": x["plan"]})
    return out


def distribution(rows):
    metrics, dims = Counter(), Counter()
    for x in rows:
        p = x["plan"]
        metrics.update(p.get("metrics", []))
        for d in [p.get("dimension"), p.get("dimension2")] + [f["dimension"] for f in p.get("filters", [])]:
            if d:
                dims[d] += 1
    n = len(rows)
    noisy = sum(x["style"] in ("typo", "caps", "lower", "grammar") for x in rows)
    pct = lambda k: f"{k} ({k / n:.1%})"  # noqa: E731
    return {
        "examples": n,
        "categories": dict(Counter(x["category"] for x in rows).most_common()),
        "intents": dict(Counter(x["plan"]["intent"] for x in rows).most_common()),
        "not_answerable_reasons": dict(Counter(x["plan"].get("reason") for x in rows
                                               if x["plan"]["intent"] == "not_answerable").most_common()),
        "metrics": dict(metrics.most_common()), "dimensions": dict(dims.most_common()),
        "sources": dict(Counter(x["source"] for x in rows)), "styles": dict(Counter(x["style"] for x in rows).most_common()),
        "follow_up": pct(sum(x["previous"] is not None for x in rows)),
        "unsupported_or_ambiguous": pct(sum(x["category"] in ("unsupported", "ambiguous") for x in rows)),
        "adversarial": pct(sum(x["category"] == "adversarial" for x in rows)),
        "noisy": pct(noisy),
        "with_visualization": pct(sum("visualization" in x["plan"] for x in rows)),
        "template_ids": len({x["template_id"] for x in rows}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out_dir = args.out or DATASET
    freeze = args.out is None
    train_path, val_path = out_dir / "train.jsonl", out_dir / "validation.jsonl"
    manifest_path, report_path = out_dir / "train_manifest.json", out_dir / "train_generation_report.json"
    if freeze and (train_path.exists() or val_path.exists() or manifest_path.exists()):
        print("train.jsonl / validation.jsonl are frozen; generate a new version under a new name instead.")
        return 1
    if sha256(DATASET / "eval_frozen.jsonl") != EVAL_SHA256:
        print("FROZEN EVALUATION DATASET CHANGED — PHASE 3 ABORTED")
        return 2
    pf.backend()
    import vocab
    problems = vocab.check_vocab()
    if problems:
        print("\n".join(problems))
        return 1

    train, val, rejected, samples, stats, checker = generate()
    train, val = finish(train, "trn"), finish(val, "val")

    # Final integrity pass over exactly what is written (independent of the generation-time checks).
    eval_q = [json.loads(l)["question"] for l in (DATASET / "eval_frozen.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_index = leakage.Index(eval_q)
    integrity = Counter()
    for x in train + val:
        integrity["validate_plan_pass"] += checker.reason(x["question"], x["plan"], x["previous"]) is None
        if x["plan"]["intent"] != "not_answerable":
            integrity["answerable"] += 1
            integrity["execute_plan_pass"] += checker.reason(x["question"], x["plan"], x["previous"]) is None
        integrity["eval_leaks"] += eval_index.conflict(x["question"]) is not None
    keys = [(leakage.normalize(x["question"]), json.dumps(x["previous"], sort_keys=True)) for x in train + val]
    integrity["normalized_duplicates"] = len(keys) - len(set(keys))
    exact = [(x["question"], json.dumps(x["previous"], sort_keys=True)) for x in train + val]
    integrity["exact_duplicates"] = len(exact) - len(set(exact))
    integrity["train_validation_template_overlap"] = len({x["template_id"] for x in train} & {x["template_id"] for x in val})
    train_norm = {leakage.normalize(x["question"]) for x in train}
    integrity["validation_questions_in_train"] = sum(leakage.normalize(x["question"]) in train_norm for x in val)
    ok = (integrity["validate_plan_pass"] == len(train) + len(val)
          and integrity["execute_plan_pass"] == integrity["answerable"] and not integrity["eval_leaks"]
          and not integrity["normalized_duplicates"] and not integrity["exact_duplicates"]
          and not integrity["train_validation_template_overlap"] and not integrity["validation_questions_in_train"])
    if not ok:
        print(json.dumps(integrity, indent=2))
        print("Integrity check failed; nothing frozen.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in ((train_path, train), (val_path, val)):
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    generated = stats["candidates"]
    report = {"candidates_generated": generated, "rejected": sum(rejected.values()),
              "rejection_reasons": dict(rejected.most_common()), "rejection_samples": samples,
              "draws_not_applicable": stats["draws_not_applicable"],
              "families_accepted": {k[7:-9]: v for k, v in stats.items() if k.startswith("family_")},
              "known_backend_issue": "campaign_list sorted by a condition field outside chart_builder.CAMPAIGN_METRICS "
                                     "(ctr, clicks, conversions) can pass validate_plan() but raise in execute_plan(); "
                                     f"{fam_bug_note(rejected)}"}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "version": VERSION, "generation_seed": SEED, "created_on": date.today().isoformat(),
        "generator": "experiments/tiny_model/generation/generate_training.py (families.py, curated.py, vocab.py, leakage.py)",
        "train": {"file": train_path.name, "sha256": sha256(train_path), **distribution(train)},
        "validation": {"file": val_path.name, "sha256": sha256(val_path), **distribution(val)},
        "eval_sha256_checked": EVAL_SHA256,
        "checks": {
            "validate_plan": f"{integrity['validate_plan_pass']}/{len(train) + len(val)} pass (not_answerable refused with its reason)",
            "execute_plan": f"{integrity['execute_plan_pass']}/{integrity['answerable']} answerable plans return status ok",
            "post_processing": "no label is changed by query_planner.answer()'s post-processing",
            "eval_leakage": "none: normalized text, word-set Jaccard >= 0.8, identical or near (>= 0.75) skeleton vs eval_frozen.jsonl",
            "eval_templates_reused": 0,
            "exact_duplicates": integrity["exact_duplicates"], "normalized_duplicates": integrity["normalized_duplicates"],
            "train_validation_template_overlap": integrity["train_validation_template_overlap"],
            "validation_questions_in_train": integrity["validation_questions_in_train"],
        },
        "generation": {"candidates": generated, "rejected": sum(rejected.values()), "report": report_path.name},
        "labels": "deterministic from the same semantic spec as the question, or hand-written; no LLM (Groq or other)",
        "target_format": "sparse canonical plan (plan_format.to_sparse); to_canonical() -> validate_plan() -> execute_plan()",
        "rule": "Frozen. Do not edit or regenerate in place; eval_frozen.jsonl stays untouched.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if sha256(DATASET / "eval_frozen.jsonl") != EVAL_SHA256:
        print("FROZEN EVALUATION DATASET CHANGED — PHASE 3 FAILED")
        return 2
    print(f"train {len(train)} ({sha256(train_path)})\nvalidation {len(val)} ({sha256(val_path)})")
    return 0


def fam_bug_note(rejected) -> str:
    n = sum(v for k, v in rejected.items() if k.startswith("execute_plan raised"))
    return f"{n} such generated examples were rejected by the execution check (see rejection_reasons)."


if __name__ == "__main__":
    sys.exit(main())
