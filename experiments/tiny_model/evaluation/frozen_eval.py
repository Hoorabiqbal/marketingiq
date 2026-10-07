"""
Phase 5: the one-time final evaluation of the Phase 4 model on eval_frozen.jsonl. No training.

    question (+ previous plan) -> plan_format.messages() -> Qwen3 chat template, thinking OFF
      -> fine-tuned Qwen3-0.6B + LoRA, greedy -> raw text (new tokens only)
      -> parse -> plan_format.to_canonical() -> validate_plan() -> execute_plan() on DuckDB

Everything model-independent lives here so the Colab notebook stays thin and the scoring can be
re-run on the laptop from the saved generations:

    python frozen_eval.py --self-test                 integrity check + every gold plan scored as a prediction
    python frozen_eval.py --rescore generations.jsonl rescore saved raw outputs (same code, no model)

Scoring reuses Phase 2 evaluation/score.py unchanged: `exact` (every plan field means the same after
validation, the headline "exact plan match") and `execution_match` (identical executed answer; a
not_answerable gold counts as matched when refused with the same reason). Each stage is recorded
separately and nothing is repaired: the raw output is parsed with json.loads exactly as Phase 4
validation did, and the only normalization is what production already does (to_canonical fills
defaults, validate_plan canonicalizes values, answer()'s post-processing).

One deviation from score.score(): it lets a crash inside execute_plan() escape (README: a
campaign_list sorted by ctr/clicks/conversions raises KeyError). Here the predicted plan is
executed first; a crash is recorded as an execution error (exact = execution_match = False)
instead of stopping the benchmark. Gold plans never crash (validate_dataset.py --execute).
"""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as scorer  # noqa: E402  (adds dataset/ to sys.path)
import plan_format as pf  # noqa: E402

FROZEN = HERE.parent / "dataset" / "eval_frozen.jsonl"
FROZEN_SHA256 = "51966831e505de8ac59fec49dad53cb1c2d4daa8744fa46d75794be831c7f2b1"
FROZEN_COUNT = 429
ROW_FIELDS = ["id", "template_id", "category", "source", "style", "question", "previous", "plan"]

EOS = "<|im_end|>"
THINK_OFF = "<think>\n\n</think>\n\n"  # the empty think block Qwen3 adds with enable_thinking=False
GROUPS = ("category", "gold_intent", "follow_up", "source", "style")


class IntegrityError(RuntimeError):
    pass


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(path=FROZEN) -> str:
    """Abort unless the file is exactly the frozen eval-v1 set. Checked before any row is parsed."""
    path = Path(path)
    if not path.is_file():
        raise IntegrityError(f"FROZEN EVALUATION INTEGRITY: FAIL - {path} is missing")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    count = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
    problems = []
    if digest != FROZEN_SHA256:
        problems.append(f"SHA256 {digest} != expected {FROZEN_SHA256}")
    if count != FROZEN_COUNT:
        problems.append(f"{count} questions != expected {FROZEN_COUNT}")
    if problems:
        raise IntegrityError("FROZEN EVALUATION INTEGRITY: FAIL - " + "; ".join(problems))
    print(f"FROZEN EVALUATION INTEGRITY: PASS\nQuestions: {count}\nSHA256: {digest}")
    return digest


def load_frozen(path=FROZEN) -> list:
    verify_frozen(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [r.get("id") for r in rows if list(r) != ROW_FIELDS]
    if bad or len({r["id"] for r in rows}) != len(rows):
        raise IntegrityError(f"unexpected record structure or duplicate ids, e.g. {bad[:3]}")
    return rows


# ---------------------------------------------------------------- model side (torch only imported here)

def build_prompt(tokenizer, row) -> str:
    """The Phase 4 inference prompt: plan_format.messages() through the Qwen3 chat template, thinking OFF."""
    prompt = tokenizer.apply_chat_template(pf.messages(row["question"], row["previous"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith("<|im_start|>assistant\n" + THINK_OFF):
        raise IntegrityError(f"{row['id']}: prompt does not end with the thinking-OFF assistant header")
    if not prompt.startswith("<|im_start|>system\n" + pf.SYSTEM_PROMPT):
        raise IntegrityError(f"{row['id']}: prompt does not start with the MarketingIQ system prompt")
    return prompt


def generate(model, tokenizer, prompts, max_new_tokens, batch_size, progress=True) -> list:
    """Greedy decoding with the Phase 4 validation settings (left padding, <|im_end|> stop, same pad id).
    raw_output is decoded from the new tokens only, exactly as Phase 4 did; the rest is for auditing."""
    import torch
    tokenizer.padding_side = "left"
    eos_id = tokenizer.convert_tokens_to_ids(EOS)
    results = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 eos_token_id=eos_id, pad_token_id=tokenizer.pad_token_id)
        for ids in gen[:, enc["input_ids"].shape[1]:].tolist():
            finished = eos_id in ids
            used = ids[:ids.index(eos_id) + 1] if finished else ids
            results.append({"raw_output": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                            "raw_with_special": tokenizer.decode(used, skip_special_tokens=False),
                            "finished": finished, "new_tokens": len(used)})
        if progress:
            print(f"\r{len(results)}/{len(prompts)} generated", end="")
    if progress:
        print()
    return results


def _layer_key(name: str) -> str:
    return name[name.index("layers."):].replace(".default", "") if "layers." in name else name


def lora_state(model) -> dict:
    return {_layer_key(n): p for n, p in model.named_parameters() if "lora_" in n}


def lora_digest(model) -> str:
    """Fingerprint of the in-memory LoRA weights, taken before and after generation."""
    h = hashlib.sha256()
    for k, p in sorted(lora_state(model).items()):
        h.update(k.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def check_lora_loaded(model, adapter_dir) -> dict:
    """The live LoRA tensors must be exactly the saved Phase 4 adapter (no missing/random/extra weights)."""
    import torch
    from safetensors.torch import load_file
    saved = {_layer_key(k): v for k, v in load_file(str(Path(adapter_dir) / "adapter_model.safetensors")).items()}
    live = lora_state(model)
    missing, extra = sorted(set(saved) - set(live)), sorted(set(live) - set(saved))
    differ = [k for k in saved.keys() & live.keys()
              if not torch.allclose(saved[k].float().cpu(), live[k].detach().float().cpu(), rtol=1e-2, atol=1e-4)]
    if not saved or missing or extra or differ:
        raise IntegrityError(f"loaded LoRA differs from the saved adapter: {len(saved)} saved tensors, "
                             f"missing {missing[:3]}, extra {extra[:3]}, different {differ[:3]}")
    return {"lora_tensors": len(saved), "lora_parameters": int(sum(v.numel() for v in saved.values()))}


# ---------------------------------------------------------------- scoring (no model needed)

def _parse(text):
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _outcome(r) -> str:
    """First stage at which the example went wrong (or how it went right)."""
    if not r["generated"]:
        return "empty_output"
    if not r["parsed"]:
        return "not_json" if r["finished"] else "truncated_not_json"
    if not r["canonicalized"]:
        return "unknown_keys"
    if not r["validate_plan_ok"]:
        return "rejected_by_validate_plan"
    if not r["gold_answerable"]:
        if r["correct_refusal"]:
            return "correct_refusal"
        return "refused_wrong_reason" if r["refused"] else "answered_unanswerable"
    if r["refused"]:
        return "false_refusal"
    if not r["exec_success"]:
        return "execution_error"
    return "correct" if r["execution_match"] else "execution_mismatch"


def score_example(row, gen) -> dict:
    gold, question, raw = row["plan"], row["question"], gen["raw_output"]
    r = {k: row[k] for k in ("id", "template_id", "category", "source", "style", "question", "previous")}
    r.update(gold=gold, gold_intent=gold["intent"], gold_answerable=gold["intent"] != "not_answerable",
             follow_up=row["previous"] is not None, raw_output=raw, raw_with_special=gen["raw_with_special"],
             finished=gen["finished"], new_tokens=gen["new_tokens"])
    r["generated"] = bool(raw)
    r["thinking_in_output"] = "<think>" in gen["raw_with_special"] or "</think>" in gen["raw_with_special"]
    plan = _parse(raw)
    r["parsed"] = plan is not None
    r["unknown_keys"] = sorted(set(plan) - set(pf.field_order())) if plan is not None else []
    r["canonicalized"] = plan is not None and not r["unknown_keys"]
    r["canonical_plan"] = json.loads(pf.to_canonical(plan)) if r["canonicalized"] else None
    r["exact_strict"] = plan == gold  # compact JSON identical to the gold compact plan

    crashed = None
    try:
        p_plan, p_reason, p_err = scorer._validated(raw, question)
    except Exception as e:  # noqa: BLE001 - a validator crash on model output is a validation failure
        p_plan, p_reason, p_err = None, None, f"validate_plan raised {type(e).__name__}: {e}"
        crashed = p_err
    r["validate_plan_ok"], r["validator_error"] = p_err is None, p_err
    r["refused"], r["refusal_reason"] = p_reason is not None, p_reason

    r["exec_status"] = r["exec_error"] = None
    if r["gold_answerable"] and p_plan is not None:
        try:
            out, _ = scorer.execute(p_plan)
            r["exec_status"] = out[0]  # ok / no_data, or "error" when execute_plan raised PlanError
            if out[0] == "error":
                r["exec_error"] = f"{out[1]}: {out[2]}"
        except Exception as e:  # noqa: BLE001 - known backend KeyError etc.: the user would get an error
            r["exec_status"], r["exec_error"] = "exception", f"{type(e).__name__}: {e}"
            crashed = r["exec_error"]
    r["exec_attempted"] = r["exec_status"] is not None
    r["exec_success"] = r["exec_status"] in ("ok", "no_data")

    s = scorer.score(gold, raw, question) if crashed is None else {"exact": False, "execution_match": False}
    r["exact"], r["execution_match"] = s["exact"], s["execution_match"]
    r["exec_mismatch"] = r["exec_success"] and not r["execution_match"]
    for f in scorer.FIELDS:
        r[f"field_{f}"] = s.get(f, False) if r["gold_answerable"] else None
    r["correct_refusal"] = (not r["gold_answerable"]) and bool(s.get("reason"))
    r["outcome"] = _outcome(r)
    return r


def score_all(rows, generations) -> list:
    if len(rows) != len(generations):
        raise IntegrityError(f"{len(generations)} generations for {len(rows)} questions")
    return [score_example(row, gen) for row, gen in zip(rows, generations)]


def self_test(rows) -> None:
    """Score every gold plan as if the model had produced it: must be 100% on every stage, else the
    harness (not the model) is broken. Uses gold labels only; no model output is looked at."""
    fake = [{"raw_output": json.dumps(r["plan"], separators=(",", ":")), "finished": True, "new_tokens": 0,
             "raw_with_special": json.dumps(r["plan"], separators=(",", ":")) + EOS} for r in rows]
    bad = [r["id"] for r in score_all(rows, fake)
           if not (r["exact"] and r["execution_match"] and r["exact_strict"] and r["validate_plan_ok"])
           or r["outcome"] not in ("correct", "correct_refusal")]
    if bad:
        raise IntegrityError(f"scoring self-test failed: gold plans do not match themselves: {bad[:5]}")
    print(f"Scoring self-test: all {len(rows)} gold plans score as exact + execution match against themselves.")


# ---------------------------------------------------------------- reporting

def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _m(rows, key, ci=False) -> dict:
    k, n = sum(bool(r[key]) for r in rows), len(rows)
    out = {"correct": k, "total": n, "rate": round(k / n, 4) if n else None}
    if ci:
        out["wilson_95"] = wilson(k, n)
    return out


def _table(records, group) -> dict:
    keys = ("parsed", "validate_plan_ok", "exact", "execution_match")
    out = {}
    for value in sorted({str(r[group]) for r in records}):
        rows = [r for r in records if str(r[group]) == value]
        out[value] = {"n": len(rows), **{k: round(sum(bool(r[k]) for r in rows) / len(rows), 4) for k in keys}}
    return out


def summarize(records) -> dict:
    ans = [r for r in records if r["gold_answerable"]]
    na = [r for r in records if not r["gold_answerable"]]
    attempted = [r for r in ans if r["exec_attempted"]]
    return {
        "examples": len(records),
        "pipeline": {
            "generation_success": _m(records, "generated"),
            "stopped_at_eos": _m(records, "finished"),
            "thinking_in_output": _m(records, "thinking_in_output"),
            "parse_success": _m(records, "parsed"),
            "canonicalization_success": _m(records, "canonicalized"),
            "validate_plan_success": _m(records, "validate_plan_ok"),
        },
        "exact_plan_match": _m(records, "exact", ci=True),
        "exact_plan_match_strict_json": _m(records, "exact_strict", ci=True),
        "execution_match": _m(records, "execution_match", ci=True),
        "answerable": {
            "examples": len(ans),
            "execution_attempted": _m(ans, "exec_attempted"),
            "execution_successful": _m(ans, "exec_success"),
            "execution_match": _m(ans, "execution_match", ci=True),
            "execution_mismatch": _m(ans, "exec_mismatch"),
            "execution_error": {"total": sum(r["exec_attempted"] and not r["exec_success"] for r in ans),
                                "of_attempted": len(attempted)},
            "no_data_results": sum(r["exec_status"] == "no_data" for r in ans),
            "false_refusal": _m(ans, "refused"),
            "exact_plan_match": _m(ans, "exact", ci=True),
            "field_accuracy": {f: _m(ans, f"field_{f}") for f in scorer.FIELDS},
        },
        "not_answerable": {
            "examples": len(na),
            "correctly_refused": _m(na, "correct_refusal", ci=True),
            "outcomes": dict(Counter(r["outcome"] for r in na).most_common()),
            "by_reason": {reason: _m([r for r in na if r["gold"].get("reason") == reason], "correct_refusal")
                          for reason in sorted({r["gold"].get("reason") for r in na})},
        },
        "outcomes": dict(Counter(r["outcome"] for r in records).most_common()),
        "breakdowns": {g: _table(records, g) for g in GROUPS},
    }


def _pct(m) -> str:
    ci = f" (95% CI {m['wilson_95'][0]:.1%}-{m['wilson_95'][1]:.1%})" if m.get("wilson_95") else ""
    return f"{m['correct']} / {m['total']} = {m['rate']:.2%}{ci}" if m["total"] else "n/a"


def render_markdown(report) -> str:
    s = report["metrics"]
    p, a, na = s["pipeline"], s["answerable"], s["not_answerable"]
    lines = ["# Phase 5 - frozen evaluation (eval-v1, 429 questions)", "",
             f"Model: `{report['model']['base_model']}` + LoRA `{report['model']['phase4_run']}` "
             f"(adapter SHA-256 unchanged: {report['model']['adapter_unchanged']}); thinking OFF; greedy.",
             f"Frozen set SHA-256: `{report['frozen_eval']['sha256']}`", "",
             "## Headline", "",
             f"- **Exact plan match** (score.py `exact`): {_pct(s['exact_plan_match'])}",
             f"- **Execution match** (identical DuckDB answer; correct refusals count): {_pct(s['execution_match'])}",
             f"- Exact compact-JSON match (strict): {_pct(s['exact_plan_match_strict_json'])}", "",
             "## Pipeline stages", ""]
    lines += [f"- {k.replace('_', ' ')}: {_pct(v)}" for k, v in p.items()]
    lines += ["", f"## Answerable questions ({a['examples']})", "",
              f"- execution attempted: {_pct(a['execution_attempted'])}",
              f"- execution successful: {_pct(a['execution_successful'])}",
              f"- execution match: {_pct(a['execution_match'])}",
              f"- execution mismatch (valid plan, different answer): {_pct(a['execution_mismatch'])}",
              f"- execution error: {a['execution_error']['total']} of {a['execution_error']['of_attempted']} attempted",
              f"- false refusals: {_pct(a['false_refusal'])}", "",
              "| field | accuracy |", "|---|---|"]
    lines += [f"| {f} | {_pct(m)} |" for f, m in a["field_accuracy"].items()]
    lines += ["", f"## Not-answerable questions ({na['examples']})", "",
              f"- refused with the right reason: {_pct(na['correctly_refused'])}"]
    lines += [f"- {reason}: {_pct(m)}" for reason, m in na["by_reason"].items()]
    lines += ["", "## Outcomes", "", "| outcome | n |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["outcomes"].items()]
    for g, table in s["breakdowns"].items():
        lines += ["", f"## By {g}", "", "| value | n | parse | valid | exact | execution |", "|---|---|---|---|---|---|"]
        lines += [f"| {v} | {t['n']} | {t['parsed']:.1%} | {t['validate_plan_ok']:.1%} | {t['exact']:.1%} | "
                  f"{t['execution_match']:.1%} |" for v, t in table.items()]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir, records, report) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "frozen_eval_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (out / "frozen_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "frozen_eval_summary.md").write_text(render_markdown(report), encoding="utf-8")


def main(argv) -> int:
    rows = load_frozen()
    self_test(rows)
    if argv[:1] == ["--rescore"] and len(argv) == 2:
        gens = [json.loads(line) for line in Path(argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
        if [g["id"] for g in gens] != [r["id"] for r in rows]:
            raise IntegrityError("generations are not the 429 frozen questions in order")
        records = score_all(rows, gens)
        print(json.dumps({k: v for k, v in summarize(records).items() if k != "breakdowns"}, indent=1))
    elif argv not in ([], ["--self-test"]):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
