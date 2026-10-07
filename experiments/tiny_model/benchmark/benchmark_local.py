"""
Phase 6: local CPU benchmark of the exported MarketingIQ GGUFs through llama.cpp `llama-server`.

    venv\\Scripts\\python experiments\\tiny_model\\benchmark\\benchmark_local.py ^
        --server C:\\llama.cpp\\llama-server.exe ^
        --model experiments\\tiny_model\\export\\marketingiq-qwen3-0.6b-Q8_0.gguf ^
        --model experiments\\tiny_model\\export\\marketingiq-qwen3-0.6b-Q4_K_M.gguf ^
        --threads 4

Per model (and per --threads value) it starts a fresh llama-server, then sends every question of
benchmark_questions.jsonl (30 validation questions, never eval_frozen.jsonl) one at a time:

  startup      process start -> /health ok (model load + warmup)
  prompt       prompt tokens and prompt-processing time (server `timings`)
  TTFT         client time to the first streamed token
  generation   generated tokens, generation time, tokens/s (server `timings`)
  end-to-end   client time from sending the request to the last streamed chunk
  RAM          server process RSS after load, after the run, and Windows peak working set (psutil)
  quality      parse, canonicalization, validate_plan(), EOS stop, thinking leak, and (for reference
               only) exact / execution match against the validation gold plan, all via
               evaluation/frozen_eval.score_example() - the Phase 5 scoring code, unchanged

Results go to benchmark/results/<timestamp>/ (git-ignored): requests.jsonl, summary.json, summary.md.
No Groq call, no network beyond 127.0.0.1, no backend change. Only local inference is timed:
validation and scoring run after the timing of each request.
"""
import argparse
import datetime
import json
import platform
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TINY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TINY / "evaluation"))
import llama_client as lc  # noqa: E402

QUESTIONS = HERE / "benchmark_questions.jsonl"
RUNTIME = TINY / "export" / "marketingiq_runtime.json"   # written by the Phase 6 export notebook
WARMUP_QUESTION = "warm-up request, please ignore"


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "p50": pct(values, 50), "p95": pct(values, 95),
            "min": min(values), "max": max(values)}


def check_model(path: Path, runtime: dict) -> dict:
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 2**20, 1), "sha256": lc.sha256_file(path)}
    expected = runtime.get("gguf", {}).get(path.name, {}).get("sha256") if runtime else None
    info["sha256_matches_export_report"] = None if expected is None else expected == info["sha256"]
    if expected and not info["sha256_matches_export_report"]:
        sys.exit(f"{path.name}: SHA-256 {info['sha256']} differs from the Phase 6 export report ({expected})")
    return info


def run_model(args, model: Path, threads: int, rows, n_predict, out_dir, fe):
    tag = f"{model.stem}_t{threads}"
    server = lc.LlamaServer(args.server, model, threads=threads, threads_batch=args.threads_batch or threads,
                            ctx=args.ctx, port=args.port, log_path=out_dir / f"{tag}_server.log")
    print(f"\n=== {model.name} | threads {threads} (batch {args.threads_batch or threads}) | ctx {args.ctx} ===")
    records = []
    with server:
        print(f"startup {server.startup_seconds:.2f}s")
        mem_loaded = server.memory()
        prompts = [lc.render_prompt(r["question"], r["previous"]) for r in rows]
        # Untimed warm-up on a question outside the set: first-request allocations / page faults, and (with
        # prompt caching) the shared system-prompt prefix, never a benchmark question's own tokens.
        for _ in range(args.warmup):
            server.complete(lc.render_prompt(WARMUP_QUESTION), n_predict, cache_prompt=True)
        for rep in range(args.repeats):
            for row, prompt in zip(rows, prompts):
                res = server.complete(prompt, n_predict, cache_prompt=args.cache_prompt)
                t = res["timings"]
                rec = {"model": model.name, "threads": threads, "repeat": rep, "id": row["id"],
                       "category": row["category"], "follow_up": row["previous"] is not None,
                       "question": row["question"], "raw_output": res["text"], "stop_type": res["stop_type"],
                       "prompt_tokens": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                       "prompt_tokens_per_s": t.get("prompt_per_second"),
                       "generated_tokens": t.get("predicted_n"), "generation_ms": t.get("predicted_ms"),
                       "generation_tokens_per_s": t.get("predicted_per_second"),
                       "cached_tokens": t.get("cache_n"),
                       "ttft_ms": None if res["ttft_s"] is None else res["ttft_s"] * 1000,
                       "e2e_ms": res["e2e_s"] * 1000}
                s = fe.score_example(row, lc.as_generation(res))     # after timing; not part of latency
                rec.update({k: s[k] for k in ("parsed", "canonicalized", "validate_plan_ok", "validator_error",
                                              "finished", "thinking_in_output", "exact", "execution_match",
                                              "outcome")})
                records.append(rec)
                print(f"  {row['id']:10s} {rec['e2e_ms']:7.0f} ms  ttft {rec['ttft_ms'] or 0:6.0f}  "
                      f"{rec['generated_tokens']:3d} tok {rec['generation_tokens_per_s'] or 0:5.1f} tok/s  "
                      f"{'valid' if rec['validate_plan_ok'] else 'INVALID':7s} {rec['outcome']:22s} {res['text'][:60]!r}")
        mem_end = server.memory()
    n = len(records)
    count = lambda k: sum(bool(r[k]) for r in records)  # noqa: E731
    return records, {
        "model": model.name, "threads": threads, "threads_batch": args.threads_batch or threads, "ctx": args.ctx,
        "startup_s": server.startup_seconds, "requests": n,
        "ram_after_load_mb": mem_loaded.get("rss_mb"), "ram_after_run_mb": mem_end.get("rss_mb"),
        "peak_working_set_mb": mem_end.get("peak_working_set_mb"), "private_mb": mem_end.get("private_mb"),
        **{k: stats([r[k] for r in records]) for k in
           ("e2e_ms", "ttft_ms", "prompt_ms", "prompt_tokens", "generation_ms", "generated_tokens",
            "generation_tokens_per_s", "prompt_tokens_per_s")},
        "stopped_at_eos": f"{sum(r['stop_type'] == 'eos' for r in records)}/{n}",
        "thinking_in_output": f"{count('thinking_in_output')}/{n}",
        "parse_success": f"{count('parsed')}/{n}", "canonicalization_success": f"{count('canonicalized')}/{n}",
        "validate_plan_success": f"{count('validate_plan_ok')}/{n}",
        "exact_vs_validation_gold": f"{count('exact')}/{n}",
        "execution_match_vs_validation_gold": f"{count('execution_match')}/{n}",
    }


def markdown(summaries, meta) -> str:
    f = lambda v, d=0: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = ["# Phase 6 local CPU benchmark", "",
             f"{meta['created']} | {meta['machine']} | llama-server `{meta['server']}` | "
             f"{meta['questions']} validation questions x {meta['repeats']} | n_predict {meta['n_predict']} | "
             f"cache_prompt {meta['cache_prompt']}", "",
             "| model | threads | startup s | e2e p50 ms | e2e p95 ms | TTFT p50 ms | prompt p50 ms | gen tok/s p50 "
             "| gen tokens p50 | RSS MB | peak WS MB | EOS | parse | validate_plan | exact* | exec match* |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        g = lambda k, q="p50": (s[k] or {}).get(q)  # noqa: E731
        lines.append(f"| {s['model']} | {s['threads']} | {f(s['startup_s'], 2)} | {f(g('e2e_ms'))} | "
                     f"{f(g('e2e_ms', 'p95'))} | {f(g('ttft_ms'))} | {f(g('prompt_ms'))} | "
                     f"{f(g('generation_tokens_per_s'), 1)} | {f(g('generated_tokens'))} | {f(s['ram_after_run_mb'])} | "
                     f"{f(s['peak_working_set_mb'])} | {s['stopped_at_eos']} | {s['parse_success']} | "
                     f"{s['validate_plan_success']} | {s['exact_vs_validation_gold']} | "
                     f"{s['execution_match_vs_validation_gold']} |")
    lines += ["", "*Against validation-split gold plans (development data); a sanity check of the export, "
              "not the model's accuracy. The accuracy result is Phase 5's frozen evaluation."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--server", required=True, help="path to llama-server(.exe)")
    ap.add_argument("--model", action="append", required=True, type=Path, help="GGUF file (repeat for several)")
    ap.add_argument("--threads", type=int, nargs="+", default=[lc.default_threads()],
                    help="generation threads; several values = one run each (default: physical cores)")
    ap.add_argument("--threads-batch", type=int, default=None, help="prompt-processing threads (default: --threads)")
    ap.add_argument("--ctx", type=int, default=512, help="context size (prompts are <= ~170 tokens; default 512)")
    ap.add_argument("--n-predict", type=int, default=None,
                    help="max generated tokens (default: Phase 4 max_new_tokens from export/marketingiq_runtime.json)")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick check)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1, help="untimed requests after startup")
    ap.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false",
                    help="re-process the whole prompt every request (default: reuse the shared system-prompt prefix)")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--allow-base-model", action="store_true",
                    help="harness dry run with a GGUF that is not in the export report (e.g. the untuned base model)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    runtime = json.loads(RUNTIME.read_text(encoding="utf-8")) if RUNTIME.is_file() else {}
    if runtime and runtime.get("system_prompt") != lc.pf.SYSTEM_PROMPT:
        sys.exit("export/marketingiq_runtime.json was made with a different system prompt than plan_format.py")
    n_predict = args.n_predict or runtime.get("max_new_tokens")
    if not n_predict:
        sys.exit("No --n-predict given and export/marketingiq_runtime.json not found (copy it from the Phase 6 "
                 "Drive folder next to the GGUFs).")
    models = []
    for m in args.model:
        if not m.is_file():
            sys.exit(f"model not found: {m}")
        info = check_model(m, runtime)
        if info["sha256_matches_export_report"] is None and not args.allow_base_model:
            sys.exit(f"{m.name} is not listed in export/marketingiq_runtime.json; pass --allow-base-model for a "
                     "harness dry run with another GGUF.")
        models.append((m, info))
    rows = lc.load_questions(args.questions)[:args.limit]

    import frozen_eval as fe   # Phase 5 scoring (loads the backend + CSV for validate_plan / execute_plan)
    fe.pf.backend()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or HERE / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"created": stamp, "machine": f"{platform.node()} | {platform.processor()} | {platform.platform()}",
            "server": str(args.server), "questions": len(rows), "questions_file": str(args.questions),
            "questions_sha256": lc.sha256_file(args.questions), "repeats": args.repeats, "warmup": args.warmup,
            "n_predict": n_predict, "cache_prompt": args.cache_prompt, "ctx": args.ctx,
            "decoding": "greedy: temperature 0, top_k 1, stop at <|im_end|>", "thinking": "OFF (empty think block in prompt)",
            "models": [info for _, info in models], "runtime_report": runtime or None}
    try:
        import psutil
        meta["cpu"] = {"physical": psutil.cpu_count(logical=False), "logical": psutil.cpu_count(),
                       "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    except ImportError:
        meta["cpu"] = "psutil not installed: RAM not measured"

    all_records, summaries = [], []
    for model, _ in models:
        for threads in args.threads:
            records, summary = run_model(args, model, threads, rows, n_predict, out_dir, fe)
            all_records += records
            summaries.append(summary)

    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summaries": summaries}, indent=2), encoding="utf-8")
    md = markdown(summaries, meta)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
