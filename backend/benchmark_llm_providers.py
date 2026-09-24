"""
Benchmark LLM providers on the LLM_REQUIRED path with identical inputs.

    python benchmark_llm_providers.py --providers gemini,groq --out results.json   (groq needs GROQ_API_KEY)

For each question: the Query Router classifies it (must be LLM_REQUIRED) and builds the
compact analysis from the DuckDB data layer ONCE; that exact block goes to every provider
through the LLM Adapter. Nothing else is sent. One call per question per provider (Gemini
and Groq keep their normal bounded 5xx retry), so a full run costs ~len(QUESTIONS) requests
per hosted provider. Rate-limit (429) answers are recorded, never retried or worked around.

Recorded per answer: latency (local analysis vs provider), success, length, and a numeric
grounding check: every number in the answer is matched against the numbers present in the
supplied analysis (at the precision written, incl. "$34.3 million"-style scaling). Numbers
with no match are listed as `unsupported` for manual review — they may be invented, or
derived (e.g. a difference the model computed). System RAM is recorded before and after
each provider's run.

Past results (including the removed local Qwen provider) are in benchmark_results/.
"""
import argparse
import hashlib
import ctypes
import json
import logging
import os
import re
import statistics
import sys
import time
from ctypes import wintypes
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(HERE / ".env")

QUESTIONS = [
    "Why is TikTok performing better than other platforms?",
    "What might explain the ROAS difference between platforms?",
    "Explain the decline in CTR.",
    "Explain what the monthly revenue trend means.",
    "Why does TikTok have a higher ROAS than LinkedIn?",
    "Why do campaigns with spend over 10000 have lower ROAS?",
    "Give me a business interpretation of overall performance.",
    "What insights can you draw about device performance?",
    "Explain how creative age affects CTR.",
    "Why is CPA higher for some campaign objectives?",
]

NUM_RE = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(million|billion|m\b|bn\b|k\b)?", re.IGNORECASE)
SCALE = {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "k": 1e3}


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

def analysis_numbers(obj) -> list:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += analysis_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += analysis_numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj)]  # months, buckets, ids
    return out


def check_numbers(text: str, supplied: list) -> dict:
    cited, unsupported = [], []
    for m in NUM_RE.finditer(text):
        whole, frac, scale = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        raw = m.group(0).strip()
        n = float(whole.replace(",", "") + frac)
        decimals = len(frac) - 1 if frac else 0
        if not scale and not frac and n < 10 and not raw.startswith("$"):
            continue  # list numbering / "two parts" etc.
        factor = SCALE.get(scale, 1)
        tol = 0.5 * 10 ** -decimals * factor + 1e-9
        target = n * factor
        ok = any(abs(v - target) <= tol for v in supplied)
        (cited if ok else unsupported).append(raw)
    return {"numbers_cited": len(cited) + len(unsupported), "supported": len(cited),
            "unsupported": unsupported}


# ---------------------------------------------------------------------------
# System memory (Windows; degrades to None elsewhere)
# ---------------------------------------------------------------------------

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def system_mem():
    try:
        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return {"used_gb": round((s.ullTotalPhys - s.ullAvailPhys) / 1e9, 2),
                "avail_gb": round(s.ullAvailPhys / 1e9, 2), "load_pct": s.dwMemoryLoad}
    except Exception:
        return None


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="gemini,groq")
    ap.add_argument("--out", default="llm_benchmark_results.json")
    ap.add_argument("--gemini-pause", type=float, default=3.0, help="seconds between Gemini calls (free-tier RPM)")
    ap.add_argument("--groq-pause", type=float, default=3.0, help="seconds between Groq calls (free-tier RPM/TPM)")
    args = ap.parse_args()

    import data_tools as dt
    import llm_providers
    import query_router as qr
    from gemini_rotator import GeminiKeyRotator, load_keys
    from llm_adapter import LLMAdapter, serialize_analysis
    for name in ("marketingiq.query_router", "marketingiq.data"):
        logging.getLogger(name).setLevel(logging.WARNING)
    provider_logs = []  # provider-reported timings/usage and rate-limit events
    adapter_logger = logging.getLogger("marketingiq.llm_adapter")
    handler = logging.Handler()

    def capture(record):
        msg = record.getMessage()
        if any(e in msg for e in ('"groq_generation"', '"groq_rate_limited"')):
            provider_logs.append(json.loads(msg))
    handler.emit = capture
    adapter_logger.addHandler(handler)

    dt.load_data(str(HERE / ".." / "data" / "tech_advertising_campaigns_dataset.csv"))

    # Build every question's analysis once (local processing), exactly as the router does.
    cases = []
    for q in QUESTIONS:
        t = time.perf_counter()
        r = qr.route_query(q)
        local_ms = (time.perf_counter() - t) * 1000
        assert r["route"] == "LLM_REQUIRED", (q, r["route"])
        llm_input = {"filters_applied": r["filters_applied"], "focus_metrics": r["focus_metrics"], "results": r["analysis"]}
        cases.append({"question": q, "tools": [a["tool"] for a in r["analysis"]], "local_analysis_ms": round(local_ms, 1),
                      "analysis_bytes": r["llm_context_bytes"], "llm_input": llm_input,
                      "analysis_sha256": hashlib.sha256(serialize_analysis(llm_input).encode()).hexdigest()[:16],
                      "supplied_numbers": analysis_numbers(llm_input)})

    results = {"questions": [{k: c[k] for k in ("question", "tools", "local_analysis_ms", "analysis_bytes", "analysis_sha256")}
                             for c in cases],
               "providers": {}}
    rotator = GeminiKeyRotator(load_keys())

    for name in [p.strip() for p in args.providers.split(",") if p.strip()]:
        provider = llm_providers.build_provider(name, rotator, "gemini-3.6-flash")
        adapter = LLMAdapter(provider, timeout_s=llm_providers.timeout_seconds(name))
        run = {"model": provider.model, "system_before": system_mem(), "answers": []}
        for i, c in enumerate(cases):
            if name in ("gemini", "groq") and i:
                time.sleep(args.gemini_pause if name == "gemini" else args.groq_pause)
            provider_logs.clear()
            t = time.perf_counter()
            out = adapter.explain(c["question"], c["llm_input"])
            provider_ms = (time.perf_counter() - t) * 1000
            text = out.get("text") or ""
            entry = {"question": c["question"], "status": out["status"], "error": out.get("error"),
                     "provider_ms": round(provider_ms, 1), "local_analysis_ms": c["local_analysis_ms"],
                     "words": len(text.split()), "chars": len(text), "text": text,
                     # grounding.py's verdict; when it failed, `text` is the data summary users would see.
                     "grounding": out.get("grounding"),
                     **check_numbers(text, c["supplied_numbers"])}
            for log in provider_logs:
                key = "groq_usage" if log["event"] == "groq_generation" else "rate_limit"
                entry[key] = log
            run["answers"].append(entry)
            print(f"[{name}] {i + 1}/{len(cases)} {out['status']:5} {provider_ms:8.0f} ms  "
                  f"words={entry['words']:3}  numbers={entry['numbers_cited']:2} unsupported={entry['unsupported']}  "
                  f"grounding={(entry['grounding'] or {}).get('action')}",
                  flush=True)
        run["system_after"] = system_mem()
        ok = [a["provider_ms"] for a in run["answers"] if a["status"] == "ok"]
        run["summary"] = {"success": f"{len(ok)}/{len(cases)}",
                          "provider_ms_median": round(statistics.median(ok), 1) if ok else None,
                          "provider_ms_min": round(min(ok), 1) if ok else None,
                          "provider_ms_max": round(max(ok), 1) if ok else None,
                          "words_median": statistics.median([a["words"] for a in run["answers"]]),
                          "numbers_cited": sum(a["numbers_cited"] for a in run["answers"]),
                          "numbers_supported": sum(a["supported"] for a in run["answers"]),
                          "numbers_unsupported": sum(len(a["unsupported"]) for a in run["answers"]),
                          "rate_limited": sum(a["error"] == "rate_limited" for a in run["answers"]),
                          "grounding_replaced": sum((a["grounding"] or {}).get("action") == "replaced_with_data_summary"
                                                    for a in run["answers"])}
        results["providers"][name] = run
        print(f"[{name}] summary: {run['summary']}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
