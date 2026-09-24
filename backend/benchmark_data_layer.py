"""
Benchmark the analytics tools (no LLM involved).

    python benchmark_data_layer.py                        # current data_tools, default backend
    python benchmark_data_layer.py --backend pandas       # force the Pandas reference backend
    python benchmark_data_layer.py --tools-dir <dir>      # benchmark another data_tools.py copy
    python benchmark_data_layer.py --json out.json        # also write raw results

Each operation is warmed up, then timed over --runs iterations (median and p95 in ms).
Memory is the process RSS after loading the data (Windows/Linux), plus the
Python-heap peak while loading (tracemalloc; excludes DuckDB's native memory).
"""
import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE / ".." / "data" / "tech_advertising_campaigns_dataset.csv"

OPERATIONS = [
    ("total revenue", "get_totals", {}),
    ("total spend (filtered: platform=TikTok)", "get_totals", {"filters": {"platform": "TikTok"}}),
    ("campaign count (multi-filter)", "get_totals",
     {"filters": {"platform": "Facebook", "budget": "High", "retargeting": "Retargeting Only", "device": "Mobile"}}),
    ("platform ranking by ROAS", "rank_dimension", {"dimension": "platform", "metric": "roas", "limit": 10}),
    ("monthly revenue", "trend_over_time", {"metric": "revenue"}),
    ("filtered campaigns (spend>4996, revenue<665)", "filter_campaigns",
     {"conditions": [{"field": "spend", "operator": ">", "value": 4996.16},
                     {"field": "revenue", "operator": "<", "value": 665}], "limit": 10}),
    ("ROAS threshold (roas>8)", "filter_campaigns",
     {"conditions": [{"field": "roas", "operator": ">", "value": 8}], "sort_by": "roas", "order": "desc", "limit": 10}),
    ("compare 3 platforms", "compare_entities", {"dimension": "platform", "names": ["TikTok", "LinkedIn", "Facebook"]}),
    ("numeric stats (spend, revenue, roas)", "get_numeric_field_stats", {"fields": ["spend", "revenue", "roas"]}),
    ("creative fatigue", "get_creative_fatigue", {}),
    ("list available fields", "list_available_fields", {}),
]


def rss_mb():
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.WorkingSetSize / 1e6, 1)
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1e3, 1)
    except Exception:
        return None


def shape(result):
    if isinstance(result, dict):
        parts = []
        for k, v in result.items():
            if isinstance(v, list):
                parts.append(f"{k}[{len(v)}]")
            elif isinstance(v, dict):
                parts.append(f"{k}{{{len(v)}}}")
        return f"dict({len(result)} keys" + (", " + ", ".join(parts) if parts else "") + ")"
    return type(result).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-dir", default=str(HERE))
    ap.add_argument("--backend", choices=["duckdb", "pandas"])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.backend:
        os.environ["MARKETINGIQ_DATA_BACKEND"] = args.backend
    sys.path.insert(0, args.tools_dir)
    rss_before_import = rss_mb()
    import data_tools as dt

    tracemalloc.start()
    t0 = time.perf_counter()
    dt.load_data(str(CSV))
    load_ms = (time.perf_counter() - t0) * 1000
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    backend = getattr(dt, "data_backend_name", lambda: "pandas (original)")()
    out = {"backend": backend, "load_ms": round(load_ms, 1), "rss_before_import_mb": rss_before_import,
           "rss_after_load_mb": rss_mb(), "load_python_heap_peak_mb": round(heap_peak / 1e6, 1), "ops": []}

    for label, tool, kwargs in OPERATIONS:
        fn = dt.TOOL_REGISTRY[tool]
        for _ in range(5):
            result = fn(**kwargs)
        times = []
        for _ in range(args.runs):
            t = time.perf_counter()
            fn(**kwargs)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        out["ops"].append({"operation": label, "tool": tool, "median_ms": round(statistics.median(times), 3),
                           "p95_ms": round(times[int(len(times) * 0.95) - 1], 3), "shape": shape(result)})
    out["rss_after_benchmark_mb"] = rss_mb()

    print(f"backend: {out['backend']}   load: {out['load_ms']} ms   RSS after load: {out['rss_after_load_mb']} MB   "
          f"(before import: {out['rss_before_import_mb']} MB)   Python heap peak during load: {out['load_python_heap_peak_mb']} MB")
    print(f"{'operation':48} {'median ms':>10} {'p95 ms':>9}  shape")
    for op in out["ops"]:
        print(f"{op['operation']:48} {op['median_ms']:>10} {op['p95_ms']:>9}  {op['shape']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
