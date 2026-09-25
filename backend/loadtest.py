"""
HTTP load test for the MarketingIQ backend (real FastAPI app over real HTTP, no real LLM calls).

    python loadtest.py --mode baseline                    # sequential latency per request type
    python loadtest.py --mode direct                      # DIRECT_DATABASE mix at 1/5/10/15/20 users
    python loadtest.py --mode llm --mock-delay-ms 1500    # LLM_REQUIRED path with a mocked provider
    options: --levels 1,5,10 --requests 600 --cooldown 30 --soak-rounds 3 --cold-burst 40
             --duckdb-threads 1 --json out.json

It starts loadtest_server.py (the real app, with Groq calls counted and refused) as a
separate process and fires --cold-burst simultaneous requests the moment it is up (as after a
cold start). Then for each concurrency level it sends the SAME fixed workload (--requests
requests cycling through the same request list) from that many concurrent clients. Per level it
records throughput, latency percentiles, errors, and the server process's CPU and memory.

Every response is compared with a reference answer captured before the load, so a race
condition or corrupted shared state shows up as `wrong_answer`. After each level the server
must still answer /api/health. Provider call counters must stay 0 (mock calls excepted).

The load generator runs on the same machine as the server, so both compete for the same CPU.
Needs psutil (dev only: pip install psutil).
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil

HERE = Path(__file__).parent


def chat(q):
    return ("POST", "/api/chat", {"messages": [{"role": "user", "content": q}], "filters": {}})


def query(q, filters=None):
    return ("POST", "/api/query", {"query": q, **({"filters": filters} if filters else {})})


DIRECT_WORKLOAD = {
    "total revenue": query("What is total revenue?"),
    "total spend": query("What is total spend?"),
    "campaign count": query("How many campaigns are there?"),
    "platform ranking": query("Which platform has the highest ROAS?"),
    "monthly revenue": query("Show monthly revenue."),
    "filtered campaigns": query("Show campaigns with spend over 10000."),
    "ROAS threshold": query("Show campaigns with ROAS over 8."),
    "filtered total (dashboard filter)": query("What is total revenue?", {"platform": "TikTok"}),
    "chat: platform ranking": chat("Which platform has the highest ROAS?"),
    "health": ("GET", "/api/health", None),
}
LLM_WORKLOAD = {
    "why TikTok": query("Why is TikTok performing better than other platforms?"),
    "filtered subset": query("Why do campaigns with spend over 10000 have lower ROAS?"),
    "monthly trend": query("Explain what the monthly revenue trend means."),
    "creative age": query("Explain how creative age affects CTR."),
    "chat: device insights": chat("What insights can you draw about device performance?"),
}


def canonical(body):
    """The response without timing fields, for comparing answers across requests."""
    if isinstance(body, dict):
        return {k: canonical(v) for k, v in body.items() if not k.endswith("_ms")}
    if isinstance(body, list):
        return [canonical(v) for v in body]
    return body


def expected_route(mode, method, path):
    if path == "/api/health":
        return None
    return "DIRECT_DATABASE" if mode == "direct" else "LLM_REQUIRED"


def pct(sorted_values, p):
    if not sorted_values:
        return None
    return round(sorted_values[min(len(sorted_values) - 1, max(0, int(round(p / 100 * len(sorted_values))) - 1))], 2)


def latency_stats(values):
    v = sorted(values)
    if not v:
        return {}
    return {"min": round(v[0], 2), "median": round(statistics.median(v), 2), "p95": pct(v, 95), "p99": pct(v, 99),
            "max": round(v[-1], 2)}


class Sampler(threading.Thread):
    """Samples the server process's CPU (% of one core) and RSS while a level runs."""

    def __init__(self, proc, interval=0.25):
        super().__init__(daemon=True)
        self.proc, self.interval, self.samples, self._stop = proc, interval, [], threading.Event()
        self.proc.cpu_percent(None)

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self.samples.append((self.proc.cpu_percent(None), self.proc.memory_info().rss / 1e6))
            except psutil.Error:
                return

    def stop(self):
        self._stop.set()
        self.join()
        cpu = [c for c, _ in self.samples] or [0.0]
        rss = [r for _, r in self.samples] or [self.proc.memory_info().rss / 1e6]
        return {"sampled_cpu_pct_of_one_core_peak": round(max(cpu), 1),
                "rss_mb_avg": round(statistics.mean(rss), 1), "rss_mb_peak": round(max(rss), 1)}


class ExternalServer:
    """A server started separately: a local loadtest_server.py (several load generators sharing
    one server), or a deployed backend, where only client-side measurements are possible."""

    def __init__(self, url):
        self.base = url.rstrip("/")
        self.proc = None
        r = httpx.get(self.base + "/__loadtest/counters", timeout=90)
        if r.status_code == 200 and "pid" in r.json():
            self.proc = psutil.Process(r.json()["pid"])
        self.startup_s, self.startup_rss_mb = None, self.rss()

    def rss(self):
        return round(self.proc.memory_info().rss / 1e6, 1) if self.proc else None

    def counters(self):
        if not self.proc:
            return "not available (deployed server)"
        return httpx.get(self.base + "/__loadtest/counters", timeout=5).json()

    def log_problems(self):
        return "not collected (external server)"

    def stop(self):
        pass


class Server:
    def __init__(self, port, mock_delay_ms, log_dir, duckdb_threads=None):
        cmd = [sys.executable, str(HERE / "loadtest_server.py"), "--port", str(port)]
        if duckdb_threads:
            cmd += ["--duckdb-threads", str(duckdb_threads)]
        if mock_delay_ms is not None:
            cmd += ["--mock-llm-delay-ms", str(mock_delay_ms)]
        self.log_path = Path(log_dir) / f"server_{port}.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.base = f"http://127.0.0.1:{port}"
        started = time.perf_counter()
        self.popen = subprocess.Popen(cmd, cwd=HERE, stdout=self._log, stderr=subprocess.STDOUT)
        self.proc = psutil.Process(self.popen.pid)
        while True:
            if self.popen.poll() is not None:
                raise RuntimeError(f"server exited during startup; see {self.log_path}")
            try:
                if httpx.get(self.base + "/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - started > 120:
                raise RuntimeError("server did not start within 120 s")
            time.sleep(0.05)
        self.startup_s = round(time.perf_counter() - started, 2)
        # A Windows venv python.exe is a launcher that runs the real interpreter as a child
        # process: measure the process that actually serves requests.
        children = self.proc.children(recursive=True)
        if children:
            self.proc = max(children, key=lambda p: p.memory_info().rss)
        self.startup_rss_mb = self.rss()

    def rss(self):
        return round(self.proc.memory_info().rss / 1e6, 1)

    def counters(self):
        return httpx.get(self.base + "/__loadtest/counters", timeout=5).json()

    def log_problems(self):
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        keys = ("Traceback", "ERROR", "Exception", "duckdb query failed", "database is locked")
        return {k: text.count(k) for k in keys if text.count(k)}

    def stop(self):
        self.popen.terminate()
        try:
            self.popen.wait(10)
        except subprocess.TimeoutExpired:
            self.popen.kill()
        self._log.close()


def _cpu_s(proc):
    t = proc.cpu_times()
    return t.user + t.system


def send(client, spec):
    method, path, body = spec
    r = client.request(method, path, json=body) if body is not None else client.request(method, path)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None)


def cold_burst(server, workload, n):
    """n simultaneous requests the moment the server is up (as after a cold start)."""
    if n <= 0:
        return None
    specs = list(workload.values())
    barrier = threading.Barrier(n)

    def one(i):
        with httpx.Client(base_url=server.base, timeout=60) as c:
            barrier.wait()
            try:
                return send(c, specs[i % len(specs)])[0]
            except httpx.HTTPError as e:
                return type(e).__name__
    with ThreadPoolExecutor(n) as ex:
        statuses = list(ex.map(one, range(n)))
    return {"requests": n, "ok": statuses.count(200), "failed": [s for s in statuses if s != 200]}


def references(server, workload):
    refs = {}
    with httpx.Client(base_url=server.base, timeout=60) as c:
        for label, spec in workload.items():
            status, body = send(c, spec)
            if status != 200:
                raise RuntimeError(f"reference request '{label}' failed: {status} {body}")
            refs[label] = canonical(body)
    return refs


def run_level(server, mode, workload, refs, users, n_requests, timeout_s):
    labels = list(workload)
    plan = [labels[i % len(labels)] for i in range(n_requests)]  # identical at every level
    results = [None] * n_requests
    next_i, lock = [0], threading.Lock()

    def worker():
        with httpx.Client(base_url=server.base, timeout=timeout_s) as c:
            while True:
                with lock:
                    i = next_i[0]
                    next_i[0] += 1
                if i >= n_requests:
                    return
                label = plan[i]
                method, path, _ = workload[label]
                s = time.perf_counter()
                try:
                    status, body = send(c, workload[label])
                    ms = (time.perf_counter() - s) * 1000
                    if status >= 500:
                        outcome = "http_5xx"
                    elif status != 200:
                        outcome = f"http_{status}"
                    elif canonical(body) != refs[label]:
                        outcome = "wrong_answer"
                    elif expected_route(mode, method, path) and body.get("route") != expected_route(mode, method, path):
                        outcome = "wrong_route"
                    elif mode == "llm" and path == "/api/query" and body.get("llm", {}).get("status") != "ok":
                        outcome = "llm_not_ok"
                    else:
                        outcome = "ok"
                except httpx.TimeoutException:
                    ms, outcome = (time.perf_counter() - s) * 1000, "timeout"
                except httpx.HTTPError as e:
                    ms, outcome = (time.perf_counter() - s) * 1000, f"transport_{type(e).__name__}"
                results[i] = (label, ms, outcome)

    local = server.proc is not None  # a deployed server can't be sampled from here
    sampler = Sampler(server.proc) if local else None
    if sampler:
        sampler.start()
    me = psutil.Process()
    cpu0, client_cpu0 = (_cpu_s(server.proc) if local else 0.0), _cpu_s(me)
    psutil.cpu_percent(None)
    started = time.perf_counter()
    with ThreadPoolExecutor(users) as ex:
        for f in [ex.submit(worker) for _ in range(users)]:
            f.result()
    wall = time.perf_counter() - started
    server_cpu, client_cpu = (_cpu_s(server.proc) - cpu0 if local else None), _cpu_s(me) - client_cpu0
    machine_cpu = psutil.cpu_percent(None)
    res = sampler.stop() if sampler else {}
    ok = [ms for _, ms, o in results if o == "ok"]
    outcomes = {}
    for _, _, o in results:
        outcomes[o] = outcomes.get(o, 0) + 1
    # The server must still answer normally after the level.
    with httpx.Client(base_url=server.base, timeout=10) as c:
        status, body = send(c, workload[labels[0]])
        health_status, _ = send(c, ("GET", "/api/health", None))
    return {"users": users, "total_requests": n_requests, "successful": len(ok), "failed": n_requests - len(ok),
            "outcomes": outcomes, "wall_s": round(wall, 2), "requests_per_s": round(n_requests / wall, 1),
            "latency_ms": latency_stats(ok), **res,
            # Exact averages from CPU-time deltas (100 = one core fully busy); the sampled peak
            # above is coarse on Windows (15.6 ms clock ticks).
            "server_cpu_pct_of_one_core": round(server_cpu / wall * 100, 1) if local else None,
            "server_cpu_ms_per_request": round(server_cpu * 1000 / n_requests, 2) if local else None,
            "load_generator_cpu_pct_of_one_core": round(client_cpu / wall * 100, 1),
            "machine_cpu_pct_all_cores": machine_cpu,  # server + load generator + everything else
            "rss_mb_after": server.rss(),
            "healthy_after": health_status == 200 and status == 200 and canonical(body) == refs[labels[0]]}


def baseline(server, workload, reps):
    out = {}
    with httpx.Client(base_url=server.base, timeout=60) as c:
        for label, spec in workload.items():
            for _ in range(5):
                send(c, spec)
            times = []
            for _ in range(reps):
                s = time.perf_counter()
                status, _ = send(c, spec)
                times.append((time.perf_counter() - s) * 1000)
                assert status == 200, (label, status)
            out[label] = latency_stats(times)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "direct", "llm"], required=True)
    ap.add_argument("--levels", default="1,5,10,15,20")
    ap.add_argument("--requests", type=int, default=600, help="fixed workload per level")
    ap.add_argument("--reps", type=int, default=100, help="baseline: sequential requests per type")
    ap.add_argument("--mock-delay-ms", type=float, default=1500, help="llm mode: mocked provider delay")
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--cooldown", type=float, default=30, help="idle seconds before the final memory reading")
    ap.add_argument("--soak-rounds", type=int, default=0, help="repeat the highest level N more times (memory trend)")
    ap.add_argument("--cold-burst", type=int, default=40, help="simultaneous requests right after startup")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--url", help="use an already running loadtest_server.py instead of starting one")
    ap.add_argument("--duckdb-threads", type=int, help="override DuckDB's thread count in the server")
    ap.add_argument("--log-dir", default=tempfile.gettempdir())
    ap.add_argument("--json")
    args = ap.parse_args()

    llm = args.mode == "llm"
    workload = LLM_WORKLOAD if llm else DIRECT_WORKLOAD
    server = (ExternalServer(args.url) if args.url
              else Server(args.port, args.mock_delay_ms if llm else None, args.log_dir, args.duckdb_threads))
    report = {"mode": args.mode, "machine": {"cpu_logical": psutil.cpu_count(),
                                             "cpu_physical": psutil.cpu_count(logical=False),
                                             "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)},
              "server_startup_s": server.startup_s, "server_rss_at_startup_mb": server.startup_rss_mb,
              "workload": list(workload), "mock_delay_ms": args.mock_delay_ms if llm else None}
    try:
        report["cold_start_burst"] = cold_burst(server, workload, 0 if args.url else args.cold_burst)
        refs = references(server, workload)
        report["rss_after_warmup_mb"] = server.rss()
        if args.mode == "baseline":
            report["baseline"] = baseline(server, workload, args.reps)
        else:
            levels = [int(x) for x in args.levels.split(",")]
            report["requests_per_level"] = args.requests
            report["levels"] = []
            for users in levels + [levels[-1]] * args.soak_rounds:
                row = run_level(server, args.mode, workload, refs, users, args.requests, args.timeout)
                report["levels"].append(row)
                print(f"users={users:3}  req/s={row['requests_per_s']:7.1f}  ok={row['successful']}/{row['total_requests']}"
                      f"  median={row['latency_ms'].get('median')}  p95={row['latency_ms'].get('p95')}"
                      f"  p99={row['latency_ms'].get('p99')}  max={row['latency_ms'].get('max')}"
                      f"  srv_cpu%={row['server_cpu_pct_of_one_core']} ({row['server_cpu_ms_per_request']}ms/req)"
                      f"  client_cpu%={row['load_generator_cpu_pct_of_one_core']}  machine%={row['machine_cpu_pct_all_cores']}  rss_peak={row.get('rss_mb_peak')}  {row['outcomes']}",
                      flush=True)
            report["rss_immediately_after_mb"] = server.rss()
            time.sleep(args.cooldown)
            report["rss_after_cooldown_mb"] = server.rss()
            report["cooldown_s"] = args.cooldown
        report["provider_counters"] = server.counters()
        report["server_log_problems"] = server.log_problems()
    finally:
        server.stop()
    text = json.dumps(report, indent=1)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("levels", "baseline")}, indent=1))
    if args.mode == "baseline":
        for label, s in report["baseline"].items():
            print(f"{label:36} {s}")


if __name__ == "__main__":
    main()
