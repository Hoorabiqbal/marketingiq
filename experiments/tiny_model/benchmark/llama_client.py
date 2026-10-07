"""
Phase 6: minimal llama.cpp `llama-server` client shared by the export notebook's smoke test (Colab,
Linux) and benchmark_local.py (ThinkPad, Windows). Standard library only; psutil is optional (RAM).

Prompt: the exact Phase 4/5 prompt string. Phase 4 built it with
`apply_chat_template(plan_format.messages(q, prev), add_generation_prompt=True, enable_thinking=False)`
and checked it ends with the empty think block. render_prompt() writes the same Qwen3 ChatML text
directly (the export notebook asserts byte equality with the HF tokenizer for every benchmark row
and that llama.cpp tokenizes it to the same ids), and sends it to /completion as raw text. So the
GGUF's embedded chat template and llama.cpp's reasoning handling are not involved: thinking is OFF
because the prompt already contains `<think>\\n\\n</think>\\n\\n`, exactly as in training.

Decoding: greedy (temperature 0, top_k 1), stop at <|im_end|> (the GGUF EOS), n_predict = the
Phase 4 max_new_tokens.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
import plan_format as pf  # noqa: E402  (prompt contract; the backend is only loaded when validating)

EOS = "<|im_end|>"
THINK_OFF = "<think>\n\n</think>\n\n"


def render_prompt(question: str, previous: dict = None) -> str:
    """Qwen3 ChatML for plan_format.messages(), generation prompt, thinking OFF (Phase 4 format)."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}{EOS}\n" for m in pf.messages(question, previous)]
    return "".join(parts) + "<|im_start|>assistant\n" + THINK_OFF


def greedy_request(prompt: str, n_predict: int, cache_prompt: bool = True) -> dict:
    return {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
            "repeat_penalty": 1.0, "seed": 0, "cache_prompt": cache_prompt, "return_tokens": True}


class LlamaServer:
    """Starts llama-server on one GGUF, waits until it serves, stops it. Measures startup time."""

    def __init__(self, server_exe, model, threads=4, threads_batch=None, ctx=512, port=8089, log_path=None,
                 extra_args=()):
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [str(server_exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
                    "-c", str(ctx), "-t", str(threads), "-tb", str(threads_batch or threads),
                    "-np", "1",            # one request at a time: per-request latency, smallest KV cache
                    "-ngl", "0",           # CPU only (no layers offloaded even on a GPU build)
                    "--no-webui", *map(str, extra_args)]
        self.log_path = Path(log_path) if log_path else None
        self.proc = None
        self.startup_seconds = None

    def start(self, timeout=300):
        log = open(self.log_path, "w", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT)
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if get_json(self.url + "/health", timeout=2).get("status") == "ok":
                    self.startup_seconds = time.perf_counter() - t0
                    return self
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.05)
        self.stop()
        raise TimeoutError(f"llama-server not ready after {timeout}s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- endpoints
    def props(self) -> dict:
        return get_json(self.url + "/props")

    def tokenize(self, text: str) -> list:
        return post_json(self.url + "/tokenize", {"content": text, "add_special": False,
                                                  "parse_special": True})["tokens"]

    def apply_template(self, messages: list, **template_kwargs) -> str:
        body = {"messages": messages}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        return post_json(self.url + "/apply-template", body)["prompt"]

    def complete(self, prompt: str, n_predict: int, cache_prompt: bool = True) -> dict:
        """Streams one greedy completion. Returns the text, stop reason, server timings and client-side
        time to first token / end-to-end latency (perf_counter, seconds)."""
        body = json.dumps({**greedy_request(prompt, n_predict, cache_prompt), "stream": True}).encode()
        req = urllib.request.Request(self.url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        ttft, pieces, tokens, final = None, [], [], {}
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if ttft is None and (chunk.get("content") or chunk.get("tokens")):
                    ttft = time.perf_counter() - t0
                pieces.append(chunk.get("content", ""))
                tokens += chunk.get("tokens", [])
                if chunk.get("stop"):
                    final = chunk
                    break
        e2e = time.perf_counter() - t0
        timings = final.get("timings", {})
        stop_type = final.get("stop_type") or ("eos" if final.get("stopped_eos") else
                                               "limit" if final.get("stopped_limit") else "unknown")
        text = "".join(pieces)
        return {"text": text, "tokens": tokens, "stop_type": stop_type,
                "tokens_predicted": final.get("tokens_predicted", timings.get("predicted_n")),
                "tokens_evaluated": final.get("tokens_evaluated", timings.get("prompt_n")),
                "timings": timings, "ttft_s": ttft, "e2e_s": e2e}

    # ---- process memory (optional psutil)
    def memory(self) -> dict:
        try:
            import psutil
        except ImportError:
            return {}
        info = psutil.Process(self.proc.pid).memory_info()
        out = {"rss_mb": info.rss / 2**20}
        if hasattr(info, "peak_wset"):          # Windows: peak working set of the server process
            out["peak_working_set_mb"] = info.peak_wset / 2**20
        if hasattr(info, "private"):
            out["private_mb"] = info.private / 2**20
        return out


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def post_json(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def as_generation(result: dict) -> dict:
    """llama-server result -> the generation record evaluation/frozen_eval.score_example() scores
    (same fields Phase 5 recorded: raw_output, raw_with_special, finished, new_tokens)."""
    finished = result["stop_type"] == "eos"
    return {"raw_output": result["text"].strip(),
            "raw_with_special": result["text"] + (EOS if finished else ""),
            "finished": finished, "new_tokens": result["tokens_predicted"]}


def sha256_file(path, chunk=1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_questions(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def default_threads() -> int:
    """Physical cores if psutil knows them (Ryzen 5 3500U: 4), else half the logical CPUs."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 2) // 2)
    except ImportError:
        return max(1, (os.cpu_count() or 2) // 2)
