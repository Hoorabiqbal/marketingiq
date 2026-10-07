"""
Test double for the local llama-server, shared by the test modules.

A small HTTP server on 127.0.0.1 answers /completion the way llama-server does, with a scripted
compact plan per question, so the real LocalPlanner client runs unchanged (HTTP, timeouts, parsing)
and no GGUF is needed. Importing this module first also points the app's model URL at a port where
nothing listens, so a test can never reach a real model that happens to be running.

    import fake_llama            # first: before main.py creates its planner
    import main
    with fake_llama.Model({"total revenue": {"intent": "total", "metrics": ["revenue"]}}) as m:
        ...                      # analyst.planner now talks to the fake; m.calls counts model calls
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["MARKETINGIQ_LLAMA_URL"] = "http://127.0.0.1:9"   # discard port: nothing answers
os.environ["MARKETINGIQ_LLAMA_TIMEOUT_S"] = "2"


class FakeLlama:
    """/completion like llama-server. `respond(question, previous_plan)` returns the compact plan (dict ->
    JSON), raw text, or (text, stop_type); raising drops the connection. Records every request body."""

    def __init__(self, respond, delay_s=0.0):
        self.respond, self.delay_s, self.requests = respond, delay_s, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                time.sleep(fake.delay_s)
                question, previous = fake.parse(body["prompt"])
                out = fake.respond(question, previous)
                text, stop = out if isinstance(out, tuple) else (out, "eos")
                text = json.dumps(text, separators=(",", ":")) if isinstance(text, dict) else text
                self._send({"content": text, "stop_type": stop, "tokens_predicted": 20,
                            "timings": {"prompt_n": 30, "predicted_n": 20}})

            def do_GET(self):
                self._send({"status": "ok"})

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def parse(prompt: str):
        """(question, previous sparse plan or None) from the ChatML training prompt."""
        user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        if user.startswith("Previous analysis in this conversation"):
            head, question = user.split("\nQuestion: ", 1)
            return question, json.loads(head.split(": ", 1)[1])
        return user, None

    @property
    def calls(self):
        return len(self.requests)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def refuse(question, previous):
    raise AssertionError(f"the model must not be called for {question!r}")


def scripted(plans):
    """respond() from {question: plan or callable(previous) -> plan}; unknown questions fail loudly."""
    def respond(question, previous):
        if question not in plans:
            raise AssertionError(f"no scripted plan for {question!r}")
        p = plans[question]
        return p(previous) if callable(p) else p
    return respond


class Model:
    """The app's planner pointed at a FakeLlama for the duration of a `with` block."""

    def __init__(self, respond=refuse, delay_s=0.0, timeout_s=None, url=None):
        import analyst
        import local_planner
        if isinstance(respond, dict):
            respond = scripted(respond)
        self.analyst, self.fake = analyst, FakeLlama(respond, delay_s)
        self.planner = local_planner.LocalPlanner(url or self.fake.url, timeout_s or local_planner.DEFAULT_TIMEOUT_S)

    def __enter__(self):
        self.saved, self.analyst.planner = self.analyst.planner, self.planner
        return self

    def __exit__(self, *exc):
        self.analyst.planner = self.saved
        self.fake.close()

    @property
    def calls(self):
        return self.fake.calls

    @property
    def prompts(self):
        return [r["prompt"] for r in self.fake.requests]
