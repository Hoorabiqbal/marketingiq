"""
Gemini API key fallback/rotation.

Supports GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... (any number —
just keep adding GEMINI_API_KEY_<n> and this picks them up automatically with no
code changes). Falls back to a single GEMINI_API_KEY for backward compatibility
with a single-key deployment.

Behavior:
- Requests use the current "known-good" key by default — keys are NOT rotated
  on every request, only when the current key actually hits a quota error.
- On a quota/rate-limit error (429 / RESOURCE_EXHAUSTED), moves to the next key
  and retries, up to once per configured key (bounded — never infinite).
- On an authentication error, does NOT rotate — that key's credential itself is
  broken, which is a configuration problem to report, not something a different
  key fixes by coincidence.
- On any other client error (bad request, etc.), does not rotate — retrying with
  a different key wouldn't fix a malformed request either.
- Server errors (5xx, transient overload) are left to the caller to retry with
  backoff on the SAME key, since they're an infra issue, not a key-specific one.
- Remembers the last key that worked, so the NEXT request starts there instead
  of re-trying an already-exhausted key from the top every time.
"""
import os
import threading
from google import genai
from google.genai import errors as genai_errors


def load_keys():
    keys = []
    i = 1
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    if not keys:
        single = os.environ.get("GEMINI_API_KEY")
        if single:
            keys.append(single)
    return keys


def is_quota_error(e: genai_errors.ClientError) -> bool:
    msg = str(e).lower()
    return e.code == 429 or "resource_exhausted" in msg or "quota" in msg


def is_auth_error(e: genai_errors.ClientError) -> bool:
    msg = str(e).lower()
    return e.code in (401, 403) or "api key" in msg or "permission" in msg or "unauthenticated" in msg


class GeminiKeyRotator:
    def __init__(self, keys):
        self.keys = keys
        self.clients = [genai.Client(api_key=k) for k in keys]
        self._lock = threading.Lock()
        self._current = 0  # index of the last known-good key

    @property
    def configured(self):
        return len(self.clients) > 0

    def call(self, model, contents, config):
        """Returns (response, error_tag). error_tag is None on success, else one of:
        'no_key', 'auth:<detail>', 'quota_exhausted:<detail>', 'client:<detail>',
        'server:<detail>'. 'server' errors are NOT rotated/retried here — the
        caller (main.py) owns the backoff-and-retry policy for those."""
        if not self.configured:
            return None, "no_key"

        n = len(self.clients)
        with self._lock:
            start = self._current

        last_quota_error = None
        for attempt in range(n):
            idx = (start + attempt) % n
            try:
                response = self.clients[idx].models.generate_content(model=model, contents=contents, config=config)
                with self._lock:
                    self._current = idx  # remember this as the good key for next time
                return response, None
            except genai_errors.ClientError as e:
                if is_auth_error(e):
                    return None, f"auth:{e}"
                if is_quota_error(e):
                    last_quota_error = e
                    continue  # try the next key
                return None, f"client:{e}"  # malformed request etc. — rotating won't help
            except genai_errors.ServerError as e:
                return None, f"server:{e}"

        return None, f"quota_exhausted:{last_quota_error}"
