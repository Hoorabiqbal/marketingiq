# MarketingIQ AI Analyst — Backend

A small FastAPI service that gives the AI Analyst chat real, tool-grounded access to the
full MarketingIQ dataset via **Google Gemini's** tool-use (function calling) — using
Gemini's genuine, ongoing, **free tier** (no credit card, no billing setup required).

## Why Gemini instead of Claude/OpenAI

Anthropic's and OpenAI's APIs both require billing to be enabled for any real usage —
neither has an ongoing free tier (Anthropic gives a small one-time trial credit for new
accounts, but nothing beyond that). Google's Gemini API is the one major-provider option
with a genuine, no-card-required free tier (Flash-class models, ~1,500 requests/day) that
also supports the same tool-use/function-calling pattern this project needs. Nothing here
uses an unofficial, shared, or bypassed key — it's Google's own free tier, used as intended.

## How it works

```
Your question + chat history + active dashboard filters
        │
        ▼
Gemini (gemini-2.5-flash) — decides which tool(s) answer your question
        │
        ▼
FastAPI executes that tool against the real pandas dataframe
(the same dashboard filters you have active are applied automatically)
        │
        ▼
Real numbers go back to Gemini
        │
        ▼
Gemini writes the final answer using ONLY those real numbers
```

Gemini is never given the raw dataset and never answers from memory — every number in
every answer comes from one of the tool functions in `data_tools.py` (completely
unchanged from the original design — only the LLM-calling code in `main.py` differs),
which query the actual CSV with pandas.

## Setup

1. **Install dependencies** (from inside this `backend/` folder, with your venv activated):
   ```
   pip install -r requirements.txt
   ```

2. **Get one or more free API keys** — no credit card, no billing setup:
   - Go to https://aistudio.google.com/apikey (repeat with a different Google account for
     additional keys, if you want automatic fallback — see below)
   - Sign in with any Google account
   - Click "Create API key" — that's it, no payment method requested

3. **Add your key(s)**:
   ```
   copy .env.example .env
   ```
   Open `.env` and either set `GEMINI_API_KEY=` for a single key, or
   `GEMINI_API_KEY_1=`, `GEMINI_API_KEY_2=`, `GEMINI_API_KEY_3=`, ... for automatic
   fallback across multiple keys (see "Automatic key fallback" below).
   **Never commit `.env`** — it's already in `.gitignore`.

4. **Run the server**:
   ```
   uvicorn main:app --reload --port 8000
   ```

5. **Verify it's working** — open http://localhost:8000/api/health. You should see:
   ```
   {"status":"ok","campaigns_loaded":10000,"gemini_key_configured":true,"gemini_keys_configured":1}
   ```
   If `gemini_key_configured` is `false`, the `.env` file isn't being found or is empty —
   double check it's named exactly `.env` (not `.env.txt`) and sits directly in this folder.

6. **Open the dashboard** (`site/index.html`, served locally — see the main project docs
   for why `file://` doesn't work for the AI Analyst) and try the AI Analyst tab.

## Automatic key fallback

If you're doing heavy testing (or expect real usage) and don't want a single free-tier
quota to be a bottleneck, set `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, etc.
instead of a single `GEMINI_API_KEY`. Behavior (implemented in `gemini_rotator.py`):

- Requests use one key normally — **keys are not rotated on every request**, only when
  the current one actually hits its quota.
- On a quota/rate-limit error (429), the backend automatically retries with the next
  configured key. It remembers which key last worked, so future requests start there
  directly rather than re-trying an already-exhausted key every time.
- An authentication error (a genuinely invalid key) does **not** trigger rotation — that's
  a configuration problem reported directly, since a different key wouldn't fix a typo'd one.
- If every configured key is out of quota, you get one clean message rather than a raw error.
- Add as many `GEMINI_API_KEY_<n>` as you want — no code changes needed.

## Free tier limits to know

- ~1,500 requests/day, ~15 requests/minute (exact limits are set per-project by Google and
  can change — check your live quota at https://aistudio.google.com)
- If you hit the limit, the chat will show a clear rate-limit message rather than crash —
  just wait a bit and try again
- Google's terms allow using free-tier prompts to improve their models. If you're testing
  with anything sensitive, keep that in mind (this project's dataset is synthetic/public,
  so it's a non-issue here)

## Testing without an API key

`test_app.py` exercises the entire tool-use loop (routing, filter injection, hallucination
guarding, multi-tool chains, the infinite-loop safety cap, and the "no key configured"
path) using **mocked** Gemini responses built from the real SDK's own types, so you can
verify the plumbing works before you have a real key:
```
python test_app.py
```

## How `/api/chat` answers (Query Router first, Gemini tool-use loop as fallback)

The chat API is unchanged for the dashboard: it still takes `{messages, filters}` and
returns `{answer}`, now with an extra `route` field. The latest question goes through
`chat_routing.py`:

| Route | When | LLM calls |
|---|---|---|
| `DIRECT_DATABASE` | The router maps the question to one data tool ("total revenue", "highest ROAS by platform", "monthly revenue") | **0**. The answer is built from the DuckDB result |
| `LLM_REQUIRED` | "why" / "explain" questions | **1** explanation through the LLM Adapter, using the compact analysis |
| `UNSUPPORTED` | Asks for data the dataset doesn't have (e.g. countries) | 0 |
| `FALLBACK` | Everything the router can't plan confidently: judgement calls ("high spend", "underperforming"), relative dates ("last month"), follow-ups that depend on earlier messages ("what about TikTok?"), small talk | The original Gemini tool-use loop |

A request never goes down two LLM paths: once a question is answered or explained on the
routed path, it never reaches the fallback. Set `CHAT_ROUTER_ENABLED=0` to send every
chat request straight to the original tool-use loop, as a rollback switch.

**Conversation history:** the router only looks at the latest question. When there is
earlier conversation and the question leans on it (pronouns like "it"/"that", or openers
like "and…" / "what about…"), it goes to the fallback, which receives the full history.
There is no separate conversation memory.

**Who owns retries and timeouts** (so one failure is retried in exactly one place):

| Concern | Owner |
|---|---|
| Quota errors: switch to the next API key | `GeminiKeyRotator` |
| Per-request HTTP timeout and overall deadline | `call_gemini()` in `gemini_provider.py` |
| Transient 5xx / network errors: at most 3 attempts, with backoff, within the deadline | `call_gemini()` |
| Turning an error into a user message | `LLMAdapter` (explanations), or `_message_for_error` (fallback) |

The SDK's own retries are off. The endpoint, adapter and router add none. With
`LLM_PROVIDER=qwen`, each explanation is one Ollama request within the deadline, with no
retries: retrying a slow local model would only load the machine further. Deadlines are
`LLM_TIMEOUT_SECONDS` (default 25) for an explanation and
`CHAT_FALLBACK_TIMEOUT_SECONDS` (default 45) for the whole fallback loop.

## Data layer (DuckDB)

```
Query Router / Gemini tool calls
        │  (known tools only — no free-form SQL)
        ▼
data_tools.py          business logic: metric formulas, what each dashboard filter means,
        │              ranking and rounding (unchanged names, inputs and outputs)
        ▼
campaign_repository.py data access: generic filter / aggregate / group operations
        │
        ▼
DuckDB (in-memory)     primary engine  ·  Pandas: automatic fallback + correctness reference
```

- **Source of truth is still the CSV.** At startup it is read once with Pandas, and an
  in-memory DuckDB table is built from that same parsed data. There is no database file.
  `/api/health` reports the active `data_backend`. Set `MARKETINGIQ_DATA_BACKEND=pandas` to
  use the Pandas backend instead; the app also falls back to it on its own (with a log
  line) if DuckDB can't start.
- **All SQL lives in `campaign_repository.py`.** Column names are checked against the
  table's real schema, operators against a fixed list, and every value is a bound `?`
  parameter. The router, the LLM Adapter and the endpoints never build SQL, and there is
  no text-to-SQL. Questions reach the data only through the fixed set of tools, which keeps
  the data layer's behaviour predictable, testable and safe.
- **The LLM still never receives the dataset.** Tools return small aggregates, and the
  router and adapter cap what is passed on (see below).
- **Correctness:** `test_data_layer.py` checks both backends against
  `data_layer_golden.json`, which holds the original Pandas implementation's output for 51
  cases.
- **Speed:** `benchmark_data_layer.py` measures each tool on either backend.

## Query Router (`/api/query`)

`query_router.py` is a deterministic routing layer that decides where a question goes
**without calling any LLM** (keyword/regex rules, no API call to classify). It runs
alongside `/api/chat`, which is unchanged and still uses its own Gemini tool-use loop.

```
POST /api/query   {"query": "Which platform has the highest ROAS?", "filters": {...}}
```

| Route | When | What comes back |
|---|---|---|
| `DIRECT_DATABASE` | The question maps to one existing tool ("total revenue", "highest ROAS by platform", "monthly revenue", "campaigns with ROAS above 8") | `tool`, `tool_input` and the exact `result` from that tool |
| `LLM_REQUIRED` | The question asks for an explanation ("why", "explain", "what's causing") | `analysis` (a few small tool results), `explanation` (text from the LLM Adapter, or `null` if it failed) and `llm` (status, provider, error code) |
| `NEEDS_CLARIFICATION` / `UNSUPPORTED` | Ambiguous, or asks for something the dataset doesn't have (e.g. geography) | A `message` explaining what can be asked |

The router is not a second analytics engine: every number comes from an existing
`data_tools.py` function (`TOOL_REGISTRY`), with the active dashboard filters applied
exactly as in `/api/chat`.

**Why the LLM never gets the full dataset:** for `LLM_REQUIRED` the router runs at most 3
tools and caps each list (at most 10 campaign rows and 30 series points, 16 KB total), so
the context stays the same size however large the dataset gets. It's also cheaper and
faster, and it keeps the LLM limited to aggregates that were actually computed.

Errors return a safe message with no stack trace: `400` for an empty or invalid query,
`422` for a malformed body, `503` if the data or a tool isn't available, and `500` if a
tool fails. Each request writes one structured log line (query, route, tools,
elapsed_ms, status).

### LLM providers (Gemini or local Qwen)

Explanations for `LLM_REQUIRED` questions come from **one** provider, chosen at startup:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` (Google API) or `qwen` (local model through Ollama). Any other value stops startup |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama listens. `host:port` or a full URL |
| `OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` | The Ollama model tag |
| `LLM_TIMEOUT_SECONDS` | 25 (gemini) / 90 (qwen) | Deadline per explanation. Local CPU inference, including a cold model load, needs longer |

To use Qwen: install [Ollama](https://ollama.com), run `ollama pull qwen2.5:1.5b-instruct`
once, keep Ollama running, and start the backend with `LLM_PROVIDER=qwen`.
`/api/health` shows the active `llm_provider` and `llm_model`.

- Both providers get the **same input**: the question plus the same compact analysis
  (`build_user_prompt`). Qwen gets a shorter system prompt with the same grounding rules.
- There is **no automatic failover.** If the selected provider fails, the answer says so.
  `DIRECT_DATABASE` questions never use either provider. The `/api/chat` fallback for
  unresolved questions is always the Gemini tool-use loop, whichever provider is selected.
- **Local inference limits:** Qwen runs on the CPU here (no supported GPU). The first
  request after the model unloads (Ollama's default is 5 minutes idle) pays the model load
  time. Requests compete for the same CPU, so throughput under several simultaneous users
  still needs its own load test.
- `benchmark_llm_providers.py` runs the same 10 questions through each provider and reports
  latency, grounding checks and resource use.

### LLM Adapter

`LLM_REQUIRED` answers are explained through `llm_adapter.py`: a provider-independent
`LLMProvider` interface (`generate_explanation(question, analysis, timeout_s)`) plus an
`LLMAdapter` that validates the analysis before anything is sent. It refuses anything that
isn't plain JSON (such as a DataFrame), anything over 16 KB, and empty results. It also
turns every provider failure into a structured `llm` result, so the analysis is still
returned when the LLM is down.

Gemini is the only provider so far (`gemini_provider.py`). It reuses the app's existing
`GeminiKeyRotator`, so it has the same keys and quota rotation as `/api/chat`. It makes one
call per question, with shared grounding instructions: use only the supplied numbers, keep
facts and interpretation separate, and stay short. Each call has a hard timeout
(`LLM_TIMEOUT_SECONDS`, default 25) that also bounds the server-error retries.
`DIRECT_DATABASE` questions never reach the adapter.

To add a provider, subclass `LLMProvider`, raise the `llm_adapter` error types, and pass it
to `LLMAdapter` in `main.py`.

Tests (no API key or quota needed, since Gemini is mocked and fails if called for real):
```
python test_query_router.py
python test_llm_adapter.py
python test_data_layer.py
python test_chat_migration.py
python test_qwen_provider.py
```

## Adding a new askable dimension or metric

You do **not** need to add a new question/answer branch. Add one line to
`DIMENSION_COLUMNS` in `data_tools.py` for a new dimension, or extend the `metric` enum
in `main.py`'s tool definitions for a new metric — every existing tool (`rank_dimension`,
`compare_entities`, etc.) picks it up automatically.

## Before deploying publicly

- Change `allow_origins=["*"]` in `main.py` to your actual deployed frontend URL.
- Set `GEMINI_API_KEY` (or `GEMINI_API_KEY_1`, `_2`, `_3`, ...) as environment
  variables/secrets on whatever host you use (Render, Railway, Fly.io, etc.) — in Render
  specifically, this is the "Environment Variables" section when creating/editing the Web
  Service — never bake keys into the code or a committed file.
- Update `AI_BACKEND_URL` in `site/index.html` (near the top of the AI Analyst script
  section) to point at your deployed backend URL instead of `127.0.0.1:8000`.
- If the project grows beyond the free tier's request volume, Gemini's paid tier (or
  switching back to a paid Claude/OpenAI key using the same tool-use pattern) is a
  drop-in upgrade — the `data_tools.py` layer doesn't change either way.
