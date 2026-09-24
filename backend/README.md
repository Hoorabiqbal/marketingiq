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
`LLM_PROVIDER=groq`, each explanation gets at most one retry on a 5xx or network error,
within the deadline, and a 429 is never retried. Deadlines are
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
  The parsed DataFrame is then released, so only DuckDB's copy stays in memory (with the
  Pandas backend the DataFrame is kept, since it is the data). `dt.get_dataframe()` is for
  tests and offline tools: with DuckDB it re-reads the CSV on each call.
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

### Performance and load testing

Dev-only tools (`pip install -r requirements-dev.txt`, which adds `psutil`). No real Gemini or
Groq call is ever made.

```
python perf_profile.py                        # startup phases, memory inventory, DuckDB ops,
                                              # endpoint time split by stage (in-process)
python perf_profile.py --threads 1,2,4,8      # DuckDB thread settings, interleaved rounds
python loadtest.py --mode baseline            # sequential latency per request type, over HTTP
python loadtest.py --mode direct              # DIRECT_DATABASE mix at 1/5/10/15/20 users
python loadtest.py --mode llm --mock-delay-ms 1500 --requests 200   # LLM path, mocked provider
```

`loadtest.py` starts `loadtest_server.py`: the real app under uvicorn, with Gemini and Groq
calls counted and refused, and optionally a fixed-delay mock provider. Each level sends the
same fixed workload. Every response is compared with a reference answer taken before the load
(so races or corrupted shared state show up as `wrong_answer`), the server must stay healthy
after each level, and the provider counters must stay at 0. `--duckdb-threads N` overrides
DuckDB's thread count so the setting can be re-measured on the deployment host.

Measured on a 4-core laptop (server and load generator on the same machine; the laptop's speed
varied by up to ~50% between sessions, so compare only interleaved runs):
- Memory: about 150–160 MB RSS after startup, mostly libraries (pandas + DuckDB ~65 MB, FastAPI
  / google-genai / httpx ~42 MB). The dataset is ~12 MB in DuckDB. Under sustained load RSS
  settles around 180–235 MB; the Python heap and DuckDB's own memory stay flat, and the growth
  slows over time (native allocator behaviour, not a Python leak).
- `DIRECT_DATABASE`: 3–6 ms server time, 6–15 ms over HTTP; about 200–390 req/s at 5–20
  concurrent users with no errors.
- `LLM_REQUIRED` with a 1.5 s mock provider: about 15–35 ms of local work on top of the
  provider time. Each explanation holds one of the server's 40 worker threads while the provider
  answers, so real capacity for explanations is set by the provider's latency and quota.
- DuckDB's thread setting (1, 2, 4, 8) made no consistent difference at this data size, so the
  default is kept.
- The router's entity index is built once at startup (`qr.warm_up()`) and under a lock. Before
  that, simultaneous first requests after a cold start each ran the profiling query and could
  exhaust DuckDB's 256 MB limit (HTTP 500s).

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

### LLM providers (Gemini by default, Groq optional)

Explanations for `LLM_REQUIRED` questions come from **one** provider, chosen at startup:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` (Google API) or `groq` (Groq API). Any other value stops startup |
| `GROQ_API_KEY` | none | Groq API key (console.groq.com). Put it in `.env`, never in code. Without it, `groq` still starts but explanations report `not_configured` |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model ID. Which models are available depends on the Groq account (check `GET /openai/v1/models`). Reasoning models (`openai/gpt-oss-*`) run with `reasoning_effort=low`, reasoning hidden |
| `LLM_TIMEOUT_SECONDS` | 25 | Deadline per explanation |

`/api/health` shows the active `llm_provider` and `llm_model` (never a key).

- Both providers get the **same input**: the question plus the same compact analysis
  (`build_user_prompt`) and the same system prompt (`GROUNDING_INSTRUCTIONS`).
- Groq: one request per explanation within the deadline, plus one retry on a 5xx or network
  error. A 429 rate limit is reported, never retried. The key is only sent in the
  Authorization header and is never logged. Free-tier token limits per minute can be low
  (8K tokens/min was seen for `openai/gpt-oss-120b`, about 5 explanations a minute).
- There is **no automatic failover.** If the selected provider fails, the answer says so.
  `DIRECT_DATABASE` questions never use any provider. The `/api/chat` fallback for
  unresolved questions is always the Gemini tool-use loop, whichever provider is selected.
- `benchmark_llm_providers.py` runs the same 10 questions through each provider and reports
  latency and grounding checks. Past results, including a local Qwen model through Ollama
  that was evaluated and then removed (too slow on CPU, not grounded enough), are in
  `benchmark_results/`.

### LLM Adapter

`LLM_REQUIRED` answers are explained through `llm_adapter.py`: a provider-independent
`LLMProvider` interface (`generate_explanation(question, analysis, timeout_s)`) plus an
`LLMAdapter` that validates the analysis before anything is sent. It refuses anything that
isn't plain JSON (such as a DataFrame), anything over 16 KB, and empty results. It also
turns every provider failure into a structured `llm` result, so the analysis is still
returned when the LLM is down.

There are two providers: `gemini_provider.py` (default) and `groq_provider.py` (optional).
The Gemini provider reuses the app's existing
`GeminiKeyRotator`, so it has the same keys and quota rotation as `/api/chat`. It makes one
call per question, with shared grounding instructions: use only the supplied numbers, keep
facts and interpretation separate, and stay short. Each call has a hard timeout
(`LLM_TIMEOUT_SECONDS`, default 25) that also bounds the server-error retries.
`DIRECT_DATABASE` questions never reach the adapter.

To add a provider, subclass `LLMProvider`, raise the `llm_adapter` error types, and pass it
to `LLMAdapter` in `main.py`.

### Grounding (`grounding.py`)

Every provider gets the same **typed context**, and every provider answer goes through the
same **numerical-claim validator**. Both run in the adapter, so Gemini and Groq are treated
identically and `DIRECT_DATABASE` never touches either.

**Typed context** (built from the router's analysis. It describes the metrics and never
recomputes one):
- `metric_units`: `$` money, `x` ratio (ROAS: revenue / spend, never a percentage), `%`
  already a percentage (CTR and conversion rate are ×100 in the data: 2.16 means 2.16%), and
  `count`. The dataset has no currency column; `$` follows the dashboard, and no currency
  code is claimed.
- `sections`, one per tool: `aggregate` (all campaigns in scope), `comparison` (dimension,
  ranking metric, order, entity rows), `time_series` (metric, unit, month granularity, entity
  scope), `creative_age_buckets` (unit **days**, from the `creative_age_days` column: the age
  of the ad creative, not audience age), and `filtered_subset` (conditions with units, the
  matched count, `aggregates` for the whole matching group, and separately labelled
  `examples`).
- `active_filters` only lists filters that narrow the data (the dashboard's "All Platforms"
  placeholders are dropped).
- `analysis_scope` states what the data supports and what it doesn't. Causes are never
  supported. It also names a change over time for a metric with no time series (e.g. CTR).

**Validator**: every number in the answer must match a supplied value, rounded or truncated
at the precision written (`$284.16M`, `284.2 million`, `284,157,706`, `10 000` with narrow
spaces), with a compatible unit. Detected: `unsupported_value`, `unit_mismatch` (ROAS 6.54x
written as 6.54%, creative-age days as years), `metric_label_mismatch` (a CPA value called
CTR, revenue called spend), `derived_value` ("3.4 times higher", "20% lower"),
`example_as_aggregate`, `scope_mismatch` (an overall figure attributed to a filtered
subset), `entity_mismatch`, `month_mismatch` and `unsupported_date`. List numbering (including inline "1) … 2) …"), "3
observations", "step 2" and "top 3" (within the supplied ranking) are ignored.

**When validation fails**, the answer is never shown. The adapter returns a deterministic
summary built from the context ("What the data shows: …") plus a note that the explanation
was withheld. There is no second LLM call. An answer whose numbers pass but that states a
cause without hedging ("because", "drives") is kept, with a caveat that the data shows what
happened, not why. The `llm.grounding` field reports `passed`, `action` (`none` /
`caveat_added` / `replaced_with_data_summary`), issue categories and `local_ms`. A
`grounding_check` log line records the provider, the categories and the offending numbers,
never the question, the answer text or any key.

**Limits**: numbers written as words ("three times") aren't checked. A comparison between two
supported values ("A is higher than B") isn't checked. Bounds such as "ROAS below 1" or
"over $6" count as unsupported unless that number is in the data. The causal check is a
keyword heuristic. The `/api/chat` fallback (Gemini tool-use loop) is not validated.

Tests (no API key or quota needed, since Gemini is mocked and fails if called for real):
```
python test_query_router.py
python test_llm_adapter.py
python test_data_layer.py
python test_chat_migration.py
python test_groq_provider.py
python test_grounding.py
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
