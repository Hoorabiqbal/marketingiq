# MarketingIQ AI Analyst — Backend

A small FastAPI service behind the dashboard's AI Analyst. Every question goes through a
deterministic **Query Router** and is answered from the real MarketingIQ dataset in
**DuckDB**. Only "why / explain" questions use an LLM — **Groq**, the only provider — and
only to explain a compact analysis the router computed. Every LLM answer is checked by a
deterministic **grounding validator** before anyone sees it.

## How it works

```
Your question (+ active dashboard filters)
        │
        ▼
Query Router (query_router.py) — keyword/regex rules, no LLM
        ├── DIRECT_DATABASE ──► DuckDB ──► exact answer (0 LLM calls)
        ├── LLM_REQUIRED ─────► DuckDB ──► compact typed analysis ──► Groq
        │                                   ──► grounding validator ──► safe answer
        └── UNSUPPORTED / NEEDS_CLARIFICATION ──► the router's message (0 LLM calls)
```

The LLM never chooses tools, never sees the dataset and never answers from memory: it only
explains the figures it is handed, and an answer with a number the data doesn't support is
replaced by a deterministic summary of the data.

## Setup

1. **Install dependencies** (from inside this `backend/` folder, with your venv activated):
   ```
   pip install -r requirements.txt
   ```

2. **Get a Groq API key** at https://console.groq.com/keys.

3. **Add it**:
   ```
   copy .env.example .env
   ```
   Open `.env` and set `GROQ_API_KEY=`. **Never commit `.env`** — it's already in
   `.gitignore`. Without a key the app still starts and answers database questions;
   explanations report the AI service as unavailable.

4. **Run the server**:
   ```
   uvicorn main:app --reload --port 8000
   ```

5. **Verify it's working** — open http://localhost:8000/api/health. You should see:
   ```
   {"status":"ok","campaigns_loaded":10000,"data_backend":"duckdb","llm_provider":"groq",
    "llm_model":"openai/gpt-oss-120b","llm_configured":true}
   ```
   If `llm_configured` is `false`, the `.env` file isn't being found or has no key —
   double check it's named exactly `.env` (not `.env.txt`) and sits directly in this folder.

6. **Open the dashboard**, served locally (`python -m http.server 5500 --bind 127.0.0.1
   --directory site` from the project root, then http://127.0.0.1:5500/dashboard.html).
   When served from localhost / 127.0.0.1 it calls this local backend.

## Groq limits to know

On Groq's free plan, `openai/gpt-oss-120b` and `openai/gpt-oss-20b` both allow 30
requests/minute, 1,000/day, **8,000 tokens/minute** and 200,000 tokens/day (check your
live limits at https://console.groq.com/settings/limits). One explanation uses roughly
2,000 tokens (prompt plus the model's hidden reasoning), so the token limit allows only a
few explanations per minute for the whole app. When it is reached, users see the figures
with a "temporarily rate-limited" notice; the request is not retried. Database questions
never use Groq and are unaffected.

## How `/api/chat` answers

The dashboard sends `{messages, filters}` and gets `{answer, route}`; `answer` is HTML
(every value escaped). Only the latest question is answered, through `chat_routing.py`:

| Route | When | LLM calls |
|---|---|---|
| `DIRECT_DATABASE` | The router maps the question to one data tool ("total revenue", "highest ROAS by platform", "monthly revenue") | **0**. The answer is built from the DuckDB result |
| `LLM_REQUIRED` | "why" / "explain" questions | **1** Groq call through the LLM Adapter, on the compact analysis, grounding-validated |
| `UNSUPPORTED` | Data the dataset doesn't have (e.g. countries), or not about the campaign data | 0: the router's message |
| `NEEDS_CLARIFICATION` | Judgement calls ("underperforming"), relative dates ("last month"), follow-ups that depend on earlier messages ("what about TikTok?"), empty or invalid input | 0: a message saying what can be asked |

There is no free-form LLM fallback. **Conversation history** is only used to spot
follow-ups: a question that leans on an earlier turn (pronouns like "it"/"that", or
openers like "and…" / "what about…") gets a request to ask the full question, because
answering it without that context could silently answer the wrong question.

When an explanation fails (rate limit, timeout, too many running, service unavailable),
the chat still shows the figures from DuckDB, followed by a neutral notice. Users never see
the provider's name, model or configuration; the server log has the details.

**Retries and timeouts:** one request per explanation within `LLM_TIMEOUT_SECONDS`
(default 25), plus at most one retry on a 5xx or network error inside that deadline. A 429
rate limit, a timeout, an auth error or a malformed response is never retried. The
endpoint, adapter and router add no retries of their own.

## Data layer (DuckDB)

```
Query Router
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

Dev-only tools (`pip install -r requirements-dev.txt`, which adds `psutil`). No real Groq
call is ever made.

```
python perf_profile.py                        # startup phases, memory inventory, DuckDB ops,
                                              # endpoint time split by stage (in-process)
python perf_profile.py --threads 1,2,4,8      # DuckDB thread settings, interleaved rounds
python loadtest.py --mode baseline            # sequential latency per request type, over HTTP
python loadtest.py --mode direct              # DIRECT_DATABASE mix at 1/5/10/15/20 users
python loadtest.py --mode llm --mock-delay-ms 1500 --requests 200   # LLM path, mocked provider
```

`loadtest.py` starts `loadtest_server.py`: the real app under uvicorn, with Groq calls
counted and refused, and optionally a fixed-delay mock provider. Each level sends the
same fixed workload. Every response is compared with a reference answer taken before the load
(so races or corrupted shared state show up as `wrong_answer`), the server must stay healthy
after each level, and the provider counters must stay at 0. `--duckdb-threads N` overrides
DuckDB's thread count so the setting can be re-measured on the deployment host.

Measured on a 4-core laptop (server and load generator on the same machine; the laptop's speed
varied by up to ~50% between sessions, so compare only interleaved runs):
- Memory: about 130 MB RSS after startup (about 152 MB before google-genai was removed),
  mostly libraries (pandas + DuckDB ~65 MB). The dataset is ~12 MB in DuckDB. Under sustained
  load (measured before the removal) RSS settled around 180–235 MB; the Python heap and
  DuckDB's own memory stay flat, and the growth slows over time (native allocator behaviour,
  not a Python leak).
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
**without calling any LLM** (keyword/regex rules, no API call to classify). `/api/chat`
uses it for every question; `/api/query` exposes its full JSON result.

```
POST /api/query   {"query": "Which platform has the highest ROAS?", "filters": {...}}
```

| Route | When | What comes back |
|---|---|---|
| `DIRECT_DATABASE` | The question maps to one existing tool ("total revenue", "highest ROAS by platform", "monthly revenue", "campaigns with ROAS above 8") | `tool`, `tool_input` and the exact `result` from that tool |
| `LLM_REQUIRED` | The question asks for an explanation ("why", "explain", "what's causing") | `analysis` (a few small tool results), `explanation` (grounding-validated text, or `null` if the LLM failed) and `llm` (status, provider, error code, grounding verdict) |
| `NEEDS_CLARIFICATION` / `UNSUPPORTED` | Ambiguous, or asks for something the dataset doesn't have (e.g. geography) | A `message` explaining what can be asked |

The router is not a second analytics engine: every number comes from an existing
`data_tools.py` function (`TOOL_REGISTRY`), with the active dashboard filters applied.

**Why the LLM never gets the full dataset:** for `LLM_REQUIRED` the router runs at most 3
tools and caps each list (at most 10 campaign rows and 30 series points, 16 KB total), so
the context stays the same size however large the dataset gets. It's also cheaper and
faster, and it keeps the LLM limited to aggregates that were actually computed.

Errors return a safe message with no stack trace: `400` for an empty or invalid query,
`422` for a malformed body, `503` if the data or a tool isn't available, and `500` if a
tool fails. Each request writes one structured log line (query, route, tools,
elapsed_ms, status).

### LLM provider: Groq

Groq is the only LLM provider; there is no provider selection and no failover.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | Groq API key. Put it in `.env` (or Render's environment), never in code. Without it the app starts, and explanations report the AI service as unavailable |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model ID (a production model, not preview; chosen over `openai/gpt-oss-20b` for better grounding accuracy in `benchmark_results/`). Reasoning models (`openai/gpt-oss-*`) run with `reasoning_effort=low`, reasoning hidden |
| `LLM_TIMEOUT_SECONDS` | 25 | Deadline per explanation |
| `LLM_MAX_CONCURRENT` | 20 | Explanations allowed to wait on Groq at once (see below) |

`/api/health` shows `llm_provider`, `llm_model` and `llm_configured` (never the key). A
leftover `LLM_PROVIDER` variable from older configurations is ignored (logged at startup).

- The key is only sent in the Authorization header and is never logged; auth-error bodies
  aren't logged either. A 429 is logged with Groq's rate-limit headers.
- At most `LLM_MAX_CONCURRENT` (default 20) requests wait on Groq at once. Extra ones get an
  immediate `busy` answer (the figures are still returned) instead of queueing, so slow
  provider calls can't take every server worker and database answers stay fast. Measured:
  with 45 explanations in flight on an 8 s provider, database answers took 6.4 s without
  the cap and 14 ms with it. `DIRECT_DATABASE` questions never use the provider.
- `benchmark_llm_providers.py` runs the same 10 questions through one or more Groq models
  and reports latency and grounding checks. Past results, including Gemini and a local Qwen
  model (both evaluated and removed), are in `benchmark_results/`.

### LLM Adapter

`LLM_REQUIRED` answers are explained through `llm_adapter.py`: a provider-independent
`LLMProvider` interface (`generate_explanation(question, analysis, timeout_s)`) plus an
`LLMAdapter` that validates the analysis before anything is sent. It refuses anything that
isn't plain JSON (such as a DataFrame), anything over 16 KB, and empty results. It also
turns every provider failure into a structured `llm` result with a neutral user-facing
message, so the analysis is still returned when the LLM is down. `groq_provider.py`
implements the interface; tests use a fake transport (`fake_groq.py`).

### Grounding (`grounding.py`)

Groq gets a **typed context**, and every Groq answer goes through a deterministic
**numerical-claim validator**. Both run in the provider-independent adapter, and
`DIRECT_DATABASE` never touches either.

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
keyword heuristic.

Tests (no API key or quota needed: Groq runs on a fake transport, `fake_groq.py`):
```
python test_app.py
python test_query_router.py
python test_llm_adapter.py
python test_data_layer.py
python test_chat_migration.py
python test_groq_provider.py
python test_grounding.py
python test_charts.py
python test_planner.py
python test_conversation.py
python test_general_analytics.py
```

## Adding a new askable dimension or metric

You do **not** need to add a new question/answer branch. Add one line to
`DIMENSION_COLUMNS` in `data_tools.py` for a new dimension, or extend the `metric` enum
in `main.py`'s tool definitions for a new metric — every existing tool (`rank_dimension`,
`compare_entities`, etc.) picks it up automatically.

## Deployment (Render)

The backend runs as a Render web service; the frontend (`site/`) is a static site on Netlify.

- **Root directory:** `backend`. **Build:** `pip install -r requirements.txt`.
  **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT` (one process; FastAPI runs the
  sync endpoints on its 40-thread pool). Health check path: `/api/health`.
- **Startup:** the CSV (`data/…csv`, in the repo) is parsed once, copied into in-memory DuckDB
  and the router's entity index is built before the first request. There is no database file.
  Under memory pressure DuckDB may spill to `backend/.tmp/` (ephemeral disk is fine).
- **Memory:** about 155 MB idle and 180–235 MB under sustained load locally; DuckDB is capped
  at 256 MB. This fits a 512 MB instance.
- **Frontend:** `site/dashboard.html` calls the deployed backend, or `http://127.0.0.1:8000`
  when the dashboard itself is served from localhost / 127.0.0.1.
- **CORS:** only `CORS_ALLOW_ORIGINS` (default `https://marketingiqp.netlify.app`) and pages
  served from localhost / 127.0.0.1 may call the API from a browser.
- **Answers are HTML-escaped** on every path (the dashboard renders them as HTML).
- **Conversation memory** (follow-ups such as "make it a chart" or "what about February?") is kept
  per browser conversation id in this process's memory only (`conversation.py`: at most 2,000
  sessions, 4 hours each, only the last request's structured plan). A restart, redeploy or free-tier
  cold start forgets it; the chat still shows the earlier messages (kept in the browser) but the next
  follow-up needs a full question.

Environment variables (see `.env.example`; never commit keys):

| Variable | Status | Default / notes |
|---|---|---|
| `GROQ_API_KEY` | **Required** | Groq explanations. Without it the app starts; explanations report the AI service as unavailable. |
| `GROQ_MODEL` | Optional | `openai/gpt-oss-120b` |
| `LLM_TIMEOUT_SECONDS` | Optional | `25` |
| `LLM_MAX_CONCURRENT` | Optional | `20`. Explanations beyond this waiting on Groq get an immediate "busy" answer. |
| `CORS_ALLOW_ORIGINS` | Optional | `https://marketingiqp.netlify.app` (comma-separated; `*` = any) |
| `MARKETINGIQ_DATA_BACKEND` | Optional | `duckdb`; `pandas` forces the fallback backend. |
| `MARKETINGIQ_CSV_PATH` | Optional | `../data/tech_advertising_campaigns_dataset.csv` |
| `PORT`, `PYTHON_VERSION` | Set by / in Render | |

Development only: `requirements-dev.txt` (`psutil`) for `perf_profile.py` and `loadtest.py`.

No longer used (safe to delete from Render's environment): `GEMINI_API_KEY`,
`GEMINI_API_KEY_1`…, `LLM_PROVIDER`, `CHAT_ROUTER_ENABLED`, `CHAT_FALLBACK_TIMEOUT_SECONDS`.

If explanations outgrow Groq's free-plan limits, a paid Groq plan raises them with no code
change; the data layer and the router don't depend on the LLM at all.
