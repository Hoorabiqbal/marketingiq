# MarketingIQ AI Analyst — Backend

A small FastAPI service behind the dashboard's AI Analyst. It is **100% local**: no cloud LLM, no
API key.

A fine-tuned **Qwen3-0.6B** (Q4_K_M GGUF, served by a local llama.cpp `llama-server`) turns each
question into a compact **analytics plan**. `validate_plan()` checks the plan, and **DuckDB**
computes every number. Answers, findings, recommendations and charts are all **deterministic**:
the model plans and never writes text or numbers.

## How it works

```text
Your question (+ active dashboard filters)
        │
        ├─ greeting / thanks / "what can you do" ............ fixed reply (0 model calls)
        ├─ exact campaign ID ("CAMP_00012") ................. exact DuckDB lookup (0 model calls)
        └─ everything else
              ▼
        Qwen3-0.6B (local_planner.py) — ONE call → compact JSON plan
              ▼   (for follow-ups the remembered validated plan is sent too; never the transcript)
        parse → canonicalize → validate_plan()   (a rejected plan never reaches DuckDB)
              ▼
        DuckDB (query_planner.execute_plan → analytics_executor → data_tools)
              ▼
        analysis ......... deterministic answer text + validated chart (chart_builder)
        "why / explain" .. deterministic findings (findings_engine)
        advice ........... findings + rule-based recommendations (recommendation_engine)
```

"Why?", "explain this" and "what should we improve?" reuse the remembered plan, so they make no
model call at all. No question ever makes a second model call.

| Module | Role |
|---|---|
| `main.py` | FastAPI: `/api/chat`, `/api/health` |
| `analyst.py` | The answer path above: small talk, campaign ID, analysis / explanation / advice |
| `local_planner.py` | Qwen client (persistent llama-server), parse, canonicalize, `validate_plan()`, loss-making rule |
| `query_planner.py` | The plan contract: `SCHEMA`, `validate_plan()`, `execute_plan()` and deterministic answer text |
| `analytics_engine.py` | Multi-metric / multi-series / extra-field shapes and their charts |
| `analytics_executor.py` | Data-tool dispatch with the dashboard filters; data-layer errors |
| `findings_engine.py` | Factual statements computed from DuckDB rows (highest / lowest, gaps, trends, correlation, loss-making) |
| `recommendation_engine.py` | Bounded rules tied to findings ("review", "test", "investigate", "monitor"); no promises, no new numbers |
| `campaign_lookup.py` | Exact campaign-ID lookup (+ benchmarks for "why" / advice about a campaign) |
| `conversation.py` | Bounded session memory, small talk, request type, follow-up detection |
| `chart_builder.py` | Chart specs and `validate_chart()` (data + metadata only, never code) |
| `data_tools.py`, `campaign_repository.py` | Metric formulas and the DuckDB data layer |
| `grounding.py` | Metric units / formatting; a numeric-claim validator kept as a tested library |

## Setup

1. **Install dependencies** (from inside this `backend/` folder, with your venv activated):

   ```text
   pip install -r requirements.txt
   ```

2. **Start the model server** once and leave it running (from the repository root; see
   `experiments/tiny_model/PHASE7_LOCAL_QWEN.md` for where the GGUF and llama.cpp come from):

   ```text
   C:\llama.cpp\llama-server.exe -m experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q4_K_M.gguf ^
       --host 127.0.0.1 --port 8089 -c 512 -t 4 -tb 4 -np 1 -ngl 0 --reasoning off --no-webui
   ```

3. **Run the API**:

   ```text
   uvicorn main:app --reload --port 8000
   ```

4. **Verify**: <http://localhost:8000/api/health> should return

   ```text
   {"status":"ok","campaigns_loaded":10000,"data_backend":"duckdb","planner":"local_qwen",
    "planner_model":"unsloth/Qwen3-0.6B + MarketingIQ LoRA (...), merged","local_model_reachable":true}
   ```

   If `local_model_reachable` is `false`, start the llama-server (step 2). Until then, analytical
   questions answer "The local analytics model is temporarily unavailable." The app never falls
   back to another model.

5. **Open the dashboard**, served locally (`python -m http.server 5500 --bind 127.0.0.1
   --directory site` from the project root, then http://127.0.0.1:5500/dashboard.html). When it is
   served from localhost / 127.0.0.1, it calls this local backend.

## How `/api/chat` answers

The dashboard sends `{messages, filters, session_id}` and gets `{answer, route, chart}`. `answer`
is HTML with every value escaped. `chart` is a validated chart spec or null, and `charts` is added
when metrics in different units need separate charts.

| `route` | When | Model calls |
|---|---|---|
| `ANALYSIS` | Totals, breakdowns, comparisons, trends, relationships, filters, campaign lists, charts, follow-ups | 1 |
| `EXPLANATION` | "Why …", "explain …" | 1, or 0 for "why?" / "explain this" about the last analysis |
| `RECOMMENDATION` | "How can we improve …", "what should we do about …" | 1, or 0 for "what should we improve?" |
| `CAMPAIGN` | A campaign ID | 0 |
| `CONVERSATION` | Greetings, thanks, "what can you do?" | 0 |
| `UNSUPPORTED` | Data the dataset doesn't have ("Browser-level analysis isn't available …"), off-topic | 1 (refused by the model or by `validate_plan()`) |
| `NEEDS_CLARIFICATION` | Malformed or ambiguous plans, references with nothing to refer to, empty or too-long input | 0–1 |
| `MODEL_UNAVAILABLE` | llama-server down or timed out (8 s) | 1 attempt, no retry |
| `DATA_ERROR` | A data-layer failure; details are logged, never shown | 1 |

Explanations say what the data shows and never claim a cause: "Based on the available metrics,
this describes what happened; the data doesn't show why". Budget recommendations follow
efficiency (ROAS), never totals. Loss-making campaigns always mean **profit below $0** (the
dashboard's Unsuccessful Campaigns rule), whatever proxy the model picks.

**Conversation memory** (`conversation.py`) is kept per browser conversation id, in this
process's memory only: at most 2,000 sessions, 4 hours each, and only the last validated plan.
A follow-up ("now profit", "chart this", "what about spend?") sends that plan to the model. A
complete new question replaces it, and a failed question leaves it as it was. A restart forgets
every session.

## Data layer (DuckDB)

```text
validated plan
        │  (known tools only — no free-form SQL, no text-to-SQL)
        ▼
analytics_executor.py  tool dispatch with the dashboard filters
        ▼
data_tools.py          business logic: metric formulas, what each dashboard filter means, rounding
        ▼
campaign_repository.py data access: generic filter / aggregate / group operations
        ▼
DuckDB (in-memory)     primary engine  ·  Pandas: automatic fallback + correctness reference
```

- **Source of truth is the CSV.** At startup it is read once with Pandas, and an in-memory DuckDB
  table is built from it. There is no database file. Set `MARKETINGIQ_DATA_BACKEND=pandas` to use
  the Pandas backend; the app also falls back to it on its own if DuckDB can't start.
- **All SQL lives in `campaign_repository.py`**: column names are checked against the real
  schema, operators against a fixed list, and every value is a bound `?` parameter.
- **The model never sees the dataset**: it gets only the question and, for follow-ups, the
  remembered plan.
- **Correctness:** `test_data_layer.py` checks both backends against `data_layer_golden.json`.
  **Speed:** `benchmark_data_layer.py`.

## Tests

No GGUF needed: the model is a fake llama-server over real HTTP (`fake_llama.py`), and every
figure is checked against independent DuckDB queries.

```text
python test_analyst.py            # final architecture: call budgets, validation gate, findings, recommendations
python test_app.py                # health / endpoints / dashboard contract, no cloud-LLM code
python test_planner.py            # plan contract, execution, security of malicious plans
python test_general_analytics.py  # advanced analytics and charts, conversation chains
python test_charts.py             # chart specs and their validation
python test_grounding.py          # the numeric-claim validator library
python test_data_layer.py         # DuckDB vs Pandas golden results
```

Real-model checks (need the running llama-server):
`experiments/tiny_model/benchmark/smoke_local_integration.py` (end-to-end) and
`findings_quality_local.py` (explanation / advice quality gate).

## Deployment

The backend needs a reachable llama-server running the Q4_K_M GGUF (about 0.4 GB on disk and
0.8 GB RAM). **On a host without one, every analytical question answers "The local analytics model is
temporarily unavailable."** Small talk and campaign-ID lookups still work.

The current public Render deployment ran the previous (cloud-LLM) version of this code and must not
be redeployed from this code unless a llama-server is reachable from it. See
`experiments/tiny_model/PHASE7_LOCAL_QWEN.md` for deployment options.

- **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`. Health check path: `/api/health`.
- **CORS:** only `CORS_ALLOW_ORIGINS` (default `https://marketingiqp.netlify.app`) and pages
  served from localhost / 127.0.0.1 may call the API from a browser.

Environment variables (see `.env.example`; none are secrets):

| Variable | Default |
|---|---|
| `MARKETINGIQ_LLAMA_URL` | `http://127.0.0.1:8089` |
| `MARKETINGIQ_LLAMA_TIMEOUT_S` | `8` |
| `MARKETINGIQ_TINY_MODEL_DIR` | `../experiments/tiny_model` (prompt contract + `export/marketingiq_runtime.json`) |
| `CORS_ALLOW_ORIGINS` | `https://marketingiqp.netlify.app` (comma-separated; `*` = any) |
| `MARKETINGIQ_DATA_BACKEND` | `duckdb` (`pandas` forces the fallback) |
| `MARKETINGIQ_CSV_PATH` | `../data/tech_advertising_campaigns_dataset.csv` |

No longer used (safe to delete from any environment): `GROQ_API_KEY`, `GROQ_MODEL`,
`LLM_TIMEOUT_SECONDS`, `LLM_MAX_CONCURRENT`, `MARKETINGIQ_PLANNER`, `GEMINI_API_KEY*`, `LLM_PROVIDER`.
If a Groq key was ever used, revoke it at <https://console.groq.com/keys>.

`benchmark_results/` holds historical records of earlier cloud/local LLM comparisons. The
validator tests use them as regression fixtures.
