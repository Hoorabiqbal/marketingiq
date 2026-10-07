# Final local architecture: MarketingIQ with the local Qwen planner

MarketingIQ's AI Analyst (`/api/chat`) is 100% local. A fine-tuned Qwen3-0.6B (Q4_K_M) on a persistent
llama-server is the **only** language component, and it only plans. It needs no cloud LLM, no API key
and no configuration switch. (This file grew out of the Phase 7 guide. Phase 7–8 ran this path
behind a temporary `MARKETINGIQ_PLANNER` switch, which has since been removed together with the
Query Router and Groq.)

## Final architecture

```text
User
 → fine-tuned Qwen3-0.6B Q4_K_M   (ONE greedy call, persistent llama-server, thinking OFF)
 → compact JSON plan
 → parse → canonicalization       (plan_format.to_canonical; unknown keys rejected)
 → validate_plan()                (rejected plans never reach DuckDB)
 → DuckDB                         (execute_plan → analytics_executor → data_tools: every number)
 → deterministic answer / chart        analysis
   deterministic findings              "why", "explain"   (findings_engine)
   findings + rule-based actions       advice             (recommendation_engine)
```

**Cloud LLM calls: 0. API keys: none. Second model calls: never.** Qwen never writes prose, SVG,
HTML, JavaScript or a number.

Side paths, explicit and narrow (not a router; they make no analytics decision):

| Message | Path | Model calls |
|---|---|---|
| Greetings, thanks, "what can you do?", personal topics | fixed replies (`conversation.small_talk`) | 0 |
| A campaign ID (`CAMP_00012`, or "it" right after a campaign answer) | exact DuckDB lookup (`campaign_lookup`), with benchmarks for "why" / advice | 0 |
| "Why?", "explain this", "what should we improve?" about the last analysis | findings / recommendations on the remembered validated plan | 0 |

Memory keeps only the last **validated** plan per conversation (2,000 sessions, 4 h, no transcript).
A follow-up sends that plan to Qwen in the training wording. A complete new question replaces it,
and a failed question leaves it unchanged.

Two deterministic semantic rules sit on top of the model's plan:

- **loss-making** ("loss-making", "losing", "at a loss", "negative-profit", "unprofitable") means
  profit below $0, the dashboard's Unsuccessful Campaigns rule;
- **budget advice follows ROAS**, never totals: test budget on the highest-ROAS segment, review the
  lowest before adding budget.

## Run it (Windows, PowerShell, from the repository root)

1. Start the model server **once** and leave it running:

   ```powershell
   C:\llama.cpp\llama-server.exe -m experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q4_K_M.gguf `
       --host 127.0.0.1 --port 8089 -c 512 -t 4 -tb 4 -np 1 -ngl 0 --reasoning off --no-webui
   ```

2. Start the backend:

   ```powershell
   cd backend; ..\venv\Scripts\uvicorn main:app --port 8000
   ```

   `GET /api/health` reports `"planner": "local_qwen"` and `local_model_reachable`. Optional settings:
   `MARKETINGIQ_LLAMA_URL` (default `http://127.0.0.1:8089`), `MARKETINGIQ_LLAMA_TIMEOUT_S` (default 8),
   `MARKETINGIQ_TINY_MODEL_DIR`. Generation is fixed by the client: temperature 0, top_k 1,
   `max_new_tokens` 85 from `export/marketingiq_runtime.json`.

For best latency, keep the laptop plugged in with Windows power mode on **Best performance**.
Measured end to end on the X395: planning p50 ≈ 0.7 s on AC power, ≈ 1.1 s on battery.

## Failures (never executed, never sent to another model)

| Failure | User sees | Route |
|---|---|---|
| llama-server down, timed out or dropped the connection | "The local analytics model is temporarily unavailable." | MODEL_UNAVAILABLE |
| Output cut off, not a JSON object, unknown keys | the standard "couldn't map that question" hint | NEEDS_CLARIFICATION |
| `validate_plan()` rejects a field outside the dataset | e.g. "Browser-level analysis isn't available in the current MarketingIQ dataset. Available breakdowns: …" | UNSUPPORTED |
| Execution error | a safe data-error message (details only in the server log) | DATA_ERROR |

## Tests and checks

- `backend/test_*.py`: 132 tests against a fake llama-server over real HTTP. No GGUF needed.
- `benchmark/smoke_local_integration.py`: real end-to-end smoke test (13 questions, none from the
  frozen set).
- `benchmark/findings_quality_local.py`: the explanation / advice quality gate, using the same 13
  questions the failed local-Qwen explainer was tested on.

## Deployment (not done; decide separately)

The ThinkPad runtime and the public Render deployment are different environments, and nothing here
changes Render or Netlify. **The current Render service runs the previous code** (Query Router + Groq).
Redeploying this code there without a reachable llama-server would answer every analytical question
with "model unavailable".

1. **Keep the public site on the previous deployment** and run the local architecture on machines
   that run llama-server (the X395 today).
2. **Host llama-server next to the API.** For example, a small VM or a private Render service with
   about 1 GB of RAM for the model, with `MARKETINGIQ_LLAMA_URL` pointing at it and the server kept
   private. CPU speed decides latency, so measure on that host first; shared or free CPUs are likely
   slower than the X395. The GGUF (≈ 0.4 GB) must be shipped to that host; it is not in git.
3. **Run the model inside the API process** (llama.cpp Python bindings). This is a different runtime
   from the one validated here, so it needs its own validation first.

Before any production switch, run the full test suite and both real-model scripts on the target host.
