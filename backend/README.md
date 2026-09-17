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

2. **Get a free API key** — no credit card, no billing setup:
   - Go to https://aistudio.google.com/apikey
   - Sign in with any Google account
   - Click "Create API key" — that's it, no payment method requested

3. **Add your key**:
   ```
   copy .env.example .env
   ```
   Open `.env` and paste your real key after `GEMINI_API_KEY=`.
   **Never commit `.env`** — it's already in `.gitignore`.

4. **Run the server**:
   ```
   uvicorn main:app --reload --port 8000
   ```

5. **Verify it's working** — open http://localhost:8000/api/health. You should see:
   ```
   {"status":"ok","campaigns_loaded":10000,"gemini_key_configured":true}
   ```
   If `gemini_key_configured` is `false`, the `.env` file isn't being found or is empty —
   double check it's named exactly `.env` (not `.env.txt`) and sits directly in this folder.

6. **Open the dashboard** (`site/index.html`, served locally — see the main project docs
   for why `file://` doesn't work for the AI Analyst) and try the AI Analyst tab.

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

## Adding a new askable dimension or metric

You do **not** need to add a new question/answer branch. Add one line to
`DIMENSION_COLUMNS` in `data_tools.py` for a new dimension, or extend the `metric` enum
in `main.py`'s tool definitions for a new metric — every existing tool (`rank_dimension`,
`compare_entities`, etc.) picks it up automatically.

## Before deploying publicly

- Change `allow_origins=["*"]` in `main.py` to your actual deployed frontend URL.
- Set `GEMINI_API_KEY` as a secret/environment variable on whatever host you use
  (Render, Railway, Fly.io, etc.) — never bake it into the code or a committed file.
- Update `AI_BACKEND_URL` in `site/index.html` (near the top of the AI Analyst script
  section) to point at your deployed backend URL instead of `127.0.0.1:8000`.
- If the project grows beyond the free tier's request volume, Gemini's paid tier (or
  switching back to a paid Claude/OpenAI key using the same tool-use pattern) is a
  drop-in upgrade — the `data_tools.py` layer doesn't change either way.
