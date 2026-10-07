# MarketingIQ

**AI-Powered Marketing Performance & Decision Intelligence Platform**

A portfolio analytics application that turns a 10,000-campaign digital advertising dataset into
an interactive, data-grounded decision-support tool — built as a real web app (not a BI-tool
export), with a genuinely data-grounded AI Analyst on top and a premium glassmorphism UI.

> Built end-to-end: data inspection → analytical layer → UI design → interactive frontend →
> data pipeline → AI integration → testing → deployment → iterative refinement.

---

## Live demo

**Landing page:** https://marketingiqp.netlify.app
**Dashboard (direct):** https://marketingiqp.netlify.app/dashboard.html
**Backend API health check:** https://marketingiq-backend.onrender.com/api/health

The deployed backend on Render's free tier still runs the **previous** version (Query Router + a
cloud LLM for explanations). This repository's backend is now fully local: a fine-tuned Qwen3-0.6B
on a local llama-server plans every analytical question, so it needs that model server to run (see
[`backend/README.md`](backend/README.md)). It has not been deployed.

---

## What it does

MarketingIQ helps a marketing team answer the questions that actually drive budget decisions:

- How is marketing performing overall, and is it improving or declining?
- Which channels are efficient, and which are quietly wasting budget?
- Which campaigns should be scaled, and which should be reviewed or killed?
- Which audiences, devices, and creative formats convert best?
- "Ask" those same questions in plain English and get a real, data-grounded answer.

## Features

- **Landing page** — a premium entry point with a live, tilted preview of the actual dashboard
  (not a mockup — a real embedded instance of it), before continuing into the app itself.
- **5 analytics pages** — Executive Overview, Campaign, Channel, Audience, and AI Analyst —
  each with KPI cards, charts, and tables computed live from the full dataset.
- **Real, working filters** — platform, objective, industry vertical, budget tier, and
  retargeting status all genuinely re-filter every KPI, chart, and table from the full
  dataset (not a decorative UI). Clicking a bar in the Audience or Channel charts (age,
  device, gender, creative format, emotion, placement, or platform) also applies as a live
  cross-filter, combinable with the sidebar filters and removable via a chip UI — similar to
  drill-down filtering in tools like Power BI.
- **AI Marketing Analyst** — ask questions in plain English; every figure is computed live
  from the real dataset in DuckDB. A small fine-tuned local model only turns the question into a
  validated analytics plan; answers, "why" findings, recommendations and charts are computed
  deterministically from the data (see below).
- **Glassmorphism design system** — a 3-level frosted-glass surface hierarchy (standard /
  elevated / focused) across both a dark (midnight) and light (pearl) theme, with the
  preference synced between the landing page and dashboard.
- **Fully responsive** — desktop, laptop, tablet, and mobile, including an off-canvas
  navigation drawer on narrow screens and controlled (non-oversized) typography in
  expanded/focused chart views.
- Hover tooltips on every chart, sortable/searchable campaign table, real creative-fatigue
  and opportunity-matrix analysis.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript (no framework) — inline SVG for all charts |
| Data pipeline | Python, Pandas |
| AI backend | FastAPI, DuckDB, a fine-tuned Qwen3-0.6B (Q4_K_M GGUF on llama.cpp) as the only planner, `validate_plan()`, deterministic findings / recommendations / charts |
| Data | A 10,000-row synthetic digital-advertising campaign dataset (41 columns) |

No paid services, cloud LLMs or API keys are required: the model runs locally on a CPU (see
[`backend/README.md`](backend/README.md)).

## Architecture

```
tech_advertising_campaigns_dataset.csv
        │
        ▼  (scripts/build_data.py — Python/Pandas)
site/data.js  — real per-campaign records + precomputed aggregates, embedded as
                a JS global (avoids a fetch-based CORS issue when opened locally)
        │
        ▼
site/index.html (landing page) ──► site/dashboard.html (the actual app)
                                     all filtering/charts computed client-side
                                     from the real data in data.js
        │
        ▼  (AI Analyst tab only)
backend/main.py (FastAPI) ── local Qwen3-0.6B (llama-server): question ──► compact plan
        ──► validate_plan() ──► DuckDB (same CSV, live filters)
        ├── analysis ─────► deterministic answer + validated chart
        ├── why / explain ─► deterministic findings
        └── advice ───────► findings + rule-based recommendations
        ▼
Answer returned to the chat UI
```

`site/dashboard.html` (charts, filters, tables) is a fully static page — it works with the
backend turned off. Only the **AI Analyst tab** needs the backend (and its local model server)
running. `site/index.html` is a lightweight landing page that
embeds a live, non-interactive preview of the real dashboard.

## The AI Analyst, in detail

This is the part of the brief most prone to hand-waving ("connect an AI"), so here's exactly
how it avoids hallucination:

1. The user's question and currently active dashboard filters (sidebar *and* chart-click
   cross-filters) are sent to the backend.
2. A **fine-tuned Qwen3-0.6B**, running locally on llama.cpp, turns the question into a compact
   **analytics plan**: metric, breakdown, filters, period, chart type. That is all it does. It never
   sees the dataset, never writes SQL and never writes an answer or a number. For a follow-up
   ("now profit"), it gets the previous validated plan, not the chat transcript.
3. **`validate_plan()`** checks every plan against allowlists and the dataset's real values. A
   malformed or invalid plan never reaches the data.
4. The plan runs on a fixed set of **data tools** in **DuckDB**, with the user's filters applied.
   The answer text and charts are built deterministically from the results.
5. "Why / explain" questions get **deterministic findings** computed from the results: highest
   and lowest, gaps, change over time, correlation, loss-making counts. Advice questions add
   **rule-based recommendations** tied to those findings (budget follows ROAS; loss-making means
   profit below $0). No model writes prose, so no figure can be invented.
6. Questions the data can't answer (fields that don't exist, such as browsers or countries) get a
   clear message about what is available, rather than a guess.

Adding a new dimension or metric to the data layer is a small change in `backend/data_tools.py` /
`schema_catalog.py`. For the model to plan questions about it, its training data would need the
new field (`experiments/tiny_model/`).

## Dataset

`data/tech_advertising_campaigns_dataset.csv` — 10,000 rows × 41 columns, 0 missing values,
0 duplicates, spanning January 2024 – January 2026. Dimensions include platform, campaign
objective, ad placement, device type, creative format/emotion, target audience age/gender,
industry vertical, budget tier, and retargeting status; metrics include impressions, clicks,
conversions, spend, revenue, profit, and pre-calculated CTR/CPC/conversion rate/CPA/ROAS.

### KPI definitions

| KPI | Formula |
|---|---|
| CTR | Clicks ÷ Impressions × 100 |
| Conversion Rate | Conversions ÷ Clicks × 100 |
| CPC | Spend ÷ Clicks |
| CPA | Spend ÷ Conversions |
| ROAS | Revenue ÷ Spend |
| ROI | (Revenue − Spend) ÷ Spend × 100 |

## Project structure

```
marketingiq/
├── data/
│   └── tech_advertising_campaigns_dataset.csv
├── scripts/
│   └── build_data.py          # regenerates site/data.js from the CSV
├── site/                       # the static frontend — deployable as-is
│   ├── index.html               # landing page (entry point)
│   ├── dashboard.html           # the actual analytics app
│   └── data.js
├── backend/                     # only needed for the AI Analyst tab
│   ├── main.py                  # FastAPI app: /api/chat, /api/health
│   ├── analyst.py               # the answer path: plan, validate, DuckDB, answer / findings / advice
│   ├── local_planner.py         # the local Qwen planner (persistent llama-server)
│   ├── query_planner.py         # plan contract: validate_plan(), execute_plan()
│   ├── findings_engine.py       # deterministic findings
│   ├── recommendation_engine.py # rule-based recommendations
│   ├── data_tools.py            # analytics tools over the DuckDB data layer
│   ├── requirements.txt
│   ├── test_*.py
│   ├── .env.example
│   └── README.md
├── experiments/tiny_model/      # how the local planner was built, evaluated and exported
└── docs/
```

## Setup

### Option A — Frontend only (no AI, no Python needed to run it)

Just open `site/index.html` (or `site/dashboard.html` directly) in a browser. Everything
except the AI Analyst tab works immediately, since all data is embedded in `site/data.js`.

*(Note: on some browsers, opening via `file://` can block the AI Analyst's network request
even with the backend running — see Option B for the fix, serving it over a local address.)*

### Option B — Full setup, including the AI Analyst

1. **Clone the repo** and open it in your editor of choice.
2. **Regenerate the data file** (optional — `site/data.js` is already committed, but if you
   change the CSV, rebuild it):
   ```
   cd scripts
   python build_data.py
   ```
3. **Set up the AI backend** — see [`backend/README.md`](backend/README.md) for full detail
   (installing dependencies, starting the local model server, running the API).
4. **Serve the frontend** over a local address (not `file://`):
   ```
   cd site
   python -m http.server 5500 --bind 127.0.0.1
   ```
5. Open `http://127.0.0.1:5500/index.html` (or go straight to `dashboard.html`), with the
   backend running at `http://127.0.0.1:8000` in a separate terminal.

## Environment variables

No secrets are needed. Optional settings (model server URL, CORS, data backend) are listed in
`backend/.env.example`; a local `backend/.env` is never committed (see `.gitignore`).

## Testing

- `backend/test_*.py` — automated tests of the chat path (one planning call at most, the
  validation gate, findings, recommendation direction, memory, charts), the plan contract
  (including malicious plans), the DuckDB data layer (checked against the original Pandas results)
  and the grounding validator. The model is replaced by a fake local llama-server, so the suite
  needs no GGUF.
- Frontend behavior (zero-match filters, search edge cases, sort correctness, chart
  cross-filtering, responsive navigation, theme persistence) was verified with an automated
  headless-browser test pass during development — 60+ test cases across multiple suites.

## Limitations

- The dataset is synthetic/public, not live production ad-platform data.
- The AI Analyst needs the local model server; without it, analytical questions answer that
  the model is unavailable. Explanations and recommendations are deterministic summaries of the
  data, not free-form analysis: they say what happened, never why.
- The 0.6B planner occasionally misreads a question. Invalid plans are always refused, but a
  valid-yet-wrong plan answers a different question correctly (about 90% exact plans on the
  frozen evaluation set).
- Filtering, charts, and the campaign ledger operate on the full 10,000-row dataset, but the
  campaign ledger table displays the top 100 matching rows at a time for performance; the KPI
  totals above it reflect the complete filtered set regardless.

## License

MIT — see [`LICENSE`](LICENSE).

---

Built by Hoorab Iqbal as a portfolio project.
