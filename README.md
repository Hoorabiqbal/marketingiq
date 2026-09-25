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

The backend runs on Render's free tier, which spins down after ~15 minutes of inactivity —
if the AI Analyst is slow to respond the first time, that's it waking back up (~30–60 seconds),
not a bug. AI explanations run on a free-tier LLM plan with a small per-minute budget: if
several "why" questions arrive at once, the AI Analyst shows the figures with a
"temporarily rate-limited" notice instead of an explanation — database answers are unaffected.

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
  from the real dataset in DuckDB. Factual questions are answered directly from the data
  with no LLM at all; "why" questions get an LLM explanation that is checked number by number
  against the data before it is shown (see below).
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
| AI backend | FastAPI, DuckDB, a deterministic query router, Groq (`openai/gpt-oss-120b`) for explanations only, plus a numerical grounding validator |
| Data | A 10,000-row synthetic digital-advertising campaign dataset (41 columns) |

No paid services are required to run this project — Groq's free plan is enough (see
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
backend/main.py (FastAPI) ── Query Router (deterministic, no LLM)
        ├── DIRECT_DATABASE ──► DuckDB (same CSV, live filters) ──► exact answer
        ├── LLM_REQUIRED ─────► DuckDB ──► compact typed analysis ──► Groq
        │                                   ──► grounding validator ──► safe answer
        └── UNSUPPORTED / NEEDS_CLARIFICATION ──► what can be asked instead
        ▼
Answer returned to the chat UI
```

`site/dashboard.html` (charts, filters, tables) is a fully static page — it works with the
backend turned off. Only the **AI Analyst tab** needs the backend running, since it's the
only feature that calls out to an LLM. `site/index.html` is a lightweight landing page that
embeds a live, non-interactive preview of the real dashboard.

## The AI Analyst, in detail

This is the part of the brief most prone to hand-waving ("connect an AI"), so here's exactly
how it avoids hallucination:

1. The user's question and currently active dashboard filters (sidebar *and* chart-click
   cross-filters) are sent to the backend.
2. A deterministic **Query Router** (rules, no LLM) maps the question onto a fixed set of
   **data tools** (`get_totals`, `rank_dimension`, `compare_entities`, `filter_campaigns`,
   `trend_over_time`, etc.), run in **DuckDB** with the user's filters applied.
3. Factual questions ("total revenue", "which platform has the highest ROAS", "monthly
   revenue") are answered straight from that result — no LLM is involved at all.
4. "Why / explain" questions get a small, typed analysis (units, scopes, aggregates vs.
   examples, what the data can and cannot establish) that is sent to **Groq** for a short
   explanation. The LLM never sees the dataset and never chooses tools.
5. Every explanation passes a deterministic **grounding validator**: each number must match
   the supplied data with the right unit, metric, entity and month. An explanation with an
   invented or mislabelled figure is never shown — the user gets a plain summary of the real
   figures instead.
6. Questions the data can't answer (fields that don't exist, judgement calls like
   "underperforming", relative dates like "last month") get a clear message about what can be
   asked, rather than a guess.

Adding a new askable dimension or metric is a one-line change in `backend/data_tools.py` —
not a new question/answer branch.

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
│   ├── main.py                  # FastAPI app: /api/chat, /api/query, /api/health
│   ├── query_router.py          # deterministic question routing (no LLM)
│   ├── data_tools.py            # analytics tools over the DuckDB data layer
│   ├── llm_adapter.py           # provider-independent explanation layer
│   ├── groq_provider.py         # Groq, the only LLM provider
│   ├── grounding.py             # typed LLM context + numerical-claim validator
│   ├── requirements.txt
│   ├── test_app.py
│   ├── .env.example
│   └── README.md
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
   (installing dependencies, getting a Groq API key, running the server).
4. **Serve the frontend** over a local address (not `file://`):
   ```
   cd site
   python -m http.server 5500 --bind 127.0.0.1
   ```
5. Open `http://127.0.0.1:5500/index.html` (or go straight to `dashboard.html`), with the
   backend running at `http://127.0.0.1:8000` in a separate terminal.

## Environment variables

All secrets live in `backend/.env` (never committed — see `.gitignore`). The only required one
is `GROQ_API_KEY`; see `backend/.env.example` for the optional settings.

## Testing

- `backend/test_*.py` — automated tests of the router, the DuckDB data layer (checked
  against the original Pandas results), the chat endpoint, the Groq provider (rate limits,
  timeouts, retries, errors), the grounding validator (including real bad answers recorded
  from earlier LLM benchmarks) and cold-start concurrency. Groq is replaced by a fake HTTP
  transport, so the suite needs no API key and uses no quota.
- Frontend behavior (zero-match filters, search edge cases, sort correctness, chart
  cross-filtering, responsive navigation, theme persistence) was verified with an automated
  headless-browser test pass during development — 60+ test cases across multiple suites.

## Limitations

- The dataset is synthetic/public, not live production ad-platform data.
- AI explanations use Groq's free plan (about 8,000 tokens a minute, roughly a few
  explanations per minute for the whole app). Beyond that, users see the figures with a
  "temporarily rate-limited" notice. Factual questions never use the LLM and are unaffected.
- The AI Analyst answers each question on its own: follow-ups that depend on an earlier
  answer ("what about TikTok?") are asked to be rephrased as a full question.
- Filtering, charts, and the campaign ledger operate on the full 10,000-row dataset, but the
  campaign ledger table displays the top 100 matching rows at a time for performance; the KPI
  totals above it reflect the complete filtered set regardless.

## License

MIT — see [`LICENSE`](LICENSE).

---

Built by Hoorab Iqbal as a portfolio project.
