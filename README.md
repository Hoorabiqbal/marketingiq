# MarketingIQ

**AI-Powered Marketing Performance & Decision Intelligence Platform**

A portfolio analytics application that turns a 10,000-campaign digital advertising dataset into
an interactive, data-grounded decision-support tool — built as a real web app (not a BI-tool
export), with a genuinely data-grounded AI Analyst on top.

> Built end-to-end: data inspection → analytical layer → UI design → interactive frontend →
> data pipeline → AI integration → testing → deployment.

---

## What it does

MarketingIQ helps a marketing team answer the questions that actually drive budget decisions:

- How is marketing performing overall, and is it improving or declining?
- Which channels are efficient, and which are quietly wasting budget?
- Which campaigns should be scaled, and which should be reviewed or killed?
- Which audiences, devices, and creative formats convert best?
- "Ask" those same questions in plain English and get a real, data-grounded answer.

## Screenshots

*(Add screenshots of the Overview, Campaigns, and AI Analyst pages here before publishing —
see `docs/` for image guidance.)*

## Features

- **Executive Overview** — KPI strip with real deltas, revenue/spend trend, ROAS-by-platform
  ranking, marketing funnel, campaign opportunity matrix, budget-vs-revenue contribution,
  top/risk campaigns, and auto-generated insights.
- **Campaign Analytics** — sortable, searchable ledger over all 10,000 real campaigns, live
  KPI recalculation, and a real ROAS-vs-conversions opportunity matrix.
- **Channel & Platform Analytics** — full platform comparison table (spend, revenue, ROAS,
  CPA, conversion rate, CTR) and ad-placement performance.
- **Audience & Device Analytics** — ROAS by age group and device, revenue by gender,
  conversion rate by income bracket.
- **Creative Performance** — ROAS/CTR by creative format, a real creative-fatigue curve
  (CTR vs. creative age), CTA impact, and conversion rate by emotional hook.
- **Real, working filters** — platform, objective, industry vertical, budget tier, and
  retargeting status all genuinely re-filter every KPI, chart, and table from the full
  dataset (not a decorative UI).
- **AI Marketing Analyst** — ask open-ended questions in plain English; answers are computed
  live from the real dataset via tool-use, not templated or hardcoded (see below).
- Light/dark theme, hover tooltips on every chart, responsive dense layout.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript (no framework) — inline SVG for charts |
| Data pipeline | Python, Pandas |
| AI backend | FastAPI, Google Gemini API (`gemini-3.6-flash`, free tier) |
| Data | A 10,000-row synthetic digital-advertising campaign dataset (41 columns) |

No paid services are required to run this project — the Gemini API's free tier requires no
credit card (see [`backend/README.md`](backend/README.md)).

## Architecture

```
tech_advertising_campaigns_dataset.csv
        │
        ▼  (scripts/build_data.py — Python/Pandas)
site/data.js  — real per-campaign records + precomputed aggregates, embedded as
                a JS global (avoids a fetch-based CORS issue when opened locally)
        │
        ▼
site/index.html — dashboard UI, all filtering/charts computed client-side from
                   the real data in data.js
        │
        ▼  (AI Analyst tab only)
backend/main.py (FastAPI) ── Gemini tool-use loop ── backend/data_tools.py
        │                                             (pandas query layer,
        ▼                                              same CSV, live filters)
Grounded, business-analyst-style answer, returned to the chat UI
```

The dashboard itself (charts, filters, tables) is a fully static site — it works with the
backend turned off. Only the **AI Analyst tab** needs the backend running, since it's the
only feature that calls out to an LLM.

## The AI Analyst, in detail

This is the part of the brief most prone to hand-waving ("connect an AI"), so here's exactly
how it avoids hallucination:

1. The user's question, conversation history, and currently active dashboard filters are
   sent to the backend.
2. Gemini is given a fixed set of **data tools** (`get_totals`, `rank_dimension`,
   `compare_entities`, `filter_campaigns`, `percentage_share`, `trend_over_time`, etc.) — it
   decides which tool(s) to call and with what parameters, based on the actual question. This
   is real intent interpretation, not a hardcoded list of recognized questions.
3. The backend executes the chosen tool(s) against the real Pandas dataframe — the same
   dashboard filters the user has active are applied automatically inside every tool call.
4. The real, computed result is sent back to Gemini as a tool result.
5. Gemini writes the final answer **using only that tool result** — a system prompt
   explicitly forbids inventing numbers, and requires an explicit "not available in the
   dataset" response when a tool returns nothing relevant (e.g. asking about a platform that
   doesn't exist in the data returns `not_found` plus what IS available, so the AI corrects
   the user rather than guessing).
6. Multi-step reasoning is supported — e.g. "underperforming campaigns" first calls
   `get_numeric_field_stats` to get real percentiles, then uses those as thresholds in
   `filter_campaigns`, rather than a guessed definition of "underperforming."

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
├── site/                       # the static dashboard — deployable as-is
│   ├── index.html
│   └── data.js
├── backend/                     # only needed for the AI Analyst tab
│   ├── main.py
│   ├── data_tools.py
│   ├── requirements.txt
│   ├── test_app.py
│   ├── .env.example
│   └── README.md
└── docs/
```

## Setup

### Option A — Dashboard only (no AI, no Python needed to run it)

Just open `site/index.html` in a browser. Everything except the AI Analyst tab works
immediately, since all data is embedded in `site/data.js`.

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
   (installing dependencies, getting a free Gemini API key, running the server).
4. **Serve the frontend** over a local address (not `file://`):
   ```
   cd site
   python -m http.server 5500 --bind 127.0.0.1
   ```
5. Open `http://127.0.0.1:5500/index.html`, with the backend running at
   `http://127.0.0.1:8000` in a separate terminal.

## Environment variables

All secrets live in `backend/.env` (never committed — see `.gitignore`). See
`backend/.env.example` for the exact variable name and where to get a free key.

## Testing

- `backend/test_app.py` — automated tests of the AI tool-use loop (filter injection,
  hallucination guarding, multi-tool chaining, retry-on-overload, graceful missing-key
  handling) using mocked Gemini responses built from the real SDK's own types.
- Frontend behavior (zero-match filters, search edge cases, sort correctness, chart
  rendering under empty data) was verified with an automated headless-browser test pass
  during development.

## Limitations

- The dataset is synthetic/public, not live production ad-platform data.
- The AI Analyst's free-tier Gemini access is rate-limited (~1,500 requests/day) and can
  occasionally return a transient "servers overloaded" message under high demand — the app
  retries automatically before showing this to the user.
- Filtering, charts, and the campaign ledger operate on the full 10,000-row dataset, but the
  campaign ledger table displays the top 100 matching rows at a time for performance; the KPI
  totals above it reflect the complete filtered set regardless.

## License

MIT — see [`LICENSE`](LICENSE).

---

Built by Hoorab Iqbal as a portfolio project.
