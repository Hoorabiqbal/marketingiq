# LLM provider benchmark results (historical)

Raw output of `benchmark_llm_providers.py` from the provider evaluation in Phases 6–7.
Every run used the same 10 `LLM_REQUIRED` questions and the same DuckDB-computed compact
analysis for each question (identical `analysis_sha256` / `analysis_bytes` across runs).

| File | Providers |
|---|---|
| `phase6_qwen_vs_gemini.json` | local `qwen2.5:1.5b-instruct` (Ollama, CPU) and `gemini-3.6-flash` |
| `phase7_gemini_vs_groq_gpt-oss-20b.json` | `gemini-3.6-flash` and Groq `openai/gpt-oss-20b` |
| `phase7_groq_gpt-oss-120b.json` | Groq `openai/gpt-oss-120b` only |

The JSON "unsupported numbers" field is the automatic check only: it confirms that each
number in an answer appears in the analysis. It does not catch wrong comparisons or numbers
attached to the wrong label, and it splits numbers written with narrow-space thousands
separators (most of Groq 20b's 17 "unsupported" numbers are that artifact). The results
below come from reading every answer against the analysis by hand.

## Results

| | Gemini | Groq gpt-oss-20b | Groq gpt-oss-120b | Qwen 2.5 1.5B (local) |
|---|---|---|---|---|
| Answered | 10/10 | 10/10 | 9/10 (Q10 hit the free-tier 8K tokens/min limit) | 5/10 (5 timeouts at 90 s) |
| Median provider latency | 8.7 s | 0.8 s | 1.0 s | 75.6 s |
| Questions with a material error | 1 | 7 | 3 | all 5 answered |

- **Gemini:** one miscount (Q6 says 5 of 10 sampled campaigns have ROAS below 1; it is 4).
  Flagged insufficient data where appropriate (Q2, Q5, Q6).
- **Groq gpt-oss-20b:** wrong comparisons (Q1, Q2, Q7), creative age read as audience age
  (Q3, Q9), values attached to the wrong month (Q4), overall ROAS attributed to a subgroup (Q6).
- **Groq gpt-oss-120b:** a wrong conversion-rate comparison (Q1), an invented monthly low
  (Q4), a wrong "exceed 4" claim (Q6). Q10, re-run once separately with the identical
  payload after the rate limit, was correct.
- **Qwen:** invented numbers and trends, wrong metric labels, generic or invented reasons.

## Decision (after Phase 7)

Gemini stays the default provider. Groq stays available as an optional hosted provider
(`LLM_PROVIDER=groq`). The local Qwen/Ollama provider is removed (too slow on CPU and not
grounded enough). There is no automatic failover between providers.

## Issues found that affect every provider

Addressed in the grounding phase (`grounding.py`; these answers are regression fixtures in
`test_grounding.py`):

1. Creative-fatigue buckets now carry `unit: "days"` and say they are not audience age.
2. Filtered questions now get `aggregates` for the whole matching group, with the example
   rows labelled separately as individual campaigns.
3. "Explain the decline in CTR": `analysis_scope` now states that no CTR time series is
   supplied, so a change over time can't be established. The data still has no CTR series.
4. A numerical-claim validator now checks units, adjacent metric labels, entity, month and
   scope, not only whether a number appears. It still can't check comparisons between two
   supported values.
