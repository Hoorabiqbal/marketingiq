# Tiny local planner experiment (Phase 1: design + scaffolding)

Can a small fine-tuned model, running on a CPU laptop, turn MarketingIQ questions into the **same
validated analytics plan** the Groq planner produces today?

This experiment is isolated: nothing under `backend/` is changed, Groq stays the only production
LLM, and no model has been trained yet.

## 1. Where the tiny model would sit

Production today (`backend/chat_routing.py`):

```
question ─ small talk / memory follow-up (conversation.py) ........ 0 LLM calls
         ─ schema parse of complex shapes (semantic_parser.py) ...... 0 LLM calls
         ─ explicit chart (chart_builder.py) ........................ 0 LLM calls
         ─ Query Router DIRECT_DATABASE ............................. 0 LLM calls
         ─ Query Router LLM_REQUIRED (explanations) ................. Groq explain + grounding
         ─ router can't map it / chart unmapped / follow-up the
           memory can't resolve ─> query_planner.answer()
                                    └ adapter.plan()  ◀── ONLY this call is the experiment's target
                                    └ validate_plan() ─ execute_plan() on DuckDB ─ answer/chart
```

The tiny model is a candidate replacement for **one call**: `adapter.plan()` inside
`query_planner.answer()`. It never sees data, never writes SQL/code, and its output passes the
unchanged `validate_plan()` (allowlists + the dataset's real values). Every fast path above keeps
0 LLM calls. Groq explanations (`LLM_REQUIRED`) are out of scope: the historical benchmark
(`backend/benchmark_results/README.md`) showed a local 1.5B model can't write grounded prose.

## 2. Model choice (checked against Unsloth's model catalog, Sept 2026)

| Candidate | Size | Notes |
|---|---|---|
| **Qwen3-0.6B** (`unsloth/Qwen3-0.6B`) | 0.6B | **Benchmark first.** Text-only, strong JSON/instruction following for its size, Unsloth 16-bit + bnb-4bit, official GGUF, runs in llama.cpp **and** Ollama. Non-thinking mode (`enable_thinking=False`). |
| Gemma 3 270M (`unsloth/gemma-3-270m-it`) | 0.27B | Floor baseline: ~2× faster, but less headroom for paraphrase generalization. Official Unsloth notebook exists. |
| Qwen3-1.7B / Gemma 3 1B / Llama 3.2 1B | 1–1.7B | Step up **only** if 0.6B misses the targets in §8. |
| Qwen3.5-0.8B / 2B | 0.8–2B | Not first: multimodal (vision tower), and Unsloth notes its GGUFs don't run in Ollama. |
| Gemma 4 E2B | ~2B effective | Too heavy for this laptop (multimodal, larger real weights). |

Why Qwen3-0.6B: the task is a closed-vocabulary semantic parse (17 metrics, 20 dimensions, ~100
values, 9 intents) into short JSON. That needs reliable instruction following and paraphrase
understanding more than world knowledge; 0.6B is the smallest size with a good track record for
structured output, and ~0.6–1 GB RAM when quantized.

## 3. Training objective

Supervised fine-tuning, completion-only loss:

```
input : fixed short system prompt + question (+ previous plan for follow-ups)
target: the canonical MarketingIQ plan in sparse form (defaults omitted)
```

- **Canonical format, not a new schema.** Targets use exactly the keys/values of
  `query_planner.SCHEMA`. `dataset/plan_format.py` drops fields equal to their defaults
  (`to_sparse`) and restores them (`to_canonical`) before `validate_plan()`. Measured on the seed
  set: 88 vs 258 characters per plan (~2.9× less to decode on CPU).
- No campaign rows, numbers or DuckDB results in any example. The model learns how people *ask*,
  not what the data *says*.
- Follow-ups use the same "Previous analysis ... as a plan" wording `query_planner.answer()`
  already sends Groq, so the conversation-memory system is unchanged.
- Short system prompt (~40 tokens) instead of Groq's ~1,300-token catalog prompt: the vocabulary
  is learned, and prefill is the main CPU cost. Trade-off: new dimension values require
  regenerating data and retraining (acceptable for the experiment; noted for any later phase).

## 4. Dataset record schema (JSONL)

```json
{"id": "brk-0142", "template_id": "brk-07", "category": "breakdown", "source": "template",
 "question": "which channel brings in the most sales?",
 "previous": null,
 "plan": {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform"}}
```

`source` ∈ `template | paraphrase | hand | production_log`. `previous` is a sparse plan or null.
`dataset/seed_examples.jsonl` holds 28 hand-labelled examples (every required category, incl.
follow-ups, typos, unsupported metric/dimension, off-topic). They define the format and smoke-test
the pipeline; they are **not** used in the test set.

## 5. How examples are generated and reviewed

1. **Templates × catalog slots** (bulk of train). ~60–80 templates per the categories below, slots
   filled from `schema_catalog` (metric aliases, dimension aliases, real values, months/years in
   2024-01..2026-01, N for top/bottom). The gold plan is built **programmatically** alongside the
   sentence, so labels are exact by construction.
2. **Paraphrase augmentation.** Template sentences reworded (casual, terse, typos, "pls",
   British spelling, reordered clauses). An LLM may propose wordings offline, but the gold plan is
   always the template's, never an LLM label; each paraphrase is spot-checked.
3. **Hand-written questions** (~200), written without looking at templates; the test set's core.
4. **Checks** — `dataset/validate_dataset.py`: every gold plan must pass production
   `validate_plan()` (not_answerable must be refused with the right reason), must be in sparse
   form, and splits are checked for leakage. Plus a manual review of a random 5% per category.

Categories: totals, filtered totals, breakdowns, top/bottom N, rankings with exclude, entity
comparisons, multi-entity trends, monthly/yearly trends, date filters, month-across-years,
year-over-year, change rankings, campaign lists with conditions, relationships/correlations,
scatter, grouped/stacked/100%/pie chart requests, two-dimension breakdowns, catalog-only metrics
(bounce rate, session duration, quality score), follow-ups (metric swap, entity swap, "make this a
chart", grain change, "compare it with X"), ambiguous, unsupported metrics/dimensions, off-topic.

Size: **~3,000–5,000 train, ~400 validation, ~400 test.** Enough to cover every
intent × slot combination several times without memorizing phrasings.

## 6. Split (leakage-proof)

- **Test (~400)**: all hand-written questions + held-out *templates* (≈20% of template IDs never
  seen in training) + held-out paraphrase styles. Test measures new wording, not recall.
- **Validation (~400)**: template-disjoint from test, used only for early stopping/checkpoint
  choice.
- **Train**: the rest.
- `validate_dataset.py train.jsonl val.jsonl test.jsonl` fails on any identical normalized
  question across files or any `template_id` shared with test, and lists near-duplicates
  (word-set Jaccard ≥ 0.8) for review. Entity *values* are a closed set and appear in all
  splits; *combinations and wordings* are what is held out.
- The test set is frozen before training starts and never tuned against.

## 7. Fine-tuning (Unsloth, Colab/Kaggle — not the laptop)

- `FastLanguageModel.from_pretrained("unsloth/Qwen3-0.6B", max_seq_length=512, load_in_4bit=False)`
- **16-bit LoRA** (Unsloth: slightly more accurate than QLoRA; a 0.6B model fits a free T4 in
  16-bit, so 4-bit QLoRA isn't needed). QLoRA only if stepping up to 1.7B on a small GPU.
- r=16, lora_alpha=16–32, dropout 0, all linear projections (`q,k,v,o,gate,up,down_proj`),
  `use_gradient_checkpointing="unsloth"`.
- lr 2e-4, 2–3 epochs, effective batch 16 (2 × 8 accumulation), weight_decay 0.01,
  `train_on_responses_only` (loss on the JSON only).
- Qwen3 chat template with `enable_thinking=False` in training **and** inference (Unsloth warns
  that a template mismatch is the most common cause of bad exported models).
- Best checkpoint by validation exact-match (not loss). Expected cost: minutes on a free T4.
- Export: `model.save_pretrained_gguf(dir, tokenizer, quantization_method="q8_0")` (also try
  `q4_k_m`). LoRA adapters and GGUFs live in `artifacts/` (git-ignored).

## 8. Local inference

**llama.cpp `llama-server`** (prebuilt Windows CPU/Vulkan binaries) on `127.0.0.1`:
- OpenAI-compatible `/v1/chat/completions`, same request shape as the Groq provider.
- `response_format` with `plan_format.sparse_schema()` → grammar-constrained decoding: output is
  always parseable JSON with allowlisted keys/enums. `validate_plan()` still decides.
- Prompt-prefix caching makes the fixed system prompt a one-time cost.
- temperature 0, max_tokens ~128, `enable_thinking=false`.

Ollama is an equivalent fallback for Qwen3 (supported; `format` takes a JSON schema). Estimated
RAM: **~0.6–1.0 GB** for Qwen3-0.6B Q8_0 at 1–2K context (~0.5 GB at Q4_K_M), measured in the
benchmark.

## 9. Evaluation: tiny model vs the current Groq planner

Same frozen test set, same scorer (`evaluation/score.py`):
- **Groq**: `query_planner.system_prompt()` + `SCHEMA` through the existing `LLMAdapter` /
  `GroqProvider` (the production planner call, unchanged).
- **Tiny**: `plan_format.messages()` to local `llama-server`.

Both outputs go through `to_canonical()` → `validate_plan()` → the post-processing `answer()`
applies, then are compared on the validated `Plan`. Reported per model and per category:

| Metric | How |
|---|---|
| JSON validity | output parses |
| Validator acceptance | `validate_plan()` accepts (or correctly refuses) |
| Intent / metric / dimension / entity / filter accuracy | field equal to gold after validation |
| Time interpretation | months, years, grain all equal |
| Chart intent | visualization equal (after the "chart only when asked" rule) |
| Unsupported handling | refused with the right reason; false-refusal rate on answerable ones |
| **Exact plan match** | every field equal — **the headline number** |
| Latency | p50/p95 per question (wall clock, laptop CPU) |
| RAM | llama-server peak working set (psutil) |

Targets to call the tiny model viable: exact match within ~5 points of Groq, ≥ 99% validator
acceptance, 0 answerable questions refused as off-topic above Groq's rate, p95 < 3 s on the X395.
A plan that fails validation is never run in production either way; the risk being measured is
**valid but wrong** plans, which only exact-match accuracy catches.

## Files

```
experiments/tiny_model/
  README.md                     this design
  requirements-train.txt        Colab/Kaggle deps (not the backend's)
  .gitignore                    artifacts/ (LoRA, GGUF), results/
  dataset/plan_format.py        prompt + sparse<->canonical plan; reuses backend validator
  dataset/seed_examples.jsonl   28 hand-labelled format/smoke examples
  dataset/validate_dataset.py   gold-label + leakage checks
  evaluation/score.py           field-level scoring of any planner output vs gold
  evaluation/frozen_eval.py     Phase 5 harness: integrity, prompts, generation, staged scoring, reports
  notebooks/                    Phase 4/5 Colab notebooks + make_colab_bundle.py / make_eval_bundle.py
  PHASE4_INSTRUCTIONS.md        step-by-step Colab training guide
  PHASE5_INSTRUCTIONS.md        step-by-step Colab frozen-evaluation guide
  PHASE6_INSTRUCTIONS.md        merge/GGUF export (Colab) + llama.cpp CPU benchmark (Windows laptop)
  benchmark/                    Phase 6: 30-question local set, llama-server client, benchmark_local.py
  export/                       Phase 6 GGUFs + marketingiq_runtime.json, downloaded from Drive (git-ignored)
```

Run checks locally with the backend venv: `python dataset/validate_dataset.py` (from this folder).

## Phase 2: frozen evaluation set (`eval-v1`)

`dataset/eval_frozen.jsonl` (429 questions) + `dataset/eval_manifest.json` (counts, SHA-256).
Built by `dataset/build_eval.py`, which refuses to overwrite a frozen file. Gold plans are
hand-written against the real plan format; no LLM produced or checked them. Every gold plan passes
`validate_plan()`, and every answerable one runs through `execute_plan()` with status ok
(`validate_dataset.py --execute`).

- 389 hand-written questions + 40 slot-filled breakdowns from 4 frames (`eval-tpl-01..04`).
  Phase 3 must not reuse eval questions, `template_id`s or frames; run
  `validate_dataset.py train.jsonl val.jsonl eval_frozen.jsonl` to check.
- Records add a `style` tag (casual, short, typo, caps, ...) to the Phase 1 fields.
- `evaluation/score.py` compares plans by what they execute to (see its docstring) and reports
  `exact` (all fields) and `execution_match` (identical executed answer).
- Backend issue found (not fixed; production is out of scope): a `campaign_list` whose conditions
  sort by a field outside `chart_builder.CAMPAIGN_METRICS` (e.g. CTR > 5%, conversions >= 500)
  passes `validate_plan()` but `execute_plan()` raises `KeyError`. Such questions are left out of
  the gold set; a model producing such a plan scores as an execution error.

## Phase 3: frozen training data (`train-v1`)

`dataset/train.jsonl` (4,524) + `dataset/validation.jsonl` (498), manifest `dataset/train_manifest.json`
(counts, SHA-256, seed 20260929), rejection log `dataset/train_generation_report.json`. Built by
`generation/generate_training.py` (`families.py` = 46 semantic families, `curated.py` = hand-written
tricky cases, `vocab.py` = catalog-checked wording, `leakage.py`). It refuses to overwrite frozen
files; `--out DIR` reproduces the same bytes elsewhere (checked twice).

- Question and gold plan are built from one semantic spec (or written by hand); no LLM labels.
- Every row: `validate_plan()` pass (not_answerable refused with its reason), unchanged by
  `answer()`'s post-processing, answerable plans `execute_plan()` status ok.
- Leakage vs `eval_frozen.jsonl`: normalized text, word-set Jaccard ≥ 0.8, and *skeleton* match
  (metric/dimension/value/month/number words replaced by slots; identical or Jaccard ≥ 0.75 is
  dropped). Consequence: the bare "`<metric> by/per <dimension>`" forms are eval template families and
  never appear in training; breakdowns are taught through other wordings.
- Train/validation are split by template frame (0 shared template_ids).
- Relative dates ("last year") are not used: the data ends in Jan 2026, so they have no stable meaning.
- The known `campaign_list` execution bug (conditions on ctr/clicks/conversions) was exercised on
  purpose: 28 such examples were rejected by the execution check and logged.

## Phase 4: Colab fine-tuning workflow (prepared, not yet run)

`notebooks/marketingiq_qwen3_0_6b_finetune.ipynb` fine-tunes `unsloth/Qwen3-0.6B` with 16-bit LoRA
(r=16, alpha=32, dropout 0, all q/k/v/o/gate/up/down projections), lr 2e-4, 2 epochs, effective batch 16,
seed 20260929, thinking OFF, answer-only loss (Unsloth `train_on_responses_only`, verified on every row).
Beginner steps: `PHASE4_INSTRUCTIONS.md`.

- Input to Colab: one zip from `notebooks/make_colab_bundle.py` (train/validation data, `plan_format.py`,
  `score.py`, the read-only backend modules + CSV for `validate_plan()`/`execute_plan()`). It never contains
  `eval_frozen.jsonl`; the notebook refuses to run if that file is present.
- Measured on the real data: longest example 165 tokens, so `max_seq_length = 256`; about 26% of tokens carry loss.
- Validation scores parse rate, `validate_plan()` rate, strict and semantic exact match, and execution match
  (real DuckDB) on `validation.jsonl` only. The final model after epoch 2 is kept (no checkpoint selection).
- Outputs (adapter, `training_report.json`, predictions) are copied to Google Drive. No GGUF export in this phase.

## Phase 5: final frozen evaluation (prepared)

`notebooks/marketingiq_qwen3_0_6b_frozen_eval.ipynb` (Colab, no training) runs the Phase 4 adapter once on the
429 frozen questions. Steps: `PHASE5_INSTRUCTIONS.md`. Input: `notebooks/make_eval_bundle.py`, which refuses to
build unless `eval_frozen.jsonl` matches its SHA-256 and `plan_format.py`/`score.py`/backend/CSV are byte-identical
to the Phase 4 bundle.

- Harness: `evaluation/frozen_eval.py` (integrity check, Phase 4 prompt via `plan_format.messages()`, greedy decoding
  with Phase 4 settings, per-stage scoring through `score.py`, reports). `--self-test` runs without a model.
- Records generation, EOS stop, parse, canonicalization, `validate_plan()`, execution attempted/successful/
  match/mismatch/error, false refusals, per-field accuracy and per-category/intent/style tables. Headline numbers:
  `exact` and `execution_match` with Wilson 95% intervals.
- A crash inside `execute_plan()` on a model plan (the known `campaign_list` KeyError) is an execution error, not
  a stop. Adapter files are hashed before/after; live LoRA weights are compared to the file and fingerprinted.

## Phase 6: merge, GGUF export, quantization, local CPU benchmark (prepared)

No training: the representation changes, not the model. Steps: `PHASE6_INSTRUCTIONS.md`.

- **Colab** `notebooks/marketingiq_qwen3_0_6b_export_gguf.ipynb` (input: `notebooks/make_export_bundle.py`, no frozen or
  training data). Verifies that every adapter file's SHA-256 equals the Phase 5 report, and that the live LoRA
  fingerprint equals Phase 5's. Loads the model exactly as Phase 5 did. Merges with Unsloth
  `save_pretrained_merged(save_method="merged_16bit")` and checks the result two ways: every merged matrix equals
  `W + (alpha/r)·B·A` (float32 recomputation), and the merged model alone reproduces the LoRA outputs. Converts with
  llama.cpp `convert_hf_to_gguf.py` (F16, lossless from the fp16 merge) and quantizes with `llama-quantize` to
  **Q8_0** and **Q4_K_M**, pinned to llama.cpp tag `b11461` (commit recorded). It then checks the GGUF metadata
  (qwen3, EOS `<|im_end|>`, no BOS, chat template embedded) and smoke-tests every GGUF through a CPU `llama-server`.
- **Prompt on llama.cpp:** `benchmark/llama_client.render_prompt()` writes the exact Phase 4 string, including the
  empty think block, and sends it to `/completion` as raw text. Thinking is OFF exactly as in training. The
  notebook asserts byte equality with the HF template and token-id equality with llama.cpp's tokenizer. Checked
  on this laptop with llama.cpp b11461: 30/30 identical token ids, no BOS added. With `enable_thinking=false`, the
  embedded template renders 30/30 identical prompts. Without that flag, the chat endpoint lets Qwen3 think by
  default, so the chat route needs `--reasoning off` or `chat_template_kwargs`.
- **Benchmark set:** `benchmark/benchmark_questions.jsonl`, 30 seeded per-category questions from `validation.jsonl`
  (`make_benchmark_set.py`), 8 flagged as the smoke set, 0 in `eval_frozen.jsonl`.
- **Laptop:** `benchmark/benchmark_local.py` starts `llama-server` per GGUF (and per thread count) and measures
  startup, prompt ms, TTFT, generation ms and tok/s, end-to-end latency, RSS and peak working set, EOS, parse and
  `validate_plan()`. It scores with the unchanged Phase 5 `score_example()`. A harness dry run on this X395 with
  the *untuned* base Qwen3-0.6B Q4_K_M (mechanics only; its plans are meaningless): startup ~3.8 s, RSS ~750 MB,
  and 4 threads decoded faster than 8 (18.0 vs 14.6 tok/s).
- Model binaries (`*.gguf`, `*.safetensors`, `export/`, `merged_16bit/`) are git-ignored.

## Phase 7: local Qwen integration (behind a switch)

*Superseded: the switch, the Query Router and Groq were removed in the final architecture below.*

`MARKETINGIQ_PLANNER=local_qwen` routes analytical `/api/chat` questions to `backend/local_planner.py`. That module
makes one greedy call to a persistent llama-server running the Q4_K_M GGUF, using the exact training prompt from
`plan_format` + `llama_client`. The output then goes through parse, canonicalization, `validate_plan()`, and the
existing `execute_plan()` on DuckDB. A follow-up sends only the previous validated plan. Explanations, advice and
campaign IDs keep the existing path. The default `existing` mode is unchanged, and the router is still present.
Run guide and routing table: `PHASE7_LOCAL_QWEN.md`. Tests: `backend/test_local_planner.py` (26, fake
llama-server). Real smoke test: `benchmark/smoke_local_integration.py`.

## Phase 8: final local architecture (router bypass, release validation)

In `local_qwen` mode, normal analytics go **User → Qwen3-0.6B Q4_K_M → compact plan → canonicalization →
`validate_plan()` → DuckDB → deterministic answer/chart**, with 0 Groq calls. The Query Router, schema parser,
old chart path, deterministic follow-up editor and Groq planner no longer take part. Phase 7 still reached them
through its side-path delegation: in a session, "explain that by device" was answered by the old follow-up editor.
The three side paths are now explicit and narrow: small talk (deterministic), campaign-ID lookup (router
`get_campaign` only) and the grounded explanation/advice path (Groq). Neither lookup path ever falls back to router
analytics or the Groq planner. The conversation resolver only detects follow-ups, and Qwen gets the previous
validated plan. Execution errors are caught safely. A validator rejection of an unknown metric or breakdown shows
the validator's own refusal text. `existing` stays the default because production (Render) runs it.
Details, run commands and deployment options: `PHASE7_LOCAL_QWEN.md`. Tests: `backend/test_final_architecture.py`.

## Final architecture (100% local; Query Router and Groq removed)

**User → Qwen3-0.6B Q4_K_M → compact plan → canonicalization → `validate_plan()` → DuckDB →
deterministic answer / findings / recommendation / chart.** Cloud LLM calls: 0. API keys: none.

- A local Qwen explainer was built and tested first (`benchmark/explain_quality_local.py`, kept as
  evidence). It **failed** its quality gate: backwards budget advice, loss-making read as revenue < 0,
  garbled CPA advice. So Qwen only plans. "Why" and advice answers come from
  `backend/findings_engine.py` and `backend/recommendation_engine.py`, which compute deterministically
  from DuckDB rows. On the same 13 gate questions (`benchmark/findings_quality_local.py`) every
  directional check passes and no causal claim is made.
- The Query Router's question classification, the schema parser, the deterministic follow-up editor,
  the Groq planner and explainer, `/api/query` and the `MARKETINGIQ_PLANNER` switch were removed. Data
  execution moved unchanged to `backend/analytics_executor.py`. Small talk, exact campaign-ID lookup
  and follow-up detection remain as narrow helpers that make no analytics decision.
- Run guide, routing table and deployment options: `PHASE7_LOCAL_QWEN.md`. Real end-to-end check:
  `benchmark/smoke_local_integration.py`.

**Historical scripts.** `notebooks/make_colab_bundle.py`, `make_eval_bundle.py` and
`make_export_bundle.py` reproduce the Phase 4–6 experiment state, and their frozen hashes were left
unchanged on purpose. `evaluation/score.py` / `frozen_eval.py` (frozen, unmodified) still score saved
generations. One path is historical: `score.py`'s handler for a data-layer error names the old router
module, which only matters if an execution error occurs. The Phase 5 results and `eval_frozen.jsonl`
were not touched.
