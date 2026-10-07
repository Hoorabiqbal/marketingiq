# Phase 5: final frozen evaluation in Google Colab

Phase 5 runs the Phase 4 model (Qwen3-0.6B + LoRA) **once** on the 429 frozen questions in
`eval_frozen.jsonl` and scores every answer against the gold plan with the real `validate_plan()` and
DuckDB `execute_plan()`. **Nothing is trained.** The adapter is hashed before and after, and the
notebook stops if anything changed. It takes about 15–20 minutes on a T4.

> The results are final. Do not change the prompt, the settings, the model or the checkpoint because
> of them. Doing so would turn the frozen set into development data.

## Before you start: build the evaluation bundle on your laptop

From the repository root (`marketingiq/`):

```
python experiments/tiny_model/notebooks/make_eval_bundle.py
```

This creates `experiments/tiny_model/artifacts/marketingiq_eval_bundle.zip` (about 0.8 MB), containing:

- `eval_frozen.jsonl` (checked against SHA-256 `51966831…c7f2b1`, 429 questions) and `eval_manifest.json`;
- `plan_format.py`, `score.py` and the backend modules and CSV. Each is checked to be byte-identical to
  what Phase 4 trained and validated with. If the backend has changed since, the script refuses to build;
- `evaluation/frozen_eval.py`, the Phase 5 harness.

It never packs training data or secrets.

You can optionally check the harness locally first. No model is needed; this uses the backend venv:

```
venv\Scripts\python experiments/tiny_model/evaluation/frozen_eval.py --self-test
```

It prints `FROZEN EVALUATION INTEGRITY: PASS` and confirms that all 429 gold plans score as matches
against themselves.

## Steps

1. Upload `marketingiq_eval_bundle.zip` to `My Drive/marketingiq_tiny/`, next to the Phase 4 bundle.
   The Phase 4 run folder must still be at `My Drive/marketingiq_tiny/runs/<date-time>_marketingiq-qwen3-0.6b-lora/`.
2. In Colab, choose *File → Upload notebook* and pick
   `experiments/tiny_model/notebooks/marketingiq_qwen3_0_6b_frozen_eval.ipynb`.
3. Set *Runtime → Change runtime type* to **T4 GPU** (the GPU Phase 4 used, so the numerics match).
4. If Drive holds more than one Phase 4 run, set `PHASE4_RUN` in section 0 to the run Phase 4 produced.
   Decide this from the Phase 4 records alone, never from frozen-eval results.
5. Run all cells from top to bottom. Each check stops the notebook with `STOPPED: …` if something is wrong:
   - section 2: the Phase 4 report, prompt contract and base model match;
   - section 4: `FROZEN EVALUATION INTEGRITY: PASS`, `Questions: 429`, and the SHA-256;
   - section 5: the gold-plan self-test is 429/429;
   - section 6: the loaded LoRA tensors are identical to `adapter_model.safetensors`;
   - section 7: all 429 prompts are identical to the Hub chat template, with thinking OFF;
   - section 8: generation, a batch-size-1 consistency diagnostic, and confirmation that the LoRA weights are unchanged;
   - sections 9–10: the scores, then the results are saved to Drive.
6. The results land in `My Drive/marketingiq_tiny/phase5/<date-time>_frozen_eval/`:

```
frozen_eval_summary.md          headline numbers, stage rates, per-category tables
frozen_eval_report.json         everything above + model/adapter hashes, settings, versions, code hashes
frozen_eval_predictions.jsonl   one row per question: raw output and every stage result
frozen_eval_generations.jsonl   the raw generations only (re-score offline with --rescore)
```

Copy the folder to `experiments/tiny_model/results/` (git-ignored) to keep it with the repo. To
re-score the same outputs on the laptop with identical code:
`frozen_eval.py --rescore results/<run>/frozen_eval_generations.jsonl`.

## What the numbers mean

| Metric | Meaning |
|---|---|
| generation / stopped at EOS | The model produced text and ended with `<|im_end|>` (not cut off). |
| parse | The raw output is a JSON object (`json.loads`; no repair). |
| canonicalization | The object has only plan keys, and `to_canonical()` fills in the defaults. |
| validate_plan | Production validation accepts it, or refuses it as not answerable. |
| **Exact plan match** | `score.py` `exact`: every plan field means the same as gold after validation. |
| strict JSON match | The compact JSON is identical to the gold compact plan. |
| **Execution match** | The predicted and gold plans give an identical DuckDB answer (lines and charts). For a not-answerable gold, the question must be refused with the same reason. |
| execution error | A valid plan that `execute_plan()` cannot run, including the known `campaign_list` KeyError. |
| false refusal | An answerable question that the model refused. |

## Not part of Phase 5

Phase 5 does not cover training, GGUF export or quantization, llama.cpp or Ollama, laptop
latency and RAM, the Groq comparison, FastAPI integration, or removing the Query Router.
