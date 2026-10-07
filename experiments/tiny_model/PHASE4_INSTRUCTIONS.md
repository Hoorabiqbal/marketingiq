# Phase 4: train the tiny planner in Google Colab

This guide assumes you have never fine-tuned a model before. You will teach `unsloth/Qwen3-0.6B`
to turn a MarketingIQ question into a compact MarketingIQ plan. Training runs on a free Google
Colab GPU, not on your laptop. It takes about 30–45 minutes in total, of which training itself
is roughly 10–20 minutes on a T4.

> **Do not upload or use `eval_frozen.jsonl`.** The 429 frozen questions are the final exam for
> a later phase. They are not needed here, and the bundle you upload does not contain them. The
> notebook refuses to run if that file shows up.

You need a Google account (for Colab and Google Drive). You do not need an API key, a Groq key,
or a Hugging Face token.

## Before you start: build the upload bundle on your laptop

From the repository root (`marketingiq/`), run:

```
python experiments/tiny_model/notebooks/make_colab_bundle.py
```

This takes a second and runs no model. It creates
`experiments/tiny_model/artifacts/marketingiq_colab_bundle.zip` (about 1 MB), which contains:

- `train.jsonl` and `validation.jsonl`,
- `plan_format.py` (the prompt format) and `score.py` (the plan scorer),
- the read-only MarketingIQ backend modules and campaign CSV that `validate_plan()` and
  `execute_plan()` need to check the model's plans.

The script refuses to run if the training data no longer matches its frozen hashes. It never
packs `eval_frozen.jsonl`, `.env` files or any secrets.

## Steps

1. **Open Google Colab.** Go to <https://colab.research.google.com> and sign in.

2. **Open the notebook.** Choose *File → Upload notebook* and pick
   `experiments/tiny_model/notebooks/marketingiq_qwen3_0_6b_finetune.ipynb`.

3. **Switch the runtime to a GPU.** Choose *Runtime → Change runtime type*, select **T4 GPU**
   (or any better GPU you have access to) and click *Save*.

4. **Confirm the GPU.** Run the first code cell of section 1 (click the ▶ button next to it). It
   prints the `nvidia-smi` table and a line such as `GPU: Tesla T4 | VRAM: 14.7 GB`. If it prints
   `STOPPED: No CUDA GPU`, repeat step 3.

5. **Make the training data available.** In Google Drive, create a folder named
   `marketingiq_tiny` directly under *My Drive*, and upload `marketingiq_colab_bundle.zip` into it.
   The expected path is `My Drive/marketingiq_tiny/marketingiq_colab_bundle.zip`. Do not upload
   `eval_frozen.jsonl`.

6. **Run the cells from top to bottom.** Click each ▶ in order, or use *Runtime → Run all*. The
   install cell takes 1–3 minutes and prints nothing (its output is hidden on purpose). In
   section 2, Colab asks for permission to access Google Drive; allow it.

7. **Check the hashes (section 3).** Both files must show `OK`:
   ```
   train.jsonl       SHA-256 55452c8e687b9fb78c4ca33b466534cbf63e4039a0be40ba8814c628c9aca007  OK
   validation.jsonl  SHA-256 e247278e7ca147e9e2ea8786784d8409a930c2b2b357afa8d54ddbe672c35e73  OK
   ```
   If either shows `MISMATCH`, the notebook stops with `Dataset integrity check failed - DO NOT
   TRAIN`. Rebuild the bundle on your laptop and upload it again.

8. **Confirm the row counts (sections 3 and 4).** You should see `rows: 4524 (expected 4524)` for
   training and `rows: 498 (expected 498)` for validation. Section 4 also prints
   `DuckDB plan execution available: True`.

9. **Review the token-length analysis (section 6).** A table shows the longest example and the
   p50/p90/p95/p99 lengths. With the current data, the longest example is about 165 tokens, so
   the notebook selects `MAX_SEQ_LENGTH = 256` and no example is cut off.

   Sections 7–10 load the model and check that the loss is answer-only. Section 10 must print
   `prompt tokens never trained` for both splits, followed by an example in which only the plan
   tokens are marked `[LOSS]`.

10. **Start training.** Run the cell under **▶ START GPU TRAINING HERE** (section 11).

11. **Wait for the 2 epochs to finish.** A progress bar counts about 566 steps. Training loss is
    printed every 10 steps and validation loss every 50. Keep the browser tab open: Colab may
    disconnect a session that is idle for a long time.

12. **Run validation (section 12).** The model writes a plan for each of the 498 validation
    questions, and each plan is checked by the real MarketingIQ `validate_plan()` and executed on
    DuckDB. This takes a few minutes.

13. **Review the metrics.** Section 12 prints the overall scores and a per-category table.
    Section 13 shows a few example predictions.

    | Metric | Meaning |
    |---|---|
    | parse rate | The output is valid JSON. |
    | validate_plan rate | Production validation accepts the plan. |
    | exact match (strict) | The predicted plan is identical to the gold plan. |
    | exact match (semantic) | Every field means the same thing after validation. |
    | execution match | Running the predicted plan gives the same answer as the gold plan. |

    Validation loss alone does not show whether the plans are right; these metrics do.

14. **Save the LoRA to Google Drive (sections 14 and 15).** Section 14 saves the adapter,
    tokenizer, report, training log and predictions to `/content/outputs/`, which is **deleted when
    the Colab session ends**. Section 15 copies everything to
    `My Drive/marketingiq_tiny/runs/<date-time>_marketingiq-qwen3-0.6b-lora/` and verifies the
    copy. Do not close Colab before section 15 prints `Saved to Google Drive`.

15. **Download or keep the training report.** `training_report.json` in that Drive folder records
    the model ID, data hashes, settings, losses and validation metrics. You can also download it
    with the command that section 15 prints.

## What you end up with (in Google Drive)

```
marketingiq_tiny/runs/<date-time>_marketingiq-qwen3-0.6b-lora/
  adapter/                       LoRA weights, adapter config, tokenizer, prompt contract
  training_report.json           settings, losses, validation metrics
  training_log_history.json      loss and learning rate per logging step
  validation_predictions.jsonl   every validation question, gold plan and model output
  checkpoints/                   end-of-epoch checkpoints
```

## If something goes wrong

- **`STOPPED: ...` messages** mean a safety check failed; the message says which one. Fix the
  cause and re-run from the top.
- **Out of GPU memory:** in section 0, set `BATCH_SIZE = 4` and `GRAD_ACCUM = 4` (the effective
  batch stays 16), then use *Runtime → Restart session* and run everything again.
- **Colab disconnected during training:** the files in `/content` are lost. Reconnect and run
  everything again from the top.
- **Error while installing packages:** use *Runtime → Disconnect and delete runtime*, then run
  again from the top.

## Not part of Phase 4

Phase 4 does not run the frozen 429-question evaluation, export to GGUF, use llama.cpp or Ollama,
measure laptop latency, integrate with FastAPI or deploy anything. Those are later phases.
