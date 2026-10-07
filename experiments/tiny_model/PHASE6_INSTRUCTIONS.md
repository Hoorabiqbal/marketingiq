# Phase 6: merge, GGUF export, quantization and local CPU benchmark

Phase 6 changes how the model is stored, not what it learned:

```
unsloth/Qwen3-0.6B + Phase 4 LoRA  ->  merged 16-bit model  ->  GGUF (F16)  ->  Q8_0 and Q4_K_M  ->  llama.cpp on the laptop CPU
```

**Nothing is trained.** The notebook checks that the adapter is byte-identical to the one Phase 5 evaluated
before it exports anything. `eval_frozen.jsonl` is not used. The smoke test and the benchmark use 30 questions
from `validation.jsonl` (`benchmark/benchmark_questions.jsonl`). Production code (backend, FastAPI, frontend,
Query Router, Netlify/Render) is not touched, and no Groq call is made.

The work happens in two places:

- **Part A (Google Colab, about 30 minutes):** merge, convert, quantize and smoke-test. This runs in Colab
  because the adapter is stored in Google Drive.
- **Part B (the ThinkPad X395, Windows 11):** run the GGUFs with llama.cpp on the CPU and benchmark them.

---

## Part A: export in Google Colab

### 1. Build the export bundle on your laptop

From the repository root:

```
venv\Scripts\python experiments\tiny_model\notebooks\make_export_bundle.py
```

This writes `experiments/tiny_model/artifacts/marketingiq_export_bundle.zip` (about 0.8 MB). The script refuses
to build if `plan_format.py`, `score.py`, `frozen_eval.py`, the backend modules or the CSV differ from what
Phase 4/5 used. The zip contains no frozen evaluation set, no training data and no secrets.

### 2. Upload and run

1. Upload the zip to `My Drive/marketingiq_tiny/`. The Phase 4 run (`runs/…_marketingiq-qwen3-0.6b-lora/`) and the
   Phase 5 results (`phase5/…_frozen_eval/`) must still be in that folder.
2. In Colab, choose *File → Upload notebook* and select
   `experiments/tiny_model/notebooks/marketingiq_qwen3_0_6b_export_gguf.ipynb`.
3. Choose *Runtime → Change runtime type → T4 GPU*. This is the GPU Phase 4 and 5 used, so the merge runs in the
   same fp16 precision Phase 5 evaluated in.
4. If Drive holds more than one Phase 4 run or Phase 5 result, set `PHASE4_RUN` / `PHASE5_RUN` in section 0.
5. Run every cell from top to bottom. Each check stops the notebook with `STOPPED: …` if something is wrong.

| Section | What it checks |
|---|---|
| 2 | `ADAPTER VERIFICATION: PASS`. All adapter files are present, every adapter file's SHA-256 matches `frozen_eval_report.json`, and Phase 5 recorded 391/429 exact and 383/429 execution. |
| 3 | The bundle code matches the code Phase 5 recorded, and the gold self-test passes on the 30 benchmark questions. |
| 4 | Loads the model exactly as Phase 5 did. The LoRA tensors match the adapter file and the Phase 5 fingerprint. Every prompt is identical across the HF chat template (thinking OFF), `llama_client.render_prompt()` and the Hub template. |
| 5 | Reference plans from the unmerged model (greedy, batch 1). |
| 6 | Unsloth `save_pretrained_merged(save_method="merged_16bit")`. **WEIGHT CHECK: PASS**: every merged matrix equals `W_base + (alpha/r)·B·A`, and every other tensor equals the base. Also reports how many of the 30 merged-model outputs are identical to the reference. |
| 7 | Clones llama.cpp at tag `b11461` (the commit is recorded), builds it for CPU, then runs `convert_hf_to_gguf.py` (F16) and `llama-quantize` (Q8_0, Q4_K_M). **GGUF METADATA: PASS** means: qwen3, EOS = `<\|im_end\|>`, no BOS, chat template embedded. |
| 8 | **Smoke test** per GGUF through `llama-server`: llama.cpp tokenization equals the HF token ids, and on the 8 smoke questions every output stops at EOS, has no thinking, parses, canonicalizes and passes `validate_plan()`. |
| 9 | Saves everything to `My Drive/marketingiq_tiny/phase6/<date-time>_export/` and verifies each copy's SHA-256. |

Nothing is tuned on the smoke results. If a GGUF fails, diagnose the export (tokenizer, template, EOS,
conversion). Do not change the model or the training settings.

### 3. Bring the files to the laptop

From the Drive folder `phase6/<date-time>_export/export/`, download these files into
`experiments\tiny_model\export\` (create the folder). It is git-ignored, so the model files are never committed.

```
experiments\tiny_model\export\
    marketingiq-qwen3-0.6b-Q8_0.gguf      (~0.6 GB)
    marketingiq-qwen3-0.6b-Q4_K_M.gguf    (~0.4 GB)
    marketingiq_runtime.json              prompt contract, n_predict, GGUF SHA-256, llama.cpp tag
```

Keep `phase6_export_report.json` with them (or in `experiments\tiny_model\results\`).

---

## Part B: llama.cpp on the ThinkPad X395 (CPU)

The ThinkPad has a Ryzen 5 3500U (4 cores / 8 threads, AVX2), a Radeon Vega 8 iGPU and about 13.6 GB of usable
RAM. Inference runs on the CPU. CUDA does not apply (there is no NVIDIA GPU). A Vulkan build exists, but it is
not needed for this benchmark.

All commands are for **PowerShell**, run from the repository root.

### 1. Get llama.cpp (prebuilt, same tag as the export)

1. Download `llama-b11461-bin-win-cpu-x64.zip` from <https://github.com/ggml-org/llama.cpp/releases/tag/b11461>.
2. Extract it to `C:\llama.cpp\`, then unblock the files:
   ```powershell
   Get-ChildItem C:\llama.cpp -Recurse | Unblock-File
   C:\llama.cpp\llama-server.exe --version     # version: ... (build 11461, commit 4d756bc72)
   ```
   The zip contains one CPU backend per instruction set. llama.cpp picks the best one for the Ryzen
   automatically (AVX2).

(Optional) Build from source instead. You need Git, CMake and Visual Studio 2022 Build Tools (C++):

```powershell
git clone --depth 1 --branch b11461 https://github.com/ggml-org/llama.cpp C:\src\llama.cpp
cmake -S C:\src\llama.cpp -B C:\src\llama.cpp\build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build C:\src\llama.cpp\build --config Release -j 8 --target llama-server llama-cli llama-completion
# binaries: C:\src\llama.cpp\build\bin\Release\
```

### 2. Settings for compact plans

| Setting | Value | Why |
|---|---|---|
| Threads `-t` | **4** to start | One per physical core. For token generation, using all 8 SMT threads is usually slower on this CPU: a base-model dry run on this laptop measured 18.0 tok/s at 4 threads vs 14.6 at 8. Configurable. |
| Batch threads `-tb` | 4 (also try 8) | Prompt processing can gain from SMT (the same dry run: 436 ms → 337 ms). |
| Context `-c` | 512 | Prompts are ≤ ~110 tokens and plans ≤ `max_new_tokens`, so 512 has headroom and keeps the KV cache small. |
| Max tokens `-n` / `n_predict` | `max_new_tokens` from `marketingiq_runtime.json` | The Phase 4 contract (longest training plan + 32). Plans end at `<\|im_end\|>` long before this. |
| Sampling | `--temp 0 --top-k 1` | Greedy and deterministic, like Phase 4/5. No other samplers needed. |
| GPU layers `-ngl` | 0 | CPU only. |
| Thinking | **OFF** | See below. |

**How thinking stays OFF.** The model was trained on prompts that end with an empty think block
(`<|im_start|>assistant\n<think>\n\n</think>\n\n`). Each run mode below reproduces that:

- *Raw prompt* (`llama-completion -f`, server `/completion`): the prompt itself contains the empty think block.
  The benchmark uses this mode.
- *Chat* (`llama-cli`, server `/v1/chat/completions`): pass `--reasoning off` or
  `chat_template_kwargs: {"enable_thinking": false}`. **If you leave this out, Qwen3 thinks by default.** In the
  dry run, a request without it spent the whole token budget on reasoning and returned no plan.

With thinking disabled, llama.cpp's embedded template produces exactly the training prompt (checked: 30/30
identical, with token ids identical to Hugging Face's).

### 3. One question with llama-completion (raw prompt, exact training format)

The prompt contains special tokens and newlines, so write it to a file first:

```powershell
venv\Scripts\python -c "import sys; sys.path.insert(0, 'experiments/tiny_model/benchmark'); import llama_client as lc; open('prompt.txt', 'w', encoding='utf-8', newline='\n').write(lc.render_prompt('revenue by platform'))"

C:\llama.cpp\llama-completion.exe -m experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q8_0.gguf `
    -f prompt.txt -no-cnv --no-display-prompt -n 96 --temp 0 --top-k 1 -t 4 -c 512 -ngl 0
```

To use Q4_K_M, swap in `marketingiq-qwen3-0.6b-Q4_K_M.gguf`. Use the `max_new_tokens` value from
`marketingiq_runtime.json` for `-n`. The output is one compact JSON plan followed by `[end of text]` (the EOS stop).
Timings are printed at the end of stderr.

### 4. Interactive check with llama-cli (chat mode, thinking off)

```powershell
C:\llama.cpp\llama-cli.exe -m experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q4_K_M.gguf `
    -sys "You are the MarketingIQ planner. Translate the question into a MarketingIQ analytics plan as compact JSON. Never answer the question and never write data values." `
    -p "revenue by platform" -st --reasoning off --temp 0 --top-k 1 -n 96 -t 4 -c 512 -ngl 0
```

`-st` exits after one answer. Leave out `-p` and `-st` to chat. Follow-ups need the previous plan in the user
message, in the `plan_format.user_prompt()` wording; the benchmark handles that for you.

### 5. Optional: llama-server

```powershell
C:\llama.cpp\llama-server.exe -m experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q8_0.gguf `
    --host 127.0.0.1 --port 8089 -c 512 -t 4 -tb 4 -np 1 -ngl 0 --reasoning off --no-webui
```

```powershell
# raw prompt (exact training format), greedy
$prompt = [IO.File]::ReadAllText("$PWD\prompt.txt")   # not Get-Content: PowerShell 5.1 would JSON-encode extra properties
$body = @{ prompt = $prompt; n_predict = 96; temperature = 0; top_k = 1 } | ConvertTo-Json
(Invoke-RestMethod http://127.0.0.1:8089/completion -Method Post -Body $body -ContentType "application/json").content

# OpenAI-compatible chat (thinking off by --reasoning off, or per request:)
$body = @{ messages = @(@{role="system"; content="You are the MarketingIQ planner. Translate the question into a MarketingIQ analytics plan as compact JSON. Never answer the question and never write data values."},
                        @{role="user"; content="revenue by platform"});
           max_tokens = 96; temperature = 0; top_k = 1; chat_template_kwargs = @{enable_thinking = $false} } | ConvertTo-Json -Depth 4
(Invoke-RestMethod http://127.0.0.1:8089/v1/chat/completions -Method Post -Body $body -ContentType "application/json").choices[0].message.content
```

The server binds to `127.0.0.1` only. Stop it with Ctrl+C.

### 6. Run the benchmark

Before you start, plug in the laptop, set Windows power mode to **Best performance**, and close heavy apps.
Laptop CPU speed depends on power and heat. In the dry run, generation speed varied between 6 and 18 tok/s
across requests.

```powershell
venv\Scripts\python experiments\tiny_model\benchmark\benchmark_local.py `
    --server C:\llama.cpp\llama-server.exe `
    --model experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q8_0.gguf `
    --model experiments\tiny_model\export\marketingiq-qwen3-0.6b-Q4_K_M.gguf `
    --threads 4
```

The script checks each GGUF's SHA-256 against `marketingiq_runtime.json`. Then it starts a fresh `llama-server`
for each model (and for each `--threads` value), sends one untimed warm-up request, and sends the 30 questions one
at a time with greedy decoding.

Useful options:

| Option | Effect |
|---|---|
| `--threads 4 8` | One run per thread count (thread sweep). |
| `--threads-batch 8` | Prompt-processing threads (default: same as `--threads`). |
| `--repeats 3` | Repeat the 30 questions for steadier percentiles. |
| `--no-cache-prompt` | Re-process the system prompt on every request. By default the shared system-prompt prefix (38 tokens) is reused, as a long-running server would. |
| `--limit 5` | Quick check on the first 5 questions. |
| `--ctx`, `--n-predict`, `--port` | Override the defaults (512, the contract's `max_new_tokens`, 8089). |

The script reports, per model × threads:

- startup time (process start until `/health` returns ok);
- prompt tokens and prompt-processing ms;
- time to first token (client side, streamed);
- generated tokens, generation ms and tokens/s;
- end-to-end request latency;
- server RSS after load and after the run, and the Windows peak working set;
- EOS stops, thinking leaks, parse success, `validate_plan()` success;
- exact and execution match against the validation gold plans, for reference only.

Results go to `experiments\tiny_model\benchmark\results\<timestamp>\` (git-ignored): `requests.jsonl`,
`summary.json`, `summary.md` and the server logs.

Compare Q8_0 and Q4_K_M on latency, RAM **and** `validate_plan` / EOS / parse. Neither one is assumed to be
better. The README's viability bar is p95 under 3 s on the X395. Accuracy is Phase 5's number. The benchmark's
agreement with the validation gold is only a sanity check that quantization did not break the model. Do not pick
settings using the frozen set.

## Not part of Phase 6

FastAPI integration, removing or replacing the Query Router, production deployment, Groq comparisons, retraining
and frozen re-evaluation. Whether to replace the router is decided after the laptop benchmark.
