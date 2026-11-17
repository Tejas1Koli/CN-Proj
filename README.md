# Cloud-local hybrid chatbot (API)

Laya decides the tier for every query:

| Query | Tier | Model |
|---|---|---|
| easy | **local** | Gemma 4 E2B, on-device via LiteRT-LM (Metal GPU) |
| complex | **cloud** | Nemotron-3-Ultra 550B via the NVIDIA API |

Routing rule (see `app.py` -> `decide_route`):
`cloud` if Laya's difficulty score >= 1.5, **or** the tool-needs probability >= 0.14, **or** the query is sensitive.

## Run

```bash
source .venv/bin/activate
uvicorn app:app --reload --port 8000
```

## Call

```bash
# let Laya choose
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "What is the capital of France?"}'

# force a tier (testing)
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "Say OK", "force_route": "cloud"}'
```

Response:

```json
{
  "reply": "The capital of France is **Paris**.",
  "route": "local",
  "reason": {"difficulty": 1.1084, "domain": "factual_lookup", "needs_tools": false, "is_sensitive": false},
  "timings_ms": {"decision": 1381.3, "total": 2486.6}
}
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | answer a message, returns the reply + which tier was used |
| GET | `/health` | liveness check |

## Config (`.env`)

| Key | Default |
|---|---|
| `CLOUD_PROVIDER` | `nvidia` (or `openrouter`) |
| `NVIDIA_API_KEY` | required when `CLOUD_PROVIDER=nvidia` |
| `CLOUD_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `CLOUD_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` |
| `LOCAL_MODEL` | `~/.litert-lm/models/gemma4-e2b/model.litertlm` |

## Notes

- The first call is slow (~30 s) because Laya and Gemma both load on demand. Both stay warm after that.
- **Local inference runs on the Metal GPU**, not the CPU: `app.py` passes `backend=GPU(gpu_decode_steps_per_sync=4)` to the LiteRT-LM `Engine` and falls back to `CPU()` if no GPU is available. Measured on an M2 MacBook Air: **51.6 tok/s on Metal vs 30.7 tok/s on CPU** (1.7x). The logs confirm `Selected adapter: Apple M2, arch=metal-3, backend=Metal`.
- `gemma4-e2b` has **no function-calling support**, so tool use can only happen on the cloud tier.
- **Routing to the cloud does not give the model tools.** `nemotron-3-ultra` has no web access, so "search the web for..." is routed to the cloud and the cloud model replies that it cannot browse. Wiring real tools needs a tool-calling model or an actual search tool.
- The cloud model is served by `integrate.api.nvidia.com`, which has no daily request cap.
  Latency still varies: a short factual query returns in ~1 s, long generations take 30 s+.
- Reasoning is disabled on the cloud model (`reasoning: {enabled: false}`) so it cannot
  spend hidden tokens the local model has no way to produce. That keeps the arms comparable.
- Laya logs a warning that one checkpoint temperature is out of range, so confidence for wide `choice` questions is uncalibrated.
- The `needs_tools` threshold is fitted to a small sample (12 prompts) and should be re-tuned against real traffic.

---

# Benchmark (`benchmark.ipynb`)

Research comparison of three systems on identical items:

| Arm | Description |
|---|---|
| **Local only** | Gemma 4 E2B via LiteRT-LM (Metal GPU) |
| **Cloud only** | Nemotron-3-Ultra 550B via the NVIDIA API |
| **Hybrid** | Laya routes each query to local or cloud |

The report is in **`paper/main.pdf`** (9 pages, LaTeX) and **`slides/slides.pptx`**
(13 slides). Every number in both is generated from `results/summary.json`
and `results/raw_paired.jsonl` by `tools/make_paper.py` and
`tools/make_slides.py`, so the documents cannot drift from the experiment.

Tasks: MMLU (5-shot), GSM8K (5-shot), ARC-Challenge (5-shot), BoolQ (3-shot),
HumanEval (0-shot), MBPP (3-shot). Code tasks are scored by sandboxed
execution; choice tasks by strict answer-marker extraction. Accuracy is
reported with 95% Wilson intervals, and arm-vs-arm differences with McNemar's
exact test on paired items.

Two prompt-format details were fixed after measuring them against the local
model, and both apply identically to all three arms:

* **Answer directive on multiple choice.** Without
  `Answer the following multiple-choice question with a single letter
  (A, B, C, D). Do not explain.`, Gemma 4 E2B pre-ambles instead of
  answering. MMLU went from **0/20 to 9/20** scored-correct (and 5/20 → 20/20
  "contains a letter").
* **`[BEGIN]`/`[END]` delimiters on MBPP.** With a prose few-shot format the
  model answers an *exemplar* task instead of the query. MBPP went from
  **0/25 to 10/25** pass@1.

Every result record carries a `prompt_hash`, so changing a prompt invalidates
that item's old record instead of silently mixing two prompts under one id.

## Run it

```bash
source .venv/bin/activate

# notebook (analysis + figures)
jupyter lab benchmark.ipynb

# or from the shell, for accumulating samples across daily quota resets
.venv/bin/python run_bench.py --max-calls 40
```

## Headline results (46 paired items)

| Arm | Accuracy | Wall clock | Output tokens (text) |
|---|---|---|---|
| Local only | 58.7% | 456.1 s | 6,556 |
| Cloud only | 95.7% | 185.3 s | 2,818 |
| Hybrid | 93.5% | 352.8 s | 3,308 |
| Oracle (perfect router) | 95.7% | — | — |

Output tokens are counted by re-tokenising the stored answer text. The runtime's
own counter reports 6,997 / 2,667 because LiteRT-LM includes ~10 chat-template
tokens per item (`tools/recount_tokens.py`).

* **The oracle equals cloud-only**, so no router could have gained accuracy on this
  sample: the cloud's two misses were also missed on-device. Routing bought latency
  and privacy, not correctness. The hybrid lost **2.2 points** by sending one
  HumanEval item to the device, capturing **94.1%** of the local→oracle headroom.
* The cloud is faster overall because it **says less** (2.33× fewer output tokens),
  not because it decodes much faster (2.06× per token).
* Per task the ranking reverses: **MMLU and ARC are 1.6–2.0× faster on-device**;
  **GSM8K and MBPP are 4.6–5.9× faster in the cloud**.
* Routing is **not free**: 156.0 s of the hybrid's 352.8 s (44%) is router time.
  A decision costs 3.1 s in the request path vs 0.36 s on an idle GPU, because the
  router shares the accelerator with the local model (`tools/measure_router.py`).
* Speed tracks **expected output length**, not difficulty: on MMLU/ARC/BoolQ both
  arms emit one token and the device wins on prefill; on MBPP the local model writes
  254 tokens where the cloud writes 21 and loses 5.9×.

Two measurement defects found after the first run are corrected here, both from
stored text with no new API calls:

* **Code scorer join bug.** The extracted program was concatenated straight onto
  the test assertions, so a completion not ending in a newline became
  `return xassert ...` and raised `SyntaxError`. MBPP cloud accuracy was recorded
  as 2/8; it is 7/8. This inverted the earlier claim that the router ignored a task
  where the local model was better. Five correctness labels changed
  (`tools/rescore.py`; the pre-fix file is kept as `raw_paired.original.jsonl`).
* **Token counter.** `conv.token_count` includes the chat template, so a one-letter
  answer was counted as 11 tokens, inflating both local output volume and apparent
  throughput (`tools/recount_tokens.py`).

## Sample size and the API budget

The study is a pilot: **46 items, 6–8 per task**. Only GSM8K reaches significance
(McNemar, p = 0.031); every other per-task test is p ≥ 0.25. Pooled gaps are large
enough to believe, per-task claims are indicative. `BENCH_MODE=full` accumulates a
larger sample incrementally, because the runner is resumable.

`run_bench.py` is quota-aware for OpenRouter (`quota.py`); on the NVIDIA endpoint
there is no daily cap, so the budget is only the wall clock of the run.

## The paired protocol

Rather than three independent passes, every item is evaluated **once per
backend** and the hybrid arm is *derived* from those same results:

```
local_answer  = Gemma 4 E2B     (no API call)
cloud_answer  = Nemotron         (1 API call)
route         = Laya.decision    (no API call)
hybrid_answer = cloud_answer if route == "cloud" else local_answer
```

That halves the API cost (one request per item instead of two), makes the
comparison exactly paired, and yields the **oracle routing** upper bound for
free. A small end-to-end hybrid pass validates that the derivation matches
the real service.

## Layout

```
benchmark.ipynb      the research notebook
run_bench.py         CLI runner (resumable, quota-aware)
app.py               the FastAPI hybrid service
bench/
  config.py          paths, thresholds, sample sizes, seed
  data.py            dataset loading, prompt construction, scorers
  models.py          LocalModel / CloudModel / HybridModel adapters
  protocol.py        the paired protocol + end-to-end validation
  runner.py          execution, aggregation, statistics
  stats.py           Wilson intervals, McNemar, bootstrap
  hardware.py        host profiling, peak-RSS sampling
  quota.py           provider-aware quota management
  plots.py           figures
  exec_sandbox.py    sandboxed execution of model-generated code
paper/
  main.tex           generated source of the report
  main.pdf           the compiled report (9 pages)
slides/
  slides.pptx        the deck (13 slides)
tools/
  make_paper.py      rebuilds paper/main.tex from results/, then compiles
  make_slides.py     rebuilds slides/slides.pptx from results/
  measure_router.py  isolated + deployed-path router latency
  build_notebook.py  regenerates the notebook cells
results/
  raw_paired.jsonl   one JSON record per evaluated item (resumable)
  raw_systems.jsonl  per-system passes
  summary.json       aggregated numbers behind every table and figure
  router_latency_clean.json  router cost, idle vs shared accelerator
  figures/           generated PNGs
```

## Rebuild the write-up

```bash
PYTHONPATH="$PWD" .venv/bin/python tools/make_paper.py
cd paper && pdflatex main.tex && pdflatex main.tex

PYTHONPATH="$PWD" .venv/bin/python tools/make_slides.py
```

## Modes

| `BENCH_MODE` | Items per task | Notes |
|---|---|---|
| `smoke` | 4–6 | pipeline check |
| `quick` (default) | 6 | 36 items, ~46 paired records — the pilot reported here |
| `full` | 164–500 | ~1864 items, several hours on the NVIDIA endpoint |
