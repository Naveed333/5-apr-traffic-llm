# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MS Thesis implementation of [TrafficLLM (arXiv:2408.10390v2)](https://arxiv.org/abs/2408.10390) at LUMS SBASSE. The system uses an LLM (Llama-3.1-8B-Instruct via vLLM) to forecast internet traffic using iterative refinement with feedback — applied to Milan hourly traffic data (grid cell 4455).

## Running the Pipeline

All commands run from `trafficllm_milan/`:

```bash
python main.py --phase data          # Validate dataset (expects 720 values, 21 train/8 test windows)
python main.py --phase connection    # Test vLLM API at localhost:8000/v1
python main.py --phase dry-run       # Algorithm 1 on window 1 only (quick smoke test)
python main.py --phase train         # Run Algorithm 1 on all 21 training windows (resumable)
python main.py --phase pexam         # Build frozen p_exam files for each context size
python main.py --phase test          # Single test-window prediction + metrics
python main.py --phase evaluate      # Evaluate all 8 test windows, save MAE/MSE
python main.py --all                 # Run all phases sequentially
```

Additional flags: `--context-size [4096|65536|131072]`, `--no-skip` (re-run already-done windows).

**Prerequisite**: vLLM must be running (`localhost:8000/v1`) serving a model at `/workspace/models/llama-3.1-8b`.

## Architecture

The pipeline has two distinct phases: **training** (Algorithm 1 iterative refinement) and **inference** (single-shot with frozen examples).

### Key Data Flow

```
milan_traffic.csv → 720 hourly values → 56 sliding windows (W=24, L=24)
                                         ↓           ↓
                                   21 train      8 test
                                         ↓
                              Algorithm 1 (per window)
                                         ↓
                              training_records/window_{t}.json
                                         ↓
                              p_exam_{context_size}.txt  (frozen)
                                         ↓
                              inference: p_exam + p_input + p_ques → LLM → parse → MAE/MSE
```

### Module Responsibilities

| File | Role |
|------|------|
| `config.py` | All parameters: dataset path, window sizes, convergence, vLLM endpoint, context sizes |
| `data_loader.py` | CSV loading, window creation, train/test split |
| `llm_client.py` | vLLM API wrapper (OpenAI-compatible); logs all calls to `outputs/llm_calls_log.jsonl` |
| `prompt_builder.py` | Prompt components: `p_input`, `p_ques`, `p_feed`, `p_refine`, validation/critique prompts |
| `algorithm1.py` | Algorithm 1: initial prediction (Eq. 3) then iterative feedback→validate→critique→refine loop (Eq. 5) until convergence or MAX_ITERATIONS |
| `training.py` | Runs Algorithm 1 over all training windows; caches results; supports resume |
| `pexam_builder.py` | Greedy token-budget packing of training records into p_exam for each context size |
| `inference.py` | Test-time: loads frozen p_exam, single LLM call per window |
| `evaluator.py` | Runs inference on 8 test windows, aggregates MAE/MSE, prints thesis-format tables |
| `main.py` | CLI orchestrator for all phases |

### Algorithm 1 (core loop in `algorithm1.py`)

1. **Eq. 3** — Initial prediction: `p_exam + p_input + p_ques → y_hat_0`
2. **Eq. 5 loop** (up to `MAX_ITERATIONS=10`, convergence threshold `delta_MAE < 0.1`):
   - Generate 4-question feedback (MAE analysis, Fourier analysis, alignment check, method ID)
   - Self-validate feedback
   - Self-critique (correct mistakes)
   - Refine prediction using full Eq. 5 context
3. Save complete record (all iterations, predictions, MAE progression, convergence flag)

### p_exam Construction (Option 3 — Full Eq. 5 context)

`pexam_builder.py` greedily fills the token budget (`context_size - 500`) with complete training examples that include the entire refinement history (feedback + all iterations). More context → more examples packed in.

## Outputs

```
outputs/
├── training_records/window_{1..42}.json   # Per-window Algorithm 1 records
├── p_exam/p_exam_{4096,65536,131072}.txt  # Frozen few-shot context per context size
├── results/results_{context_size}.json   # Evaluation results (MAE/MSE per test window)
└── llm_calls_log.jsonl                   # All LLM calls (gitignored; can be large)
```

Training records and p_exam files are gitignored (reproducible from scratch). Results are also gitignored.

## Experiment Matrix

The thesis evaluates 3 context sizes × 2 models (8B and 1B) = 6 runs:

| Context | Token budget | Approx. examples |
|---------|-------------|-----------------|
| 4096    | 3596        | ~6              |
| 65536   | 65036       | ~114            |
| 131072  | 130572      | ~229            |

Currently configured model: `Llama-3.1-8B-Instruct` at context `65536` (edit `config.py` to change).

## Paper Reference Baselines

- GPT-4 no refinement: MAE = 14.92 MB
- TrafficLLM (GPT-4 + Algorithm 1): MAE = 12.37 MB (17.09% improvement)
