# TrafficLLM Milan

MS Thesis implementation — LUMS SBASSE
Supervisor: Dr. Naveed Ul Hassan
Paper: arXiv:2408.10390v2 — TrafficLLM

Dataset: Milan internet traffic (Barlacchi et al. 2015), grid cell 4455, central Milan
Model: Llama-3.1-8B-Instruct via vLLM on Vast.ai
Option: 3 — Full Equation 5 context in p_exam

---

## Setup

```bash
pip install -r requirements.txt
```

Place your dataset file in this directory:
```
trafficllm_milan/milan_traffic.csv
```
Expected columns: `timestamp`, `traffic_mb`

---

## Running the Pipeline

### Step 1 — Verify dataset
```bash
cd trafficllm_milan
python main.py --phase data
```
Expected: 720 total values, 21 train windows, 8 test windows.

### Step 2 — Start vLLM on Vast.ai
```bash
# On Vast.ai instance:
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 131072

# SSH tunnel:
ssh -L 8000:localhost:8000 user@vastai_ip
```

### Step 3 — Test connection
```bash
python main.py --phase connection
```

### Step 4 — Dry run (window 1 only)
```bash
python main.py --phase dry-run
```

### Step 5 — Full training (Algorithm 1 on Days 1–21)
```bash
python main.py --phase train
```
Saves records to `outputs/training_records/window_{t}.json` after each window.
Safe to interrupt and resume — already-completed windows are skipped.

### Step 6 — Build p_exam
```bash
python main.py --phase pexam
```
Builds `outputs/p_exam/p_exam_{4096,65536,131072}.txt`

### Step 7 — Single test prediction
```bash
python main.py --phase test
python main.py --phase test --context-size 65536
```

### Step 8 — Evaluate all test windows
```bash
python main.py --phase evaluate
python main.py --phase evaluate --context-size 65536
python main.py --phase evaluate --context-size 131072
```

### Run everything at once
```bash
python main.py --all
```

---

## Switching Context Size

Training records are reused across all context sizes. To switch:

```python
# Edit config.py:
CONTEXT_SIZE = 65536   # or 131072
```

Or pass on the command line:
```bash
python main.py --phase evaluate --context-size 65536
```

---

## Output Files

| Path | Contents |
|------|----------|
| `outputs/training_records/window_{t}.json` | Algorithm 1 result per training window |
| `outputs/p_exam/p_exam_{ctx}.txt` | Frozen p_exam string for each context size |
| `outputs/results/results_{ctx}.json` | MAE/MSE results per context size |
| `outputs/llm_calls_log.jsonl` | Every LLM call logged (prompt tail + response head) |

---

## Experiment Matrix (3×2)

| Run | Model | Context | p_exam examples (est.) |
|-----|-------|---------|------------------------|
| 1 | Llama-3.1-8B | 4K | ~6 |
| 2 | Llama-3.1-8B | 64K | ~114 |
| 3 | Llama-3.1-8B | 128K | ~229 |
| 4 | Llama-3.1-1B | 4K | ~6 |
| 5 | Llama-3.1-1B | 64K | ~114 |
| 6 | Llama-3.1-1B | 128K | ~229 |

To switch model, update `VLLM_MODEL` in `config.py`.

---

## Expected Results (reference)

Paper reports for Milan dataset:
- GPT-4 (no refinement): MAE = 14.92 MB
- TrafficLLM (GPT-4): MAE = 12.37 MB (17.09% improvement)
