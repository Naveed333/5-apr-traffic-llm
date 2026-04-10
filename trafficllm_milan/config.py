# config.py — all parameters in one place
import os

DATASET_PATH = 'milan_traffic.csv'
GRID_CELL_ID = '4455'
LOCATION = '45.4642N 9.1900E'

TOTAL_DAYS = 57   # TRAIN_DAYS + TEST_DAYS + 1 = 42 + 14 + 1
HOURS_PER_DAY = 24
W = 24       # input window (lookback hours) — override with --window-size
L = 24       # predict horizon: one day ahead

TRAIN_DAYS = 42
TEST_DAYS = 14

CONVERGENCE_THRESHOLD = 0.1
MAX_ITERATIONS = 10

VLLM_BASE_URL = 'http://localhost:8000/v1'
VLLM_MODEL = '/workspace/models/llama-3.1-8b'
# VLLM_MODEL = '/workspace/models/llama-3.1-1b'

TEMPERATURE = 0.0

CONTEXT_SIZES = [4096, 65536]
# CONTEXT_SIZE = 4096   # active context — change per experiment
CONTEXT_SIZE = 65536   # active context — change per experiment
# CONTEXT_SIZE = 131072   # active context — change per experiment

TOKENS_RESERVE = 500  # for p_input + p_ques + output

# Output directory — named by W so each window-size ablation is isolated.
# main.py patches this when --window-size is passed.
OUTPUTS_BASE = os.path.join('outputs', f'W{W}')
