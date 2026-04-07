# config.py — all parameters in one place

DATASET_PATH = 'milan_traffic.csv'
GRID_CELL_ID = '4455'

TOTAL_DAYS = 60
HOURS_PER_DAY = 24
W = 24       # input window: one day of history
L = 24       # predict horizon: one day ahead

TRAIN_DAYS = 42
TEST_DAYS = 14

CONVERGENCE_THRESHOLD = 0.1
MAX_ITERATIONS = 10

VLLM_BASE_URL = 'http://localhost:8000/v1'
# VLLM_MODEL = '/workspace/models/llama-3.1-8b'
VLLM_MODEL = '/workspace/models/llama-3.1-1b'

TEMPERATURE = 0.0

CONTEXT_SIZES = [4096, 65536, 131072]
CONTEXT_SIZE = 4096   # active context — change per experiment
# CONTEXT_SIZE = 65536   # active context — change per experiment
# CONTEXT_SIZE = 131072   # active context — change per experiment

TOKENS_RESERVE = 500  # for p_input + p_ques + output
