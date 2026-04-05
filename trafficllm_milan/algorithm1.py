# algorithm1.py — full Algorithm 1 implementation (TrafficLLM arXiv:2408.10390v2)

import re
import numpy as np

from config import CONVERGENCE_THRESHOLD, MAX_ITERATIONS, L
from llm_client import call_llm
from prompt_builder import (
    format_input,
    format_output,
    build_p_ques,
    build_p_feed_prompt,
    build_p_refine,
    build_validation_prompt,
    build_critique_prompt,
)


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true, dtype=float) - np.array(y_pred, dtype=float))))


def compute_mse(y_true, y_pred):
    return float(np.mean((np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)) ** 2))


def parse_prediction(text):
    """Parse 24-value prediction from LLM text. Returns list[float] or None."""
    # Pattern 1: H01: 123.45 MB  (canonical)
    matches = re.findall(r'H\d{1,2}:\s*([\d.]+)\s*(?:MB)?', text, re.IGNORECASE)
    if len(matches) >= L:
        return [float(v) for v in matches[:L]]

    # Pattern 2: bare numbers separated by whitespace/commas
    matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if len(matches) >= L:
        return [float(v) for v in matches[:L]]

    # Pattern 3: Python list  [1, 2, 3, ...]
    list_match = re.search(r'\[([^\]]+)\]', text)
    if list_match:
        nums = re.findall(r'[\d.]+', list_match.group(1))
        if len(nums) >= L:
            return [float(v) for v in nums[:L]]

    return None


def call_and_parse(prompt, max_tokens=800):
    """Call LLM and parse prediction. Raises ValueError with raw response if parsing fails."""
    raw = call_llm(prompt, max_tokens=max_tokens)
    result = parse_prediction(raw)
    if result is None:
        raise ValueError(f'parse_prediction failed. Raw LLM response:\n{raw}')
    return result


def extract_method(text):
    """Identify prediction method name from feedback text."""
    patterns = [
        (r'seasonal\s+arima',        'Seasonal ARIMA'),
        (r'arima',                    'ARIMA'),
        (r'lstm\+arima|arima\+lstm',  'LSTM+ARIMA'),
        (r'lstm',                     'LSTM'),
        (r'prophet',                  'Prophet'),
        (r'exponential\s+smoothing',  'Exponential Smoothing'),
        (r'fourier',                  'Fourier'),
        (r'linear\s+regression',      'Linear Regression'),
    ]
    lower = text.lower() if text else ''
    for pattern, label in patterns:
        if re.search(pattern, lower):
            return label
    return 'unknown'


def run_algorithm1(window, p_exam_so_far=''):
    """
    Runs Algorithm 1 (TrafficLLM arXiv:2408.10390v2) on one training window.
    Returns a complete record dict with all iterations for building p_exam.
    """
    x = window['x']
    y = window['y']
    t = window['timestep']

    record = {
        'timestep':     t,
        'date_start':   window.get('date_start', ''),
        'date_end':     window.get('date_end', ''),
        'x':            x,
        'y':            y,
        'iterations':   [],
        'converged':    False,
        'converged_at': None,
    }

    p_input = format_input(x, t)
    p_ques  = build_p_ques()

    # LINE 2: initial prediction — Eq.(3): p_exam + p_input + p_ques
    prompt_eq3 = (p_exam_so_far + '\n' if p_exam_so_far else '') + p_input + '\n' + p_ques
    y_hat_0    = call_and_parse(prompt_eq3)
    mae_prev   = compute_mae(y, y_hat_0)
    y_hat_i    = y_hat_0

    record['y_hat_0'] = y_hat_0
    record['mae_0']   = mae_prev

    p_feed_i   = ''
    y_hat_next = y_hat_0

    # LINE 3-8: refinement loop
    for i in range(MAX_ITERATIONS):
        # LINE 5: generate feedback (uses p_input + current prediction as context)
        feed_context = p_input + '\n' + format_output(y_hat_i) + '\n'
        feedback_raw = call_llm(feed_context + build_p_feed_prompt(y_hat_i, y, mae_prev, i), max_tokens=1024)

        # LINE 6: validate and correct feedback
        validated = call_llm(feed_context + feedback_raw + '\n' + build_validation_prompt(), max_tokens=512)
        p_feed_i  = call_llm(feed_context + feedback_raw + '\n' + validated + '\n' + build_critique_prompt(), max_tokens=512)

        # LINE 7: refine — fresh focused prompt so model doesn't see rambling context
        p_refine_i    = build_p_refine(p_feed_i, target_time=f'Day {window.get("date_end", t + 1)}')
        refine_prompt = p_input + '\n' + p_refine_i
        y_hat_next    = call_and_parse(refine_prompt)

        mae_curr  = compute_mae(y, y_hat_next)
        delta_mae = abs(mae_curr - mae_prev)

        record['iterations'].append({
            'i':          i,
            'y_hat':      y_hat_i,
            'mae':        mae_prev,
            'p_feed':     p_feed_i,
            'p_refine':   p_refine_i,
            'y_hat_next': y_hat_next,
            'mae_next':   mae_curr,
            'delta_mae':  delta_mae,
        })

        # LINE 8: convergence check
        if delta_mae < CONVERGENCE_THRESHOLD:
            record['converged']    = True
            record['converged_at'] = i
            break

        mae_prev = mae_curr
        y_hat_i  = y_hat_next

    # LINE 10: return final prediction
    record['y_hat_final']  = y_hat_next
    record['mae_final']    = compute_mae(y, y_hat_next)
    record['method_found'] = extract_method(p_feed_i)

    return record
