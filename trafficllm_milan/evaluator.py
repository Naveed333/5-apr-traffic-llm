# evaluator.py — MAE/MSE computation and result reporting

import json
import os
from datetime import datetime

import numpy as np

from config import CONTEXT_SIZE
from data_loader import load_and_split
from inference import load_pexam, predict

RESULTS_DIR = os.path.join('outputs', 'results')


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def compute_mse(y_true, y_pred):
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def evaluate_all(context_size=None, verbose=True):
    """
    Run predictions on all test windows and compute MAE/MSE.
    Returns dict with per-window and aggregate results.
    """
    if context_size is None:
        context_size = CONTEXT_SIZE

    _, test_windows, _ = load_and_split()
    p_exam = load_pexam(context_size)

    print(f'\n=== Evaluation — context {context_size} ===')
    print(f'Test windows: {len(test_windows)}\n')

    per_window = []
    skipped    = 0

    for win in test_windows:
        t      = win['timestep']
        x_test = win['x']
        y_true = win['y']
        date_s = win.get('date_start', f'Day {t}')
        date_e = win.get('date_end',   f'Day {t+1}')

        y_pred = predict(x_test, t, p_exam)

        if y_pred is None:
            print(f'  Window {date_s}→{date_e}: SKIPPED (parse error)')
            skipped += 1
            continue

        mae = compute_mae(y_true, y_pred)
        mse = compute_mse(y_true, y_pred)

        per_window.append({
            'timestep':   t,
            'date_start': date_s,
            'date_end':   date_e,
            'mae':        mae,
            'mse':        mse,
            'y_true':     y_true,
            'y_pred':     y_pred,
        })

        print(f'  Window {date_s}→{date_e}: MAE={mae:.4f}  MSE={mse:.4f}')

        if verbose:
            _print_side_by_side(y_true, y_pred)

    if not per_window:
        print('No valid predictions produced.')
        return {}

    avg_mae = float(np.mean([r['mae'] for r in per_window]))
    avg_mse = float(np.mean([r['mse'] for r in per_window]))

    results = {
        'context_size':          context_size,
        'num_test_windows':      len(test_windows),
        'num_skipped':           skipped,
        'per_window_results':    per_window,
        'avg_mae':               avg_mae,
        'avg_mse':               avg_mse,
        'timestamp':             datetime.utcnow().isoformat() + 'Z',
    }

    _save_results(results, context_size)
    _print_table(per_window, avg_mae, avg_mse)
    _print_thesis_line(results)

    return results


def _print_side_by_side(y_true, y_pred):
    print(f'    {"Hour":>4}  {"y_true":>10}  {"y_pred":>10}  {"|error|":>10}')
    for h, (yt, yp) in enumerate(zip(y_true, y_pred), start=1):
        err = abs(yt - yp)
        print(f'    H{h:02d}   {yt:>10.4f}  {yp:>10.4f}  {err:>10.4f}')
    print()


def _print_table(per_window, avg_mae, avg_mse):
    border = '+' + '-' * 20 + '+' + '-' * 10 + '+' + '-' * 12 + '+'
    print(f'\n{border}')
    print(f'| {"Test Window":<18} | {"MAE":>8} | {"MSE":>10} |')
    print(border)
    for r in per_window:
        label = f'{r["date_start"]} → {r["date_end"]}'
        print(f'| {label:<18} | {r["mae"]:>8.4f} | {r["mse"]:>10.4f} |')
    print(border)
    print(f'| {"AVERAGE":<18} | {avg_mae:>8.4f} | {avg_mse:>10.4f} |')
    print(border)


def _print_thesis_line(results):
    ctx  = results['context_size']
    label = f'{ctx // 1024}K' if ctx >= 1024 else str(ctx)
    n_ex = results.get('num_pexam_examples', '?')
    print(
        f'\nThesis line: '
        f'Context {label} | p_exam examples: {n_ex} | '
        f'Avg MAE: {results["avg_mae"]:.4f} MB | '
        f'Avg MSE: {results["avg_mse"]:.4f}'
    )


def _save_results(results, context_size):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'results_{context_size}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {path}')


if __name__ == '__main__':
    evaluate_all()
