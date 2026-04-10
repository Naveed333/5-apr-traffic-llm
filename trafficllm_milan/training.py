# training.py — run Algorithm 1 on all training windows, save records

import json
import os
import time

import numpy as np

import config
from data_loader import load_and_split
from algorithm1 import run_algorithm1


def _records_dir():
    return os.path.join(config.OUTPUTS_BASE, 'training_records')


def record_path(timestep):
    return os.path.join(_records_dir(), f'window_{timestep}.json')


def already_done(timestep):
    p = record_path(timestep)
    return os.path.exists(p) and 'y_hat_final' in json.load(open(p))


def save_record(record):
    os.makedirs(_records_dir(), exist_ok=True)
    with open(record_path(record['timestep']), 'w') as f:
        json.dump(record, f, indent=2)


def run_training(skip_existing=True, dry_run=False):
    train_windows, _, _ = load_and_split()
    total = len(train_windows)

    print(f'\n=== TrafficLLM Training Phase ===')
    print(f'Training windows: {total}  (skip_existing={skip_existing})\n')

    maes_0, maes_final = [], []
    windows_to_run = train_windows[:1] if dry_run else train_windows

    for win in windows_to_run:
        t = win['timestep']

        if skip_existing and already_done(t):
            rec = json.load(open(record_path(t)))
            maes_0.append(rec['mae_0'])
            maes_final.append(rec['mae_final'])
            print(f'  Window {t}/{total}: already done — skipping.')
            continue

        print(f'  Window {t}/{total}: running Algorithm 1 '
              f'({win["date_start"]} → {win["date_end"]}) ...', flush=True)
        t_start = time.time()

        record = run_algorithm1(win, p_exam_so_far='')
        save_record(record)

        mae_0     = record['mae_0']
        mae_final = record['mae_final']
        maes_0.append(mae_0)
        maes_final.append(mae_final)

        print(f'    MAE_0={mae_0:.4f} → MAE_final={mae_final:.4f}  '
              f'iters={len(record["iterations"])}  '
              f'converged={record["converged"]}  '
              f'method={record["method_found"]}  '
              f'[{time.time() - t_start:.1f}s]')

        if dry_run:
            print('\n--- Dry run complete (one window only) ---\n')
            return

    if maes_0:
        print(f'\n=== Training Summary ===')
        print(f'  Windows processed : {len(maes_0)}')
        print(f'  Avg MAE_0         : {float(np.nanmean(maes_0)):.4f} MB')
        print(f'  Avg MAE_final     : {float(np.nanmean(maes_final)):.4f} MB')


if __name__ == '__main__':
    run_training()
