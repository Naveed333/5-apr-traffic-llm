#!/usr/bin/env python3
# main.py — orchestrates the full TrafficLLM pipeline
#
# Usage:
#   python main.py --phase data       # verify dataset loads
#   python main.py --phase train      # run Algorithm 1 on training windows
#   python main.py --phase pexam      # build p_exam for each context size
#   python main.py --phase test       # run single test-window prediction
#   python main.py --phase evaluate   # evaluate all 9 test windows
#   python main.py --all              # run all phases in order

import argparse
import sys
import os

# Make sure we run from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


# -----------------------------------------------------------------------
# Phase: data
# -----------------------------------------------------------------------
def phase_data():
    print('\n=== Phase: data ===')
    from data_loader import load_and_split
    train_windows, test_windows, values = load_and_split()

    print(f'Total hourly values : {len(values)}')
    print(f'Total windows       : {len(train_windows) + len(test_windows)}')
    print(f'Train windows       : {len(train_windows)}')
    print(f'Test windows        : {len(test_windows)}')

    w = train_windows[0]
    print(f'\nSample window 1:')
    print(f'  date_start : {w["date_start"]}')
    print(f'  date_end   : {w["date_end"]}')
    print(f'  x[0:6]     : {[round(v, 2) for v in w["x"][:6]]}')
    print(f'  y[0:6]     : {[round(v, 2) for v in w["y"][:6]]}')
    print('\n[data] OK')


# -----------------------------------------------------------------------
# Phase: connection test
# -----------------------------------------------------------------------
def phase_connection():
    print('\n=== Phase: connection ===')
    from llm_client import test_connection
    ok = test_connection()
    if not ok:
        sys.exit(1)


# -----------------------------------------------------------------------
# Phase: dry-run (window 1 only)
# -----------------------------------------------------------------------
def phase_dry_run():
    print('\n=== Phase: dry-run (window 1) ===')
    from training import run_training
    run_training(skip_existing=False, dry_run=True)


# -----------------------------------------------------------------------
# Phase: train
# -----------------------------------------------------------------------
def phase_train(skip_existing=True):
    print('\n=== Phase: train ===')
    from training import run_training
    run_training(skip_existing=skip_existing)


# -----------------------------------------------------------------------
# Phase: pexam
# -----------------------------------------------------------------------
def phase_pexam(context_sizes=None):
    print('\n=== Phase: pexam ===')
    from pexam_builder import build_and_save_all
    build_and_save_all(context_sizes=context_sizes)

    # Show first example of 4096 context for verification
    import os
    p4k = os.path.join('outputs', 'p_exam', 'p_exam_4096.txt')
    if os.path.exists(p4k):
        with open(p4k) as f:
            preview = f.read()[:1500]
        print(f'\n--- Preview: first ~1500 chars of p_exam_4096.txt ---\n{preview}\n---')


# -----------------------------------------------------------------------
# Phase: test (single window)
# -----------------------------------------------------------------------
def phase_test(context_size=None):
    from config import CONTEXT_SIZE, TRAIN_DAYS
    if context_size is None:
        context_size = CONTEXT_SIZE

    print(f'\n=== Phase: test (single window, context={context_size}) ===')

    from data_loader import load_and_split
    from inference import load_pexam, predict
    from evaluator import compute_mae, compute_mse
    from prompt_builder import format_input, build_p_ques
    from llm_client import count_tokens

    _, test_windows, _ = load_and_split()
    win = test_windows[0]

    p_exam = load_pexam(context_size)

    x_test = win['x']
    y_true = win['y']
    t      = win['timestep']

    print(f'\nTest window: {win["date_start"]} → {win["date_end"]}')
    print(f'x_test (H01-H24): {[round(v, 2) for v in x_test]}')

    # Show total prompt token count
    from prompt_builder import format_input, build_p_ques
    p_input = format_input(x_test, t)
    p_ques  = build_p_ques()
    full_prompt = (p_exam + '\n' if p_exam else '') + p_input + '\n' + p_ques
    print(f'\nTotal prompt tokens : {count_tokens(full_prompt)}')
    print(f'Prompt tail (last 200 chars):\n  ...{full_prompt[-200:]}')

    y_pred = predict(x_test, t, p_exam)

    if y_pred is None:
        print('\nERROR: prediction failed — check outputs/llm_calls_log.jsonl')
        return

    print(f'\nParsed predictions:')
    print('  ' + '  '.join(f'H{i+1:02d}:{v:.4f}' for i, v in enumerate(y_pred)))

    mae = compute_mae(y_true, y_pred)
    mse = compute_mse(y_true, y_pred)

    print(f'\n{"Hour":>4}  {"y_true":>10}  {"y_pred":>10}  {"|error|":>10}')
    for h, (yt, yp) in enumerate(zip(y_true, y_pred), start=1):
        print(f'H{h:02d}   {yt:>10.4f}  {yp:>10.4f}  {abs(yt-yp):>10.4f}')

    print(f'\nTest window {win["date_start"]}→{win["date_end"]}: '
          f'MAE={mae:.4f} MB, MSE={mse:.4f} MB²')


# -----------------------------------------------------------------------
# Phase: evaluate
# -----------------------------------------------------------------------
def phase_evaluate(context_size=None):
    from config import CONTEXT_SIZE
    if context_size is None:
        context_size = CONTEXT_SIZE

    from evaluator import evaluate_all
    evaluate_all(context_size=context_size, verbose=False)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='TrafficLLM Milan Pipeline')
    parser.add_argument(
        '--phase',
        choices=['data', 'connection', 'dry-run', 'train', 'pexam', 'test', 'evaluate'],
        help='Which phase to run',
    )
    parser.add_argument('--all', action='store_true', help='Run all phases in order')
    parser.add_argument('--context-size', type=int, default=None,
                        help='Override CONTEXT_SIZE from config.py')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-run training even if records already exist')

    args = parser.parse_args()

    ctx = args.context_size

    if args.all:
        phase_data()
        phase_connection()
        phase_train(skip_existing=not args.no_skip)
        phase_pexam()
        phase_test(context_size=ctx)
        phase_evaluate(context_size=ctx)
    elif args.phase == 'data':
        phase_data()
    elif args.phase == 'connection':
        phase_connection()
    elif args.phase == 'dry-run':
        phase_dry_run()
    elif args.phase == 'train':
        phase_train(skip_existing=not args.no_skip)
    elif args.phase == 'pexam':
        phase_pexam()
    elif args.phase == 'test':
        phase_test(context_size=ctx)
    elif args.phase == 'evaluate':
        phase_evaluate(context_size=ctx)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
