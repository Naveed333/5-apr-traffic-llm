# pexam_builder.py — assemble Option 3 p_exam from training records

import json
import os
import glob

import config
from config import CONTEXT_SIZES, TOKENS_RESERVE, GRID_CELL_ID, LOCATION
from llm_client import count_tokens
from prompt_builder import format_input, format_output


def _records_dir():
    return os.path.join(config.OUTPUTS_BASE, 'training_records')


def _pexam_dir():
    return os.path.join(config.OUTPUTS_BASE, 'p_exam')


def load_all_records():
    paths   = sorted(glob.glob(os.path.join(_records_dir(), 'window_*.json')))
    records = [json.load(open(p)) for p in paths]
    records.sort(key=lambda r: r.get('timestep', 0))
    return records


def _record_to_example_text(rec, n):
    t     = rec['timestep']
    x     = rec['x']
    y     = rec['y']
    iters = rec.get('iterations', [])

    lines = [
        f'--- Example {n} ---',
        f'INPUT (past 24h, timestep {t}, Milan {GRID_CELL_ID} {LOCATION}):',
        format_input(x, t),
        '',
        f'ITERATION 0 — first prediction:',
        f'{format_output(rec["y_hat_0"])}  [MAE={rec["mae_0"]:.4f} MB]',
        '',
    ]

    for it in iters:
        i = it['i']
        lines += [
            f'Feedback (iter {i}):',
            it['p_feed'].strip(),
            f'Refinement (iter {i}):',
            it['p_refine'].strip(),
            f'ITERATION {i+1} — refined prediction:',
            f'{format_output(it["y_hat_next"])}  [MAE={it["mae_next"]:.4f} MB]',
            '',
        ]

    method = rec.get('method_found', 'unknown')
    if rec.get('converged'):
        lines.append(f'CONVERGED at iteration {rec["converged_at"]}. Method: {method}.')
    else:
        lines.append(f'MAX_ITERATIONS reached. Method: {method}.')

    lines += [
        f'Final prediction: {format_output(rec["y_hat_final"])}',
        f'Real ground truth: {format_output(y)}',
        '---',
        '',
    ]

    return '\n'.join(lines)


def build_pexam(context_size, records):
    token_budget = context_size - TOKENS_RESERVE
    parts, total_tokens, n_examples = [], 0, 0

    for n, rec in enumerate(records, start=1):
        text      = _record_to_example_text(rec, n)
        ex_tokens = count_tokens(text)
        if total_tokens + ex_tokens > token_budget:
            break
        parts.append(text)
        total_tokens += ex_tokens
        n_examples   += 1

    return '\n'.join(parts), n_examples, total_tokens


def build_and_save_all(context_sizes=None):
    if context_sizes is None:
        context_sizes = CONTEXT_SIZES

    records = load_all_records()
    print(f'Loaded {len(records)} training records.\n')

    pexam_dir = _pexam_dir()
    os.makedirs(pexam_dir, exist_ok=True)

    for ctx in context_sizes:
        p_exam, n_ex, n_tok = build_pexam(ctx, records)
        path = os.path.join(pexam_dir, f'p_exam_{ctx}.txt')
        with open(path, 'w') as f:
            f.write(p_exam)
        label = f'{ctx // 1024}K' if ctx >= 1024 else str(ctx)
        print(f'  Context {label:>6}: {n_ex:3d} examples, {n_tok:6d} tokens  → {path}')


if __name__ == '__main__':
    build_and_save_all()
