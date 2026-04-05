# inference.py — test-time inference using frozen p_exam (Equation 3 only)

import os

from config import CONTEXT_SIZE
from prompt_builder import format_input, build_p_ques
from algorithm1 import call_and_parse

PEXAM_DIR = os.path.join('outputs', 'p_exam')


def load_pexam(context_size=None):
    if context_size is None:
        context_size = CONTEXT_SIZE
    path = os.path.join(PEXAM_DIR, f'p_exam_{context_size}.txt')
    with open(path) as f:
        return f.read()


def predict(x_test, timestep, p_exam, max_tokens=512):
    """Single LLM call — Equation 3 only, no refinement."""
    p_input = format_input(x_test, timestep)
    p_ques  = build_p_ques()
    prompt  = (p_exam + '\n' if p_exam else '') + p_input + '\n' + p_ques
    return call_and_parse(prompt, max_tokens=max_tokens)
