# inference.py — test-time inference using frozen p_exam (Equation 3 only)

import os

import config
from config import CONTEXT_SIZE
from llm_client import call_llm
from prompt_builder import format_input, build_p_ques
from algorithm1 import parse_prediction


def load_pexam(context_size=None):
    if context_size is None:
        context_size = CONTEXT_SIZE
    path = os.path.join(config.OUTPUTS_BASE, 'p_exam', f'p_exam_{context_size}.txt')
    with open(path) as f:
        return f.read()


def predict(x_test, timestep, p_exam, max_tokens=512):
    """Single LLM call — Equation 3 only, no refinement."""
    p_input = format_input(x_test, timestep)
    p_ques  = build_p_ques()
    prompt  = (p_exam + '\n' if p_exam else '') + p_input + '\n' + p_ques
    return parse_prediction(call_llm(prompt, max_tokens=max_tokens))
