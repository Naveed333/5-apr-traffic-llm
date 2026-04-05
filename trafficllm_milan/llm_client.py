# llm_client.py — vLLM API calls via openai client

import json
import time
import os
from datetime import datetime

import tiktoken
from openai import OpenAI
from config import VLLM_BASE_URL, VLLM_MODEL, TEMPERATURE

_enc = tiktoken.get_encoding('cl100k_base')

_client = None
LOG_PATH = os.path.join('outputs', 'llm_calls_log.jsonl')


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key='token')
    return _client


def count_tokens(text):
    return len(_enc.encode(text))


def call_llm(prompt, max_tokens=512):
    client = _get_client()
    os.makedirs('outputs', exist_ok=True)

    t0 = time.time()
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    text = response.choices[0].message.content or ''

    entry = {
        'timestamp':       datetime.utcnow().isoformat() + 'Z',
        'prompt_tokens':   count_tokens(prompt),
        'response_length': len(text),
        'elapsed_s':       round(elapsed, 2),
        'prompt_tail':     prompt[-200:],
        'response_head':   text[:300],
    }
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    return text


def test_connection():
    result = call_llm('Say "OK" and nothing else.', max_tokens=10)
    print(f'  [LLM] connection OK — response: {result.strip()!r}')
    return True
