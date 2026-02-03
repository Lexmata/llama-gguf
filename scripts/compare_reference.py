#!/usr/bin/env python3
"""Compare llama-cpp-python output with our implementation"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Load model with logits output
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_batch=512,
    verbose=False,
    logits_all=True,  # Return logits for all tokens
)

# Test single token "="
print("=== Testing single token '=' ===")
prompt = "="
output = llm.create_completion(
    prompt=prompt,
    max_tokens=1,
    temperature=0.0,
    logprobs=10,  # Return top 10 logprobs
)

print(f"Input: '{prompt}'")
print(f"Generated: '{output['choices'][0]['text']}'")
if 'logprobs' in output['choices'][0]:
    top_logprobs = output['choices'][0]['logprobs']['top_logprobs'][0]
    print(f"Top logprobs: {top_logprobs}")

# Test "1+1="
print("\n=== Testing '1+1=' ===")
prompt = "1+1="
output = llm.create_completion(
    prompt=prompt,
    max_tokens=1,
    temperature=0.0,
    logprobs=10,
)

print(f"Input: '{prompt}'")
print(f"Generated: '{output['choices'][0]['text']}'")
if 'logprobs' in output['choices'][0]:
    top_logprobs = output['choices'][0]['logprobs']['top_logprobs'][0]
    print(f"Top logprobs: {top_logprobs}")

# Also show the tokens
print(f"\nTokens: {llm.tokenize(prompt.encode())}")

# Test what our Rust implementation produces
print("\n=== Comparing with our Rust implementation ===")
print("Run: cargo run --release -- run ~/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p '1+1=' --max-tokens 1")
