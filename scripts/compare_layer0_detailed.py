#!/usr/bin/env python3
"""Detailed layer 0 comparison using llama-cpp-python internals"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Load model
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_batch=512,
    verbose=False,
)

# Get token for "=" 
tokens = llm.tokenize(b"=", add_bos=False)
print(f"Token for '=': {tokens}")

# Run eval on single token and get logits
llm.reset()  # Reset state

# Process token and get logits
llm.eval(tokens)
logits = llm.scores[0]  # Get logits for first token position

# Get top predictions
top_indices = np.argsort(logits)[::-1][:10]

print("\nTop 10 predictions for single token '=':")
for i, idx in enumerate(top_indices):
    try:
        tok_str = llm.detokenize([idx]).decode('utf-8', errors='replace')
    except:
        tok_str = f"<token {idx}>"
    print(f"  {i+1}. Token {idx} '{tok_str}': {logits[idx]:.4f}")

# Now try "1+1="
llm.reset()
tokens = llm.tokenize(b"1+1=", add_bos=False)
print(f"\nTokens for '1+1=': {tokens}")

llm.eval(tokens)

# Get logits for last position
logits = llm.scores[len(tokens) - 1]

# Get top predictions
top_indices = np.argsort(logits)[::-1][:10]

print("\nTop 10 predictions for '1+1=':")
for i, idx in enumerate(top_indices):
    try:
        tok_str = llm.detokenize([idx]).decode('utf-8', errors='replace')
    except:
        tok_str = f"<token {idx}>"
    print(f"  {i+1}. Token {idx} '{tok_str}': {logits[idx]:.4f}")

# Check rank of "2"
tok_2 = llm.tokenize(b"2", add_bos=False)[0]
print(f"\nToken for '2': {tok_2}")
rank_2 = np.where(top_indices == tok_2)[0]
if len(rank_2) > 0:
    print(f"Token '2' rank: {rank_2[0] + 1}")
else:
    all_sorted = np.argsort(logits)[::-1]
    rank = np.where(all_sorted == tok_2)[0][0] + 1
    print(f"Token '2' rank: {rank}")
    print(f"Token '2' logit: {logits[tok_2]:.4f}")

# Print logit stats
print(f"\nLogit stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
