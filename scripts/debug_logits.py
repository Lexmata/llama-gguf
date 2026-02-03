#!/usr/bin/env python3
"""Debug logit distributions between our impl and llama-cpp"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Load reference logits 
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

tokens = [16, 10, 16, 28]  # "1+1="

llm.reset()
llm.eval(tokens)
ref_logits = np.array(llm._scores[-1])

print("=== llama-cpp-python logits ===")
print(f"  Shape: {ref_logits.shape}")
print(f"  Min: {ref_logits.min():.4f}")
print(f"  Max: {ref_logits.max():.4f}")
print(f"  Mean: {ref_logits.mean():.4f}")
print(f"  Std: {ref_logits.std():.4f}")
print(f"  Sum: {ref_logits.sum():.4f}")

# Show histogram
print(f"\n  Logit distribution:")
for threshold in [-5, 0, 5, 10, 15, 17]:
    count = np.sum(ref_logits > threshold)
    print(f"    > {threshold}: {count} tokens")

print(f"\n  Sample logits [0:10]: {ref_logits[:10]}")
print(f"  Sample logits [16:20]: {ref_logits[16:20]}")  # Numeric tokens

# Get top tokens
top_idx = np.argsort(ref_logits)[::-1][:10]
print(f"\n  Top 10 tokens:")
for idx in top_idx:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"    {idx} ({repr(tok)}): {ref_logits[idx]:.4f}")
