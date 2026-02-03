#!/usr/bin/env python3
"""Try to get internal states from llama-cpp-python"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Load model
llm = Llama(model_path=model_path, n_ctx=512, verbose=True, logits_all=True, n_gpu_layers=0)

# Process "1+1="
tokens = [16, 10, 16, 28]
print(f"\n=== Processing tokens: {tokens} ===")

llm.reset()
llm.eval(tokens)

# Get all logits
all_logits = np.array(llm._scores)
print(f"\nAll logits shape: {all_logits.shape}")  # Should be (4, vocab_size)

# For each position, show top predictions
for pos in range(all_logits.shape[0]):
    logits = all_logits[pos]
    top_idx = np.argsort(logits)[::-1][:3]
    print(f"\nPosition {pos}:")
    for idx in top_idx:
        tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
        print(f"  {idx} ({repr(tok)}): {logits[idx]:.4f}")

# Final position stats  
print(f"\nFinal position (after '='):")
final_logits = all_logits[-1]
print(f"  min={final_logits.min():.4f}, max={final_logits.max():.4f}, mean={final_logits.mean():.4f}")
print(f"  Token 17 ('2'): {final_logits[17]:.4f}, rank: {int(np.sum(final_logits > final_logits[17])) + 1}")
