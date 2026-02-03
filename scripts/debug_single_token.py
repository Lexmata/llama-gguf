#!/usr/bin/env python3
"""Debug single token forward pass"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

# Single token "="
tokens = [28]

llm.reset()
llm.eval(tokens)
logits = np.array(llm._scores[-1])

print("=== Single token '=' (28) ===")
print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")

# Show top predictions
top_idx = np.argsort(logits)[::-1][:5]
print("\nTop 5 predictions:")
for idx in top_idx:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  Token {idx} ({repr(tok)}): {logits[idx]:.4f}")

print("\nNumeric tokens [16-20]:")
for i in range(16, 21):
    tok = llm.detokenize([i]).decode('utf-8', errors='replace')
    rank = int(np.sum(logits > logits[i])) + 1
    print(f"  Token {i} ('{tok}'), rank {rank}: {logits[i]:.4f}")
