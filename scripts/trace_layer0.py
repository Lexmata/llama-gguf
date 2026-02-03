#!/usr/bin/env python3
"""
Try to trace internal layer states using llama-cpp-python's low-level API
"""

import numpy as np
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Unfortunately llama-cpp-python doesn't expose internal layer states easily
# Let's at least verify the final output and see if we can find any hooks

llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)

# Process single token
tokens = [28]  # "="
llm.reset()
llm.eval(tokens)
logits = np.array(llm._scores[-1])

print("=== llama-cpp-python single token '=' ===")
print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}")
print(f"Logits: mean={logits.mean():.4f}, std={logits.std():.4f}")
print(f"Sum: {logits.sum():.4f}")

# Top predictions
top_idx = np.argsort(logits)[::-1][:10]
print("\nTop 10 predictions:")
for idx in top_idx:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  Token {idx} ({repr(tok)}): {logits[idx]:.4f}")

# Now compare with our output
print("\n=== Our implementation comparison ===")
print("Our logits: min=-15.46, max=9.82")
print("Our logits: mean=-2.50, std=2.94")
print("Our sum: (not computed)")

# Key differences:
# llama-cpp: max=13.16, sum around -230k (151936 * -1.53)
# our impl: max=9.82, sum around -380k (151936 * -2.50)

print("\n=== Analysis ===")
print("The logit scales are different but in a similar ballpark.")
print("More critically, our top predictions are completely wrong.")
print("\nllama-cpp top: ' ', '1', '2', '0', '3' (all sensible after '=')")
print("Our top: 100258, 48888, 33044... (seemingly random tokens)")

# Let's also check the vocab to understand what those random tokens are
print("\n=== Decoding our top predicted tokens ===")
our_top = [100258, 48888, 33044, 119099, 28341]
for idx in our_top:
    try:
        tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
        print(f"  Token {idx}: {repr(tok)}")
    except:
        print(f"  Token {idx}: <decode error>")
