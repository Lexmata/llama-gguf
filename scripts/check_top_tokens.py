#!/usr/bin/env python3
"""Check what tokens are our top predictions"""

from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
llm = Llama(model_path=model_path, n_ctx=512, verbose=False)

# Our top tokens
our_top = [25428, 105181, 77056, 5567, 125767]

print("=== Our top token predictions ===")
for tok_id in our_top:
    try:
        tok_str = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
        print(f"  Token {tok_id}: {repr(tok_str)}")
    except Exception as e:
        print(f"  Token {tok_id}: ERROR - {e}")

# Also check some numeric tokens
print("\n=== Numeric tokens ===")
for tok_id in range(15, 26):
    try:
        tok_str = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
        print(f"  Token {tok_id}: {repr(tok_str)}")
    except Exception as e:
        print(f"  Token {tok_id}: ERROR - {e}")
