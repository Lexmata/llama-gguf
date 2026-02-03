#!/usr/bin/env python3
"""Compare our Q/K/V computation with llama-cpp-python's internal state."""

import numpy as np
from llama_cpp import Llama

def main():
    model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    print("Loading model with llama-cpp-python...")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_batch=512,
        verbose=False,
    )
    
    # Tokenize "1+1="
    prompt = "1+1="
    tokens = llm.tokenize(prompt.encode(), add_bos=False)
    print(f"Tokens for '{prompt}': {tokens}")
    
    # Run a simple generation
    print("\nGenerating with llama-cpp-python:")
    output = llm(prompt, max_tokens=1, temperature=0.0, top_k=1)
    generated = output['choices'][0]['text']
    print(f"Generated: '{generated}'")
    
    # Check what the model predicts
    # With temperature=0 and top_k=1, this should be deterministic
    print("\nExpected output from a working model: '2'")
    print(f"Actual output: '{generated}'")
    
    if generated.strip() == '2':
        print("SUCCESS: llama-cpp-python produces correct output")
    else:
        print(f"NOTE: llama-cpp-python produces '{generated}' instead of '2'")
        print("This could indicate the model itself has issues, or different tokenization")
    
    # Let's also check what logits look like
    print("\n=== Checking logits with logits_all=True ===")
    llm2 = Llama(
        model_path=model_path,
        n_ctx=512,
        n_batch=512,
        logits_all=True,
        verbose=False,
    )
    
    # Evaluate the tokens
    llm2.reset()
    llm2.eval(tokens)
    
    # Get logits for the last position
    logits = np.array(llm2.scores[-1])
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    
    # Find top tokens
    top_indices = np.argsort(logits)[::-1][:10]
    print("\nTop 10 tokens by logit:")
    for i, idx in enumerate(top_indices):
        token_str = llm2.detokenize([idx]).decode('utf-8', errors='replace')
        print(f"  {i+1}. Token {idx} ('{token_str}'): {logits[idx]:.4f}")
    
    # Check token 17 (which should be "2" based on our earlier testing)
    print(f"\nToken 17 logit: {logits[17]:.4f}")
    print(f"Token 17 rank: {np.sum(logits > logits[17]) + 1}")
    
    # Decode token 17
    token_17_str = llm2.detokenize([17]).decode('utf-8', errors='replace')
    print(f"Token 17 decoded: '{token_17_str}'")

if __name__ == "__main__":
    main()
