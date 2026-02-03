#!/usr/bin/env python3
"""
Layer-by-layer reference implementation using llama-cpp-python.

This script uses llama-cpp-python's internal state access to get
intermediate values for comparison with our Rust implementation.

Usage:
    python scripts/layer_by_layer_reference.py
"""

import numpy as np
from llama_cpp import Llama

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def main():
    print("=== Layer-by-Layer Reference (Python) ===")
    print(f"Model: {MODEL_PATH}")
    
    # Load model with logits_all=True to get all token logits
    print("Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_batch=512,
        logits_all=True,
        verbose=False
    )
    
    # Tokenize "1+1="
    prompt = "1+1="
    tokens = llm.tokenize(prompt.encode(), add_bos=False)
    print(f"Tokens: {tokens} (len={len(tokens)})")
    
    # Run inference
    print("\nRunning inference...")
    output = llm.create_completion(
        prompt,
        max_tokens=1,
        temperature=0.0,
        top_k=1,
        logprobs=10
    )
    
    # Get the generated text
    generated = output['choices'][0]['text']
    print(f"\nGenerated: '{generated}'")
    
    # Get logprobs for top tokens
    if 'logprobs' in output['choices'][0]:
        logprobs = output['choices'][0]['logprobs']
        print("\nTop tokens from llama.cpp:")
        if 'top_logprobs' in logprobs and logprobs['top_logprobs']:
            for tok, prob in sorted(logprobs['top_logprobs'][0].items(), key=lambda x: -x[1])[:10]:
                print(f"  '{tok}': {prob:.4f}")
    
    # Evaluate to get raw logits
    print("\n--- Raw Logits Analysis ---")
    llm.reset()
    llm.eval(tokens)
    
    # Get logits for the last position
    n_vocab = llm.n_vocab()
    logits = np.array([llm._scores[-1, i] for i in range(min(n_vocab, 151936))])
    
    print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.6f}")
    
    # Top predictions
    top_indices = np.argsort(logits)[::-1][:10]
    print("\nTop 10 predictions:")
    for idx in top_indices:
        try:
            tok_str = llm.detokenize([idx]).decode('utf-8', errors='replace')
        except:
            tok_str = "<?>"
        print(f"  Token {idx} ('{tok_str}'): logit={logits[idx]:.4f}")
    
    # Check token 17
    token_2_rank = np.where(np.argsort(logits)[::-1] == 17)[0]
    if len(token_2_rank) > 0:
        print(f"\nToken 17 ('2'): logit={logits[17]:.4f}, rank={token_2_rank[0] + 1}")
    
    # Also print logits at specific positions for comparison
    print("\n--- Logits at specific positions ---")
    specific_tokens = [17, 16, 18, 1402, 22925]  # "2", "1", "3", "AM", "ande"
    for tok_id in specific_tokens:
        if tok_id < len(logits):
            try:
                tok_str = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
            except:
                tok_str = "<?>"
            print(f"  Token {tok_id} ('{tok_str}'): logit={logits[tok_id]:.4f}")

if __name__ == "__main__":
    main()
