#!/usr/bin/env python3
"""Compare exact logits between different input sequences using llama-cpp-python."""

import numpy as np
from llama_cpp import Llama

def get_token_rank(logits, token_id):
    """Get the rank of a specific token in the logit distribution."""
    sorted_indices = np.argsort(logits)[::-1]
    rank = np.where(sorted_indices == token_id)[0][0] + 1
    return rank

def main():
    model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_batch=512,
        verbose=False,
        logits_all=False,  # We only need logits for last token
    )
    
    print()
    print("Token sequence               | Token 17 Logit | Rank")
    print("-----------------------------+----------------+------")
    
    test_sequences = [
        ("'=' at pos 0", [28]),
        ("'1=' (1 at pos 0)", [16, 28]),
        ("'+=' (+ at pos 0)", [10, 28]),
        ("'a=' (a at pos 0)", [64, 28]),
        ("' =' (space at pos 0)", [220, 28]),
        ("'==' (= at pos 0,1)", [28, 28]),
        ("'===' (= at pos 0,1,2)", [28, 28, 28]),
        ("'1+1=' (full)", [16, 10, 16, 28]),
        ("'11' (1 at pos 0,1)", [16, 16]),
    ]
    
    for desc, tokens in test_sequences:
        llm.reset()
        llm.eval(tokens)
        
        # Get logits for last position
        # In newer llama-cpp-python, scores is a list of numpy arrays
        logits = np.array(llm._scores[-1])
        
        logit_17 = logits[17]
        rank = get_token_rank(logits, 17)
        
        print(f"{desc:28} | {logit_17:14.4f} | {rank:4d}")
    
    # Also check what the top predictions are for each
    print()
    print("Top 5 predictions for each sequence:")
    print("=" * 60)
    
    for desc, tokens in test_sequences[:4]:
        llm.reset()
        llm.eval(tokens)
        logits = np.array(llm._scores[-1])
        
        top_indices = np.argsort(logits)[::-1][:5]
        print(f"\n{desc}:")
        for i, idx in enumerate(top_indices):
            token_str = llm.detokenize([idx]).decode('utf-8', errors='replace')
            print(f"  {i+1}. Token {idx} ('{token_str}'): {logits[idx]:.4f}")

if __name__ == "__main__":
    main()
