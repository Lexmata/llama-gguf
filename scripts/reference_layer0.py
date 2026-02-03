#!/usr/bin/env python3
"""
Reference implementation of layer 0 forward pass using llama-cpp-python's internal state.
Instead of dequantizing manually, we'll compare final outputs at key checkpoints.
"""

import numpy as np
from llama_cpp import Llama

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def main():
    print("Loading model with llama-cpp-python...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        verbose=False,
        embedding=True,  # Enable embeddings
    )
    
    # Token "=" is 28
    token_id = 28
    
    # Get the embedding for this token
    print(f"\n=== Testing token {token_id} ('=') ===")
    
    # Use the embed method to get the final hidden state after all layers
    text = "="  # This should tokenize to [28]
    
    # Get embedding (final hidden state after all layers)
    embedding = llm.embed(text)
    embedding = np.array(embedding, dtype=np.float32)
    
    print(f"Final hidden state (after all layers):")
    print(f"  Shape: {embedding.shape}")
    print(f"  Stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
    print(f"  First 10: {embedding[:10]}")
    
    # Now let's also run a completion to see what the model predicts
    print("\n=== Predictions ===")
    output = llm.create_completion(
        prompt="=",
        max_tokens=1,
        temperature=0.0,
    )
    print(f"After '=': predicted '{output['choices'][0]['text']}'")
    
    # Test with "1+1="
    output = llm.create_completion(
        prompt="1+1=",
        max_tokens=1,
        temperature=0.0,
    )
    print(f"After '1+1=': predicted '{output['choices'][0]['text']}'")
    
    # Get embedding for "1+1="
    embedding_full = llm.embed("1+1=")
    embedding_full = np.array(embedding_full, dtype=np.float32)
    
    print(f"\nFinal hidden state for '1+1=':")
    print(f"  Shape: {embedding_full.shape}")
    print(f"  Stats: min={embedding_full.min():.4f}, max={embedding_full.max():.4f}, mean={embedding_full.mean():.4f}")
    print(f"  First 10: {embedding_full[:10]}")
    
    # Let's also get the embedding for just "1" to compare
    embedding_1 = llm.embed("1")
    embedding_1 = np.array(embedding_1, dtype=np.float32)
    
    print(f"\nFinal hidden state for '1':")
    print(f"  Shape: {embedding_1.shape}")
    print(f"  Stats: min={embedding_1.min():.4f}, max={embedding_1.max():.4f}, mean={embedding_1.mean():.4f}")
    print(f"  First 10: {embedding_1[:10]}")

if __name__ == "__main__":
    main()
