#!/usr/bin/env python3
"""Verify embeddings between our implementation and llama-cpp-python."""

import numpy as np
from llama_cpp import Llama
import struct

def read_gguf_embeddings():
    """Read embeddings directly from GGUF file."""
    model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    # Use llama-cpp to get model info
    llm = Llama(model_path=model_path, n_ctx=4, embedding=True, verbose=False)
    
    # Get embeddings for some tokens
    tokens = [16, 28, 10]  # "1", "=", "+"
    
    print("Embeddings via llama-cpp-python:")
    for tok in tokens:
        # Create a context and get the embedding
        llm.reset()
        emb = llm.embed([tok])  # This returns the embedding for the sequence
        if emb is not None:
            emb = np.array(emb[0])  # First sequence
            print(f"  Token {tok}: shape={emb.shape}, first5={emb[:5]}")

if __name__ == "__main__":
    read_gguf_embeddings()
