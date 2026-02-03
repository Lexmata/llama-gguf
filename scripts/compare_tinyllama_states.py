#!/usr/bin/env python3
"""Compare TinyLlama hidden states between llama.cpp and llama-rs"""

import numpy as np
import struct
import os

# TinyLlama model parameters
HIDDEN_SIZE = 2048
NUM_LAYERS = 22
HEAD_DIM = 64  # 2048 / 32
NUM_HEADS = 32
NUM_KV_HEADS = 4
INTERMEDIATE_SIZE = 5632
ROPE_THETA = 10000.0
NORM_EPS = 1e-5

def load_gguf_tensors(path):
    """Load tensor data from GGUF file"""
    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=512, n_gpu_layers=0, verbose=False)
    return llm

def rms_norm(x, weight, eps=1e-5):
    """RMS normalization"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

def rope_neox(x, pos, theta=10000.0):
    """Apply NeoX-style RoPE to x of shape [num_heads, 1, head_dim]"""
    num_heads, seq_len, head_dim = x.shape
    half_dim = head_dim // 2
    
    # Compute frequencies
    freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    t = np.array([pos], dtype=np.float32)
    freqs = np.outer(t, freqs)  # [1, half_dim]
    
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    
    out = np.zeros_like(x)
    for h in range(num_heads):
        for s in range(seq_len):
            x1 = x[h, s, :half_dim]
            x2 = x[h, s, half_dim:]
            out[h, s, :half_dim] = x1 * cos[s] - x2 * sin[s]
            out[h, s, half_dim:] = x2 * cos[s] + x1 * sin[s]
    
    return out

def main():
    path = os.path.expanduser('~/.cache/llama-rs/models/TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
    
    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=512, n_gpu_layers=0, verbose=False, embedding=True)
    
    # Get embeddings for "1+1="
    embeddings = np.array(llm.embed("1+1="))
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Last token embedding stats: min={embeddings[-1].min():.4f}, max={embeddings[-1].max():.4f}, mean={embeddings[-1].mean():.4f}")
    
    # The embeddings from llm.embed() are the FINAL hidden states after all layers
    # For comparison, we need to use llama_get_embeddings_seq to get intermediate states
    # But llama-cpp-python doesn't expose that
    
    # Instead, let's run inference and check the output
    output = llm("1+1=", max_tokens=1, temperature=0)
    print(f"\nllama.cpp generates: {repr(output['choices'][0]['text'])}")
    
    # Token IDs
    tokens = llm.tokenize(b"1+1=")
    print(f"\nTokens for '1+1=': {tokens}")
    
if __name__ == "__main__":
    main()
