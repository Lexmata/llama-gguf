#!/usr/bin/env python3
"""Extract hidden states from llama-cpp-python to compare with our implementation"""

import numpy as np
import ctypes
from llama_cpp import Llama

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Load model
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_batch=512,
    verbose=False,
    embedding=True,  # Enable embedding mode to get hidden states
)

# Test single token
print("=== Single token test ===")
prompt = "1"
tokens = llm.tokenize(prompt.encode())
print(f"Token for '{prompt}': {tokens}")

# Get embedding (this is the output hidden state)
embedding = llm.embed(prompt)
embedding = np.array(embedding)
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
print(f"Embedding first 10: {embedding[:10]}")

# Also try with "1+1="
print("\n=== Multi-token test '1+1=' ===")
prompt = "1+1="
tokens = llm.tokenize(prompt.encode())
print(f"Tokens: {tokens}")

embedding = llm.embed(prompt)
embedding = np.array(embedding)
print(f"Final embedding shape: {embedding.shape}")
print(f"Final embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
print(f"Final embedding first 10: {embedding[:10]}")

# Now compare with completion output to verify the embedding relates to logits
print("\n=== Completion test ===")
output = llm.create_completion(
    prompt="1+1=",
    max_tokens=1,
    temperature=0.0,
)
print(f"Generated: '{output['choices'][0]['text']}'")
