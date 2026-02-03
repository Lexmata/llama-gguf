#!/usr/bin/env python3
"""Check if output.weight exists or if it's tied to token_embd.weight"""

from gguf import GGUFReader
import numpy as np

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

print("=== Tensor List ===")
for tensor in reader.tensors:
    if 'embd' in tensor.name or 'output' in tensor.name:
        print(f"  {tensor.name}: shape={tensor.shape}, type={tensor.tensor_type}")

print("\n=== Checking for output.weight ===")
output_weight = None
token_embd = None
for tensor in reader.tensors:
    if tensor.name == "output.weight":
        output_weight = tensor
        print(f"Found output.weight: shape={tensor.shape}")
    if tensor.name == "token_embd.weight":
        token_embd = tensor
        print(f"Found token_embd.weight: shape={tensor.shape}")

if output_weight is None:
    print("\nNo output.weight found - using weight tying with token_embd.weight")
else:
    print(f"\noutput.weight exists separately")
    # Check if they're the same
    if token_embd is not None:
        emb_data = token_embd.data
        out_data = output_weight.data
        print(f"  token_embd.data.shape: {emb_data.shape}")
        print(f"  output.data.shape: {out_data.shape}")
        if emb_data.shape == out_data.shape:
            if np.allclose(emb_data, out_data):
                print("  => They are IDENTICAL (weight tying)")
            else:
                print("  => They are DIFFERENT")
