#!/usr/bin/env python3
"""Check the RMS norm weight values"""

import numpy as np
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

def get_tensor_by_name(name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None

# Check layer 0 and layer 1 norm weights
for layer in [0, 1]:
    for norm_name in [f'blk.{layer}.attn_norm.weight', f'blk.{layer}.ffn_norm.weight']:
        tensor = get_tensor_by_name(norm_name)
        if tensor is None:
            print(f"{norm_name}: NOT FOUND")
            continue
        
        data = np.frombuffer(tensor.data.tobytes(), dtype=np.float32)
        print(f"\n{norm_name}:")
        print(f"  Shape: {tensor.shape}, Type: {tensor.tensor_type}")
        print(f"  Min: {data.min():.6f}, Max: {data.max():.6f}")
        print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")
        print(f"  First 10: {data[:10]}")

# Also check output norm
tensor = get_tensor_by_name('output_norm.weight')
if tensor:
    data = np.frombuffer(tensor.data.tobytes(), dtype=np.float32)
    print(f"\noutput_norm.weight:")
    print(f"  Shape: {tensor.shape}, Type: {tensor.tensor_type}")
    print(f"  Min: {data.min():.6f}, Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")
