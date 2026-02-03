#!/usr/bin/env python3
"""Check the actual bias values in the GGUF file"""

import numpy as np
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

def get_tensor_by_name(name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None

# Check layer 0 Q, K, V biases
for bias_name in ['blk.0.attn_q.bias', 'blk.0.attn_k.bias', 'blk.0.attn_v.bias']:
    tensor = get_tensor_by_name(bias_name)
    if tensor is None:
        print(f"{bias_name}: NOT FOUND")
        continue
    
    print(f"\n{bias_name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.tensor_type}")
    
    # For F32 tensors, read directly
    if tensor.tensor_type.name == 'F32':
        data = np.frombuffer(tensor.data.tobytes(), dtype=np.float32)
        print(f"  Min: {data.min():.6f}")
        print(f"  Max: {data.max():.6f}")
        print(f"  Mean: {data.mean():.6f}")
        print(f"  Std: {data.std():.6f}")
        print(f"  First 10: {data[:10]}")
        print(f"  Last 10: {data[-10:]}")
        
        # Check how many values are close to the extreme
        extreme_thresh = 50
        num_extreme = np.sum(np.abs(data) > extreme_thresh)
        print(f"  Values > {extreme_thresh} in magnitude: {num_extreme} / {len(data)} ({100*num_extreme/len(data):.1f}%)")
    else:
        print(f"  (Non-F32 tensor, skipping detailed analysis)")

# Also check the weight magnitudes for comparison
print("\n\n=== Weight magnitudes for comparison ===")
for weight_name in ['blk.0.attn_q.weight', 'blk.0.attn_k.weight', 'blk.0.attn_v.weight']:
    tensor = get_tensor_by_name(weight_name)
    if tensor is None:
        print(f"{weight_name}: NOT FOUND")
        continue
    
    print(f"\n{weight_name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.tensor_type}")
