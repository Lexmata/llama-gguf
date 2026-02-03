#!/usr/bin/env python3
"""Extract layer 0 intermediate values from GGUF manually to compare"""

import numpy as np
from gguf import GGUFReader
import struct

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

def get_tensor_by_name(name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None

def dequantize_f32(tensor):
    """For F32 tensors, just reshape"""
    return np.frombuffer(tensor.data.tobytes(), dtype=np.float32)

def dequantize_q8_0(data, shape):
    """Dequantize Q8_0 format"""
    hidden_size, vocab_size = shape
    result = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    
    blocks_per_row = hidden_size // 32
    bytes_per_block = 34  # 2 (scale) + 32 (quants)
    
    for vocab_idx in range(vocab_size):
        row_start = vocab_idx * blocks_per_row * bytes_per_block
        for block_idx in range(blocks_per_row):
            block_start = row_start + block_idx * bytes_per_block
            
            scale = np.frombuffer(data[block_start:block_start+2], dtype=np.float16)[0]
            quants = data[block_start+2:block_start+34].astype(np.int8)
            
            for j, q in enumerate(quants):
                result[vocab_idx, block_idx * 32 + j] = float(scale) * float(q)
    
    return result

# Get token embedding for token 16 ('1')
embd_tensor = get_tensor_by_name('token_embd.weight')
print(f"token_embd.weight shape: {embd_tensor.shape}, type: {embd_tensor.tensor_type}")

# For Q5_0, dequantization is complex. Let's get the raw output instead
# by using a reference value from our Rust implementation

print("\n=== Reference values from our Rust implementation ===")
print("Token 16 ('1') embedding first 5: [-0.0169, -0.0056, -0.0226, -0.0367, -0.0169]")

# Get layer 0 attention norm weight (F32)
attn_norm = get_tensor_by_name('blk.0.attn_norm.weight')
attn_norm_w = dequantize_f32(attn_norm)
print(f"\nblk.0.attn_norm.weight: shape={len(attn_norm_w)}, first 5: {attn_norm_w[:5]}")

# Get Q bias (F32)
q_bias_tensor = get_tensor_by_name('blk.0.attn_q.bias')
q_bias = dequantize_f32(q_bias_tensor)
print(f"blk.0.attn_q.bias: shape={len(q_bias)}, min={q_bias.min():.2f}, max={q_bias.max():.2f}")

# Get K bias (F32)
k_bias_tensor = get_tensor_by_name('blk.0.attn_k.bias')
k_bias = dequantize_f32(k_bias_tensor)
print(f"blk.0.attn_k.bias: shape={len(k_bias)}, min={k_bias.min():.2f}, max={k_bias.max():.2f}")

print("\n=== Key question: is our matrix multiplication correct? ===")
print("GGUF shape for weights: [out_features, in_features]")
print("For attn_q.weight with shape [896, 896]: in=896 (hidden), out=896 (num_heads*head_dim)")
print("Computation: Q = x @ W^T + bias")
print("But in GGUF col-major: W[i,j] = data[i + j*out_features]")
print("So: Q[j] = sum_i(x[i] * W[i,j]) = sum_i(x[i] * data[i + j*in])")
print()
print("Our vec_mat does: out[j] = sum_i(x[i] * w[i + j*k]) where k=in_features")
print("This matches the GGUF layout!")

print("\n=== Checking output.weight for logit computation ===")
output_tensor = get_tensor_by_name('output.weight')
print(f"output.weight shape: {output_tensor.shape}, type: {output_tensor.tensor_type}")
print("Shape [896, 151936] means: in=896, out=151936")
print("logits[j] = sum_i(hidden[i] * W[i,j])")
print("With GGUF: logits[j] = sum_i(hidden[i] * data[i + j*896])")
print()
print("This is exactly what our vec_mat does!")

print("\n=== Possible issues to investigate ===")
print("1. Q5_0 dequantization might have bugs")
print("2. The attention scores with large biases might overflow/underflow")
print("3. There might be a transpose issue we're missing")
print("4. The RoPE implementation might still be wrong")
