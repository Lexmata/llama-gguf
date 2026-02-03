#!/usr/bin/env python3
"""Test RMSNorm computation to compare with Rust implementation"""

import numpy as np
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

def get_tensor_by_name(name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None

def dequantize_q5_0(data, shape):
    """Dequantize Q5_0 format - 5-bit quantization with 32-element blocks"""
    # Q5_0 block: 2 bytes (fp16 scale) + 4 bytes (high bits) + 16 bytes (low nibbles) = 22 bytes
    # 32 elements per block
    BLOCK_SIZE = 32
    BYTES_PER_BLOCK = 22
    
    # Shape is [hidden_size, vocab_size] in GGUF but stored as [vocab_size][hidden_size]
    hidden_size, vocab_size = shape
    result = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    
    blocks_per_row = hidden_size // BLOCK_SIZE
    
    for v in range(vocab_size):
        row_start = v * blocks_per_row * BYTES_PER_BLOCK
        for b in range(blocks_per_row):
            block_start = row_start + b * BYTES_PER_BLOCK
            
            # Read scale (fp16)
            scale = np.frombuffer(data[block_start:block_start+2], dtype=np.float16)[0]
            
            # Read high bits (4 bytes = 32 bits, one bit per element)
            qh = np.frombuffer(data[block_start+2:block_start+6], dtype=np.uint8)
            qh_bits = np.unpackbits(qh, bitorder='little')[:32]
            
            # Read low nibbles (16 bytes = 32 nibbles)
            ql = data[block_start+6:block_start+22]
            
            for j in range(32):
                # Low 4 bits from nibble
                if j < 16:
                    low_nibble = ql[j] & 0x0F
                else:
                    low_nibble = (ql[j-16] >> 4) & 0x0F
                
                # High bit
                high_bit = qh_bits[j]
                
                # Combine: 5-bit value = (high_bit << 4) | low_nibble, then subtract 16
                quant_val = (high_bit << 4) | low_nibble
                dequant_val = (quant_val - 16) * float(scale)
                
                result[v, b * 32 + j] = dequant_val
    
    return result

# Get embedding for token 16 ("1")
embd_tensor = get_tensor_by_name('token_embd.weight')
print(f"token_embd.weight shape: {embd_tensor.shape}, type: {embd_tensor.tensor_type}")

# Dequantize and get embedding for token 16
embd_data = dequantize_q5_0(embd_tensor.data.tobytes(), embd_tensor.shape)
token_16_emb = embd_data[16]
print(f"\nToken 16 embedding stats:")
print(f"  min: {token_16_emb.min():.6f}, max: {token_16_emb.max():.6f}")
print(f"  mean: {token_16_emb.mean():.6f}, std: {token_16_emb.std():.6f}")
print(f"  first 5: {token_16_emb[:5]}")

# Get layer 0 attn_norm weight
norm_tensor = get_tensor_by_name('blk.0.attn_norm.weight')
norm_weight = np.frombuffer(norm_tensor.data.tobytes(), dtype=np.float32)
print(f"\nblk.0.attn_norm.weight stats:")
print(f"  min: {norm_weight.min():.6f}, max: {norm_weight.max():.6f}")
print(f"  mean: {norm_weight.mean():.6f}, std: {norm_weight.std():.6f}")
print(f"  first 5: {norm_weight[:5]}")

# Compute RMSNorm
eps = 1e-6
def rms_norm(x, weight, eps):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

normed = rms_norm(token_16_emb, norm_weight, eps)
print(f"\nAfter RMSNorm:")
print(f"  min: {normed.min():.6f}, max: {normed.max():.6f}")
print(f"  mean: {normed.mean():.6f}, std: {normed.std():.6f}")
print(f"  first 5: {normed[:5]}")

# Also print the RMS value itself
rms = np.sqrt(np.mean(token_16_emb ** 2) + eps)
print(f"\nRMS of embedding: {rms:.6f}")
print(f"Normalized embedding (before weight) first 5: {(token_16_emb[:5] / rms)}")
