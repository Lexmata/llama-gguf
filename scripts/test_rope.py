#!/usr/bin/env python3
"""Test RoPE computation to compare with Rust implementation"""

import numpy as np

def apply_rope_neox(x, position, freq_base=1000000.0, freq_scale=1.0):
    """Apply NeoX-style RoPE to a vector"""
    head_dim = len(x)
    half_dim = head_dim // 2
    
    position = position * freq_scale
    result = np.zeros_like(x)
    
    for i in range(half_dim):
        # Frequency for this dimension
        freq = 1.0 / (freq_base ** ((2 * i) / head_dim))
        theta = position * freq
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # NeoX style: pairs are (i, i+half_dim)
        x0 = x[i]
        x1 = x[i + half_dim]
        
        result[i] = x0 * cos_theta - x1 * sin_theta
        result[i + half_dim] = x0 * sin_theta + x1 * cos_theta
    
    return result

def apply_rope_llama(x, position, freq_base=1000000.0, freq_scale=1.0):
    """Apply LLaMA-style RoPE to a vector (consecutive pairs)"""
    head_dim = len(x)
    
    position = position * freq_scale
    result = np.zeros_like(x)
    
    for i in range(head_dim // 2):
        # Frequency for this dimension
        freq = 1.0 / (freq_base ** ((2 * i) / head_dim))
        theta = position * freq
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # LLaMA style: pairs are (2i, 2i+1)
        x0 = x[2*i]
        x1 = x[2*i + 1]
        
        result[2*i] = x0 * cos_theta - x1 * sin_theta
        result[2*i + 1] = x0 * sin_theta + x1 * cos_theta
    
    return result

# Test with simple values
head_dim = 64  # Qwen2 0.5B has head_dim=64
np.random.seed(42)
x = np.random.randn(head_dim).astype(np.float32)

print("=== RoPE Test ===")
print(f"Input first 5: {x[:5]}")
print(f"Input last 5: {x[-5:]}")

# Test position 0 (should not change values much since theta=0)
print("\n=== Position 0 ===")
neox_p0 = apply_rope_neox(x, 0)
llama_p0 = apply_rope_llama(x, 0)
print(f"NeoX first 5: {neox_p0[:5]}")
print(f"LLaMA first 5: {llama_p0[:5]}")
print(f"Difference: {np.max(np.abs(neox_p0 - x)):.6f} (should be ~0)")

# Test position 1
print("\n=== Position 1 ===")
neox_p1 = apply_rope_neox(x, 1)
llama_p1 = apply_rope_llama(x, 1)
print(f"NeoX first 5: {neox_p1[:5]}")
print(f"LLaMA first 5: {llama_p1[:5]}")
print(f"NeoX differs from LLaMA: {np.max(np.abs(neox_p1 - llama_p1)):.6f}")

# Test position 3 (the '=' token position in "1+1=")
print("\n=== Position 3 ===")
neox_p3 = apply_rope_neox(x, 3)
llama_p3 = apply_rope_llama(x, 3)
print(f"NeoX first 5: {neox_p3[:5]}")
print(f"LLaMA first 5: {llama_p3[:5]}")
print(f"NeoX differs from LLaMA: {np.max(np.abs(neox_p3 - llama_p3)):.6f}")

# Show the rotation amounts for different dimensions at position 3
print("\n=== Rotation angles at position 3 ===")
freq_base = 1000000.0
for i in [0, 1, 2, 31, 32]:  # First few and middle/last dimensions
    freq = 1.0 / (freq_base ** ((2 * i) / head_dim))
    theta = 3 * freq
    print(f"Dim {i}: freq={freq:.6f}, theta={theta:.6f}, cos={np.cos(theta):.4f}, sin={np.sin(theta):.4f}")
