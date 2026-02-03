#!/usr/bin/env python3
"""
Verify layer 0 computation step by step.
Uses correct Q5_0 dequantization that matches Rust.
"""

import numpy as np
from gguf import GGUFReader

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def dequantize_q5_0_block(block_data):
    """Dequantize a Q5_0 block matching the Rust implementation"""
    scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
    qh = int.from_bytes(block_data[2:6], 'little')
    qs = block_data[6:22]
    
    result = np.zeros(32, dtype=np.float32)
    for i in range(16):
        byte_val = qs[i]
        lo4 = byte_val & 0x0F
        hi4 = (byte_val >> 4) & 0x0F
        lo5 = (qh >> i) & 1
        hi5 = (qh >> (i + 16)) & 1
        lo = (lo4 | (lo5 << 4)) - 16
        hi = (hi4 | (hi5 << 4)) - 16
        result[i] = float(scale) * lo
        result[i + 16] = float(scale) * hi
    return result

def dequantize_q5_0_tensor(raw_data, shape):
    """Dequantize entire Q5_0 tensor"""
    num_elements = np.prod(shape)
    num_blocks = num_elements // 32
    bytes_per_block = 22
    
    result = np.zeros(num_elements, dtype=np.float32)
    for b in range(num_blocks):
        block_data = raw_data[b * bytes_per_block:(b + 1) * bytes_per_block]
        result[b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    
    return result.reshape(shape)

def main():
    print("Loading GGUF file...")
    reader = GGUFReader(MODEL_PATH)
    
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t
    
    # Model params
    hidden_size = 896
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    
    # === Step 1: Get embedding for token 28 ===
    print("\n=== Step 1: Embedding ===")
    emb_data = np.array(tensors["token_embd.weight"].data)  # (vocab, bytes_per_token)
    token_28_raw = bytes(emb_data[28])
    
    embedding = np.zeros(hidden_size, dtype=np.float32)
    for b in range(hidden_size // 32):
        block_data = token_28_raw[b * 22:(b + 1) * 22]
        embedding[b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    
    print(f"Embedding: min={embedding.min():.6f}, max={embedding.max():.6f}")
    print(f"First 5: {embedding[:5]}")
    
    # === Step 2: RMSNorm ===
    print("\n=== Step 2: RMSNorm ===")
    norm_weight = np.frombuffer(bytes(tensors["blk.0.attn_norm.weight"].data), dtype=np.float32)
    
    eps = 1e-6
    rms = np.sqrt(np.mean(embedding ** 2) + eps)
    normed = (embedding / rms) * norm_weight
    
    print(f"RMS value: {rms:.6f}")
    print(f"After norm: min={normed.min():.6f}, max={normed.max():.6f}")
    print(f"First 5: {normed[:5]}")
    
    # Rust comparison
    rust_norm_first10 = [-0.033233173, -0.09551787, 0.077177264, 0.16835451, -0.10561742,
                        -0.047722254, -0.04460435, 0.079641014, -0.05736941, 0.02641047]
    print(f"\nRust first 10: {rust_norm_first10}")
    if np.allclose(normed[:10], rust_norm_first10, rtol=1e-3):
        print("MATCH!")
    else:
        print(f"MISMATCH! Diff: {normed[:10] - rust_norm_first10}")
    
    # === Step 3: Q, K, V projections ===
    print("\n=== Step 3: Q, K, V Projections ===")
    
    # Load biases
    bq = np.frombuffer(bytes(tensors["blk.0.attn_q.bias"].data), dtype=np.float32)
    bk = np.frombuffer(bytes(tensors["blk.0.attn_k.bias"].data), dtype=np.float32)
    bv = np.frombuffer(bytes(tensors["blk.0.attn_v.bias"].data), dtype=np.float32)
    
    print(f"Bq: min={bq.min():.2f}, max={bq.max():.2f}")
    print(f"Bk: min={bk.min():.2f}, max={bk.max():.2f}")
    print(f"Bv: min={bv.min():.2f}, max={bv.max():.2f}")
    
    # Load Q weight - shape is [896, 896] in GGUF
    # Q weight as Q5_0: need to dequantize
    wq_tensor = tensors["blk.0.attn_q.weight"]
    print(f"\nWq tensor: type={wq_tensor.tensor_type}, shape={wq_tensor.shape}")
    
    # For Q5_0 with shape [896, 896]:
    # Each row is 896 elements = 28 blocks = 616 bytes
    # Total rows = 896
    wq_raw = np.array(wq_tensor.data)
    print(f"Wq raw shape: {wq_raw.shape}")
    
    # Dequantize Wq
    # Shape [out_features, in_features] = [896, 896]
    # But in GGUF column-major, this is actually stored as [in, out] = [896, 896]
    num_rows = 896  # out_features
    wq_dequant = np.zeros((num_rows, hidden_size), dtype=np.float32)
    
    for row in range(num_rows):
        row_data = bytes(wq_raw[row])
        for b in range(hidden_size // 32):
            block_data = row_data[b * 22:(b + 1) * 22]
            wq_dequant[row, b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    
    print(f"Wq dequantized: shape={wq_dequant.shape}")
    print(f"Wq stats: min={wq_dequant.min():.4f}, max={wq_dequant.max():.4f}")
    
    # Compute Q = normed @ Wq^T + bq
    # Since Wq is [out, in], we need normed @ Wq^T where normed is [in]
    # Result is [out]
    q = normed @ wq_dequant.T + bq
    
    print(f"\nQ (after bias): min={q.min():.2f}, max={q.max():.2f}")
    print(f"Q first 10: {q[:10]}")
    
    # Rust comparison
    rust_q_first10 = [0.053115353, 0.242939, -0.040524334, -0.73274565, -14.9995775,
                     0.59721935, 0.44047412, 0.3104411, -15.542654, -34.987522]
    print(f"\nRust Q first 10: {rust_q_first10}")
    if np.allclose(q[:10], rust_q_first10, rtol=0.1):
        print("MATCH (within 10%)!")
    else:
        print(f"MISMATCH! Python Q: {q[:10]}")
        print(f"Ratio: {q[:10] / rust_q_first10}")
    
    # Compute K and V similarly
    wk_raw = np.array(tensors["blk.0.attn_k.weight"].data)
    wv_raw = np.array(tensors["blk.0.attn_v.weight"].data)
    
    wk_dequant = np.zeros((num_kv_heads * head_dim, hidden_size), dtype=np.float32)
    wv_dequant = np.zeros((num_kv_heads * head_dim, hidden_size), dtype=np.float32)
    
    for row in range(num_kv_heads * head_dim):
        row_data = bytes(wk_raw[row])
        for b in range(hidden_size // 32):
            block_data = row_data[b * 22:(b + 1) * 22]
            wk_dequant[row, b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
            
        row_data = bytes(wv_raw[row])
        for b in range(hidden_size // 32):
            block_data = row_data[b * 22:(b + 1) * 22]
            wv_dequant[row, b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    
    k = normed @ wk_dequant.T + bk
    v = normed @ wv_dequant.T + bv
    
    print(f"\nK (after bias): min={k.min():.2f}, max={k.max():.2f}")
    print(f"K first 10: {k[:10]}")
    
    print(f"\nV: min={v.min():.4f}, max={v.max():.4f}")
    print(f"V first 10: {v[:10]}")
    
    # Rust V comparison
    rust_v_first10 = [-0.0154732065, -0.011354765, 0.025334416, 0.012548599, 0.0367594,
                     0.028623417, 0.007893015, 0.0054122787, -0.01558469, 0.0059777]
    print(f"\nRust V first 10: {rust_v_first10}")
    if np.allclose(v[:10], rust_v_first10, rtol=0.1):
        print("V MATCH!")
    else:
        print(f"V MISMATCH!")
    
    # === Step 4: At position 0, attention output = V (softmax of single element = 1) ===
    print("\n=== Step 4: Attention at pos=0 ===")
    # At pos=0, we only attend to ourselves, so attention weights are all 1.0
    # Attention output for each head is just V
    # For GQA: heads 0-6 use KV head 0, heads 7-13 use KV head 1
    
    v_heads = v.reshape(num_kv_heads, head_dim)
    attn_out = np.zeros((num_heads, head_dim), dtype=np.float32)
    
    num_queries_per_kv = num_heads // num_kv_heads
    for h in range(num_heads):
        kv_h = h // num_queries_per_kv
        attn_out[h] = v_heads[kv_h]
    
    attn_out_flat = attn_out.flatten()
    print(f"Attention output: min={attn_out_flat.min():.4f}, max={attn_out_flat.max():.4f}")
    print(f"First 10: {attn_out_flat[:10]}")
    
    # Rust comparison
    rust_attn_first10 = [-0.0154732065, -0.011354765, 0.025334416, 0.012548599, 0.0367594,
                        0.028623417, 0.007893015, 0.0054122787, -0.01558469, 0.0059777]
    if np.allclose(attn_out_flat[:10], rust_attn_first10, rtol=0.1):
        print("Attention output MATCH!")
    
    # === Step 5: Output projection ===
    print("\n=== Step 5: Output Projection ===")
    wo_raw = np.array(tensors["blk.0.attn_output.weight"].data)
    
    wo_dequant = np.zeros((hidden_size, num_heads * head_dim), dtype=np.float32)
    for row in range(hidden_size):
        row_data = bytes(wo_raw[row])
        for b in range((num_heads * head_dim) // 32):
            block_data = row_data[b * 22:(b + 1) * 22]
            wo_dequant[row, b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    
    output = attn_out_flat @ wo_dequant.T
    
    print(f"Output projection: min={output.min():.4f}, max={output.max():.4f}")
    print(f"First 10: {output[:10]}")
    
    # Rust comparison
    rust_out_first10 = [-0.016385693, 0.0012340598, 0.00031158322, 0.009895786, -0.0055726185,
                       -0.0048790635, -0.0040900544, 0.0039386232, -0.001069307, -0.00047134276]
    print(f"\nRust output first 10: {rust_out_first10}")
    if np.allclose(output[:10], rust_out_first10, rtol=0.1):
        print("Output projection MATCH!")
    else:
        print(f"MISMATCH! Python: {output[:10]}")
    
    # === Step 6: Residual connection ===
    print("\n=== Step 6: Residual (after attention) ===")
    hidden_after_attn = embedding + output
    
    print(f"After residual: min={hidden_after_attn.min():.4f}, max={hidden_after_attn.max():.4f}")
    print(f"First 10: {hidden_after_attn[:10]}")
    
    # Rust comparison
    rust_residual_first10 = [-0.010862011, -0.015336985, -0.016259462, 0.029228672, 0.013760267,
                            -0.013164585, 0.004195468, -0.015394263, 0.004454375, -0.0059950245]
    print(f"\nRust residual first 10: {rust_residual_first10}")
    if np.allclose(hidden_after_attn[:10], rust_residual_first10, rtol=0.1):
        print("Residual MATCH!")
    else:
        print(f"MISMATCH!")
        print(f"Diff: {hidden_after_attn[:10] - rust_residual_first10}")

if __name__ == "__main__":
    main()
