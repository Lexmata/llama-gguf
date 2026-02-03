#!/usr/bin/env python3
"""
Step-by-step comparison of layer 0 forward pass.
Manually computes each step using raw GGUF weights to compare with Rust.
"""

import numpy as np
from gguf import GGUFReader

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def dequantize_q5_0(data: bytes, shape: tuple) -> np.ndarray:
    """Dequantize Q5_0 format (5-bit quantization)"""
    # Q5_0 block: 2 bytes scale (f16) + 4 bytes high bits + 16 bytes low bits = 22 bytes per 32 elements
    block_size = 32
    num_elements = np.prod(shape)
    num_blocks = num_elements // block_size
    
    result = np.zeros(num_elements, dtype=np.float32)
    
    for block_idx in range(num_blocks):
        block_offset = block_idx * 22  # 22 bytes per block
        
        # Scale is f16 (2 bytes)
        scale = np.frombuffer(data[block_offset:block_offset+2], dtype=np.float16)[0]
        scale = float(scale)
        
        # High bits: 4 bytes = 32 bits, one per element
        qh = np.frombuffer(data[block_offset+2:block_offset+6], dtype=np.uint8)
        qh_bits = np.unpackbits(qh, bitorder='little')[:32]
        
        # Low bits: 16 bytes = 32 nibbles (4 bits each)
        ql = np.frombuffer(data[block_offset+6:block_offset+22], dtype=np.uint8)
        
        for i in range(32):
            # Low 4 bits
            if i < 16:
                low_bits = ql[i] & 0x0F
            else:
                low_bits = (ql[i - 16] >> 4) & 0x0F
            
            # High bit
            high_bit = qh_bits[i]
            
            # Combine: 5-bit value = high_bit << 4 | low_bits
            q_val = (high_bit << 4) | low_bits
            
            # Dequantize: value = scale * (q_val - 16)
            result[block_idx * block_size + i] = scale * (q_val - 16)
    
    return result.reshape(shape)

def dequantize_q4_k(data: bytes, shape: tuple) -> np.ndarray:
    """Dequantize Q4_K format"""
    # Q4_K block structure (256 elements per super-block):
    # - d: f16 (2 bytes) - super-block scale
    # - dmin: f16 (2 bytes) - super-block min
    # - scales: 12 bytes (6-bit scales for 8 sub-blocks, packed)
    # - qs: 128 bytes (4-bit quantized values)
    # Total: 144 bytes per 256 elements
    
    block_size = 256
    num_elements = np.prod(shape)
    num_blocks = num_elements // block_size
    
    result = np.zeros(num_elements, dtype=np.float32)
    
    for block_idx in range(num_blocks):
        block_offset = block_idx * 144
        
        # Read d and dmin (f16)
        d = np.frombuffer(data[block_offset:block_offset+2], dtype=np.float16)[0]
        dmin = np.frombuffer(data[block_offset+2:block_offset+4], dtype=np.float16)[0]
        d = float(d)
        dmin = float(dmin)
        
        # Read packed scales (12 bytes for 8 sub-blocks)
        scales_raw = np.frombuffer(data[block_offset+4:block_offset+16], dtype=np.uint8)
        
        # Unpack 6-bit scales and mins
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        
        # First 4 scales: lower 6 bits of bytes 0-3
        for i in range(4):
            scales[i] = scales_raw[i] & 0x3F
        # Next 4 scales: lower 6 bits of bytes 4-7
        for i in range(4):
            scales[i + 4] = scales_raw[i + 4] & 0x3F
        
        # First 4 mins: upper 2 bits of bytes 0-3 combined with bytes 8-9
        for i in range(4):
            mins[i] = ((scales_raw[i] >> 6) | ((scales_raw[8 + i // 2] >> (4 * (i % 2))) & 0x0F) << 2)
        # Next 4 mins: upper 2 bits of bytes 4-7 combined with bytes 10-11
        for i in range(4):
            mins[i + 4] = ((scales_raw[i + 4] >> 6) | ((scales_raw[10 + i // 2] >> (4 * (i % 2))) & 0x0F) << 2)
        
        # Read quantized values (128 bytes = 256 4-bit values)
        qs = np.frombuffer(data[block_offset+16:block_offset+144], dtype=np.uint8)
        
        # Dequantize each sub-block (32 elements each)
        for sb in range(8):
            sc = scales[sb]
            m = mins[sb]
            
            for j in range(32):
                idx = sb * 32 + j
                
                # Get 4-bit quantized value
                byte_idx = idx // 2
                if idx % 2 == 0:
                    q = qs[byte_idx] & 0x0F
                else:
                    q = (qs[byte_idx] >> 4) & 0x0F
                
                # Dequantize
                result[block_idx * block_size + idx] = d * sc * q - dmin * m
    
    return result.reshape(shape)

def dequantize_f32(data: bytes, shape: tuple) -> np.ndarray:
    """Load F32 tensor"""
    return np.frombuffer(data, dtype=np.float32).reshape(shape)

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS Normalization"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

def rope_neox(x: np.ndarray, pos: int, head_dim: int, freq_base: float = 1000000.0) -> np.ndarray:
    """Apply NeoX-style RoPE to a single head's Q or K vector"""
    half_dim = head_dim // 2
    result = x.copy()
    
    for i in range(half_dim):
        freq = 1.0 / (freq_base ** ((2 * i) / head_dim))
        theta = pos * freq
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x0 = x[i]
        x1 = x[i + half_dim]
        
        result[i] = x0 * cos_theta - x1 * sin_theta
        result[i + half_dim] = x0 * sin_theta + x1 * cos_theta
    
    return result

def main():
    print("Loading GGUF file...")
    reader = GGUFReader(MODEL_PATH)
    
    # Build tensor dictionary
    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = tensor
    
    # Get model parameters
    hidden_size = 896
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    
    # Token to test: "=" which is token 28
    token_id = 28
    pos = 0
    
    print(f"\n=== Testing token {token_id} ('=') at position {pos} ===")
    
    # Step 1: Get embedding
    print("\n--- Step 1: Token Embedding ---")
    emb_tensor = tensors["token_embd.weight"]
    emb_data = dequantize_q5_0(bytes(emb_tensor.data), (151936, hidden_size))
    embedding = emb_data[token_id]
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding stats: min={embedding.min():.6f}, max={embedding.max():.6f}, mean={embedding.mean():.6f}")
    print(f"Embedding first 10: {embedding[:10]}")
    
    # Step 2: Apply attention norm
    print("\n--- Step 2: Attention RMSNorm ---")
    norm_tensor = tensors["blk.0.attn_norm.weight"]
    norm_weight = dequantize_f32(bytes(norm_tensor.data), (hidden_size,))
    
    normed = rms_norm(embedding, norm_weight)
    print(f"After norm stats: min={normed.min():.6f}, max={normed.max():.6f}, mean={normed.mean():.6f}")
    print(f"After norm first 10: {normed[:10]}")
    
    # Step 3: Compute Q, K, V projections
    print("\n--- Step 3: Q, K, V Projections ---")
    
    # Load weights (stored as [out_features, in_features] in column-major = [in_features, out_features] row-major)
    wq_tensor = tensors["blk.0.attn_q.weight"]
    wk_tensor = tensors["blk.0.attn_k.weight"]
    wv_tensor = tensors["blk.0.attn_v.weight"]
    
    # Q weight: [hidden_size, num_heads * head_dim] = [896, 896]
    wq = dequantize_q5_0(bytes(wq_tensor.data), (num_heads * head_dim, hidden_size))
    wk = dequantize_q5_0(bytes(wk_tensor.data), (num_kv_heads * head_dim, hidden_size))
    wv = dequantize_q5_0(bytes(wv_tensor.data), (num_kv_heads * head_dim, hidden_size))
    
    print(f"Wq shape (GGUF): {wq.shape}")
    print(f"Wk shape (GGUF): {wk.shape}")
    print(f"Wv shape (GGUF): {wv.shape}")
    
    # Load biases
    bq_tensor = tensors["blk.0.attn_q.bias"]
    bk_tensor = tensors["blk.0.attn_k.bias"]
    bv_tensor = tensors["blk.0.attn_v.bias"]
    
    bq = dequantize_f32(bytes(bq_tensor.data), (num_heads * head_dim,))
    bk = dequantize_f32(bytes(bk_tensor.data), (num_kv_heads * head_dim,))
    bv = dequantize_f32(bytes(bv_tensor.data), (num_kv_heads * head_dim,))
    
    print(f"Bq stats: min={bq.min():.2f}, max={bq.max():.2f}")
    print(f"Bk stats: min={bk.min():.2f}, max={bk.max():.2f}")
    print(f"Bv stats: min={bv.min():.2f}, max={bv.max():.2f}")
    
    # Compute Q = normed @ Wq.T + bq
    # GGUF stores weights as [out, in] in column-major, which is [in, out] in row-major
    # So for y = x @ W where x is [in], W should be [in, out]
    # We have wq as [out, in] from reshape, so we need wq.T
    q = normed @ wq.T + bq
    k = normed @ wk.T + bk
    v = normed @ wv.T + bv
    
    print(f"\nQ (before RoPE) stats: min={q.min():.2f}, max={q.max():.2f}")
    print(f"K (before RoPE) stats: min={k.min():.2f}, max={k.max():.2f}")
    print(f"V stats: min={v.min():.2f}, max={v.max():.2f}")
    print(f"Q first 10: {q[:10]}")
    print(f"K first 10: {k[:10]}")
    print(f"V first 10: {v[:10]}")
    
    # Step 4: Apply RoPE to Q and K
    print("\n--- Step 4: Apply RoPE ---")
    q_heads = q.reshape(num_heads, head_dim)
    k_heads = k.reshape(num_kv_heads, head_dim)
    
    q_rope = np.zeros_like(q_heads)
    k_rope = np.zeros_like(k_heads)
    
    for h in range(num_heads):
        q_rope[h] = rope_neox(q_heads[h], pos, head_dim)
    for h in range(num_kv_heads):
        k_rope[h] = rope_neox(k_heads[h], pos, head_dim)
    
    print(f"Q after RoPE, head 0 first 10: {q_rope[0, :10]}")
    print(f"K after RoPE, head 0 first 10: {k_rope[0, :10]}")
    
    # At position 0 with cos(0)=1, sin(0)=0, RoPE should be identity
    print(f"\nAt pos=0, RoPE should be identity. Checking...")
    print(f"Q diff from original: {np.abs(q_rope - q_heads).max():.6f}")
    print(f"K diff from original: {np.abs(k_rope - k_heads).max():.6f}")
    
    # Step 5: Self-attention at position 0
    print("\n--- Step 5: Self-Attention (pos=0) ---")
    # At position 0, we only attend to ourselves
    # attention_score = Q @ K.T / sqrt(head_dim)
    # For GQA: each Q head attends to its corresponding KV head
    
    scale = 1.0 / np.sqrt(head_dim)
    
    # For head 0, which uses KV head 0
    attn_score = np.dot(q_rope[0], k_rope[0]) * scale
    print(f"Head 0 attention score (raw): {attn_score:.4f}")
    
    # Softmax (with single element, output is always 1.0)
    attn_weight = 1.0  # softmax of single element
    
    # Output = attention_weight * V
    # Head 0 uses KV head 0
    v_heads = v.reshape(num_kv_heads, head_dim)
    attn_out_head0 = attn_weight * v_heads[0]
    print(f"Attention output head 0 first 10: {attn_out_head0[:10]}")
    
    # Step 6: Compute full attention output
    print("\n--- Step 6: Full Attention Output ---")
    attn_out = np.zeros((num_heads, head_dim), dtype=np.float32)
    
    num_queries_per_kv = num_heads // num_kv_heads  # 14 // 2 = 7
    
    for h in range(num_heads):
        kv_h = h // num_queries_per_kv
        score = np.dot(q_rope[h], k_rope[kv_h]) * scale
        # Softmax of single element = 1.0
        attn_out[h] = v_heads[kv_h]
    
    attn_out_flat = attn_out.flatten()
    print(f"Attention output (flat) stats: min={attn_out_flat.min():.4f}, max={attn_out_flat.max():.4f}")
    print(f"Attention output first 10: {attn_out_flat[:10]}")
    
    # Step 7: Output projection
    print("\n--- Step 7: Output Projection ---")
    wo_tensor = tensors["blk.0.attn_output.weight"]
    wo = dequantize_q5_0(bytes(wo_tensor.data), (hidden_size, num_heads * head_dim))
    
    # o = attn_out @ Wo.T (Wo is [hidden, attn_dim] stored as [attn_dim, hidden] row-major)
    output = attn_out_flat @ wo.T
    print(f"Output projection stats: min={output.min():.4f}, max={output.max():.4f}")
    print(f"Output projection first 10: {output[:10]}")
    
    # Step 8: Residual connection
    print("\n--- Step 8: Residual Connection ---")
    hidden_after_attn = embedding + output
    print(f"After attention+residual stats: min={hidden_after_attn.min():.4f}, max={hidden_after_attn.max():.4f}")
    print(f"After attention+residual first 10: {hidden_after_attn[:10]}")
    
    # Save these values for comparison with Rust
    print("\n=== REFERENCE VALUES FOR RUST COMPARISON ===")
    print(f"Embedding[0:5]: {embedding[:5]}")
    print(f"After norm[0:5]: {normed[:5]}")
    print(f"Q before RoPE[0:5]: {q[:5]}")
    print(f"K before RoPE[0:5]: {k[:5]}")
    print(f"V[0:5]: {v[:5]}")
    print(f"Attn output[0:5]: {attn_out_flat[:5]}")
    print(f"Output proj[0:5]: {output[:5]}")
    print(f"After residual[0:5]: {hidden_after_attn[:5]}")

if __name__ == "__main__":
    main()
