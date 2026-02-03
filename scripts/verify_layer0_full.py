#!/usr/bin/env python3
"""
Verify full layer 0 computation including FFN.
"""

import numpy as np
from gguf import GGUFReader

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def dequantize_q5_0_block(block_data):
    """Dequantize a Q5_0 block"""
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

def dequantize_q8_0_block(block_data):
    """Dequantize a Q8_0 block"""
    scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
    qs = np.frombuffer(block_data[2:34], dtype=np.int8)
    return qs.astype(np.float32) * float(scale)

def dequantize_tensor(raw_data, shape, tensor_type, out_features, in_features):
    """Dequantize a tensor based on its type"""
    result = np.zeros((out_features, in_features), dtype=np.float32)
    
    if tensor_type == 6:  # Q5_0
        bytes_per_block = 22
        for row in range(out_features):
            row_data = bytes(raw_data[row])
            for b in range(in_features // 32):
                block_data = row_data[b * bytes_per_block:(b + 1) * bytes_per_block]
                result[row, b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    elif tensor_type == 8:  # Q8_0
        bytes_per_block = 34
        for row in range(out_features):
            row_data = bytes(raw_data[row])
            for b in range(in_features // 32):
                block_data = row_data[b * bytes_per_block:(b + 1) * bytes_per_block]
                result[row, b * 32:(b + 1) * 32] = dequantize_q8_0_block(block_data)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")
    
    return result

def silu(x):
    """SiLU activation (x * sigmoid(x))"""
    return x * (1.0 / (1.0 + np.exp(-x)))

def main():
    print("Loading GGUF file...")
    reader = GGUFReader(MODEL_PATH)
    
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t
    
    # Model params
    hidden_size = 896
    intermediate_size = 4864  # Typical for Qwen2.5
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    
    # Check intermediate size from FFN weights
    ffn_gate = tensors["blk.0.ffn_gate.weight"]
    print(f"FFN gate shape: {ffn_gate.shape}, type: {ffn_gate.tensor_type}")
    intermediate_size = ffn_gate.shape[0]
    print(f"Intermediate size: {intermediate_size}")
    
    # === Step 1: Get embedding ===
    emb_data = np.array(tensors["token_embd.weight"].data)
    embedding = np.zeros(hidden_size, dtype=np.float32)
    for b in range(hidden_size // 32):
        block_data = bytes(emb_data[28])[b * 22:(b + 1) * 22]
        embedding[b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    
    print(f"\nEmbedding: min={embedding.min():.4f}, max={embedding.max():.4f}")
    
    # === Step 2: Attention (we already verified this) ===
    # Apply RMSNorm
    norm_weight = np.frombuffer(bytes(tensors["blk.0.attn_norm.weight"].data), dtype=np.float32)
    eps = 1e-6
    rms = np.sqrt(np.mean(embedding ** 2) + eps)
    normed = (embedding / rms) * norm_weight
    
    # V projection (Q8_0)
    wv_raw = np.array(tensors["blk.0.attn_v.weight"].data)
    wv = dequantize_tensor(wv_raw, None, 8, num_kv_heads * head_dim, hidden_size)
    bv = np.frombuffer(bytes(tensors["blk.0.attn_v.bias"].data), dtype=np.float32)
    v = normed @ wv.T + bv
    
    # At pos=0, attention output = V replicated for all heads
    v_heads = v.reshape(num_kv_heads, head_dim)
    attn_out = np.zeros((num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        kv_h = h // (num_heads // num_kv_heads)
        attn_out[h] = v_heads[kv_h]
    attn_out_flat = attn_out.flatten()
    
    # Output projection (Q5_0)
    wo_raw = np.array(tensors["blk.0.attn_output.weight"].data)
    wo = dequantize_tensor(wo_raw, None, 6, hidden_size, num_heads * head_dim)
    output = attn_out_flat @ wo.T
    
    # Residual connection
    hidden_after_attn = embedding + output
    
    print(f"After attention+residual: min={hidden_after_attn.min():.4f}, max={hidden_after_attn.max():.4f}")
    print(f"First 5: {hidden_after_attn[:5]}")
    
    # === Step 3: FFN ===
    print("\n=== FFN ===")
    
    # FFN norm
    ffn_norm_weight = np.frombuffer(bytes(tensors["blk.0.ffn_norm.weight"].data), dtype=np.float32)
    rms_ffn = np.sqrt(np.mean(hidden_after_attn ** 2) + eps)
    normed_ffn = (hidden_after_attn / rms_ffn) * ffn_norm_weight
    
    print(f"After FFN norm: min={normed_ffn.min():.4f}, max={normed_ffn.max():.4f}")
    
    # FFN gate, up, down projections
    print(f"\nLoading FFN weights...")
    ffn_gate_tensor = tensors["blk.0.ffn_gate.weight"]
    ffn_up_tensor = tensors["blk.0.ffn_up.weight"]
    ffn_down_tensor = tensors["blk.0.ffn_down.weight"]
    
    print(f"FFN gate: type={ffn_gate_tensor.tensor_type}, shape={ffn_gate_tensor.shape}")
    print(f"FFN up: type={ffn_up_tensor.tensor_type}, shape={ffn_up_tensor.shape}")
    print(f"FFN down: type={ffn_down_tensor.tensor_type}, shape={ffn_down_tensor.shape}")
    
    ffn_gate_raw = np.array(ffn_gate_tensor.data)
    ffn_up_raw = np.array(ffn_up_tensor.data)
    ffn_down_raw = np.array(ffn_down_tensor.data)
    
    # Dequantize (assuming Q5_0 for now, type 6)
    ffn_gate = dequantize_tensor(ffn_gate_raw, None, 6, intermediate_size, hidden_size)
    ffn_up = dequantize_tensor(ffn_up_raw, None, 6, intermediate_size, hidden_size)
    
    # Down is [hidden_size, intermediate_size]
    ffn_down = dequantize_tensor(ffn_down_raw, None, ffn_down_tensor.tensor_type, hidden_size, intermediate_size)
    
    print(f"\nFFN gate dequant: shape={ffn_gate.shape}")
    print(f"FFN up dequant: shape={ffn_up.shape}")
    print(f"FFN down dequant: shape={ffn_down.shape}")
    
    # SwiGLU: out = (gate(x) * sigmoid(gate(x))) * up(x) then down()
    gate_out = normed_ffn @ ffn_gate.T
    up_out = normed_ffn @ ffn_up.T
    
    print(f"\nGate output: min={gate_out.min():.4f}, max={gate_out.max():.4f}")
    print(f"Up output: min={up_out.min():.4f}, max={up_out.max():.4f}")
    
    # SwiGLU activation
    swiglu_out = silu(gate_out) * up_out
    print(f"After SwiGLU: min={swiglu_out.min():.4f}, max={swiglu_out.max():.4f}")
    
    # Down projection
    ffn_out = swiglu_out @ ffn_down.T
    print(f"FFN output: min={ffn_out.min():.4f}, max={ffn_out.max():.4f}")
    
    # Final residual
    layer0_output = hidden_after_attn + ffn_out
    print(f"\n=== Layer 0 Final Output ===")
    print(f"min={layer0_output.min():.4f}, max={layer0_output.max():.4f}")
    print(f"First 10: {layer0_output[:10]}")
    
    # Compare with Rust layer 0 output from trace_full_forward
    # (Looking at the output after layer 0)
    # After layer  0: min=   -0.69, max=    0.82, mean= -0.0003, std=  0.0863
    #             first 5: [0.0040, -0.0122, -0.0078, 0.0312, 0.0162]
    # Hmm, those don't look right for the full model - let me get the actual values
    
if __name__ == "__main__":
    main()
