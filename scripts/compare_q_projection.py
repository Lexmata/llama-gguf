#!/usr/bin/env python3
"""
Compare Q projection output between our implementation and NumPy reference.

This loads the actual GGUF weights and computes Q = RMSNorm(embed) @ Wq + bq
to verify the computation is correct.
"""

import struct
import numpy as np

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def read_gguf_minimal(path):
    """Minimal GGUF reader that extracts just what we need."""
    with open(path, 'rb') as f:
        # Header
        magic = f.read(4)
        assert magic == b'GGUF', f"Invalid magic: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"GGUF v{version}: {tensor_count} tensors, {metadata_count} metadata")
        
        # Skip metadata
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            if key_len > 1000000:  # Sanity check
                raise ValueError(f"Invalid key_len: {key_len}")
            key = f.read(key_len).decode('utf-8')
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # Skip value based on type
            if value_type == 8:  # string
                val_len = struct.unpack('<Q', f.read(8))[0]
                f.read(val_len)
            elif value_type in (4, 10):  # uint32, float32
                f.read(4)
            elif value_type in (6, 7):  # uint64, int64
                f.read(8)
            elif value_type == 11:  # bool
                f.read(1)
            elif value_type == 9:  # array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 8:  # string array
                    for _ in range(arr_len):
                        s_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(s_len)
                else:
                    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 8, 7: 8, 10: 4, 11: 1}
                    f.read(arr_len * sizes.get(arr_type, 4))
            else:
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 5: 4}
                f.read(sizes.get(value_type, 4))
        
        # Alignment to 32 bytes
        pos = f.tell()
        if pos % 32 != 0:
            f.seek(32 - (pos % 32), 1)
        
        # Read tensor info
        tensors = {}
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': offset}
        
        # Alignment after tensor info
        pos = f.tell()
        if pos % 32 != 0:
            f.seek(32 - (pos % 32), 1)
        
        data_start = f.tell()
        
    return tensors, data_start

def dequantize_q5_0(data, n_elements):
    """Dequantize Q5_0."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)
    
    bytes_per_block = 22  # 2 (scale) + 4 (high bits) + 16 (nibbles)
    
    for block_idx in range(n_blocks):
        block_start = block_idx * bytes_per_block
        
        d = struct.unpack('<e', data[block_start:block_start+2])[0]
        qh = int.from_bytes(data[block_start+2:block_start+6], 'little')
        qs = data[block_start+6:block_start+22]
        
        out_offset = block_idx * block_size
        for j in range(block_size):
            byte_idx = j // 2
            nibble = (qs[byte_idx] >> (4 * (j % 2))) & 0x0F
            high_bit = (qh >> j) & 1
            q = nibble | (high_bit << 4)
            result[out_offset + j] = d * (q - 16)
    
    return result

def dequantize_q4_k(data, n_elements):
    """Dequantize Q4_K properly."""
    # Q4_K has super-blocks of 256 elements
    # Each super-block: 2 bytes d (f16), 2 bytes dmin (f16), 12 bytes scales, 128 bytes quantized
    block_size = 256
    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_elements, dtype=np.float32)
    
    bytes_per_block = 2 + 2 + 12 + 128  # 144 bytes per block
    
    for block_idx in range(n_blocks):
        block_start = block_idx * bytes_per_block
        if block_start + bytes_per_block > len(data):
            break
            
        d = struct.unpack('<e', data[block_start:block_start+2])[0]
        dmin = struct.unpack('<e', data[block_start+2:block_start+4])[0]
        
        scales = data[block_start+4:block_start+16]
        qs = data[block_start+16:block_start+144]
        
        out_offset = block_idx * block_size
        for sub in range(8):  # 8 sub-blocks of 32 elements each
            # Decode scales
            sc_idx = sub
            if sc_idx < 4:
                sc = scales[sc_idx] & 0x3F
                m = scales[sc_idx + 4] & 0x3F
            else:
                sc = (scales[sc_idx - 4] >> 6) | ((scales[sc_idx] & 0xF) << 2)
                m = (scales[sc_idx] >> 4) | ((scales[sc_idx + 4] & 0xF) << 2)
            
            d2 = d * sc
            m2 = dmin * m
            
            qs_offset = sub * 16
            for j in range(32):
                qi = (qs[qs_offset + j // 2] >> (4 * (j % 2))) & 0xF
                idx = out_offset + sub * 32 + j
                if idx < n_elements:
                    result[idx] = d2 * qi - m2
    
    return result

def load_tensor(path, tensors, data_start, name):
    """Load and dequantize a tensor."""
    if name not in tensors:
        return None
    
    info = tensors[name]
    n_elements = int(np.prod(info['dims']))
    
    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        
        if info['dtype'] == 0:  # F32
            data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32)
        elif info['dtype'] == 6:  # Q5_0
            bytes_per_block = 22
            n_blocks = n_elements // 32
            raw = f.read(n_blocks * bytes_per_block)
            data = dequantize_q5_0(raw, n_elements)
        elif info['dtype'] == 12:  # Q4_K
            bytes_per_block = 144
            n_blocks = (n_elements + 255) // 256
            raw = f.read(n_blocks * bytes_per_block)
            data = dequantize_q4_k(raw, n_elements)
        else:
            print(f"Warning: Unknown dtype {info['dtype']} for {name}")
            return None
    
    return data

def rms_norm(x, w, eps=1e-6):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * w

def main():
    print("=== Q Projection Comparison ===")
    
    tensors, data_start = read_gguf_minimal(MODEL_PATH)
    
    # Config
    hidden_size = 896
    num_heads = 14
    head_dim = 64
    eps = 1e-6
    
    # Load tensors
    print("\nLoading tensors...")
    
    emb = load_tensor(MODEL_PATH, tensors, data_start, "token_embd.weight")
    print(f"Embedding: dtype={tensors['token_embd.weight']['dtype']}, dims={tensors['token_embd.weight']['dims']}")
    
    attn_norm = load_tensor(MODEL_PATH, tensors, data_start, "blk.0.attn_norm.weight")
    print(f"Attn norm: dtype={tensors['blk.0.attn_norm.weight']['dtype']}, dims={tensors['blk.0.attn_norm.weight']['dims']}")
    
    wq = load_tensor(MODEL_PATH, tensors, data_start, "blk.0.attn_q.weight")
    print(f"Wq: dtype={tensors['blk.0.attn_q.weight']['dtype']}, dims={tensors['blk.0.attn_q.weight']['dims']}")
    
    q_bias = load_tensor(MODEL_PATH, tensors, data_start, "blk.0.attn_q.bias")
    print(f"Q bias: dtype={tensors['blk.0.attn_q.bias']['dtype']}, dims={tensors['blk.0.attn_q.bias']['dims']}")
    
    # Token 16 embedding
    emb_reshaped = emb.reshape(hidden_size, -1, order='F')
    tok_emb = emb_reshaped[:, 16].copy()
    print(f"\nToken 16 embedding: min={tok_emb.min():.6f}, max={tok_emb.max():.6f}")
    print(f"First 5: {tok_emb[:5]}")
    
    # RMS norm
    normed = rms_norm(tok_emb, attn_norm, eps)
    print(f"\nAfter RMS norm: min={normed.min():.6f}, max={normed.max():.6f}")
    print(f"First 5: {normed[:5]}")
    
    # Q projection: Q = normed @ Wq + bias
    # GGUF stores Wq as [hidden_size, num_heads * head_dim] in column-major
    # So for y = x @ W, we need W.shape = (in_features, out_features) in column-major
    # Which means W.reshape(in, out, order='F') gives the correct matrix
    wq_matrix = wq.reshape(hidden_size, num_heads * head_dim, order='F')
    
    q = normed @ wq_matrix
    print(f"\nQ (before bias): min={q.min():.6f}, max={q.max():.6f}")
    print(f"First 10: {q[:10]}")
    
    q = q + q_bias
    print(f"\nQ (after bias): min={q.min():.4f}, max={q.max():.4f}")
    print(f"First 10: {q[:10]}")
    
    # Compare with what Rust reports
    print("\n=== Expected from Rust layer_by_layer_debug (trace_layer0 without bias) ===")
    print("Q (before bias): min=-6.4981, max=7.7531")
    print("First 5: [0.016891703, 0.029571375, -0.119339295, -0.2899048, -1.0193536]")
    
    print("\n=== Comparison ===")
    rust_q_first5 = np.array([0.016891703, 0.029571375, -0.119339295, -0.2899048, -1.0193536])
    py_q_first5 = q[:5] - q_bias[:5]  # Remove bias for comparison
    
    print(f"Rust Q (no bias) first 5: {rust_q_first5}")
    print(f"Python Q (no bias) first 5: {py_q_first5}")
    print(f"Difference: {py_q_first5 - rust_q_first5}")
    print(f"Max abs diff: {np.abs(py_q_first5 - rust_q_first5).max():.6f}")

if __name__ == "__main__":
    main()
