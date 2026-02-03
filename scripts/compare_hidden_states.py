#!/usr/bin/env python3
"""
Compare hidden states between llama-cpp and our implementation.

Since llama-cpp-python doesn't expose intermediate hidden states directly,
we'll compute them manually using the GGUF weights and compare with
our Rust implementation's output.
"""

import numpy as np
import json
import subprocess
import sys
from pathlib import Path
from gguf import GGUFReader

MODEL_PATH = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Dequantization Functions
# =============================================================================

def dequantize_q5_0_block(block_data):
    """Dequantize a Q5_0 block (32 elements from 22 bytes)"""
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
    """Dequantize a Q8_0 block (32 elements from 34 bytes)"""
    scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
    qs = np.frombuffer(block_data[2:34], dtype=np.int8)
    return qs.astype(np.float32) * float(scale)

def dequantize_q4_k_block(block_data):
    """Dequantize a Q4_K block (256 elements from 144 bytes)"""
    # Q4_K structure:
    # - d: 2 bytes (f16 super-block scale)
    # - dmin: 2 bytes (f16 super-block min)
    # - scales: 12 bytes (6-bit scales and mins for 8 sub-blocks, packed)
    # - qs: 128 bytes (4-bit quantized values)
    # Total: 144 bytes for 256 elements
    
    d = np.frombuffer(block_data[:2], dtype=np.float16)[0]
    dmin = np.frombuffer(block_data[2:4], dtype=np.float16)[0]
    scales_raw = block_data[4:16]
    qs = np.frombuffer(block_data[16:144], dtype=np.uint8)
    
    # Unpack 6-bit scales and mins from 12 bytes
    # This is complex packing - simplified version
    scales = np.zeros(8, dtype=np.float32)
    mins = np.zeros(8, dtype=np.float32)
    
    for i in range(4):
        scales[i] = scales_raw[i] & 0x3F
        scales[i + 4] = scales_raw[i + 4] & 0x3F
    
    for i in range(4):
        m_lo = (scales_raw[i] >> 6) & 0x03
        m_hi = (scales_raw[8 + i // 2] >> (4 * (i % 2))) & 0x0F
        mins[i] = m_lo | (m_hi << 2)
        
        m_lo = (scales_raw[i + 4] >> 6) & 0x03
        m_hi = (scales_raw[10 + i // 2] >> (4 * (i % 2))) & 0x0F
        mins[i + 4] = m_lo | (m_hi << 2)
    
    result = np.zeros(256, dtype=np.float32)
    
    for sb in range(8):
        sc = scales[sb]
        m = mins[sb]
        
        for j in range(32):
            idx = sb * 32 + j
            byte_idx = idx // 2
            
            if idx % 2 == 0:
                q = qs[byte_idx] & 0x0F
            else:
                q = (qs[byte_idx] >> 4) & 0x0F
            
            result[idx] = float(d) * sc * q - float(dmin) * m
    
    return result

def dequantize_q6_k_block(block_data):
    """Dequantize a Q6_K block (256 elements from 210 bytes)
    
    Layout matches llama.cpp's dequantize_row_q6_K.
    """
    ql = np.frombuffer(block_data[:128], dtype=np.uint8)
    qh = np.frombuffer(block_data[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_data[192:208], dtype=np.int8)
    d = np.frombuffer(block_data[208:210], dtype=np.float16)[0]
    
    result = np.zeros(256, dtype=np.float32)
    
    # Process 256 elements in two groups of 128
    for n in range(2):
        ql_base = n * 64
        qh_base = n * 32
        sc_base = n * 8
        out_base = n * 128
        
        for l in range(32):
            is_idx = l // 16
            
            # Extract 4 quantized values using interleaved pattern
            # CRITICAL: Cast to int before subtracting 32 to avoid uint8 overflow!
            q1 = int((int(ql[ql_base + l]) & 0x0F) | ((int(qh[qh_base + l]) & 0x03) << 4)) - 32
            q2 = int((int(ql[ql_base + l + 32]) & 0x0F) | (((int(qh[qh_base + l]) >> 2) & 0x03) << 4)) - 32
            q3 = int((int(ql[ql_base + l]) >> 4) | (((int(qh[qh_base + l]) >> 4) & 0x03) << 4)) - 32
            q4 = int((int(ql[ql_base + l + 32]) >> 4) | (((int(qh[qh_base + l]) >> 6) & 0x03) << 4)) - 32
            
            # Apply scales with correct interleaved pattern
            result[out_base + l] = float(d) * int(scales[sc_base + is_idx]) * q1
            result[out_base + l + 32] = float(d) * int(scales[sc_base + is_idx + 2]) * q2
            result[out_base + l + 64] = float(d) * int(scales[sc_base + is_idx + 4]) * q3
            result[out_base + l + 96] = float(d) * int(scales[sc_base + is_idx + 6]) * q4
    
    return result

def dequantize_tensor(tensor, out_features, in_features):
    """Dequantize a tensor based on its type"""
    raw_data = np.array(tensor.data)
    tensor_type = tensor.tensor_type
    
    result = np.zeros((out_features, in_features), dtype=np.float32)
    
    if tensor_type == 0:  # F32
        return np.frombuffer(bytes(tensor.data), dtype=np.float32).reshape(out_features, in_features)
    elif tensor_type == 6:  # Q5_0
        bytes_per_block = 22
        blocks_per_row = in_features // 32
        for row in range(out_features):
            row_data = bytes(raw_data[row])
            for b in range(blocks_per_row):
                block_data = row_data[b * bytes_per_block:(b + 1) * bytes_per_block]
                result[row, b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
    elif tensor_type == 8:  # Q8_0
        bytes_per_block = 34
        blocks_per_row = in_features // 32
        for row in range(out_features):
            row_data = bytes(raw_data[row])
            for b in range(blocks_per_row):
                block_data = row_data[b * bytes_per_block:(b + 1) * bytes_per_block]
                result[row, b * 32:(b + 1) * 32] = dequantize_q8_0_block(block_data)
    elif tensor_type == 12:  # Q4_K
        bytes_per_block = 144
        blocks_per_row = in_features // 256
        for row in range(out_features):
            row_data = bytes(raw_data[row])
            for b in range(blocks_per_row):
                block_data = row_data[b * bytes_per_block:(b + 1) * bytes_per_block]
                result[row, b * 256:(b + 1) * 256] = dequantize_q4_k_block(block_data)
    elif tensor_type == 14:  # Q6_K
        bytes_per_block = 210
        blocks_per_row = in_features // 256
        for row in range(out_features):
            row_data = bytes(raw_data[row])
            for b in range(blocks_per_row):
                block_data = row_data[b * bytes_per_block:(b + 1) * bytes_per_block]
                result[row, b * 256:(b + 1) * 256] = dequantize_q6_k_block(block_data)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")
    
    return result

# =============================================================================
# Model Operations
# =============================================================================

def rms_norm(x, weight, eps=1e-6):
    """RMS Normalization"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

def silu(x):
    """SiLU activation"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))

def rope_neox(x, pos, head_dim, freq_base=1000000.0):
    """Apply NeoX-style RoPE"""
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

# =============================================================================
# Full Forward Pass (Python Reference)
# =============================================================================

class PythonModel:
    def __init__(self, model_path):
        print("Loading GGUF model for Python reference...")
        self.reader = GGUFReader(model_path)
        self.tensors = {}
        for t in self.reader.tensors:
            self.tensors[t.name] = t
        
        # Get model config
        self.hidden_size = 896
        self.num_heads = 14
        self.num_kv_heads = 2
        self.head_dim = 64
        self.num_layers = 24
        self.freq_base = 1000000.0
        
        # Detect intermediate size from FFN
        ffn_gate = self.tensors["blk.0.ffn_gate.weight"]
        self.intermediate_size = ffn_gate.shape[1]  # [hidden, intermediate]
        print(f"Detected intermediate_size: {self.intermediate_size}")
        
        # Cache dequantized embeddings
        print("Dequantizing token embeddings...")
        emb_data = np.array(self.tensors["token_embd.weight"].data)
        self.embeddings = {}  # Lazy load
        self._emb_raw = emb_data
        
    def get_embedding(self, token_id):
        if token_id not in self.embeddings:
            embedding = np.zeros(self.hidden_size, dtype=np.float32)
            row_data = bytes(self._emb_raw[token_id])
            for b in range(self.hidden_size // 32):
                block_data = row_data[b * 22:(b + 1) * 22]
                embedding[b * 32:(b + 1) * 32] = dequantize_q5_0_block(block_data)
            self.embeddings[token_id] = embedding
        return self.embeddings[token_id].copy()
    
    def forward_layer(self, hidden, layer_idx, pos):
        """Process one transformer layer"""
        prefix = f"blk.{layer_idx}"
        
        # Attention norm
        attn_norm_w = np.frombuffer(bytes(self.tensors[f"{prefix}.attn_norm.weight"].data), dtype=np.float32)
        normed = rms_norm(hidden, attn_norm_w)
        
        # Q, K, V projections
        wq = dequantize_tensor(self.tensors[f"{prefix}.attn_q.weight"], 
                               self.num_heads * self.head_dim, self.hidden_size)
        wk = dequantize_tensor(self.tensors[f"{prefix}.attn_k.weight"],
                               self.num_kv_heads * self.head_dim, self.hidden_size)
        wv = dequantize_tensor(self.tensors[f"{prefix}.attn_v.weight"],
                               self.num_kv_heads * self.head_dim, self.hidden_size)
        
        bq = np.frombuffer(bytes(self.tensors[f"{prefix}.attn_q.bias"].data), dtype=np.float32)
        bk = np.frombuffer(bytes(self.tensors[f"{prefix}.attn_k.bias"].data), dtype=np.float32)
        bv = np.frombuffer(bytes(self.tensors[f"{prefix}.attn_v.bias"].data), dtype=np.float32)
        
        q = normed @ wq.T + bq
        k = normed @ wk.T + bk
        v = normed @ wv.T + bv
        
        # Apply RoPE
        q_heads = q.reshape(self.num_heads, self.head_dim)
        k_heads = k.reshape(self.num_kv_heads, self.head_dim)
        
        for h in range(self.num_heads):
            q_heads[h] = rope_neox(q_heads[h], pos, self.head_dim, self.freq_base)
        for h in range(self.num_kv_heads):
            k_heads[h] = rope_neox(k_heads[h], pos, self.head_dim, self.freq_base)
        
        # At pos=0, attention output = V (self-attention on single token)
        # For pos > 0, we'd need KV cache, but for single token test, this is fine
        v_heads = v.reshape(self.num_kv_heads, self.head_dim)
        attn_out = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        for h in range(self.num_heads):
            kv_h = h // num_queries_per_kv
            attn_out[h] = v_heads[kv_h]
        
        attn_out_flat = attn_out.flatten()
        
        # Output projection
        wo = dequantize_tensor(self.tensors[f"{prefix}.attn_output.weight"],
                               self.hidden_size, self.num_heads * self.head_dim)
        output = attn_out_flat @ wo.T
        
        # Residual
        hidden = hidden + output
        
        # FFN
        ffn_norm_w = np.frombuffer(bytes(self.tensors[f"{prefix}.ffn_norm.weight"].data), dtype=np.float32)
        normed_ffn = rms_norm(hidden, ffn_norm_w)
        
        ffn_gate = dequantize_tensor(self.tensors[f"{prefix}.ffn_gate.weight"],
                                     self.intermediate_size, self.hidden_size)
        ffn_up = dequantize_tensor(self.tensors[f"{prefix}.ffn_up.weight"],
                                   self.intermediate_size, self.hidden_size)
        ffn_down = dequantize_tensor(self.tensors[f"{prefix}.ffn_down.weight"],
                                     self.hidden_size, self.intermediate_size)
        
        gate_out = normed_ffn @ ffn_gate.T
        up_out = normed_ffn @ ffn_up.T
        swiglu_out = silu(gate_out) * up_out
        ffn_out = swiglu_out @ ffn_down.T
        
        # Residual
        hidden = hidden + ffn_out
        
        return hidden
    
    def forward(self, token_id, num_layers=None):
        """Full forward pass, returns hidden states after each layer"""
        if num_layers is None:
            num_layers = self.num_layers
            
        hidden = self.get_embedding(token_id)
        states = [("embedding", hidden.copy())]
        
        for layer_idx in range(num_layers):
            hidden = self.forward_layer(hidden, layer_idx, pos=0)
            states.append((f"layer_{layer_idx}", hidden.copy()))
        
        return states


def run_rust_comparison(token_id):
    """Run the Rust implementation and capture hidden states"""
    # We'll run a Rust example that outputs JSON with hidden states
    rust_cmd = [
        "cargo", "run", "--release", "--example", "compare_hidden_states",
        "--", str(token_id)
    ]
    
    result = subprocess.run(
        rust_cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Rust example failed:")
        print(result.stderr)
        return None
    
    # Parse JSON output from Rust
    try:
        # Find JSON in output
        output = result.stdout
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = output[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse Rust output: {e}")
        print(f"Output was: {result.stdout[:500]}")
    
    return None


def compare_states(python_states, rust_states):
    """Compare hidden states and report differences"""
    print("\n" + "=" * 70)
    print("HIDDEN STATE COMPARISON")
    print("=" * 70)
    
    for py_state in python_states:
        name, py_hidden = py_state
        
        if name not in rust_states:
            print(f"\n{name}: NOT FOUND IN RUST OUTPUT")
            continue
        
        rust_hidden = np.array(rust_states[name])
        
        # Compute difference metrics
        abs_diff = np.abs(py_hidden - rust_hidden)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        
        # Relative difference (avoiding division by zero)
        rel_diff = np.where(np.abs(py_hidden) > 1e-6, 
                           abs_diff / np.abs(py_hidden), 
                           abs_diff)
        max_rel_diff = rel_diff.max()
        
        # Stats
        py_stats = f"min={py_hidden.min():.4f}, max={py_hidden.max():.4f}, mean={py_hidden.mean():.4f}"
        rust_stats = f"min={rust_hidden.min():.4f}, max={rust_hidden.max():.4f}, mean={rust_hidden.mean():.4f}"
        
        # Determine if match
        matches = max_diff < 0.01  # 1% tolerance
        status = "✓ MATCH" if matches else "✗ MISMATCH"
        
        print(f"\n{name}: {status}")
        print(f"  Python: {py_stats}")
        print(f"  Rust:   {rust_stats}")
        print(f"  Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}")
        
        if not matches:
            print(f"  Python first 5: {py_hidden[:5]}")
            print(f"  Rust first 5:   {rust_hidden[:5]}")
            
            # Find first major divergence
            diverge_idx = np.argmax(abs_diff > 0.01)
            if abs_diff[diverge_idx] > 0.01:
                print(f"  First major divergence at index {diverge_idx}:")
                print(f"    Python: {py_hidden[diverge_idx]:.6f}")
                print(f"    Rust:   {rust_hidden[diverge_idx]:.6f}")


def main():
    token_id = 28  # '=' token
    
    print(f"Comparing hidden states for token {token_id}")
    print("-" * 70)
    
    # Run Python reference
    print("\n1. Running Python reference implementation...")
    model = PythonModel(MODEL_PATH)
    
    # Only compute first few layers for speed
    num_layers = 3
    print(f"   Computing {num_layers} layers...")
    python_states = model.forward(token_id, num_layers=num_layers)
    
    print("\n2. Python hidden states computed:")
    for name, hidden in python_states:
        print(f"   {name}: min={hidden.min():.4f}, max={hidden.max():.4f}")
    
    # Save Python states for Rust comparison
    states_dict = {name: hidden.tolist() for name, hidden in python_states}
    states_file = PROJECT_ROOT / "scripts" / "python_states.json"
    with open(states_file, 'w') as f:
        json.dump(states_dict, f)
    print(f"\n   Saved to {states_file}")
    
    print("\n3. Run the Rust comparison example with:")
    print(f"   cargo run --example compare_hidden_states -- {token_id}")
    print("\n   Or compare manually with the saved states.")


if __name__ == "__main__":
    main()
