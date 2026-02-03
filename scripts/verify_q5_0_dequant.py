#!/usr/bin/env python3
"""Verify Q5_0 dequantization by extracting embedding values"""

import numpy as np
from gguf import GGUFReader
import struct

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

# Find token_embd.weight which uses Q5_0
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        print(f"=== token_embd.weight ===")
        print(f"Shape (GGUF ne): {tensor.shape}")  # [896, 151936]
        print(f"Type: {tensor.tensor_type}")  # 9 = Q5_0
        print(f"Data shape: {tensor.data.shape}")  # raw data
        
        # Q5_0 format (from ggml):
        # struct block_q5_0 {
        #     ggml_half d;      // 2 bytes - scale
        #     uint8_t qh[4];    // 4 bytes - high bits of 32 values
        #     uint8_t qs[16];   // 16 bytes - low 4 bits of 32 values
        # }
        # Total: 2 + 4 + 16 = 22 bytes per 32 values
        
        # For 896 values per row:
        # 896 / 32 = 28 blocks
        # 28 * 22 = 616 bytes per row (matches data shape!)
        
        data = tensor.data  # raw bytes
        print(f"\nRaw data shape: {data.shape}")  # Should be (151936, 616)
        
        # Dequantize row 28 (token '=') manually
        row_28 = data[28]  # 616 bytes
        print(f"Row 28 has {len(row_28)} bytes")
        
        dequant_row = []
        for block_idx in range(28):  # 28 blocks of 32 values each
            block_start = block_idx * 22  # 22 bytes per block
            
            # Scale (fp16, 2 bytes)
            d_bytes = row_28[block_start:block_start+2]
            d = np.frombuffer(d_bytes, dtype=np.float16)[0]
            d = float(d)
            
            # High bits for 32 values (4 bytes = 32 bits, one bit per value)
            qh = row_28[block_start+2:block_start+6]
            
            # Low 4 bits for 32 values (16 bytes, 2 values per byte)
            qs = row_28[block_start+6:block_start+22]
            
            # Dequantize 32 values
            for j in range(32):
                # Get the 5-bit quantized value
                # Low 4 bits from qs
                q_lo = (qs[j // 2] >> (4 * (j % 2))) & 0x0F
                # High bit from qh (packed as 32 bits across 4 bytes)
                q_hi = (qh[j // 8] >> (j % 8)) & 0x01
                q = q_lo | (q_hi << 4)
                
                # Convert to signed and dequantize
                # Q5_0 uses offset of 16 to center the 5-bit range [0,31] around 0
                dequant_val = d * (q - 16)
                dequant_row.append(dequant_val)
        
        dequant_row = np.array(dequant_row)
        print(f"\nDequantized row 28 (token '='):")
        print(f"  Length: {len(dequant_row)}")
        print(f"  First 5: {dequant_row[:5]}")
        print(f"  Sum: {dequant_row.sum():.6f}")
        print(f"  Min: {dequant_row.min():.6f}, Max: {dequant_row.max():.6f}")
        
        # Save for comparison
        np.save('/tmp/py_embedding_28.npy', dequant_row)
        print("\nSaved to /tmp/py_embedding_28.npy")
        
        # Compare with what our Rust code produces
        print("\n=== Our Rust implementation produces for token 28 ===")
        print("First 5: [0.0055, -0.0166, -0.0166, 0.0193, 0.0193]")
        print("Sum: -0.230461")
        print("These should match if dequantization is correct")
        
        break
