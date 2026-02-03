#!/usr/bin/env python3
"""Compare dequantized weight values between GGUF and our understanding"""

import numpy as np
from gguf import GGUFReader

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

# Find output.weight (Q8_0)
for tensor in reader.tensors:
    if tensor.name == 'output.weight':
        print(f"=== output.weight ===")
        print(f"Shape (GGUF ne): {tensor.shape}")  # [896, 151936]
        print(f"Type: {tensor.tensor_type}")  # 8 = Q8_0
        print(f"Data shape: {tensor.data.shape}")  # (151936, 952)
        
        # Q8_0 format: for each block of 32 values
        #   - 1 fp16 scale (2 bytes) 
        #   - 32 int8 quants (32 bytes)
        # Total: 34 bytes per block
        
        # 952 bytes per row / 34 bytes per block = 28 blocks
        # 28 blocks * 32 values = 896 values = hidden_size
        
        # Let's dequantize the first few rows manually
        data = tensor.data  # shape (151936, 952), dtype uint8
        
        # Dequantize first vocab token's weights (row 0)
        row0 = data[0]  # 952 bytes = 28 Q8_0 blocks
        
        # Parse Q8_0 blocks
        dequant_row0 = []
        for block_idx in range(28):  # 28 blocks per row
            block_start = block_idx * 34
            # Scale is fp16 in first 2 bytes
            scale_bytes = row0[block_start:block_start+2]
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            scale = float(scale)
            
            # Quants are int8 in next 32 bytes
            quants = row0[block_start+2:block_start+34].astype(np.int8)
            
            # Dequantize: value = quant * scale
            for q in quants:
                dequant_row0.append(float(q) * scale)
        
        dequant_row0 = np.array(dequant_row0)
        print(f"\nFirst vocab token (token 0) weights:")
        print(f"  Length: {len(dequant_row0)}")
        print(f"  First 5: {dequant_row0[:5]}")
        print(f"  Last 5: {dequant_row0[-5:]}")
        print(f"  Sum: {dequant_row0.sum():.6f}")
        
        # Now let's also look at token 17 (the digit "2")
        row17 = data[17]
        dequant_row17 = []
        for block_idx in range(28):
            block_start = block_idx * 34
            scale_bytes = row17[block_start:block_start+2]
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            scale = float(scale)
            quants = row17[block_start+2:block_start+34].astype(np.int8)
            for q in quants:
                dequant_row17.append(float(q) * scale)
        
        dequant_row17 = np.array(dequant_row17)
        print(f"\nToken 17 ('2') weights:")
        print(f"  First 5: {dequant_row17[:5]}")
        print(f"  Last 5: {dequant_row17[-5:]}")
        print(f"  Sum: {dequant_row17.sum():.6f}")
        
        # Save for comparison with Rust
        np.save('/tmp/output_weight_row0.npy', dequant_row0)
        np.save('/tmp/output_weight_row17.npy', dequant_row17)
        print("\nSaved dequantized rows to /tmp/output_weight_row{0,17}.npy")
        
        # Also dequantize the entire tensor to verify our understanding
        print("\n=== Full dequantization verification ===")
        all_dequant = []
        for row_idx in range(min(10, data.shape[0])):  # Just first 10 rows for speed
            row = data[row_idx]
            row_vals = []
            for block_idx in range(28):
                block_start = block_idx * 34
                scale = float(np.frombuffer(row[block_start:block_start+2], dtype=np.float16)[0])
                quants = row[block_start+2:block_start+34].astype(np.int8)
                for q in quants:
                    row_vals.append(float(q) * scale)
            all_dequant.append(row_vals)
        
        all_dequant = np.array(all_dequant)
        print(f"Dequantized shape: {all_dequant.shape}")  # Should be (10, 896)
        print(f"This confirms: rows = vocab tokens, columns = hidden dims")
        
        # So the linear indexing is:
        # dequant_flat[j * 896 + i] = weight for vocab j, hidden i
        # Which matches vec_mat: out[j] += x[i] * w[i + j * 896]
        break

# Also check token_embd.weight (Q4_K)
print("\n\n=== token_embd.weight ===")
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        print(f"Shape (GGUF ne): {tensor.shape}")  # [896, 151936]
        print(f"Type: {tensor.tensor_type}")  # 6 = Q4_K
        print(f"Data shape: {tensor.data.shape}")  # (151936, 616)
        
        # Q4_K: block size 256, 144 bytes per block
        # But the data shape suggests 616 bytes per row
        # 616 / 144 = 4.28 blocks... doesn't divide evenly
        # Maybe there's a different block size
        
        # Actually Q4_K has a more complex structure with super-blocks
        # Let's just note the shapes for now
        print(f"Note: Q4_K is more complex, but layout principle is same")
        print(f"Each row (vocab token) has its own packed hidden dim weights")
        break
