#!/usr/bin/env python3
"""Test vec_mat computation against numpy"""

import numpy as np

# Create simple test case
k = 4  # input size
n = 3  # output size

# Input vector
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

# Weight matrix [k, n] in GGUF format
# Element (i, j) is at index i + j*k
W_data = np.array([
    1.0, 0.0, 0.0, 0.0,  # Column 0
    0.0, 1.0, 0.0, 0.0,  # Column 1  
    0.0, 0.0, 1.0, 0.0,  # Column 2
], dtype=np.float32)

# Reshape to [k, n] (row-major in numpy)
W = W_data.reshape(n, k).T  # This gives [k, n]
print(f"W = \n{W}")

# Standard matrix multiply
y = x @ W
print(f"x @ W = {y}")  # Should be [1.0, 2.0, 3.0]

# Now let's verify what our Rust code computes:
# out[j] = sum_i(x[i] * W_data[i + j*k])
out = np.zeros(n, dtype=np.float32)
for j in range(n):
    for i in range(k):
        out[j] += x[i] * W_data[i + j * k]
print(f"Our formula = {out}")

# They should match
print(f"Match: {np.allclose(y, out)}")

print("\n=== More realistic test ===")
k = 896  # hidden_size
n = 896  # num_heads * head_dim

np.random.seed(42)
x = np.random.randn(k).astype(np.float32)
W_data = np.random.randn(k * n).astype(np.float32)

# Reshape W_data to [n, k] and transpose to get [k, n]
W = W_data.reshape(n, k).T

# Standard matmul
y_numpy = x @ W
print(f"NumPy result: min={y_numpy.min():.4f}, max={y_numpy.max():.4f}")
print(f"NumPy first 5: {y_numpy[:5]}")

# Our formula
out = np.zeros(n, dtype=np.float32)
for j in range(n):
    for i in range(k):
        out[j] += x[i] * W_data[i + j * k]
print(f"Our formula: min={out.min():.4f}, max={out.max():.4f}")
print(f"Our first 5: {out[:5]}")

print(f"Match: {np.allclose(y_numpy, out)}")

# Now let's see what happens if GGUF actually stores as [n, k] (transposed)
print("\n=== Testing if GGUF uses different layout ===")
# If GGUF stores as W[j, i] = data[j + i*n] (row-major with dims swapped)
out2 = np.zeros(n, dtype=np.float32)
for j in range(n):
    for i in range(k):
        out2[j] += x[i] * W_data[j + i * n]  # Different indexing
print(f"Alt formula: min={out2.min():.4f}, max={out2.max():.4f}")
print(f"Alt first 5: {out2[:5]}")
print(f"Alt matches numpy: {np.allclose(y_numpy, out2)}")
