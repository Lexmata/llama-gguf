#!/usr/bin/env python3
"""Check GGML type mapping in Python's gguf library"""

from gguf import GGUFReader

# GGML tensor types from llama.cpp/ggml
# https://github.com/ggerganov/ggml/blob/master/include/ggml.h
GGML_TYPES = {
    0: "F32",
    1: "F16", 
    2: "Q4_0",
    3: "Q4_1",
    4: "Q5_0_DEPRECATED",  # Not used
    5: "Q8_0_DEPRECATED",  # Not used
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
reader = GGUFReader(model_path)

print("=== Tensor types in model ===")
for tensor in reader.tensors[:10]:  # First 10 tensors
    type_name = GGML_TYPES.get(tensor.tensor_type, f"UNKNOWN({tensor.tensor_type})")
    print(f"{tensor.name}: type={tensor.tensor_type} ({type_name})")
