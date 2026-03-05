# TODO

Planned work and known gaps for llama-gguf.

## GPU Backend Parity

Metal, DX12, and Vulkan backends fall back to CPU for several operations. Bringing these to parity with the CUDA `gpu_only.rs` engine is the highest-impact performance work remaining.

### Operations needing GPU implementation (Metal, DX12, Vulkan)

- [x] `matmul` — general matrix multiplication
- [x] `matvec` — matrix-vector multiplication
- [x] `dequantize` — quantized tensor dequantization on GPU (Q8_0, Q4_K, Q6_K)
- [x] `matvec_q` — quantized matrix-vector multiplication (2-pass: dequant + matvec)
- [x] `vec_mat_q` — quantized vector-matrix multiplication (2-pass: dequant + vec_mat)
- [x] `attention` — full attention computation (online softmax, GQA, causal masking)
- [x] `attention_cached` — KV-cached attention (online softmax, GQA)
- [x] `softmax` — GPU path for all tensor sizes
- [x] `rope` — GPU path for all configurations

### Dedicated inference engines

- [x] Metal `gpu_only.rs` equivalent — full GPU-resident inference for Apple Silicon
- [x] DX12 `gpu_only.rs` equivalent — full GPU-resident inference for Windows
- [x] Vulkan `gpu_only.rs` equivalent — full GPU-resident inference (cross-platform)

## CUDA Backend

- [x] Move attention from CPU roundtrip to fully on-GPU (QK norm, partial RoPE, attention gating)
- [x] Batched inference (batched RoPE, KV cache update, linear kernels, `forward_batch`)
- [x] Flash Attention kernel (`flash_attention_cached` with O(head_dim) shared memory)
- [x] FP16 compute path (f32↔f16 conversion, vec_mat_f16, element-wise ops, rms_norm_f16)

## Model Support

- [ ] Phi-2 / GPT-NeoX (combined QKV tensor layout)
- [ ] Gemma2 (extra norm layers, logit softcapping)
- [ ] Mamba / Mamba2 (pure SSM architecture)
- [ ] RWKV (linear attention)
- [ ] Cohere Command R (multi-head latent attention)

## Quantization

- [x] GGUF quantization writing (currently read-only)
- [x] IQ quantization formats (IQ1_S, IQ2_XXS, IQ3_XXS, etc.)
- [x] GPU-side quantization (quantize on device without host roundtrip)

## Inference Performance

- [x] Continuous batching for server mode
- [x] Speculative decoding
- [x] Prompt caching / prefix sharing across requests
- [x] KV cache quantization (INT8/FP8 KV cache to reduce VRAM)
- [x] Paged attention (vLLM-style memory management)
- [x] Tensor parallelism (split layers across GPUs on a single node)

## Tokenizer

- [x] Unigram tokenizer support
- [x] Tokenizer from `tokenizer.json` (HuggingFace format, not just GGUF metadata)

## Server

- [x] OpenAI-compatible embeddings endpoint
- [x] OpenAI-compatible completions endpoint with function calling
- [x] Token usage reporting in API responses
- [x] Request queuing and concurrency limits
- [x] Model hot-swapping

## RAG

- [x] Incremental re-indexing for updated documents
- [x] Chunking strategies (semantic, sliding window with overlap tuning)
- [x] Multi-modal embedding support (images, tables)
- [x] SQLite vector store alternative (for single-node setups without PostgreSQL)

## Distributed

- [x] Automatic model sharding based on available VRAM per node
- [x] Fault tolerance and node recovery
- [x] Load balancing across heterogeneous hardware
- [x] Multi-node tensor parallelism (complement to pipeline parallelism)

## Developer Experience

- [x] `cargo bench` coverage for all quantization formats
- [x] Integration test suite that runs without external dependencies (embedded test model)
- [x] CI matrix covering all feature flag combinations
- [x] Flamegraph-based profiling guide
