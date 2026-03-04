# TODO

Planned work and known gaps for llama-gguf.

## GPU Backend Parity

Metal, DX12, and Vulkan backends fall back to CPU for several operations. Bringing these to parity with the CUDA `gpu_only.rs` engine is the highest-impact performance work remaining.

### Operations needing GPU implementation (Metal, DX12, Vulkan)

- [ ] `matmul` — general matrix multiplication
- [ ] `matvec` — matrix-vector multiplication
- [ ] `dequantize` — quantized tensor dequantization on GPU
- [ ] `matvec_q` — quantized matrix-vector multiplication
- [ ] `vec_mat_q` — quantized vector-matrix multiplication (currently GPU for F32 only)
- [ ] `attention` — full attention computation
- [ ] `attention_cached` — KV-cached attention
- [ ] `softmax` — GPU path for all tensor sizes (currently falls back on small tensors)
- [ ] `rope` — GPU path for all configurations (currently falls back on some variants)

### Dedicated inference engines

- [ ] Metal `gpu_only.rs` equivalent — full GPU-resident inference for Apple Silicon
- [ ] DX12 `gpu_only.rs` equivalent — full GPU-resident inference for Windows
- [ ] Vulkan `gpu_only.rs` equivalent — full GPU-resident inference (cross-platform)

## CUDA Backend

- [ ] Move attention from CPU roundtrip to fully on-GPU (Qwen3Next QK norm, partial RoPE, attention gating)
- [ ] Batched inference (process multiple sequences simultaneously)
- [ ] Flash Attention kernel implementation
- [ ] FP16 compute path (currently dequantizes to F32)

## Model Support

- [ ] Phi-2 / GPT-NeoX (combined QKV tensor layout)
- [ ] Gemma2 (extra norm layers, logit softcapping)
- [ ] Mamba / Mamba2 (pure SSM architecture)
- [ ] RWKV (linear attention)
- [ ] Cohere Command R (multi-head latent attention)

## Quantization

- [ ] GGUF quantization writing (currently read-only)
- [ ] IQ quantization formats (IQ1_S, IQ2_XXS, IQ3_XXS, etc.)
- [ ] GPU-side quantization (quantize on device without host roundtrip)

## Inference Performance

- [ ] Continuous batching for server mode
- [ ] Speculative decoding
- [ ] Prompt caching / prefix sharing across requests
- [ ] KV cache quantization (INT8/FP8 KV cache to reduce VRAM)
- [ ] Paged attention (vLLM-style memory management)
- [ ] Tensor parallelism (split layers across GPUs on a single node)

## Tokenizer

- [ ] Unigram tokenizer support
- [ ] Tokenizer from `tokenizer.json` (HuggingFace format, not just GGUF metadata)

## Server

- [ ] OpenAI-compatible embeddings endpoint
- [ ] OpenAI-compatible completions endpoint with function calling
- [ ] Token usage reporting in API responses
- [ ] Request queuing and concurrency limits
- [ ] Model hot-swapping

## RAG

- [ ] Incremental re-indexing for updated documents
- [ ] Chunking strategies (semantic, sliding window with overlap tuning)
- [ ] Multi-modal embedding support (images, tables)
- [ ] SQLite vector store alternative (for single-node setups without PostgreSQL)

## Distributed

- [ ] Automatic model sharding based on available VRAM per node
- [ ] Fault tolerance and node recovery
- [ ] Load balancing across heterogeneous hardware
- [ ] Multi-node tensor parallelism (complement to pipeline parallelism)

## Developer Experience

- [ ] `cargo bench` coverage for all quantization formats
- [ ] Integration test suite that runs without external dependencies (embedded test model)
- [ ] CI matrix covering all feature flag combinations
- [ ] Flamegraph-based profiling guide
