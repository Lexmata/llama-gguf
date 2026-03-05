# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2026-03-05

### Added

- IQ quantization formats: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, TQ1_0, TQ2_0
- GGUF quantization writing — CLI `quantize` subcommand and public `GgufWriter::quantize_tensor` API
- GPU-side quantization kernels for Q8_0, Q4_K, Q6_K on CUDA, Vulkan, Metal, and DX12
- RAG incremental re-indexing with content hashing and change detection
- RAG chunking strategies: recursive character, Markdown, sliding window, sentence, paragraph splitters
- RAG multi-modal embedding support with content type detection (text, image, table, code)
- SQLite vector store alternative with built-in KNN and pure-Rust HNSW index for single-node setups
- Distributed automatic model sharding based on available VRAM per node
- Distributed fault tolerance with health checks, failure detection, and node recovery
- Distributed load balancing across heterogeneous hardware with latency-based rebalancing
- Multi-node tensor parallelism with AllReduce collective communication
- HuggingFace `tokenizer.json` compatibility — BPE, Unigram (Viterbi), and WordPiece model types
- Tokenizer pipeline components: normalizers (NFC, NFKC, Lowercase, Strip, etc.), pre-tokenizers (Whitespace, ByteLevel, Metaspace, etc.), post-processors (TemplateProcessing, BertProcessing)
- `cargo bench` coverage for all quantization formats (legacy, K-quant, IQ)
- Integration test suite with embedded GGUF test model (no external dependencies)
- CI matrix covering all feature flag combinations
- Flamegraph-based profiling guide (`docs/PROFILING.md`)
- SIMD AXPY primitive (`axpy_f32`) with AVX2, AVX-512, and NEON backends
- SIMD fused SiLU*multiply kernel (`silu_mul_inplace`) for FFN hot path

### Changed

- KV cache `attention_cached` rewritten with SIMD dot products, SIMD softmax, SIMD AXPY, and rayon parallel heads
- KV cache `shift_left` uses `copy_within` (single `memmove` per head) instead of element-wise copy
- KV cache `reset` no longer zeroes cache data — only resets the position counter
- `FeedForward::forward` fuses SiLU and multiply into a single pass, eliminating 2 intermediate tensors per layer
- `Linear::forward` bias addition is now in-place, eliminating temp tensor allocation and copy
- `TransformerLayer::forward` residual connections are in-place, eliminating 3 tensor allocations per layer
- CI workflow uses environment variables for matrix inputs to address command injection warnings

## [0.11.0] - 2026-03-04

### Added

- GPU `matmul` shaders (tiled shared-memory) for Metal, DX12, and Vulkan backends
- GPU `matvec` shaders (row-parallel) for Metal, DX12, and Vulkan backends
- GPU dequantization shaders for Q8_0, Q4_K, Q6_K formats across Metal, DX12, and Vulkan
- GPU `vec_mat_q` and `matvec_q` (2-pass: dequant + matop) for Metal, DX12, and Vulkan
- GPU `attention` and `attention_cached` shaders with online softmax, GQA, and causal masking for Metal, DX12, and Vulkan
- CUDA fully on-GPU attention pipeline (QK norm, partial RoPE, attention gating kernels)
- CUDA batched inference kernels (batched RoPE, KV cache update, linear projections)
- CUDA Flash Attention kernel with O(head_dim) shared memory and online softmax
- CUDA FP16 compute path (f32↔f16 conversion, vec_mat_f16, element-wise ops, rms_norm_f16)

### Changed

- Metal, DX12, Vulkan: all operations now execute on GPU — no CPU fallbacks remain
- GPU softmax dispatches all tensor sizes (removed small-tensor CPU threshold)
- GPU RoPE supports all configurations (relaxed shape constraints)
- CUDA `attention_cached` replaced with Flash Attention for O(n) memory usage

## [0.10.0] - 2026-03-04

### Added

- Full GPU-resident inference engine (`gpu_only.rs`) with 100% GPU execution for all layer types
- New CUDA kernels for DeltaNet, fused RMS norm, and dequantization
- CPU backend SIMD improvements for attention and matrix operations

### Changed

- MoE routing and expert dispatch improvements
- Model loader updates for Qwen3Next compatibility

### Removed

- Legacy CUDA inference engines (`fast_inference.rs`, `gpu_inference.rs`, `gpu_model.rs`, `gpu_ops.rs`)
- Legacy GPU inference examples (`fast_gpu.rs`, `gpu_inference.rs`)

## [0.9.0] - 2026-03-03

### Added

- Qwen3-Coder-Next architecture support (QK norm, partial RoPE, attention gating)
- SSM/DeltaNet recurrent layer implementation
- Qwen3 Mixture-of-Experts (MoE) support
- CUDA hybrid model support for mixed attention/recurrent architectures

### Fixed

- GPT-2 tokenizer byte-level decode for special tokens

## [0.8.0] - 2026-02-17

### Added

- Distributed pipeline-parallel inference via gRPC
- Quantized GPU weight storage for reduced VRAM usage
- `max_context_len` cap for CUDA inference
- `ChatEngine::chat_with_prefix` for assistant response prefilling

### Changed

- GPU-resident inference with `GpuModelWrapper` for reduced host-device transfers

### Fixed

- CUDA fused RMS norm kernel to process full hidden vector
- CUDA GQA-safe RoPE kernel with num_kv_heads guard
- ChatEngine prefill: batch all prompt tokens in single forward pass

## [0.6.1] - 2026-02-14

### Fixed

- CUDA backend `select_gpu_backend` variable naming

### Changed

- Updated README and AGENTS.md for v0.6.0

## [0.6.0] - 2026-02-13

### Added

- RAG support with PostgreSQL/pgvector vector store
- Hybrid search with tsvector and RRF (Reciprocal Rank Fusion)
- HNSW and IVFFlat index support
- Pipelined batch inserts with transaction batching
- Metadata filtering DSL for RAG searches
- Document reranking for retrieved chunks
- Real embedding generation via loaded LLM models
- Upsert and health check for `RagStore`
- TOML config file support for RAG
- RAG/Knowledge Base endpoints for HTTP server (AWS Bedrock-style API)
- Metal GPU backend for macOS / Apple Silicon
- DirectX 12 GPU backend for Windows
- GitHub Actions CI workflow for cross-platform testing

### Fixed

- DX12 GELU numerical stability (exp-based instead of tanh)
- DX12 RoPE readwrite buffer handling
- RAG search query bugs

## [0.5.2] - 2026-02-08

### Added

- Remote chat client and ARM64 server deployment
- TOML configuration file support for all CLI arguments
- High-level Engine API

### Changed

- Transferred project ownership to Lexmata LLC
- Gated `clap` behind `cli` feature flag

### Fixed

- Server chat template detection (use model-detected template instead of hardcoded Llama 2 format)
- CUDA GPU backend compilation errors and warnings

## [0.2.4] - 2026-02-03

### Fixed

- SentencePiece tokenization for SPM models

## [0.2.3] - 2026-02-03

### Fixed

- GPT-2 tokenizer decoding for newlines and tabs

## [0.2.2] - 2026-02-03

### Fixed

- Clippy lints and code cleanup

## [0.2.1] - 2026-02-03

### Fixed

- ARM64 implementations for `has_avx2`/`has_avx512` feature detection

## [0.2.0] - 2026-02-03

### Added

- GGUF file format reader with memory-mapped file support
- Core tensor type with storage, dtype, and quantization blocks (Q2_K through Q8_0)
- BPE tokenizer loaded from GGUF metadata
- LLaMA model architecture with KV cache inference
- CPU backend with SIMD optimizations (AVX2, AVX-512, NEON)
- CUDA backend infrastructure with RoPE, attention, and quantized kernels
- HuggingFace Hub integration for model downloads
- Qwen2 model support
- Support for tied embeddings (weight tying)
- RoPE type support (Normal and NeoX) for multi-model compatibility
- Benchmark suite with criterion

[0.12.0]: https://github.com/Lexmata/llama-gguf/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Lexmata/llama-gguf/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Lexmata/llama-gguf/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Lexmata/llama-gguf/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Lexmata/llama-gguf/compare/v0.6.1...v0.8.0
[0.6.1]: https://github.com/Lexmata/llama-gguf/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Lexmata/llama-gguf/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/Lexmata/llama-gguf/compare/v0.2.4...v0.5.2
[0.2.4]: https://github.com/Lexmata/llama-gguf/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Lexmata/llama-gguf/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Lexmata/llama-gguf/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Lexmata/llama-gguf/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Lexmata/llama-gguf/releases/tag/v0.2.0
