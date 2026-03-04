# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
