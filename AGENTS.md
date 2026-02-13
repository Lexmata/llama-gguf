# AGENTS.md

Guidelines for AI coding assistants working on llama-gguf.

## Project Overview

llama-gguf is a Rust implementation of llama.cpp - a high-performance LLM inference engine. The goal is full feature parity with llama.cpp while providing idiomatic Rust APIs suitable for ecosystem contribution.

## Architecture

Monolithic library with feature flags. Key modules:

| Module | Purpose |
|--------|---------|
| `gguf/` | GGUF file format parsing and writing |
| `onnx/` | ONNX model loading (HuggingFace Optimum exports) |
| `tensor/` | Tensor types, quantization (Q2_K through Q8_0), operations |
| `backend/` | Hardware backends (CPU, CUDA, Metal, DX12, Vulkan) |
| `model/` | Model architectures (LLaMA, Mistral, Qwen2, TinyLlama, DeepSeek) |
| `sampling/` | Token sampling strategies (greedy, top-k, top-p, temperature, grammar) |
| `tokenizer/` | BPE tokenizer loaded from GGUF metadata |
| `server/` | HTTP server with OpenAI-compatible API |
| `rag/` | RAG with PostgreSQL/pgvector vector store |
| `huggingface.rs` | HuggingFace Hub model downloading |
| `engine.rs` | High-level inference engine and chat templates |
| `config.rs` | Global configuration |

### Backend Architecture

Each backend implements the `Backend` trait from `backend/mod.rs`:

| Backend | Feature Flag | Platform | Status |
|---------|-------------|----------|--------|
| `cpu/` | `cpu` (default) | All | Production - SIMD optimized (AVX2, AVX-512, NEON) |
| `cuda/` | `cuda` | Linux/Windows | Production - NVIDIA GPUs with compute 6.0+ |
| `metal/` | `metal` | macOS | Production - Apple Silicon and AMD GPUs |
| `dx12/` | `dx12` | Windows | Production - DirectX 12 compatible GPUs |
| `vulkan/` | `vulkan` | All | Experimental - Vulkan SDK required |

### RAG Architecture

The `rag/` module provides retrieval-augmented generation backed by PostgreSQL with pgvector:

| File | Purpose |
|------|---------|
| `rag/config.rs` | `RagConfig`, `DatabaseConfig`, `EmbeddingsConfig`, `SearchConfig`, `IndexType`, `DistanceMetric`, `SearchType` |
| `rag/store.rs` | `RagStore` - connection pooling, CRUD, vector/keyword/hybrid search, `MetadataFilter` SQL generation |
| `rag/embedding.rs` | `EmbeddingGenerator` - generates embeddings using a loaded `LlamaModel` |
| `rag/knowledge_base.rs` | `KnowledgeBase` - high-level API for ingestion, chunking, and retrieve-and-generate |
| `rag/mod.rs` | Module exports and `RagError` / `RagResult` types |

Key types:
- `MetadataFilter` - Enum-based filter DSL (Eq, In, Range, Contains, AND/OR/NOT) compiled to parameterized SQL
- `RagContextBuilder` - Builds context strings from search results for prompt injection
- `TextChunker` - Splits documents into overlapping chunks

## Supported Models

Models verified to work correctly:

| Model | RoPE Type | Notes |
|-------|-----------|-------|
| Qwen2/Qwen2.5 | NeoX | Uses attention biases |
| TinyLlama | Normal | GQA with 4 KV heads |
| Mistral | Normal | Requires instruction format `[INST]...[/INST]` |
| DeepSeek-Coder | Normal | Uses `scale_linear=4.0` for RoPE |
| LLaMA/LLaMA2/LLaMA3 | Normal | Standard architecture |

**Not supported** (different architecture):
- Phi-2/GPT-NeoX (combined QKV tensors)
- Gemma2 (extra norm layers, logit softcapping)

See `docs/MODEL_COMPATIBILITY.md` for full details.

## Feature Flags

```toml
[features]
default = ["cpu", "huggingface", "cli", "client", "onnx"]
cpu = []
cuda = ["dep:cudarc"]
metal = ["dep:metal", "dep:objc"]
dx12 = ["dep:windows"]
vulkan = ["dep:ash", "dep:gpu-allocator"]
onnx = ["dep:prost"]
server = ["dep:axum", "dep:tokio", "dep:tower-http", "dep:futures"]
rag = ["dep:tokio-postgres", "dep:pgvector", "dep:deadpool-postgres", "dep:tokio", "dep:url", "dep:glob"]
huggingface = ["dep:reqwest", "dep:indicatif", "dep:directories"]
cli = ["dep:clap"]
client = ["dep:reqwest"]
```

All platform-specific backends (`metal`, `dx12`) must be gated with `#[cfg(feature = "...")]`.

## Code Style

- Follow standard Rust conventions (`cargo fmt`, `cargo clippy`)
- Use `thiserror` for error types
- Use `tracing` for logging, not `println!` or `eprintln!`
- Prefer `Result<T>` over panics
- Document public APIs with doc comments
- Use `#[repr(C)]` for types that must match llama.cpp memory layout
- Gate platform-specific code with `#[cfg(feature = "...")]` or `#[cfg(target_os = "...")]`

## Performance Guidelines

- Hot paths must avoid allocations - use scratch buffers
- Quantized operations work on blocks, not individual values
- SIMD code uses runtime feature detection (AVX2/AVX-512/NEON)
- Memory-map GGUF files, don't load entirely into RAM
- KV cache is pre-allocated, not grown dynamically
- RAG uses connection pooling via `deadpool-postgres` and pipelined batch inserts

## Testing

- Unit tests for quantization roundtrips
- Integration tests load real GGUF files (small test models)
- RAG integration tests require a PostgreSQL instance with pgvector (see `tests/rag_integration_test.rs`)
- Benchmark critical paths with `criterion` (v0.5+)
- Run all tests: `cargo test --release`
- Run RAG tests: `cargo test --release --features rag --test rag_integration_test` (requires PostgreSQL)
- Test with a model: `cargo run --release -- run <model.gguf> -p "test" -n 10`

### RAG Integration Test Setup

```bash
# Start PostgreSQL with pgvector
docker run -d --name pgvector-test -p 5434:5432 \
  -e POSTGRES_PASSWORD=testpass \
  -e POSTGRES_DB=llama_rag_test \
  pgvector/pgvector:pg16

# Enable extension
PGPASSWORD=testpass psql -h localhost -p 5434 -U postgres -d llama_rag_test \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run tests
TEST_DATABASE_URL="postgresql://postgres:testpass@localhost:5434/llama_rag_test" \
  cargo test --release --features rag --test rag_integration_test
```

## Common Tasks

### Adding a new model architecture

1. Add variant to `Architecture` enum in `model/architecture.rs`
2. Update `is_llama_like()` if architecture uses LLaMA tensor naming
3. Set correct `RopeType` in `model/loader.rs` (Normal vs NeoX)
4. Handle any architecture-specific config (biases, GQA, etc.)
5. Test with `cargo run --release -- info <model.gguf>` then inference

### RoPE Configuration

Two RoPE styles are supported:

- **Normal** (type 0): Consecutive pairs `(x[2i], x[2i+1])` - LLaMA, Mistral, TinyLlama
- **NeoX** (type 2): Split pairs `(x[i], x[i+d/2])` - Qwen2

Set in `model/loader.rs` based on architecture. Also handle:
- `freq_base`: Varies by model (10000, 100000, 1000000)
- `freq_scale`: Linear scaling factor (positions divided by this)

### Adding a new quantization format

1. Add variant to `DType` enum in `tensor/dtype.rs`
2. Define block struct in `tensor/quant/blocks.rs` with `#[repr(C)]`
3. Implement `dequantize_*` function in `tensor/quant/dequant.rs`
4. Add to match statements in `tensor/quant/mod.rs`
5. Add SIMD-optimized version in `backend/cpu/simd.rs`
6. Add roundtrip tests

### Adding a new backend

1. Create `backend/<name>/` directory
2. Implement `Backend` trait from `backend/mod.rs`
3. Add feature flag in `Cargo.toml`
4. Gate all backend code with `#[cfg(feature = "...")]`
5. Add GPU-accelerated ops: element-wise, activations, normalization, RoPE, softmax
6. Register in `backend/mod.rs` module list and `default_backend()` selection

### Working with the RAG module

**Adding a new search strategy:**
1. Add variant to `SearchType` enum in `rag/config.rs`
2. Implement search logic in `RagStore` in `rag/store.rs`
3. Wire into `KnowledgeBase::retrieve()` in `rag/knowledge_base.rs`
4. Add integration tests in `tests/rag_integration_test.rs`

**Adding a new metadata filter type:**
1. Add variant to `MetadataFilter` enum in `rag/store.rs`
2. Implement `to_sql_inner()` for the new variant - generates parameterized SQL
3. Ensure correct parameter numbering: use `param_offset + params.len()` for the new parameter index
4. Add tests for the filter in isolation and combined with AND/OR/NOT

**Important RAG implementation notes:**
- pgvector distance operators return `DOUBLE PRECISION` (f64) - cast to `f32` when storing as `Document.score`
- `min_similarity` must be cast to `f64` before binding as a SQL parameter to match operator return types
- `MetadataFilter::to_sql_inner` compound variants (AND/OR/NOT) pass `param_offset` unchanged to children - children compute their own index via `param_offset + params.len()`
- Cosine similarity ranges from -1.0 to 1.0 (not 0.0 to 1.0)

### Adding ONNX model support

1. ONNX files are parsed in `onnx/reader.rs` using protobuf (via `prost`)
2. Model config loaded from companion `config.json` in `onnx/config.rs`
3. Tensor name resolution maps ONNX graph names to llama.cpp names in `onnx/loader.rs`
4. F16/BF16 weights are auto-converted to F32 during loading

## Debugging Models

When a model produces incorrect output:

1. Check architecture: `cargo run --release -- info <model.gguf>`
2. Verify RoPE type matches llama.cpp (use `llama-cpp-python` verbose mode)
3. Compare hidden states layer-by-layer using `examples/trace_all_layers.rs`
4. Check tensor shapes match expected dimensions
5. Verify dequantization produces reasonable values

## Reference Materials

- llama.cpp source: https://github.com/ggerganov/llama.cpp
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- pgvector: https://github.com/pgvector/pgvector
- Model compatibility: `docs/MODEL_COMPATIBILITY.md`
- RAG design: `docs/plans/2026-02-13-pgvector-rag-design.md`

## Do Not

- Break GGUF compatibility with llama.cpp
- Add external tokenizer dependencies (load from GGUF only)
- Use unsafe without clear justification and safety comments
- Commit large model files (*.gguf, *.bin) to the repo
- Skip clippy warnings without `#[allow(...)]` with explanation
- Use `println!`/`eprintln!` for debug output in library code
- Use `f32` for SQL parameters compared against pgvector distance operator results (use `f64`)
- Hardcode database connection strings - always use `RagConfig`
