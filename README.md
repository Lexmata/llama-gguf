# llama-rs

A high-performance Rust implementation of [llama.cpp](https://github.com/ggerganov/llama.cpp) - an LLM inference engine with full GGUF support.

## Features

- **Full GGUF Support** - Load any GGUF model file compatible with llama.cpp
- **Multiple Architectures** - LLaMA, Mistral, Qwen2, TinyLlama, DeepSeek, and more
- **Quantization** - All K-quant formats (Q2_K through Q8_0) plus F16/F32
- **HuggingFace Integration** - Download models directly from HuggingFace Hub
- **Fast CPU Inference** - SIMD-optimized (AVX2, AVX-512, NEON)
- **CUDA GPU Acceleration** - NVIDIA GPU support with custom CUDA kernels
- **Grouped Query Attention** - Efficient KV cache for GQA models
- **Streaming Output** - Token-by-token generation

## Installation

### From Source

```bash
git clone https://github.com/pegasusheavy/llama-rs.git
cd llama-rs
cargo build --release
```

The binary will be at `target/release/llama-rs`.

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
llama-rs = { git = "https://github.com/pegasusheavy/llama-rs.git" }
```

## Quick Start

### Download a Model

```bash
# List available files in a repository
llama-rs download Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Download a specific quantized model
llama-rs download Qwen/Qwen2.5-0.5B-Instruct-GGUF -f qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Run Inference

```bash
# Basic text generation
llama-rs run model.gguf -p "Hello, world!" -n 50

# With sampling parameters
llama-rs run model.gguf -p "Once upon a time" -n 100 --temperature 0.8 --top-k 40

# Deterministic output (greedy sampling)
llama-rs run model.gguf -p "1+1=" -n 5 --temperature 0
```

### Model Information

```bash
llama-rs info model.gguf
```

## Supported Models

| Model Family | Status | Notes |
|--------------|--------|-------|
| LLaMA/LLaMA2/LLaMA3 | ✅ | Full support |
| Mistral | ✅ | Use `[INST]...[/INST]` format |
| Qwen2/Qwen2.5 | ✅ | Includes attention biases |
| TinyLlama | ✅ | GQA support |
| DeepSeek-Coder | ✅ | Linear RoPE scaling |
| CodeLlama | ✅ | LLaMA-based |
| Yi | ✅ | LLaMA-based |

See [MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md) for detailed compatibility information.

## Quantization Formats

| Format | Bits | Quality | Size (7B) |
|--------|------|---------|-----------|
| Q2_K | 2 | Low | ~2.5 GB |
| Q3_K | 3 | Fair | ~3.0 GB |
| Q4_K_M | 4 | Good | ~4.0 GB |
| Q5_K_M | 5 | Better | ~5.0 GB |
| Q6_K | 6 | High | ~5.5 GB |
| Q8_0 | 8 | Excellent | ~7.0 GB |
| F16 | 16 | Full | ~14 GB |

## Library Usage

```rust
use llama_rs::{
    backend::cpu::CpuBackend,
    gguf::GgufFile,
    model::{load_llama_model, InferenceContext},
    sampling::Sampler,
    tokenizer::Tokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model = load_llama_model("model.gguf")?;
    let gguf = GgufFile::open("model.gguf")?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    
    // Setup inference
    let backend = CpuBackend::new();
    let mut ctx = InferenceContext::new(model.config(), Box::new(backend));
    let sampler = Sampler::new(0.8, 40, 0.9); // temperature, top_k, top_p
    
    // Encode prompt
    let tokens = tokenizer.encode("Hello, world!", true)?;
    
    // Generate
    let mut output_tokens = tokens.clone();
    for _ in 0..50 {
        let logits = model.forward(&output_tokens[output_tokens.len()-1..], &mut ctx)?;
        let next_token = sampler.sample(&logits, &output_tokens);
        output_tokens.push(next_token);
        
        // Decode and print
        if let Ok(text) = tokenizer.decode(&[next_token]) {
            print!("{}", text);
        }
    }
    
    Ok(())
}
```

## CLI Reference

```
llama-rs <COMMAND>

Commands:
  run       Run inference on a model
  info      Display model information
  download  Download a model from HuggingFace Hub
  models    Manage cached models
  help      Print help

Run Options:
  -p, --prompt <PROMPT>      Input prompt
  -n, --max-tokens <N>       Maximum tokens to generate [default: 128]
  -t, --temperature <T>      Sampling temperature [default: 0.8]
  -k, --top-k <K>            Top-k sampling [default: 40]
      --top-p <P>            Top-p (nucleus) sampling [default: 0.9]
      --repeat-penalty <R>   Repetition penalty [default: 1.1]
  -s, --seed <SEED>          Random seed for reproducibility
      --gpu                  Use GPU acceleration (requires CUDA build)
```

## Building with Features

```bash
# CPU only (default)
cargo build --release

# With CUDA GPU support (requires CUDA toolkit)
CUDA_PATH=/opt/cuda cargo build --release --features cuda

# With Vulkan support (experimental)
cargo build --release --features vulkan

# With HTTP server
cargo build --release --features server
```

## GPU Acceleration

### CUDA (NVIDIA GPUs)

Enable GPU acceleration with the `--gpu` flag:

```bash
# Build with CUDA support
CUDA_PATH=/opt/cuda cargo build --release --features cuda

# Run with GPU acceleration
llama-rs run model.gguf -p "Hello" --gpu
```

**Requirements:**
- NVIDIA GPU with compute capability 6.0+
- CUDA Toolkit 12.0+ installed
- cudarc crate for CUDA bindings

**Currently GPU-accelerated operations:**
- Element-wise: add, mul, scale
- Activations: SiLU, GELU
- Normalization: RMS norm
- Softmax
- Vector-matrix multiplication (f32)

**Still using CPU fallback:**
- Quantized matrix operations (vec_mat_q)
- Attention computation
- RoPE positional embeddings

*Note: Performance gains are currently limited as quantized operations remain on CPU. Full GPU acceleration of quantized inference is planned.*

## Performance

Benchmarked on Intel i9-13900K (24 cores, AVX2) with 64GB RAM:

| Model | Quantization | Tokens/sec | Notes |
|-------|--------------|------------|-------|
| Qwen2.5-0.5B | Q4_K_M | ~1.2 t/s | 896 hidden dim |
| TinyLlama-1.1B | Q4_K_M | ~1.5 t/s | 2048 hidden dim |
| Mistral-7B | Q4_K_M | ~0.3 t/s | 4096 hidden dim |

*Current implementation prioritizes correctness over speed. Performance optimizations (batch processing, better SIMD utilization) are planned.*

Performance varies by hardware, model size, context length, and quantization.

## Contributing

Contributions are welcome! Please see [AGENTS.md](AGENTS.md) for development guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The original implementation
- [GGML](https://github.com/ggerganov/ggml) - Tensor library and GGUF format

---

**Pegasus Heavy Industries LLC** - [pegasusheavyindustries@gmail.com](mailto:pegasusheavyindustries@gmail.com)
