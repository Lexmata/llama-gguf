# llama-gguf

[![CI](https://github.com/Lexmata/llama-gguf/actions/workflows/ci.yml/badge.svg)](https://github.com/Lexmata/llama-gguf/actions/workflows/ci.yml)

A high-performance Rust implementation of [llama.cpp](https://github.com/ggerganov/llama.cpp) - an LLM inference engine with full GGUF and ONNX support.

## Features

- **Full GGUF Support** - Load any GGUF model file compatible with llama.cpp
- **ONNX Support** - Load HuggingFace Optimum ONNX exports (F32, F16, BF16 with auto-conversion)
- **Multiple Architectures** - LLaMA, Mistral, Qwen2, TinyLlama, DeepSeek, and more
- **Quantization** - All K-quant formats (Q2_K through Q8_0) plus F16/F32
- **HuggingFace Integration** - Download models directly from HuggingFace Hub
- **Fast CPU Inference** - SIMD-optimized (AVX2, AVX-512, NEON)
- **Multi-GPU Backend Support** - CUDA, Metal, Vulkan, and DirectX 12
- **Grouped Query Attention** - Efficient KV cache for GQA models
- **Streaming Output** - Token-by-token generation

## Installation

### From Source

```bash
git clone https://github.com/Lexmata/llama-gguf.git
cd llama-gguf
cargo build --release
```

The binary will be at `target/release/llama-gguf`.

### System Installation with Man Pages

**Option 1: Using cargo install (generates man pages from CLI)**

```bash
cargo install llama-gguf

# Generate and install man pages
llama-gguf manpages ~/.local/share/man/man1
mandb -u

# Or system-wide (requires sudo)
sudo llama-gguf manpages /usr/local/share/man/man1
sudo mandb
```

**Option 2: Using make (includes detailed hand-written man pages)**

```bash
git clone https://github.com/Lexmata/llama-gguf.git
cd llama-gguf

# Build and install to /usr/local (requires sudo)
sudo make install

# Or install to a custom prefix
make PREFIX=~/.local install

# Install man pages only
sudo make install-man
```

After installation, access documentation with:

```bash
man llama-gguf           # Main command overview
man llama-gguf-run       # Run inference
man llama-gguf-chat      # Interactive chat
man llama-gguf-serve     # HTTP server
man llama-gguf-rag       # RAG operations
```

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
llama-gguf = { git = "https://github.com/Lexmata/llama-gguf.git" }
```

## Quick Start

### Download a Model

```bash
# List available files in a repository
llama-gguf download Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Download a specific quantized model
llama-gguf download Qwen/Qwen2.5-0.5B-Instruct-GGUF -f qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Run Inference

```bash
# Basic text generation (GGUF)
llama-gguf run model.gguf -p "Hello, world!" -n 50

# ONNX model (requires config.json and tokenizer.json in the same directory)
llama-gguf run model.onnx -p "Hello, world!" -n 50

# With sampling parameters
llama-gguf run model.gguf -p "Once upon a time" -n 100 --temperature 0.8 --top-k 40

# Deterministic output (greedy sampling)
llama-gguf run model.gguf -p "1+1=" -n 5 --temperature 0
```

### Model Information

```bash
llama-gguf info model.gguf
llama-gguf info model.onnx
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
use llama_gguf::{
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
llama-gguf <COMMAND>

Commands:
  info         Display model information
  run          Run inference on a model
  chat         Interactive chat mode
  serve        Start HTTP server (with --features server)
  quantize     Quantize a model
  bench        Benchmark model performance
  embed        Extract embeddings
  download     Download a model from HuggingFace Hub
  models       Manage cached models
  rag          RAG operations (with --features rag)
  init-config  Generate example config file
  manpages     Generate and install man pages
  help         Print help

Run Options:
  -p, --prompt <PROMPT>      Input prompt
  -n, --max-tokens <N>       Maximum tokens to generate [default: 128]
  -t, --temperature <T>      Sampling temperature [default: 0.8]
  -k, --top-k <K>            Top-k sampling [default: 40]
      --top-p <P>            Top-p (nucleus) sampling [default: 0.9]
      --repeat-penalty <R>   Repetition penalty [default: 1.1]
  -s, --seed <SEED>          Random seed for reproducibility
      --gpu                  Use GPU acceleration (CUDA, Metal, DX12, or Vulkan)
```

## ONNX Support

llama-gguf can load models exported to ONNX format via [HuggingFace Optimum](https://huggingface.co/docs/optimum/). ONNX support is enabled by default.

**Supported formats:**
- F32, F16, and BF16 weight tensors (F16/BF16 auto-converted to F32)
- External data files (`.onnx_data`) for large models
- Graph-traced tensor name resolution for Optimum exports

**Requirements:**

An ONNX model directory must contain:
- `model.onnx` — the model graph and weights
- `config.json` — HuggingFace model configuration
- `tokenizer.json` — HuggingFace tokenizer

**Exporting a model to ONNX:**

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./tinyllama-onnx/
```

```bash
# Run the exported model
llama-gguf run ./tinyllama-onnx/model.onnx -p "Hello!" -n 50
```

## Building with Features

```bash
# CPU + ONNX (default)
cargo build --release

# Without ONNX support
cargo build --release --no-default-features --features cpu,huggingface,cli,client

# With CUDA GPU support (requires CUDA toolkit)
CUDA_PATH=/opt/cuda cargo build --release --features cuda

# With Metal support (macOS / Apple Silicon)
cargo build --release --features metal

# With Vulkan support (Linux/Windows, requires Vulkan SDK)
cargo build --release --features vulkan

# With DirectX 12 support (Windows)
cargo build --release --features dx12

# With HTTP server
cargo build --release --features server
```

## GPU Acceleration

Enable GPU acceleration with the `--gpu` flag. The engine automatically selects the best available backend in priority order: CUDA > Metal > DX12 > Vulkan > CPU.

```bash
llama-gguf run model.gguf -p "Hello" --gpu
```

### Supported Backends

| Backend | Platform | Requirements |
|---------|----------|-------------|
| **CUDA** | Linux, Windows | NVIDIA GPU (compute 6.0+), CUDA Toolkit 12.0+ |
| **Metal** | macOS | Apple Silicon or discrete AMD GPU, Xcode Command Line Tools |
| **DirectX 12** | Windows | Any DX12-capable GPU, Windows 10+, Windows SDK (dxc.exe) |
| **Vulkan** | Linux, Windows | Vulkan 1.2+ capable GPU, Vulkan SDK (glslc) |

### GPU-accelerated operations

- Element-wise: add, mul, scale
- Activations: SiLU, GELU
- Normalization: RMS norm
- Softmax
- Vector-matrix multiplication (f32)
- Rotary position embeddings (RoPE)

### CPU fallback

The following operations currently fall back to CPU across all GPU backends:

- Quantized matrix operations (vec_mat_q)
- Attention computation
- Dequantization

*Full GPU acceleration of quantized inference is planned.*

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

**Lexmata LLC** - [jquinn@lexmata.ai](mailto:jquinn@lexmata.ai)
