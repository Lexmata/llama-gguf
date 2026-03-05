# Profiling Guide

Performance profiling for llama-gguf using flamegraphs and system profilers.

## Prerequisites

### Linux (recommended)

```bash
# Install perf (Linux kernel profiler)
sudo apt install linux-tools-common linux-tools-$(uname -r)

# Install cargo-flamegraph
cargo install flamegraph

# Allow perf for current user (alternative to running as root)
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### macOS

```bash
cargo install flamegraph
# Uses dtrace instead of perf — no additional setup needed
```

## Generating Flamegraphs

### Profile Inference

Run a short inference session and capture a flamegraph:

```bash
# Profile a model run (adjust model path and token count)
cargo flamegraph --release --bin llama-gguf -- \
  run model.gguf -p "Hello" -n 50

# Output: flamegraph.svg (open in browser)
```

### Profile Benchmarks

Profile specific benchmark groups to find hot paths in quantization, SIMD, or matrix ops:

```bash
# Profile all benchmarks
cargo flamegraph --release --bench quantization

# Profile a specific benchmark group
cargo flamegraph --release --bench quantization -- --bench "dequant_kquant"
```

### Profile with Custom Arguments

```bash
# Higher sampling frequency for short operations
cargo flamegraph --release --freq 5000 --bin llama-gguf -- \
  run model.gguf -p "Test" -n 20

# Include kernel frames (Linux only, requires root)
sudo cargo flamegraph --release --root --bin llama-gguf -- \
  run model.gguf -p "Test" -n 20
```

## Using perf Directly

For more control, use `perf` directly:

```bash
# Record with call graph (DWARF unwinding for Rust)
perf record -g --call-graph dwarf \
  cargo run --release -- run model.gguf -p "Hello" -n 50

# Generate report
perf report --hierarchy

# Export for external tools
perf script > perf.data.txt
```

### Stat Counters

Quick overview of CPU utilization without flamegraphs:

```bash
perf stat cargo run --release -- run model.gguf -p "Hello" -n 50
```

This shows IPC (instructions per cycle), cache misses, branch mispredictions — useful for spotting memory-bound vs compute-bound bottlenecks.

## Interpreting Flamegraphs

### What to Look For

| Pattern | Meaning | Action |
|---------|---------|--------|
| Wide band in `dequantize_*` | Dequant is the bottleneck | Optimize SIMD path or use a less aggressive quant format |
| Wide `matvec` or `matmul` | Matrix ops dominate (expected) | Check SIMD dispatch, verify AVX2/AVX-512 is active |
| `memcpy` / `alloc` bands | Excessive allocation on hot path | Use pre-allocated scratch buffers |
| `softmax` wide band | Softmax over large vocab | Check if SIMD softmax path is used |
| Deep call stacks in `encode` | Tokenizer overhead | Profile with longer prompts to amortize |

### Key Functions to Watch

- `backend::cpu::simd::*` — SIMD-accelerated operations (should be widest)
- `tensor::quant::dequant::*` — Dequantization hot loops
- `backend::cpu::ops::*` — Fallback non-SIMD paths (should be narrow if SIMD is active)
- `model::forward` / `model::forward_batch` — Per-layer inference
- `tokenizer::encode_bpe` / `tokenizer::encode_unigram` — Tokenization

### Verifying SIMD Dispatch

If you see time in non-SIMD fallback functions, check that runtime feature detection works:

```bash
# Print detected CPU features at startup
RUST_LOG=debug cargo run --release -- info model.gguf 2>&1 | grep -i simd
```

The CPU backend should detect and use AVX2/AVX-512 (x86) or NEON (ARM) automatically.

## Profiling Specific Components

### Quantization Performance

```bash
# Benchmark all quantization formats
cargo bench --bench quantization

# Compare specific formats
cargo bench --bench quantization -- "dequant_legacy"
cargo bench --bench quantization -- "dequant_kquant"
cargo bench --bench quantization -- "dequant_iq"
```

### Memory Usage

```bash
# Track peak memory with /usr/bin/time
/usr/bin/time -v cargo run --release -- run model.gguf -p "Hello" -n 50

# Use heaptrack for allocation profiling (Linux)
heaptrack cargo run --release -- run model.gguf -p "Hello" -n 50
heaptrack_print heaptrack.*.zst | head -20
```

### Cache Performance

```bash
# L1/L2/L3 cache miss analysis
perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses \
  cargo run --release -- run model.gguf -p "Hello" -n 50
```

## Build Configuration

The release profile is already configured for profiling in `Cargo.toml`:

```toml
[profile.release]
lto = true
codegen-units = 1
```

For profiling with better debug symbols (slightly slower builds):

```bash
# Build with debug info for release (better stack traces in flamegraphs)
CARGO_PROFILE_RELEASE_DEBUG=2 cargo flamegraph --release --bin llama-gguf -- \
  run model.gguf -p "Hello" -n 50
```

## Continuous Benchmarking

Track performance across commits using criterion's baseline feature:

```bash
# Save baseline
cargo bench --bench quantization -- --save-baseline main

# Switch to feature branch, compare
cargo bench --bench quantization -- --baseline main

# HTML reports are generated in target/criterion/
open target/criterion/report/index.html
```
