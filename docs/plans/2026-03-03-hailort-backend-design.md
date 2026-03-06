# HailoRT Backend Design

## Overview

Add a Hailo AI accelerator backend to llama-gguf, using the `hailort-sys` crate for
FFI bindings to libhailort. This backend targets all Hailo device families (Hailo-8,
Hailo-8L, Hailo-10H) via the device-agnostic HailoRT runtime.

Unlike the GPU backends (CUDA, Vulkan, Metal, DX12) which run arbitrary compute
kernels, Hailo is a fixed-function inference accelerator that executes pre-compiled
HEF (Hailo Executable Format) subgraphs. The backend uses a **hybrid CPU + Hailo**
architecture with **two HEF subgraphs per transformer layer**.

## Architecture

### Module Structure

```
src/backend/hailo/
├── mod.rs          # Exports, HailoBackend stub (Backend trait), feature gates
├── context.rs      # Device lifecycle: vdevice, HEF loading, vstream setup/teardown
├── gpu_only.rs     # HailoGpuInference implementing GpuInference trait
├── compiler.rs     # Auto-compile: ONNX export, DFC invocation, HEF caching
└── config.rs       # HailoConfig, HEF path resolution, compilation settings
```

### Feature Flag

```toml
[features]
hailo = ["dep:hailort-sys"]
```

Auto-compile is a runtime config option, not a separate feature flag. It shells out
to the Hailo DFC Python toolchain at runtime.

### Execution Model

The `Backend` trait implementation is a stub returning `NotAvailable` for all
individual tensor operations. The real work uses the `GpuInference` trait.

Each `forward(token_id)` call follows this pipeline:

```
CPU: embedding lookup (token_id → [1, hidden_size])

For each transformer layer:
  ├─ Write hidden state to Hailo input buffer
  ├─ HAILO HEF 1 (pre-attention):
  │   attn_norm → Q/K/V linear projections
  │   In:  [1, hidden_size]
  │   Out: Q [1, n_heads*head_dim], K [1, n_kv_heads*head_dim], V [same]
  ├─ CPU: RoPE on Q, K
  ├─ CPU: KV cache update
  ├─ CPU: attention scoring (Q @ K^T * scale → softmax → @ V)
  ├─ CPU: O projection (linear)
  ├─ CPU: residual = input + attn_output
  ├─ Write residual to Hailo input buffer
  ├─ HAILO HEF 2 (FFN):
  │   ffn_norm → gate_proj → SiLU → up_proj → mul → down_proj
  │   In:  [1, hidden_size]
  │   Out: [1, hidden_size]
  └─ CPU: residual = residual + ffn_output

CPU: final_norm → output_projection → logits → return
```

Design rationale:
- **O projection on CPU**: Takes CPU-computed attention output; round-tripping
  to Hailo for one matmul isn't worth the transfer cost.
- **Final logits on CPU**: Single matmul, result needed on CPU for sampling.
- **KV cache on CPU**: Hailo has no stateful cache concept.
- **Reuses CPU SIMD**: RoPE, attention, and linear ops use existing
  `backend::cpu::simd` functions.

### HailoRT Context (`context.rs`)

```rust
pub struct HailoContext {
    vdevice: hailo_vdevice,
    network_groups: Vec<NetworkGroupHandle>,
    input_vstreams: HashMap<String, VstreamHandle>,
    output_vstreams: HashMap<String, VstreamHandle>,
    device_info: HailoDeviceInfo,
}
```

Lifecycle: `hailo_create_vdevice` → load HEFs → configure → activate network
groups → create vstreams → inference → Drop releases in reverse order.

All HailoRT status codes checked via `check_status(status, context) → BackendResult`.

### HEF Files

Naming convention:
```
<cache_dir>/<model_name>/
├── layer_00_attn.hef
├── layer_00_ffn.hef
├── layer_01_attn.hef
├── ...
└── manifest.json       # Model config hash, tensor shapes, target arch
```

`manifest.json` records a hash of the model config (hidden_size, num_heads,
head_dim, intermediate_size, etc.) to detect stale HEFs.

### Auto-Compile (`compiler.rs`)

When `auto_compile` is enabled:
1. Check cache for HEFs matching the model config hash
2. On cache miss: extract layer weights, export each subgraph as minimal ONNX
3. Shell out to `python -m hailo_sdk_client` (Hailo DFC) per subgraph
4. Store compiled HEFs + manifest in cache dir
5. Log progress via `tracing`

If `auto_compile` is false and HEFs are missing, return a descriptive error.

### Configuration (`config.rs`)

```rust
pub struct HailoConfig {
    pub hef_dir: Option<PathBuf>,
    pub auto_compile: bool,
    pub dfc_python: Option<PathBuf>,
    pub cache_dir: PathBuf,             // Default: ~/.cache/llama-gguf/hailo/
    pub target_arch: Option<String>,    // Auto-detected if None
    pub quantization: HailoQuantization, // INT8 (default) or INT4
}
```

### GpuInference Implementation (`gpu_only.rs`)

```rust
pub struct HailoGpuInference {
    ctx: HailoContext,
    config: InferenceConfig,
    layer_attn_hefs: Vec<LayerHefHandle>,
    layer_ffn_hefs: Vec<LayerHefHandle>,

    // CPU-resident weights
    embeddings: Vec<f32>,
    output_weight: Tensor,
    output_norm_weight: Vec<f32>,
    o_proj_weights: Vec<Tensor>,

    // KV cache (CPU)
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    position: usize,

    // Scratch buffers
    hidden: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
}
```

Implements `GpuInference` trait: `forward`, `prefill_token`, `reset`, `position`.
Wrapped by the generic `GpuModelWrapper<HailoGpuInference>` in the engine.

### Engine Integration

Hailo added to `select_gpu_model()` priority chain:
```
CUDA > Vulkan > Metal > DX12 > Hailo > CPU
```

New engine config field: `hailo_config: Option<HailoConfig>`.
New CLI flag: `--hailo` to directly select the Hailo backend.

### Error Handling

```rust
pub enum HailoError {
    DeviceNotFound,
    HefLoadFailed(String),
    HefNotFound { layer: usize, kind: &'static str },
    CompilationFailed(String),
    StreamError(String),
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    RuntimeError(hailo_status, String),
}
```

Integrates via `From<HailoError> for BackendError`.

### Testing

- Unit tests: config parsing, HEF manifest, ONNX subgraph export
- Integration tests: require Hailo device, gated behind `#[cfg(feature = "hailo")]`
- CLI subcommand: `llama-gguf hailo info` for device connectivity check
