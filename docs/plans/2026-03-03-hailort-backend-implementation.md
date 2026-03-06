# HailoRT Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Hailo AI accelerator backend using the hybrid CPU + Hailo architecture with two HEF subgraphs per transformer layer.

**Architecture:** Implements `GpuInference` trait for full forward-pass orchestration. Each layer uses two pre-compiled HEF files (pre-attention QKV projections and FFN block) executed via HailoRT vstreams. CPU handles embedding, RoPE, KV cache, attention scoring, O projection, and sampling. Auto-compile mode exports ONNX subgraphs and shells out to Hailo DFC.

**Tech Stack:** `hailort-sys` (FFI bindings), HailoRT C runtime, ONNX export via `prost`, existing CPU SIMD infrastructure.

**Design doc:** `docs/plans/2026-03-03-hailort-backend-design.md`

---

### Task 1: Add Feature Flag and Dependency

**Files:**
- Modify: `Cargo.toml` (features section ~line 22, dependencies ~line 57)
- Modify: `src/backend/mod.rs` (module declarations ~line 1-14)

**Step 1: Add hailort-sys dependency and feature flag**

In `Cargo.toml`, add to `[features]`:
```toml
hailo = ["dep:hailort-sys"]
```

In `[dependencies]`, add:
```toml
hailort-sys = { version = "0.1", optional = true }
```

**Step 2: Register hailo module in backend/mod.rs**

After the existing module declarations (after `pub mod vulkan;`), add:
```rust
#[cfg(feature = "hailo")]
pub mod hailo;
```

**Step 3: Verify it compiles**

Run: `cargo check --no-default-features`
Expected: PASS (hailo module doesn't exist yet but is feature-gated)

**Step 4: Commit**

```bash
git add Cargo.toml src/backend/mod.rs
git commit -m "Add hailo feature flag and hailort-sys dependency"
```

---

### Task 2: Create Config Types

**Files:**
- Create: `src/backend/hailo/config.rs`

**Step 1: Write HailoConfig and related types**

```rust
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum HailoQuantization {
    Int8,
    Int4,
}

impl Default for HailoQuantization {
    fn default() -> Self {
        Self::Int8
    }
}

#[derive(Debug, Clone)]
pub struct HailoConfig {
    pub hef_dir: Option<PathBuf>,
    pub auto_compile: bool,
    pub dfc_python: Option<PathBuf>,
    pub cache_dir: PathBuf,
    pub target_arch: Option<String>,
    pub quantization: HailoQuantization,
}

impl Default for HailoConfig {
    fn default() -> Self {
        let cache_dir = dirs_or_fallback();
        Self {
            hef_dir: None,
            auto_compile: false,
            dfc_python: None,
            cache_dir,
            target_arch: None,
            quantization: HailoQuantization::default(),
        }
    }
}

fn dirs_or_fallback() -> PathBuf {
    if let Some(cache) = directories::BaseDirs::new() {
        cache.cache_dir().join("llama-gguf").join("hailo")
    } else {
        PathBuf::from(".cache/llama-gguf/hailo")
    }
}

#[derive(Debug, Clone)]
pub struct HefManifest {
    pub model_config_hash: u64,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub target_arch: String,
}
```

Note: `directories` crate is already a dependency (used by `huggingface` feature).

**Step 2: Verify compilation**

Run: `cargo check --features hailo` (will fail because mod.rs doesn't exist yet — that's fine, we create it in Task 3)

**Step 3: Commit**

```bash
git add src/backend/hailo/config.rs
git commit -m "Add Hailo backend config types"
```

---

### Task 3: Create Module Root with Backend Trait Stub

**Files:**
- Create: `src/backend/hailo/mod.rs`

**Step 1: Write module root following DX12 pattern**

Model after `src/backend/dx12/mod.rs`. The active implementation is gated behind
`#[cfg(feature = "hailo")]`. The stub when hailo is disabled returns `NotAvailable`.

```rust
//! Hailo AI accelerator backend for LLM inference.
//!
//! Uses HailoRT to offload transformer layer computations (QKV projections,
//! FFN blocks) to Hailo NPUs while CPU handles attention scoring, KV cache,
//! RoPE, and sampling.
//!
//! # Requirements
//! - Hailo device (Hailo-8, Hailo-8L, or Hailo-10H)
//! - HailoRT runtime installed (`libhailort.so`)
//! - Pre-compiled HEF files or Hailo DFC for auto-compilation
//! - Build with `--features hailo`

pub mod config;

#[cfg(feature = "hailo")]
pub(crate) mod context;
#[cfg(feature = "hailo")]
pub mod gpu_only;
#[cfg(feature = "hailo")]
pub mod compiler;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};
pub use config::HailoConfig;

/// Hailo AI accelerator backend.
///
/// The Backend trait implementation is a stub — Hailo cannot execute
/// individual tensor operations. Use `HailoGpuInference` via the
/// `GpuInference` trait for full model inference.
pub struct HailoBackend {
    _config: HailoConfig,
}

impl HailoBackend {
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(HailoConfig::default())
    }

    #[cfg(feature = "hailo")]
    pub fn with_config(config: HailoConfig) -> Result<Self, BackendError> {
        // Verify a Hailo device is present
        context::check_device_available()?;
        Ok(Self { _config: config })
    }

    #[cfg(not(feature = "hailo"))]
    pub fn with_config(_config: HailoConfig) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "Hailo support not available. Build with --features hailo".to_string(),
        ))
    }
}

impl Backend for HailoBackend {
    fn name(&self) -> &str { "hailo" }
    fn is_available(&self) -> bool { cfg!(feature = "hailo") }
    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn rms_norm(&self, _x: &Tensor, _weight: &Tensor, _eps: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn rope(&self, _q: &mut Tensor, _k: &mut Tensor, _pos: usize, _freq_base: f32, _freq_scale: f32, _use_neox: bool) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
    fn attention(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor, _out: &mut Tensor, _scale: f32) -> BackendResult<()> {
        Err(BackendError::Unsupported("Hailo does not support individual tensor ops".into()))
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check --no-default-features` — should pass (hailo module is gated)
Run: `cargo check --features hailo` — will fail because context.rs doesn't exist yet, but mod.rs itself should parse

**Step 3: Commit**

```bash
git add src/backend/hailo/mod.rs
git commit -m "Add Hailo backend module with Backend trait stub"
```

---

### Task 4: Create HailoRT Context

**Files:**
- Create: `src/backend/hailo/context.rs`

**Step 1: Write the HailoRT device management context**

This wraps all HailoRT FFI calls. Key responsibilities:
- Device detection and vdevice creation
- HEF loading and network group activation
- Vstream creation and I/O
- Resource cleanup via Drop

Reference the `hailort-sys` FFI functions:
- `hailo_create_vdevice` / `hailo_release_vdevice`
- `hailo_create_hef_file` / `hailo_release_hef`
- `hailo_configure_vdevice`
- `hailo_activate_network_group` / `hailo_deactivate_network_group`
- `hailo_create_input_vstreams` / `hailo_create_output_vstreams`
- `hailo_input_vstream_write` / `hailo_output_vstream_read`
- `hailo_release_input_vstream` / `hailo_release_output_vstream`
- `hailo_get_library_version` / `hailo_identify`
- `hailo_scan_devices`

The context must handle:
- `check_device_available()` — scan for Hailo devices, return error if none found
- `HailoContext::new(config)` — create vdevice
- `load_hef(path)` — load HEF, configure, activate, create vstreams
- `write_input(hef_handle, data)` — write f32 data to input vstream
- `read_output(hef_handle, size)` — read f32 data from output vstream
- `Drop` — release all resources in reverse order

Every FFI call must check the returned `hailo_status` against `HAILO_SUCCESS`.

```rust
use hailort_sys::*;
use crate::backend::{BackendError, BackendResult};
use std::path::Path;
use std::ffi::CString;

pub fn check_status(status: hailo_status, context: &str) -> BackendResult<()> {
    if status == HAILO_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = hailo_get_status_message(status);
            if ptr.is_null() {
                format!("HailoRT error {} in {}", status, context)
            } else {
                let cstr = std::ffi::CStr::from_ptr(ptr);
                format!("{} in {}", cstr.to_string_lossy(), context)
            }
        };
        Err(BackendError::OperationFailed(msg))
    }
}

pub fn check_device_available() -> BackendResult<()> {
    let mut device_ids: [hailo_device_id_t; 16] = unsafe { std::mem::zeroed() };
    let mut count: usize = 16;
    let status = unsafe {
        hailo_scan_devices(std::ptr::null_mut(), device_ids.as_mut_ptr(), &mut count)
    };
    check_status(status, "hailo_scan_devices")?;
    if count == 0 {
        return Err(BackendError::InitializationFailed(
            "No Hailo devices found".to_string(),
        ));
    }
    tracing::info!("Found {} Hailo device(s)", count);
    Ok(())
}

// ... HailoContext struct and impl (see design doc for full shape)
```

Implement the full struct: `HailoContext`, `HefHandle` (wraps network group +
vstreams for one HEF), `HailoDeviceInfo`.

**Step 2: Verify compilation**

Run: `cargo check --features hailo`
Expected: PASS (context compiles against hailort-sys types)

Note: This will only fully link on a system with libhailort installed. On
development machines without Hailo hardware, `cargo check` verifies type
correctness but `cargo build` may fail at link time. That's expected.

**Step 3: Commit**

```bash
git add src/backend/hailo/context.rs
git commit -m "Add HailoRT context with device management and vstream I/O"
```

---

### Task 5: Create GpuInference Implementation

**Files:**
- Create: `src/backend/hailo/gpu_only.rs`

**Step 1: Write HailoGpuInference struct and from_model constructor**

Follow the pattern from `src/backend/vulkan/gpu_only.rs` and
`src/backend/dx12/gpu_only.rs`:
- `model.into_parts()` to destructure the LlamaModel
- Iterate layers to extract CPU-resident weights (O projections, norms)
- Load HEF files (pre-compiled or auto-compiled) via context
- Allocate KV caches and scratch buffers

```rust
use crate::backend::cpu::simd;
use crate::backend::{BackendError, BackendResult};
use crate::model::layers::{AttentionLayer, FfnLayer, TransformerLayer};
use crate::model::LlamaModel;
use crate::tensor::{DType, Tensor};
use super::context::{HailoContext, HefHandle};
use super::config::HailoConfig;
```

Key struct fields: see design doc section 5.

The `from_model` constructor:
1. Calls `model.into_parts()`
2. Creates `HailoContext`
3. For each layer, resolves HEF path (pre-compiled dir or auto-compile cache)
4. If HEFs not found and `auto_compile` is true, calls `compiler::compile_layer_hefs()`
5. Loads each HEF via `ctx.load_hef()`
6. Extracts CPU-side weights: embeddings (dequantized), O projection per layer,
   output norm, output weight
7. Allocates KV caches: `vec![vec![0.0f32; n_kv_heads * max_seq_len * head_dim]; num_layers]`
8. Returns `HailoGpuInference`

**Step 2: Implement GpuInference trait**

```rust
impl crate::backend::GpuInference for HailoGpuInference {
    fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> { ... }
    fn prefill_token(&mut self, token_id: u32) -> BackendResult<()> { ... }
    fn reset(&mut self) { ... }
    fn position(&self) -> usize { ... }
}
```

The `forward` method implements the execution model from the design doc:
- Embed token (CPU table lookup into `self.hidden`)
- For each layer:
  - Write `hidden` to layer attn HEF input → read Q, K, V
  - CPU RoPE via `simd::rope_normal` or `simd::rope_neox`
  - CPU KV cache write
  - CPU attention via scalar fallback (or SIMD if available)
  - CPU O projection via `simd::vec_mat_f32`
  - Residual add
  - Write residual to layer FFN HEF input → read FFN output
  - Residual add
- Final norm (CPU RMS norm) → output projection (CPU vec_mat) → logits

The `prefill_token` method is the same but skips final norm + output projection.

**Step 3: Verify compilation**

Run: `cargo check --features hailo`
Expected: PASS

**Step 4: Commit**

```bash
git add src/backend/hailo/gpu_only.rs
git commit -m "Add HailoGpuInference with hybrid CPU+Hailo forward pass"
```

---

### Task 6: Create Auto-Compiler

**Files:**
- Create: `src/backend/hailo/compiler.rs`

**Step 1: Write HEF manifest and cache management**

```rust
use std::path::{Path, PathBuf};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::model::ModelConfig;
use super::config::{HailoConfig, HefManifest};

pub fn model_config_hash(config: &ModelConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.hidden_size.hash(&mut hasher);
    config.num_heads.hash(&mut hasher);
    config.num_kv_heads.hash(&mut hasher);
    config.head_dim.hash(&mut hasher);
    config.intermediate_size.hash(&mut hasher);
    config.num_layers.hash(&mut hasher);
    hasher.finish()
}

pub fn resolve_hef_path(
    config: &HailoConfig,
    model_config: &ModelConfig,
    layer_idx: usize,
    kind: &str, // "attn" or "ffn"
) -> PathBuf {
    let dir = config.hef_dir.clone().unwrap_or_else(|| {
        let hash = model_config_hash(model_config);
        config.cache_dir.join(format!("model_{:016x}", hash))
    });
    dir.join(format!("layer_{:02}_{}.hef", layer_idx, kind))
}
```

**Step 2: Write ONNX subgraph export and DFC invocation**

```rust
pub fn compile_layer_hefs(
    config: &HailoConfig,
    model_config: &ModelConfig,
    layer: &crate::model::layers::TransformerLayer,
    layer_idx: usize,
) -> crate::backend::BackendResult<(PathBuf, PathBuf)> {
    let attn_hef = resolve_hef_path(config, model_config, layer_idx, "attn");
    let ffn_hef = resolve_hef_path(config, model_config, layer_idx, "ffn");

    if attn_hef.exists() && ffn_hef.exists() {
        return Ok((attn_hef, ffn_hef));
    }

    let dir = attn_hef.parent().unwrap();
    std::fs::create_dir_all(dir).map_err(|e| {
        crate::backend::BackendError::OperationFailed(format!(
            "Failed to create HEF cache dir: {}", e
        ))
    })?;

    // Export ONNX subgraphs and compile via DFC
    // This shells out to python:
    //   python -m hailo_sdk_client compile --hef <output> --onnx <input> --target <arch>
    compile_attn_hef(config, model_config, layer, layer_idx, &attn_hef)?;
    compile_ffn_hef(config, model_config, layer, layer_idx, &ffn_hef)?;

    // Write manifest
    write_manifest(config, model_config, dir)?;

    Ok((attn_hef, ffn_hef))
}
```

The ONNX export functions create minimal ONNX models containing just the
weight tensors and operations for each subgraph (norm + linear projections
for attn, norm + FFN ops for ffn). The DFC compilation shells out to Python.

This is the most complex part of the backend and can be iterated on after
the core inference path works with pre-compiled HEFs.

**Step 3: Verify compilation**

Run: `cargo check --features hailo`
Expected: PASS

**Step 4: Commit**

```bash
git add src/backend/hailo/compiler.rs
git commit -m "Add HEF auto-compiler with ONNX export and DFC invocation"
```

---

### Task 7: Engine Integration

**Files:**
- Modify: `src/engine.rs` (~line 60, ~line 546, ~line 684)

**Step 1: Add hailo_config to EngineConfig**

In `EngineConfig` struct (around line 60):
```rust
    /// Hailo accelerator configuration (requires `hailo` feature)
    pub hailo_config: Option<crate::backend::hailo::HailoConfig>,
```

Update `Default` impl to include `hailo_config: None`.

**Step 2: Add Hailo to select_gpu_model priority chain**

In `select_gpu_model()` (around line 642, after the DX12 block), add:
```rust
        #[cfg(feature = "hailo")]
        {
            if let Some(ref hailo_config) = engine_config.hailo_config {
                if crate::backend::hailo::HailoBackend::with_config(hailo_config.clone()).is_ok() {
                    let architecture = model.architecture();
                    match crate::backend::hailo::gpu_only::HailoGpuInference::from_model(
                        model,
                        gpu_seq_len,
                        hailo_config.clone(),
                    ) {
                        Ok(gpu) => {
                            tracing::info!("Using Hailo NPU inference");
                            let wrapper = crate::backend::GpuModelWrapper::new(
                                gpu,
                                config.clone(),
                                architecture,
                            );
                            return (
                                Arc::new(crate::backend::cpu::CpuBackend::new()),
                                Box::new(wrapper),
                            );
                        }
                        Err(e) => {
                            eprintln!("Error: Hailo inference init failed: {}", e);
                            eprintln!("The model was consumed during init. Please restart without --gpu.");
                            std::process::exit(1);
                        }
                    }
                } else {
                    tracing::info!("No Hailo device available, falling back to CPU...");
                }
            }
        }
```

**Step 3: Verify compilation**

Run: `cargo check --features hailo`
Expected: PASS

Run: `cargo check --no-default-features`
Expected: PASS (hailo is gated)

**Step 4: Commit**

```bash
git add src/engine.rs
git commit -m "Integrate Hailo backend into engine GPU selection"
```

---

### Task 8: CLI Integration

**Files:**
- Modify: `src/main.rs` (~line 72)

**Step 1: Add --hailo flag to Run subcommand**

After the `gpu` flag in the Run subcommand:
```rust
        /// Use Hailo NPU acceleration (requires --features hailo)
        #[arg(long)]
        hailo: bool,

        /// Path to pre-compiled HEF directory for Hailo
        #[arg(long)]
        hef_dir: Option<String>,
```

**Step 2: Wire into EngineConfig**

In the `Commands::Run` handler, construct `HailoConfig` when `--hailo` is set:
```rust
let hailo_config = if hailo {
    Some(crate::backend::hailo::HailoConfig {
        hef_dir: hef_dir.map(PathBuf::from),
        auto_compile: true,
        ..Default::default()
    })
} else {
    None
};
```

Pass `hailo_config` into `EngineConfig`.

Also add a `hailo info` subcommand:
```rust
        /// Show Hailo device information
        #[cfg(feature = "hailo")]
        HailoInfo,
```

That calls `check_device_available()`, `hailo_get_library_version()`,
`hailo_identify()`, and prints device info.

**Step 3: Verify compilation**

Run: `cargo check --features hailo,cli`
Expected: PASS

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "Add --hailo CLI flag and hailo-info subcommand"
```

---

### Task 9: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Step 1: Add Hailo to backend table**

In the Backend Architecture table:
```markdown
| `hailo/` | `hailo` | Linux (ARM64) | Experimental - Hailo NPU via HailoRT |
```

In Feature Flags:
```toml
hailo = ["dep:hailort-sys"]
```

**Step 2: Add "Working with the Hailo backend" section under Common Tasks**

Document:
- HEF compilation workflow
- Pre-compiled vs auto-compile
- Device requirements
- Testing with `--hailo` flag

**Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "Document Hailo backend in AGENTS.md"
```

---

### Task 10: Integration Testing

**Files:**
- Create: `tests/hailo_integration_test.rs`

**Step 1: Write device detection test**

```rust
#[cfg(feature = "hailo")]
#[test]
fn test_hailo_device_detection() {
    // Skip if no device present
    match crate::backend::hailo::HailoBackend::new() {
        Ok(backend) => {
            assert_eq!(backend.name(), "hailo");
        }
        Err(e) => {
            eprintln!("Hailo not available (expected in CI): {}", e);
        }
    }
}
```

**Step 2: Write HEF loading test (requires pre-compiled HEF + device)**

```rust
#[cfg(feature = "hailo")]
#[test]
#[ignore] // Requires Hailo device and pre-compiled HEFs
fn test_hailo_inference_tinyllama() {
    // Load a small model, run forward pass, verify logits are finite
}
```

**Step 3: Write config and manifest unit tests (no device required)**

```rust
#[test]
fn test_hailo_config_default() {
    let config = HailoConfig::default();
    assert!(!config.auto_compile);
    assert!(config.hef_dir.is_none());
}

#[test]
fn test_model_config_hash_deterministic() {
    // Same config → same hash
}

#[test]
fn test_hef_path_resolution() {
    // Verify naming convention
}
```

**Step 4: Verify tests compile**

Run: `cargo test --features hailo --lib -- hailo --no-run`
Expected: Compiles without error

**Step 5: Commit**

```bash
git add tests/hailo_integration_test.rs
git commit -m "Add Hailo integration tests"
```

---

## Execution Notes

- Tasks 1-3 can be done without a Hailo device (just type checking)
- Task 4 (context.rs) compiles against hailort-sys types but won't link without libhailort
- Task 5 (gpu_only.rs) is the core implementation — most time spent here
- Task 6 (compiler.rs) is the most complex and can be stubbed initially
- Tasks 7-8 are straightforward wiring
- Tasks 9-10 are documentation and testing

The auto-compiler (Task 6) is intentionally the least critical path. The
backend is fully functional with pre-compiled HEFs. Auto-compilation can
be iterated on separately after the core inference path works.
