# GPU-Only Inference Engine: Complete Analysis for Metal, DX12, and Vulkan Replication

This document provides a thorough analysis of the CUDA `gpu_only.rs` engine and all related backend infrastructure, enabling replication of the GPU-only inference pattern for Metal, DX12, and Vulkan backends.

---

## 1. `src/backend/cuda/gpu_only.rs` — Full Analysis

### 1.1 Struct Definition: `GpuOnlyInference`

```rust
pub struct GpuOnlyInference {
    // Core GPU infrastructure
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    weights: GpuWeightStore,
    config: InferenceConfig,

    // Position tracking
    pos: usize,

    // GPU scratch buffers (pre-allocated, reused every forward pass)
    hidden: CudaSlice<f32>,           // [hidden_size]
    hidden_norm: CudaSlice<f32>,     // [hidden_size]
    residual: CudaSlice<f32>,        // [hidden_size]
    ffn_gate: CudaSlice<f32>,        // [intermediate_size]
    ffn_up: CudaSlice<f32>,          // [intermediate_size]
    ffn_down: CudaSlice<f32>,        // [intermediate_size]
    logits: CudaSlice<f32>,          // [vocab_size]

    // CPU copy of dequantized embeddings (for token lookup)
    cpu_embeddings: Vec<f32>,

    // Owned model layers (for MoE expert weights, attention config)
    layers: Vec<TransformerLayer>,

    // CPU backend (fallback for unsupported attention configs)
    cpu_backend: CpuBackend,

    // KV caches — dual CPU/GPU for fallback
    cpu_k_caches: Vec<Option<Tensor>>,
    cpu_v_caches: Vec<Option<Tensor>>,
    gpu_k_caches: Vec<Option<CudaSlice<f32>>>,  // [num_kv_heads * max_seq_len * key_length]
    gpu_v_caches: Vec<Option<CudaSlice<f32>>>,  // [num_kv_heads * max_seq_len * value_length]

    // Attention scratch buffers
    attn_q_raw: CudaSlice<f32>,
    attn_k: CudaSlice<f32>,
    attn_v: CudaSlice<f32>,
    attn_q_proper: CudaSlice<f32>,
    attn_gate: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,

    // DeltaNet
    deltanet_config: Option<DeltaNetConfig>,
    dn_conv_states: Vec<Option<CudaSlice<f32>>>,
    dn_ssm_states: Vec<Option<CudaSlice<f32>>>,
    dn_qkv: Option<CudaSlice<f32>>,
    dn_gate_z: Option<CudaSlice<f32>>,
    dn_ba: Option<CudaSlice<f32>>,
    dn_conv_out: Option<CudaSlice<f32>>,
    dn_recurrent_out: Option<CudaSlice<f32>>,
    dn_config_gpu: Option<CudaSlice<i32>>,

    // MoE scratch buffers
    moe_hidden: CudaSlice<f32>,
    moe_expert_out: CudaSlice<f32>,
    moe_expert_gate: CudaSlice<f32>,
    moe_expert_up: CudaSlice<f32>,
    moe_expert_down: CudaSlice<f32>,

    // Per-layer flags
    has_gpu_attention: Vec<bool>,
    is_deltanet: Vec<bool>,
}
```

### 1.2 `InferenceConfig` (private)

```rust
#[derive(Clone)]
struct InferenceConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    norm_eps: f32,
    freq_base: f32,
    freq_scale: f32,
    expert_intermediate: usize,
}
```

### 1.3 Public Methods

| Method | Signature | Purpose |
|--------|------------|---------|
| `from_model` | `(model: LlamaModel, max_seq_len: usize) -> BackendResult<Self>` | Construct from model, consuming it. Uploads weights, allocates all GPU buffers. |
| `forward` | `(&mut self, token_id: u32) -> BackendResult<Vec<f32>>` | Single-token forward, returns logits. |
| `forward_batch` | `(&mut self, token_ids: &[u32]) -> BackendResult<Vec<f32>>` | Prefill + last-token forward. |
| `prefill_token` | `(&mut self, token_id: u32) -> BackendResult<()>` | Process token without returning logits. |
| `reset` | `(&mut self)` | Zero KV caches and DeltaNet states. |
| `position` | `(&self) -> usize` | Current sequence position. |

### 1.4 Full `forward()` Pipeline

```
1. embed_token(token_id)
   - htod_sync_copy_into(cpu_embeddings[offset..offset+hidden_size], hidden)

2. dtod_copy(hidden → residual)

3. For each layer 0..num_layers:
   process_layer(layer_idx)

4. Final norm:
   rms_norm_gpu(hidden → hidden_norm, "output_norm.weight")

5. Output projection:
   linear_gpu(hidden_norm → logits, "output.weight")

6. dtoh_sync_copy(logits) → return Vec<f32>

7. pos += 1
```

### 1.5 `process_layer()` Pipeline (per layer)

```
1. RMS norm: rms_norm_gpu(hidden → hidden_norm, "blk.{i}.attn_norm.weight")

2. Attention / DeltaNet:
   - If has_gpu_attention: attention_gpu_forward()
   - Else if is_deltanet: deltanet_gpu_forward()
   - Else: error

3. Add residual: add_gpu(residual, hidden_norm → hidden)
4. dtod_copy(hidden → residual)

5. FFN norm: rms_norm_gpu(hidden → hidden_norm, ffn_norm_name)

6. FFN:
   - Dense: dense_ffn_gpu_forward()
   - MoE: moe_gpu_forward()

7. Add residual: add_gpu(residual, hidden_norm → hidden)
8. dtod_copy(hidden → residual)
```

### 1.6 KV Cache Management

**Layout:**
- K cache: `[num_kv_heads * max_seq_len * key_length]` — flat, row-major per head
- V cache: `[num_kv_heads * max_seq_len * value_length]` — same

**Allocation (in `from_model`):**
```rust
let kv_size = num_kv_heads * max_seq_len * kl;
let kv_v_size = num_kv_heads * max_seq_len * vl;
gpu_k_caches.push(Some(alloc(kv_size)?));
gpu_v_caches.push(Some(alloc(kv_v_size)?));
```

**Update kernel:** `update_kv_cache` — copies `attn_k` and `attn_v` into the cache at position `pos`. When `vl != kl`, two dispatches: one for K (using kl stride), one for V (using vl stride).

**Reset:** `htod_sync_copy_into(&vec![0.0f32; len], cache)` to zero.

### 1.7 Linear Projections (vec_mat)

**`linear_gpu()`** — vector-matrix multiply: `out = x @ W`

- **Quantized path:** Uses `weights.get_quantized(weight_name)` → dispatches `vec_mat_q4k`, `vec_mat_q6k`, `vec_mat_q5k`, `vec_mat_q4_0`, or `vec_mat_q8_0` depending on dtype.
- **F32 path:** Uses `weights.get(weight_name)` → dispatches `vec_mat_f32`.
- **Bias:** If `bias_name` provided, allocates temp buffer, runs `add_f32` kernel, `dtod_copy` temp → out.
- **LaunchConfig:** `grid_dim = ((n + 255) / 256, 1, 1)`, `block_dim = (256, 1, 1)`.

### 1.8 Attention (GPU) — `attention_gpu_forward()`

1. **Q/K/V projections:** `linear_gpu` for q_proj, k_proj, v_proj (weight names: `model.layers.{i}.self_attn.{q,k,v}_proj.weight`).

2. **Split Q/gate:** If `has_gate`, `split_q_gate` kernel; else `dtod_copy(attn_q_raw → attn_q_proper)`.

3. **QK norm:** If `has_q_norm` / `has_k_norm`, upload norm weight via `htod_sync_copy`, dispatch `qk_norm_per_head`.

4. **RoPE:**
   - Partial: `partial_rope` (rope_dims < kl)
   - Full: `rope_single_pos`

5. **Update KV cache:** `update_kv_cache` (and second dispatch if vl != kl).

6. **Flash attention:** `flash_attention_cached` — reads from GPU K/V caches, writes to `attn_out`.

7. **Attention gate:** If `has_gate`, `attention_gate_sigmoid` kernel.

8. **Output projection:** `linear_gpu(attn_out → hidden_norm, o_proj.weight)`.

### 1.9 MoE — `moe_gpu_forward()`

1. **Router:** `linear_gpu(hidden_norm → router_logits_gpu, ffn_gate_inp.weight)`.
2. **Download logits:** `dtoh_sync_copy(router_logits_gpu)`.
3. **Top-k on CPU:** Normalize, sort, take top_k indices and routing weights.
4. **Zero accumulator:** `htod_sync_copy_into(zeros, moe_hidden)`.
5. **For each selected expert:**
   - `upload_expert_weight_to_store` (gate, up, down) — overwrites scratch slots.
   - `linear_gpu` (gate), `silu_gpu`, `linear_gpu` (up), `mul_gpu`, `linear_gpu` (down).
   - `scaled_add_gpu(moe_hidden, moe_expert_out, weight)`.
6. **Shared experts:** Same pattern, always active; optional sigmoid gate via CPU dot product.
7. **Copy result:** `dtod_copy(moe_hidden → hidden_norm)`.

### 1.10 DeltaNet — `deltanet_gpu_forward()`

1. **QKV projection:** `linear_gpu(hidden_norm → dn_qkv, attn_qkv.weight)`.
2. **Gate projection:** `linear_gpu(hidden_norm → dn_gate_z, attn_gate.weight)`.
3. **Beta/Alpha:** `linear_gpu(hidden_norm → dn_ba, ssm_ba.weight)`.
4. **Conv1d + SiLU:** `deltanet_conv1d_silu` kernel (updates conv_state, writes dn_conv_out).
5. **Recurrent:** `deltanet_recurrent` kernel (updates ssm_state, writes dn_recurrent_out).
6. **Output:** `linear_gpu(dn_recurrent_out → hidden_norm, ssm_out.weight)`.

### 1.11 Kernel Dispatch Summary

| Kernel | Purpose |
|--------|---------|
| `rms_norm_fused` | RMS norm with shared-memory reduction |
| `vec_mat_f32` | F32 vector-matrix multiply |
| `vec_mat_q4k`, `vec_mat_q6k`, `vec_mat_q5k`, `vec_mat_q4_0`, `vec_mat_q8_0` | Quantized vec_mat |
| `add_f32` | Element-wise add |
| `mul_f32` | Element-wise mul |
| `silu_f32` | SiLU activation |
| `scaled_add_f32` | out += scale * x |
| `split_q_gate` | Split Q into q_proper + gate |
| `qk_norm_per_head` | Per-head RMS norm for Q or K |
| `partial_rope` | RoPE for rope_dims < head_dim |
| `rope_single_pos` | Full RoPE |
| `update_kv_cache` | Write K/V into cache at pos |
| `flash_attention_cached` | Causal attention from KV cache |
| `attention_gate_sigmoid` | sigmoid(gate) * attn_out |
| `deltanet_conv1d_silu` | Conv1d + SiLU |
| `deltanet_recurrent` | SSM recurrent step |

### 1.12 GPU Buffer Management Pattern

- **Allocation:** `device.alloc_zeros::<f32>(size)` or `device.htod_sync_copy(&data)`.
- **Copy H→D:** `htod_sync_copy_into(src, dst)`.
- **Copy D→D:** `dtod_copy(src, dst)`.
- **Copy D→H:** `dtoh_sync_copy(slice)`.
- **Freeing:** Buffers are owned by struct; dropped when `GpuOnlyInference` is dropped. No explicit free.

---

## 2. `src/backend/cuda/mod.rs` — CudaBackend

### 2.1 Struct

```rust
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    config: CudaConfig,
    cpu_backend: CpuBackend,
    gpu_weights: Option<GpuWeightStore>,
    gpu_hits: AtomicUsize,
    cpu_fallbacks: AtomicUsize,
}
```

### 2.2 Trait

Implements `Backend` from `backend/mod.rs`. Per-op: upload tensor to GPU, dispatch kernel, download result. Uses `gpu_weights` when available for `vec_mat` and `vec_mat_q` to avoid re-uploading weights.

### 2.3 gpu_only Integration

**gpu_only is NOT part of CudaBackend.** It is a separate inference engine. Integration happens in `engine.rs`:

1. `select_gpu_model()` tries `GpuOnlyInference::from_model()` first.
2. On success, wraps in `GpuModelWrapper` (implements `Model` trait).
3. Backend passed to engine is `CpuBackend` (unused for forward; only for fallback paths).
4. `GpuModelWrapper::forward()` locks the `GpuOnlyInference` and calls `gpu.forward()` / `prefill_token()`.

---

## 3. `src/backend/mod.rs` — Backend Trait

### 3.1 All Methods

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;

    // Memory
    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor>;
    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor>;

    // Element-wise
    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()>;

    // Activations
    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // Normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) -> BackendResult<()>;

    // Matrix
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // Quantization
    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;
    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // RoPE
    fn rope(&self, q: &mut Tensor, k: &mut Tensor, pos: usize, freq_base: f32, freq_scale: f32, use_neox: bool) -> BackendResult<()>;

    // Attention
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor, scale: f32) -> BackendResult<()>;
    fn flash_attention(&self, ...) -> BackendResult<()>;  // default: calls attention
    fn attention_cached(&self, q: &Tensor, k_cache: &Tensor, v_cache: &Tensor, out: &mut Tensor, scale: f32, kv_len: usize) -> BackendResult<()>;
}
```

---

## 4. Vulkan Backend

### 4.1 `vulkan/mod.rs` — VulkanBackend

```rust
pub struct VulkanBackend {
    ctx: VulkanContext,
    cpu_fallback: CpuBackend,
    device_name: String,
}
```

- Implements `Backend`; delegates to `ops::*` functions.
- Per-op: create buffers, dispatch, read back, free buffers.
- **No persistent GPU buffers** — every op allocates and frees.

### 4.2 `vulkan/ops.rs` — Operations

| Op | Pattern |
|----|---------|
| add, mul, scale | create_buffer_with_data, create_output_buffer, dispatch, read_buffer, free_buffer |
| silu, gelu | Same |
| softmax | 3-pass: softmax_max, softmax_exp, softmax_div |
| rms_norm | 2-pass: rms_norm_sum, rms_norm_scale |
| vec_mat | Single dispatch |
| matmul, matvec | Single dispatch |
| dequantize | dequant_q8_0, dequant_q4_k, dequant_q6_k |
| vec_mat_q, matvec_q | 2-pass: dequant then vec_mat/matvec |
| rope | create, dispatch, read back (in-place) |
| attention, attention_cached | Full GPU |

**Buffer lifecycle:** Create → dispatch → read → free. No persistent scratch.

### 4.3 `vulkan/context.rs` — VulkanContext

```rust
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub pipelines: HashMap<String, ComputePipeline>,
    pub descriptor_pool: vk::DescriptorPool,
}
```

**Buffer API:**
- `create_buffer_with_data(data: &[f32])` → GpuBuffer
- `create_buffer_with_bytes(data, size)` → GpuBuffer
- `create_output_buffer(num_floats)` → GpuBuffer (GpuToCpu for readback)
- `read_buffer(&GpuBuffer)` → Vec<f32>
- `free_buffer(GpuBuffer)` — must be called explicitly

**Dispatch:**
- `dispatch(pipeline_name, buffers, push_constants, workgroup_count)` — allocates descriptor set, records command buffer, submits, waits on fence.

**Missing for gpu_only:** No persistent buffer API; no weight store; no quantized vec_mat kernels (Q4K, Q5K, etc.); no split_q_gate, qk_norm_per_head, partial_rope, update_kv_cache, flash_attention_cached, attention_gate_sigmoid, deltanet_conv1d_silu, deltanet_recurrent.

---

## 5. Metal Backend

### 5.1 `metal/mod.rs` — MetalBackend

```rust
pub struct MetalBackend {
    ctx: MetalContext,
    cpu_fallback: CpuBackend,
    device_name: String,
}
```

Same pattern as Vulkan: per-op, no persistent buffers.

### 5.2 `metal/ops.rs` — Operations

Same ops as Vulkan. Uses `create_buffer_with_data`, `create_output_buffer`, `read_buffer`. **Metal does NOT free buffers explicitly** — buffers are RAII (dropped when out of scope). Uses `StorageModeShared` for unified memory.

### 5.3 `metal/context.rs` — MetalContext

```rust
pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub pipelines: HashMap<String, ComputePipelineState>,
    device_name: String,
}
```

**Buffer API:**
- `create_buffer_with_data(data)` → Buffer (shared)
- `create_buffer_with_raw_bytes(data)` → Buffer
- `create_output_buffer(num_floats)` → Buffer
- `read_buffer(buf, num_floats)` → Vec<f32>

**Dispatch:**
- `dispatch(pipeline_name, buffers, param_bytes, grid, threadgroup)` — creates command buffer, encodes, commits, waits.
- `dispatch_threadgroups(...)` — for reductions.

**Missing for gpu_only:** Same as Vulkan — no persistent buffers, no weight store, no specialized kernels.

---

## 6. DX12 Backend

### 6.1 `dx12/mod.rs` — Dx12Backend

```rust
pub struct Dx12Backend {
    ctx: Dx12Context,
    cpu_fallback: CpuBackend,
    device_name: String,
}
```

### 6.2 `dx12/ops.rs` — Operations

Same pattern. Uses `create_buffer_with_data`, `create_output_buffer`, `create_readwrite_buffer` (for RoPE in-place), `read_buffer`. Buffers are RAII (GpuBuffer holds upload/default/readback resources).

### 6.3 `dx12/context.rs` — Dx12Context

```rust
pub struct Dx12Context {
    pub device: ID3D12Device,
    pub command_queue: ID3D12CommandQueue,
    pub command_allocator: ID3D12CommandAllocator,
    pub pipelines: HashMap<String, ComputePipeline>,
    pub fence: ID3D12Fence,
    pub fence_value: Mutex<u64>,
    pub fence_event: HANDLE,
}
```

**Buffer API:**
- `create_buffer_with_data(data)` → GpuBuffer (upload + default)
- `create_buffer_with_raw_bytes(data)` → GpuBuffer
- `create_readwrite_buffer(data)` → GpuBuffer (upload + default + readback)
- `create_output_buffer(num_floats)` → GpuBuffer (default + readback)
- `read_buffer(&GpuBuffer)` → Vec<f32>

**Dispatch:** Copies upload→default, barriers, dispatch, copy default→readback, barriers, fence wait.

**Missing for gpu_only:** Same gaps as Vulkan/Metal.

---

## 7. `src/model/loader.rs` — Model Loading

### 7.1 Key Structures

- **ModelLoader:** Holds `GgufFile`, `Architecture`, `ModelConfig`.
- **build_model():** Returns `LlamaModel` with `token_embedding`, `layers`, `norm`, `output`.
- **load_transformer_layer():** Produces `TransformerLayer` with `attn_norm`, `attn_layer` (FullAttention or DeltaNet), `post_attn_norm`, `ffn_norm`, `ffn_layer` (Dense or MoE).
- **LlamaModel::into_parts():** Consumes model, returns `(config, token_embedding, layers, norm, output, architecture, recurrent_mask, deltanet_config)`.

### 7.2 Weight Naming (GGUF)

| Layer | Tensor Names |
|-------|--------------|
| Attention | `blk.{i}.attn_q.weight`, `attn_k`, `attn_v`, `attn_output.weight`, `attn_norm.weight` |
| Qwen-style | `model.layers.{i}.self_attn.{q,k,v}_proj.weight`, `o_proj.weight` |
| DeltaNet | `blk.{i}.attn_qkv`, `attn_gate`, `ssm_ba`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_norm`, `ssm_out` |
| FFN | `blk.{i}.ffn_gate`, `ffn_up`, `ffn_down`, `ffn_norm` |
| MoE | `blk.{i}.ffn_gate_inp`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_shexp`, etc. |
| Output | `output_norm.weight`, `output.weight` |

---

## 8. Implementation Roadmap for Metal/DX12/Vulkan gpu_only

### Phase 1: Persistent Buffer & Weight Store

1. **Weight store** (per backend):
   - `GpuWeightStore` equivalent: HashMap of buffers keyed by tensor name.
   - `upload()`, `upload_quantized()` for f32 and quantized weights.
   - Transpose quantized blocks for kernel layout (see `dequant_weights.rs`).

2. **Persistent scratch buffers** in `GpuOnlyInference`:
   - Allocate once in `from_model`, never free until drop.
   - Map CUDA `CudaSlice<f32>` to backend buffer type (VkBuffer, MTLBuffer, ID3D12Resource).

### Phase 2: Kernel / Shader Parity

| CUDA Kernel | Vulkan | Metal | DX12 |
|-------------|--------|-------|------|
| rms_norm_fused | ✓ (rms_norm_scale) | ✓ | ✓ |
| vec_mat_f32 | ✓ | ✓ | ✓ |
| vec_mat_q4k/q5k/q6k/q4_0/q8_0 | ❌ | ❌ | ❌ |
| add_f32, mul_f32, silu_f32, scaled_add_f32 | ✓ | ✓ | ✓ |
| split_q_gate | ❌ | ❌ | ❌ |
| qk_norm_per_head | ❌ | ❌ | ❌ |
| partial_rope, rope_single_pos | rope exists | rope exists | rope exists |
| update_kv_cache | ❌ | ❌ | ❌ |
| flash_attention_cached | attention_cached | attention_cached | attention_cached |
| attention_gate_sigmoid | ❌ | ❌ | ❌ |
| deltanet_conv1d_silu | ❌ | ❌ | ❌ |
| deltanet_recurrent | ❌ | ❌ | ❌ |

**Priority:** Quantized vec_mat, update_kv_cache, then DeltaNet, then QK norm/split_q_gate/attention_gate.

### Phase 3: GpuOnlyInference Port

1. Create `GpuOnlyInferenceMetal`, `GpuOnlyInferenceDx12`, `GpuOnlyInferenceVulkan`.
2. Replace `CudaDevice` / `CudaSlice` with backend-specific types.
3. Replace `CudaKernels` with backend pipeline dispatch.
4. Implement `linear_gpu` → `linear_metal` etc. using backend buffer API.
5. Wire `engine.rs` to try Metal/DX12/Vulkan gpu_only when CUDA unavailable.

### Phase 4: Integration

- Add `GpuModelWrapper` variants or make it generic over backend.
- In `select_gpu_model`, try CUDA gpu_only first, then Metal gpu_only (macOS), then DX12 gpu_only (Windows), then Vulkan gpu_only, then fall back to per-op Backend.

---

## 9. Essential Files Reference

| File | Purpose |
|------|---------|
| `backend/cuda/gpu_only.rs` | Full GPU inference engine |
| `backend/cuda/dequant_weights.rs` | Weight upload, GpuWeightStore, expert streaming |
| `backend/cuda/kernels.rs` | CUDA kernel source (PTX) |
| `backend/mod.rs` | Backend trait |
| `backend/vulkan/context.rs` | Buffer create/read/free, dispatch |
| `backend/vulkan/ops.rs` | Per-op implementations |
| `backend/metal/context.rs` | Metal buffer and dispatch |
| `backend/metal/ops.rs` | Metal ops |
| `backend/dx12/context.rs` | DX12 buffer and dispatch |
| `backend/dx12/ops.rs` | DX12 ops |
| `model/loader.rs` | GGUF loading, tensor names |
| `model/llama.rs` | LlamaModel, into_parts |
| `engine.rs` | select_gpu_model, GpuOnlyInference integration |
