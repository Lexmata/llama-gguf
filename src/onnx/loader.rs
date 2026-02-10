//! ONNX model loader
//!
//! Loads model weights from ONNX initializer tensors and maps HuggingFace
//! tensor naming conventions to the internal representation used by `LlamaModel`.

use std::collections::HashMap;
use std::path::Path;

use crate::model::layers::{Attention, FeedForward, Linear, RMSNorm, TransformerLayer};
use crate::model::{Architecture, LlamaModel, ModelConfig, RopeType};
use crate::tensor::{DType, Tensor};

use super::config::HfConfig;
use super::error::{OnnxError, OnnxResult};
use super::reader::{self, OnnxFile};

/// Model loader for ONNX files
///
/// Parses an ONNX file and companion config.json, then maps HuggingFace
/// tensor names to internal names and builds a `LlamaModel`.
pub struct OnnxModelLoader {
    /// Parsed ONNX file
    onnx: OnnxFile,
    /// Parsed HuggingFace config
    hf_config: HfConfig,
    /// Detected architecture
    architecture: Architecture,
    /// Internal model configuration
    config: ModelConfig,
}

impl OnnxModelLoader {
    /// Load a model from an ONNX file path
    ///
    /// Expects a companion `config.json` in the same directory as the ONNX file.
    pub fn load<P: AsRef<Path>>(path: P) -> OnnxResult<Self> {
        let path = path.as_ref();

        tracing::info!("Loading ONNX model from: {}", path.display());

        // Load the ONNX file
        let onnx = OnnxFile::open(path)?;

        // Find and load config.json from the same directory
        let config_path = path
            .parent()
            .unwrap_or(Path::new("."))
            .join("config.json");

        let hf_config = HfConfig::from_file(&config_path)?;
        let architecture = hf_config.architecture();

        if matches!(architecture, Architecture::Unknown) {
            return Err(OnnxError::UnsupportedModelType(
                hf_config
                    .model_type
                    .clone()
                    .unwrap_or_else(|| "unknown".into()),
            ));
        }

        let config = hf_config.to_model_config()?;

        tracing::info!(
            "ONNX model: {} layers, {} heads, {} hidden dim, arch={:?}",
            config.num_layers,
            config.num_heads,
            config.hidden_size,
            architecture
        );

        Ok(Self {
            onnx,
            hf_config,
            architecture,
            config,
        })
    }

    /// Load with an explicit config (for non-HuggingFace models)
    pub fn load_with_config<P: AsRef<Path>>(path: P, hf_config: HfConfig) -> OnnxResult<Self> {
        let onnx = OnnxFile::open(path)?;
        let architecture = hf_config.architecture();
        let config = hf_config.to_model_config()?;

        Ok(Self {
            onnx,
            hf_config,
            architecture,
            config,
        })
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the detected architecture
    pub fn architecture(&self) -> Architecture {
        self.architecture
    }

    /// Get the HuggingFace config
    pub fn hf_config(&self) -> &HfConfig {
        &self.hf_config
    }

    /// Get the underlying ONNX file
    pub fn onnx(&self) -> &OnnxFile {
        &self.onnx
    }

    /// Build the model from loaded weights
    pub fn build_model(self) -> OnnxResult<LlamaModel> {
        let initializers = self.onnx.initializers()?;

        // Trace the computation graph to resolve opaque tensor names
        let graph_name_map = reader::trace_graph_tensor_names(&self.onnx)?;

        tracing::debug!(
            "Graph tracing found {} tensor name mappings",
            graph_name_map.len()
        );
        for (k, v) in &graph_name_map {
            tracing::debug!("  graph: {} -> {}", k, v);
        }

        let name_map = TensorNameMapper::new(&initializers, &graph_name_map);

        tracing::debug!(
            "TensorNameMapper resolved {} internal names",
            name_map.mapping.len()
        );
        for (k, v) in &name_map.mapping {
            tracing::debug!("  {} -> {}", k, v);
        }

        // Load token embeddings
        let token_embedding = self.load_mapped_tensor(&name_map, "token_embd.weight")?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(self.config.num_layers);
        for i in 0..self.config.num_layers {
            let layer = self.load_transformer_layer(&name_map, i)?;
            layers.push(layer);
        }

        // Load final normalization
        let norm_weight = self.load_mapped_tensor(&name_map, "output_norm.weight")?;
        let norm = RMSNorm::new(norm_weight, self.config.norm_eps)?;

        // Load output projection (may be tied to embeddings)
        let output = if self.config.tie_word_embeddings
            || name_map.resolve("output.weight").is_none()
        {
            Linear::new(token_embedding.clone(), None)?
        } else {
            let output_weight = self.load_mapped_tensor(&name_map, "output.weight")?;
            Linear::new(output_weight, None)?
        };

        let model = LlamaModel::new(
            self.config,
            token_embedding,
            layers,
            norm,
            output,
            self.architecture,
        )?;

        Ok(model)
    }

    /// Load a single transformer layer
    fn load_transformer_layer(
        &self,
        name_map: &TensorNameMapper,
        layer_idx: usize,
    ) -> OnnxResult<TransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention normalization
        let attn_norm_weight =
            self.load_mapped_tensor(name_map, &format!("{}.attn_norm.weight", prefix))?;
        let attn_norm = RMSNorm::new(attn_norm_weight, self.config.norm_eps)?;

        // Attention projections (with optional biases)
        let wq_bias = self.try_load_mapped_tensor(name_map, &format!("{}.attn_q.bias", prefix));
        let wq = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.attn_q.weight", prefix))?,
            wq_bias,
        )?;

        let wk_bias = self.try_load_mapped_tensor(name_map, &format!("{}.attn_k.bias", prefix));
        let wk = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.attn_k.weight", prefix))?,
            wk_bias,
        )?;

        let wv_bias = self.try_load_mapped_tensor(name_map, &format!("{}.attn_v.bias", prefix));
        let wv = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.attn_v.weight", prefix))?,
            wv_bias,
        )?;

        let wo = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.attn_output.weight", prefix))?,
            None,
        )?;

        let use_neox_rope = matches!(self.config.rope_config.rope_type, RopeType::NeoX);
        let attention = Attention::with_rope_type(
            wq,
            wk,
            wv,
            wo,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            use_neox_rope,
        );

        // FFN normalization
        let ffn_norm_weight =
            self.load_mapped_tensor(name_map, &format!("{}.ffn_norm.weight", prefix))?;
        let ffn_norm = RMSNorm::new(ffn_norm_weight, self.config.norm_eps)?;

        // FFN projections
        let w_gate = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.ffn_gate.weight", prefix))?,
            None,
        )?;
        let w_up = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.ffn_up.weight", prefix))?,
            None,
        )?;
        let w_down = Linear::new(
            self.load_mapped_tensor(name_map, &format!("{}.ffn_down.weight", prefix))?,
            None,
        )?;

        let ffn = FeedForward::new(w_gate, w_up, w_down);

        Ok(TransformerLayer {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            layer_idx,
        })
    }

    /// Load a tensor by internal name, mapping through the HF name convention
    fn load_mapped_tensor(
        &self,
        name_map: &TensorNameMapper,
        internal_name: &str,
    ) -> OnnxResult<Tensor> {
        let hf_name = name_map
            .resolve(internal_name)
            .ok_or_else(|| OnnxError::MissingTensor(internal_name.to_string()))?;

        self.load_onnx_tensor(hf_name)
    }

    /// Try to load a tensor, returning None if not found
    fn try_load_mapped_tensor(
        &self,
        name_map: &TensorNameMapper,
        internal_name: &str,
    ) -> Option<Tensor> {
        let hf_name = name_map.resolve(internal_name)?;
        self.load_onnx_tensor(hf_name).ok()
    }

    /// Load a tensor directly from ONNX initializers by its ONNX name
    ///
    /// If the tensor is F16 or BF16, it is automatically converted to F32
    /// since these formats aren't supported by most compute backends.
    fn load_onnx_tensor(&self, name: &str) -> OnnxResult<Tensor> {
        let tensor_proto = self.onnx.get_initializer(name)?;

        let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();
        let dtype = reader::onnx_dtype_to_dtype(tensor_proto.data_type)?;
        let data = reader::extract_tensor_bytes(tensor_proto, &self.onnx.base_dir)?;

        let n_elements = shape.iter().product::<usize>();
        let expected_bytes = dtype.size_for_elements(n_elements);
        if data.len() != expected_bytes {
            return Err(OnnxError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_bytes,
                actual: data.len(),
            });
        }

        // Auto-convert F16/BF16 to F32 for CPU backend compatibility
        let (data, dtype) = match dtype {
            DType::F16 => {
                tracing::debug!("Converting tensor '{}' from F16 to F32", name);
                let f32_data = convert_f16_to_f32(&data);
                (f32_data, DType::F32)
            }
            DType::BF16 => {
                tracing::debug!("Converting tensor '{}' from BF16 to F32", name);
                let f32_data = convert_bf16_to_f32(&data);
                (f32_data, DType::F32)
            }
            _ => (data, dtype),
        };

        // ONNX stores tensors in row-major (C) order, but GGUF/our backend uses
        // column-major layout for weight matrices in vec_mat operations.
        // Only transpose 2D weight tensors — NOT the embedding tensor, which uses
        // direct row-indexed lookup (token_id * hidden_size).
        let is_embedding = name.contains("embed_tokens");
        let data = if shape.len() == 2 && dtype == DType::F32 && !is_embedding {
            let rows = shape[0];
            let cols = shape[1];
            tracing::debug!(
                "Transposing tensor '{}' [{}x{}] from row-major to column-major",
                name,
                rows,
                cols
            );
            transpose_f32_to_col_major(&data, rows, cols)
        } else {
            data
        };

        let mut tensor = Tensor::new(data, shape, dtype)?;
        tensor.set_name(name);
        Ok(tensor)
    }
}

// ============================================================================
// Tensor name mapping
// ============================================================================

/// Maps internal GGUF-style tensor names to actual ONNX initializer names.
///
/// Uses two strategies:
/// 1. **Direct mapping**: If ONNX initializers have recognizable HuggingFace names
///    (e.g., `model.layers.0.self_attn.q_proj.weight`), map them directly.
/// 2. **Graph tracing**: If initializers have opaque names (e.g., `onnx::MatMul_345`),
///    trace the computation graph to discover which weight feeds which operation.
struct TensorNameMapper {
    /// Reverse mapping: internal name -> actual ONNX initializer name
    mapping: HashMap<String, String>,
}

impl TensorNameMapper {
    fn new(
        initializers: &HashMap<&str, &super::proto::TensorProto>,
        graph_name_map: &HashMap<String, String>,
    ) -> Self {
        let mut mapping = HashMap::new();

        // Phase 1: Direct mapping from recognizable HF names in initializers
        for &name in initializers.keys() {
            if let Some(internal) = hf_to_internal_name(name) {
                mapping.insert(internal, name.to_string());
            }
        }

        // Phase 2: Graph-traced mapping for opaque names
        // graph_name_map has two kinds of entries:
        //   a) MatMul-traced: opaque_onnx_name -> logical_hf_name
        //      (e.g., "onnx::MatMul_345" -> "model.layers.0.self_attn.q_proj.weight")
        //   b) Identity aliases: alias_name -> source_initializer_name
        //      (e.g., "model.norm.weight" -> "model.layers.0.input_layernorm.weight")
        for (key, value) in graph_name_map {
            if key.starts_with("onnx::") {
                // Case (a): key is opaque name (real initializer), value is HF name
                if let Some(internal) = hf_to_internal_name(value) {
                    mapping.entry(internal).or_insert_with(|| key.clone());
                }
            } else if initializers.contains_key(value.as_str()) {
                // Case (b): key is alias name, value is source initializer
                if let Some(internal) = hf_to_internal_name(key) {
                    mapping.entry(internal).or_insert_with(|| value.clone());
                }
            }
        }

        Self { mapping }
    }

    /// Resolve an internal name to the actual ONNX initializer name
    fn resolve(&self, internal_name: &str) -> Option<&str> {
        self.mapping.get(internal_name).map(|s| s.as_str())
    }
}

/// Convert a HuggingFace PyTorch tensor name to the internal (GGUF-style) name.
///
/// Supports the following naming conventions:
/// - `model.embed_tokens.weight` -> `token_embd.weight`
/// - `model.layers.{i}.self_attn.q_proj.weight` -> `blk.{i}.attn_q.weight`
/// - `model.layers.{i}.self_attn.k_proj.weight` -> `blk.{i}.attn_k.weight`
/// - `model.layers.{i}.self_attn.v_proj.weight` -> `blk.{i}.attn_v.weight`
/// - `model.layers.{i}.self_attn.o_proj.weight` -> `blk.{i}.attn_output.weight`
/// - `model.layers.{i}.mlp.gate_proj.weight` -> `blk.{i}.ffn_gate.weight`
/// - `model.layers.{i}.mlp.up_proj.weight` -> `blk.{i}.ffn_up.weight`
/// - `model.layers.{i}.mlp.down_proj.weight` -> `blk.{i}.ffn_down.weight`
/// - `model.layers.{i}.input_layernorm.weight` -> `blk.{i}.attn_norm.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight` -> `blk.{i}.ffn_norm.weight`
/// - `model.norm.weight` -> `output_norm.weight`
/// - `lm_head.weight` -> `output.weight`
fn hf_to_internal_name(hf_name: &str) -> Option<String> {
    // Token embeddings
    if hf_name == "model.embed_tokens.weight" || hf_name == "embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }

    // Output projection
    if hf_name == "lm_head.weight" {
        return Some("output.weight".to_string());
    }

    // Final normalization
    if hf_name == "model.norm.weight" || hf_name == "norm.weight" {
        return Some("output_norm.weight".to_string());
    }

    // Transformer layer tensors: model.layers.{i}.{component}
    if let Some(rest) = hf_name
        .strip_prefix("model.layers.")
        .or_else(|| hf_name.strip_prefix("layers."))
    {
        // Parse layer index
        let dot_pos = rest.find('.')?;
        let layer_idx: usize = rest[..dot_pos].parse().ok()?;
        let component = &rest[dot_pos + 1..];

        let internal = match component {
            // Attention
            "self_attn.q_proj.weight" => format!("blk.{}.attn_q.weight", layer_idx),
            "self_attn.k_proj.weight" => format!("blk.{}.attn_k.weight", layer_idx),
            "self_attn.v_proj.weight" => format!("blk.{}.attn_v.weight", layer_idx),
            "self_attn.o_proj.weight" => format!("blk.{}.attn_output.weight", layer_idx),
            "self_attn.q_proj.bias" => format!("blk.{}.attn_q.bias", layer_idx),
            "self_attn.k_proj.bias" => format!("blk.{}.attn_k.bias", layer_idx),
            "self_attn.v_proj.bias" => format!("blk.{}.attn_v.bias", layer_idx),

            // FFN (LLaMA-style: gate/up/down)
            "mlp.gate_proj.weight" => format!("blk.{}.ffn_gate.weight", layer_idx),
            "mlp.up_proj.weight" => format!("blk.{}.ffn_up.weight", layer_idx),
            "mlp.down_proj.weight" => format!("blk.{}.ffn_down.weight", layer_idx),

            // Normalization
            "input_layernorm.weight" => format!("blk.{}.attn_norm.weight", layer_idx),
            "post_attention_layernorm.weight" => format!("blk.{}.ffn_norm.weight", layer_idx),

            _ => return None,
        };

        return Some(internal);
    }

    None
}

// ============================================================================
// Tensor layout conversion
// ============================================================================

/// Transpose an F32 tensor from row-major to column-major layout.
///
/// ONNX stores tensors in row-major (C) order where `W[i,j]` is at `i * cols + j`.
/// GGUF / our backend expects column-major order where `W[i,j]` is at `i + j * rows`.
///
/// The shape stays the same `[rows, cols]` — only the memory layout changes.
fn transpose_f32_to_col_major(row_major_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let n_elements = rows * cols;
    let mut col_major = vec![0u8; n_elements * 4];

    // Read all F32 values
    let src: &[f32] =
        unsafe { std::slice::from_raw_parts(row_major_data.as_ptr() as *const f32, n_elements) };

    // Write in column-major order
    let dst: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(col_major.as_mut_ptr() as *mut f32, n_elements) };

    for i in 0..rows {
        for j in 0..cols {
            // row-major: src[i * cols + j]
            // col-major: dst[i + j * rows]
            dst[i + j * rows] = src[i * cols + j];
        }
    }

    col_major
}

// ============================================================================
// Float conversion utilities
// ============================================================================

/// Convert F16 (IEEE 754 half-precision) bytes to F32 bytes
fn convert_f16_to_f32(f16_bytes: &[u8]) -> Vec<u8> {
    let n_elements = f16_bytes.len() / 2;
    let mut f32_bytes = Vec::with_capacity(n_elements * 4);

    for chunk in f16_bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f32_val = f16_bits_to_f32(bits);
        f32_bytes.extend_from_slice(&f32_val.to_le_bytes());
    }

    f32_bytes
}

/// Convert BF16 (Brain Float 16) bytes to F32 bytes
fn convert_bf16_to_f32(bf16_bytes: &[u8]) -> Vec<u8> {
    let n_elements = bf16_bytes.len() / 2;
    let mut f32_bytes = Vec::with_capacity(n_elements * 4);

    for chunk in bf16_bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        // BF16 to F32: just shift left by 16 (BF16 is upper 16 bits of F32)
        let f32_bits = (bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        f32_bytes.extend_from_slice(&f32_val.to_le_bytes());
    }

    f32_bytes
}

/// Convert a single F16 value (as u16 bit pattern) to F32
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exponent = ((h >> 10) & 0x1F) as u32;
    let mantissa = (h & 0x3FF) as u32;

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            // Zero (positive or negative)
            sign << 31
        } else {
            // Subnormal: convert to normalized F32
            let mut e = 0u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF; // Remove implicit leading 1
            let exp = (127 - 15 - e + 1) as u32;
            (sign << 31) | (exp << 23) | (m << 13)
        }
    } else if exponent == 31 {
        // Inf or NaN
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        // Normal number
        let exp = (exponent as i32 - 15 + 127) as u32;
        (sign << 31) | (exp << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_name_mapping() {
        assert_eq!(
            hf_to_internal_name("model.embed_tokens.weight"),
            Some("token_embd.weight".to_string())
        );
        assert_eq!(
            hf_to_internal_name("lm_head.weight"),
            Some("output.weight".to_string())
        );
        assert_eq!(
            hf_to_internal_name("model.norm.weight"),
            Some("output_norm.weight".to_string())
        );
        assert_eq!(
            hf_to_internal_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".to_string())
        );
        assert_eq!(
            hf_to_internal_name("model.layers.15.mlp.gate_proj.weight"),
            Some("blk.15.ffn_gate.weight".to_string())
        );
        assert_eq!(
            hf_to_internal_name("model.layers.3.input_layernorm.weight"),
            Some("blk.3.attn_norm.weight".to_string())
        );
        assert_eq!(
            hf_to_internal_name("model.layers.3.post_attention_layernorm.weight"),
            Some("blk.3.ffn_norm.weight".to_string())
        );
        assert_eq!(hf_to_internal_name("unknown_tensor"), None);
    }
}
