//! LoRA (Low-Rank Adaptation) support for model fine-tuning
//!
//! LoRA adds trainable low-rank decomposition matrices to existing model weights,
//! allowing efficient fine-tuning without modifying the base model.
//!
//! For a weight matrix W, LoRA adds: W' = W + α * (A @ B) / r
//! where A ∈ R^(d×r), B ∈ R^(r×k), r << min(d,k), and α is a scaling factor.
//!
//! Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
//! https://arxiv.org/abs/2106.09685

use std::collections::HashMap;
use std::path::Path;

use crate::backend::Backend;
use crate::gguf::GgufFile;
use crate::tensor::{DType, Tensor};

/// LoRA configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices
    pub rank: usize,
    /// Scaling factor alpha
    pub alpha: f32,
    /// Dropout probability (0.0 = no dropout)
    pub dropout: f32,
    /// Target modules to apply LoRA to (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

impl LoraConfig {
    /// Create a LoRA config for QKV attention only
    pub fn attention_qkv(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
            ],
        }
    }

    /// Create a LoRA config for all linear layers
    pub fn all_linear(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }

    /// Compute the scaling factor: alpha / rank
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// A single LoRA adapter for one weight matrix
#[derive(Debug)]
pub struct LoraAdapter {
    /// Low-rank matrix A (input projection): [rank, in_features]
    pub lora_a: Tensor,
    /// Low-rank matrix B (output projection): [out_features, rank]
    pub lora_b: Tensor,
    /// Rank
    pub rank: usize,
    /// Scaling factor
    pub scaling: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter
    pub fn new(in_features: usize, out_features: usize, rank: usize, scaling: f32) -> Self {
        // Initialize A with small random values, B with zeros
        // This ensures the adapter starts as a no-op
        let lora_a = Tensor::zeros(vec![rank, in_features], DType::F32);
        let lora_b = Tensor::zeros(vec![out_features, rank], DType::F32);

        Self {
            lora_a,
            lora_b,
            rank,
            scaling,
        }
    }

    /// Create from existing tensors
    pub fn from_tensors(lora_a: Tensor, lora_b: Tensor, scaling: f32) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            lora_a,
            lora_b,
            rank,
            scaling,
        }
    }

    /// Apply LoRA to input: output = x @ W + scaling * (x @ A^T @ B^T)
    ///
    /// For efficiency, we compute: x @ A^T @ B^T * scaling
    /// which is equivalent to: (x @ A^T) @ B^T * scaling
    pub fn apply(&self, x: &Tensor, _backend: &dyn Backend) -> Result<Tensor, crate::backend::BackendError> {
        // x: [batch, in_features] or [in_features]
        // A: [rank, in_features] -> A^T: [in_features, rank]
        // B: [out_features, rank] -> B^T: [rank, out_features]
        //
        // Step 1: x @ A^T -> [batch, rank] or [rank]
        // Step 2: result @ B^T -> [batch, out_features] or [out_features]
        // Step 3: scale by scaling factor

        let x_shape = x.shape();
        let in_features = *x_shape.last().unwrap_or(&0);
        let out_features = self.lora_b.shape()[0];

        // For 1D input (single vector)
        if x_shape.len() == 1 {
            // x @ A^T: [in_features] @ [in_features, rank] -> [rank]
            let mut intermediate = Tensor::zeros(vec![self.rank], DType::F32);
            
            // Manual matvec with transposed A
            let x_data = x.as_f32()?;
            let a_data = self.lora_a.as_f32()?;
            let inter_data = intermediate.as_f32_mut()?;
            
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for i in 0..in_features {
                    // A is [rank, in_features], so A^T[i, r] = A[r, i]
                    sum += x_data[i] * a_data[r * in_features + i];
                }
                inter_data[r] = sum;
            }

            // intermediate @ B^T: [rank] @ [rank, out_features] -> [out_features]
            let mut output = Tensor::zeros(vec![out_features], DType::F32);
            let b_data = self.lora_b.as_f32()?;
            let out_data = output.as_f32_mut()?;

            for o in 0..out_features {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    // B is [out_features, rank], so B^T[r, o] = B[o, r]
                    sum += inter_data[r] * b_data[o * self.rank + r];
                }
                out_data[o] = sum * self.scaling;
            }

            Ok(output)
        } else {
            // Batch processing - simplified for now
            // In a full implementation, we'd handle arbitrary batch dimensions
            let batch_size = x_shape[0];
            let mut output = Tensor::zeros(vec![batch_size, out_features], DType::F32);

            let x_data = x.as_f32()?;
            let a_data = self.lora_a.as_f32()?;
            let b_data = self.lora_b.as_f32()?;
            let out_data = output.as_f32_mut()?;

            for b in 0..batch_size {
                // x[b] @ A^T -> intermediate
                let mut intermediate = vec![0.0f32; self.rank];
                for r in 0..self.rank {
                    let mut sum = 0.0f32;
                    for i in 0..in_features {
                        sum += x_data[b * in_features + i] * a_data[r * in_features + i];
                    }
                    intermediate[r] = sum;
                }

                // intermediate @ B^T -> output[b]
                for o in 0..out_features {
                    let mut sum = 0.0f32;
                    for r in 0..self.rank {
                        sum += intermediate[r] * b_data[o * self.rank + r];
                    }
                    out_data[b * out_features + o] = sum * self.scaling;
                }
            }

            Ok(output)
        }
    }

    /// Get the number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.lora_a.numel() + self.lora_b.numel()
    }
}

/// Collection of LoRA adapters for a model
#[derive(Debug)]
pub struct LoraAdapters {
    /// Adapters indexed by weight name
    adapters: HashMap<String, LoraAdapter>,
    /// Configuration
    config: LoraConfig,
    /// Whether adapters are enabled
    enabled: bool,
}

impl LoraAdapters {
    /// Create empty LoRA adapters collection
    pub fn new(config: LoraConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            config,
            enabled: true,
        }
    }

    /// Load LoRA adapters from a GGUF file
    ///
    /// Expects tensors with naming convention:
    /// - `{layer_name}.lora_a` - A matrix [rank, in_features]
    /// - `{layer_name}.lora_b` - B matrix [out_features, rank]
    pub fn load_from_gguf(path: impl AsRef<Path>, config: LoraConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let file = GgufFile::open(path.as_ref())?;
        let mut adapters = HashMap::new();

        // Look for LoRA tensors in the file
        // Convention: {layer_name}.lora_a, {layer_name}.lora_b
        let tensors = &file.data.tensors;

        for tensor_info in tensors {
            if tensor_info.name.ends_with(".lora_a") {
                let base_name = tensor_info.name.trim_end_matches(".lora_a");

                // Find corresponding B tensor
                let b_name = format!("{}.lora_b", base_name);
                if let Some(b_info) = tensors.iter().find(|t| t.name == b_name) {
                    // Get raw tensor data
                    if let (Some(a_data), Some(b_data)) = (
                        file.tensor_data(&tensor_info.name),
                        file.tensor_data(&b_name),
                    ) {
                        // Convert to F32 tensors
                        // For now, assume F32 format. A full implementation would handle
                        // different dtypes based on tensor_info.dtype
                        let a_shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
                        let b_shape: Vec<usize> = b_info.dims.iter().map(|&d| d as usize).collect();

                        if let (Ok(a_floats), Ok(b_floats)) = (
                            bytemuck::try_cast_slice::<u8, f32>(a_data),
                            bytemuck::try_cast_slice::<u8, f32>(b_data),
                        ) {
                            if let (Ok(a_tensor), Ok(b_tensor)) = (
                                Tensor::from_f32(a_floats, a_shape),
                                Tensor::from_f32(b_floats, b_shape),
                            ) {
                                let adapter = LoraAdapter::from_tensors(a_tensor, b_tensor, config.scaling());
                                adapters.insert(base_name.to_string(), adapter);
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            adapters,
            config,
            enabled: true,
        })
    }

    /// Add an adapter for a specific weight
    pub fn add_adapter(&mut self, name: &str, adapter: LoraAdapter) {
        self.adapters.insert(name.to_string(), adapter);
    }

    /// Get an adapter by name
    pub fn get(&self, name: &str) -> Option<&LoraAdapter> {
        if self.enabled {
            self.adapters.get(name)
        } else {
            None
        }
    }

    /// Check if an adapter exists for a weight
    pub fn has_adapter(&self, name: &str) -> bool {
        self.enabled && self.adapters.contains_key(name)
    }

    /// Enable all adapters
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable all adapters (use base model only)
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if adapters are enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get total number of adapters
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Get total number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.adapters.values().map(|a| a.num_parameters()).sum()
    }

    /// Get configuration
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// List all adapter names
    pub fn adapter_names(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }
}

/// Apply LoRA to a linear layer output
///
/// This computes: output = base_output + lora.apply(input)
pub fn apply_lora_to_output(
    base_output: &mut Tensor,
    input: &Tensor,
    adapter: &LoraAdapter,
    backend: &dyn Backend,
) -> Result<(), crate::backend::BackendError> {
    let lora_output = adapter.apply(input, backend)?;

    // Add LoRA output to base output
    let base_data = base_output.as_f32_mut()?;
    let lora_data = lora_output.as_f32()?;

    for (b, l) in base_data.iter_mut().zip(lora_data.iter()) {
        *b += *l;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert!((config.alpha - 16.0).abs() < 0.01);
        assert!((config.scaling() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_lora_config_attention() {
        let config = LoraConfig::attention_qkv(16, 32.0);
        assert_eq!(config.rank, 16);
        assert_eq!(config.target_modules.len(), 3);
        assert!(config.target_modules.contains(&"q_proj".to_string()));
    }

    #[test]
    fn test_lora_adapter_creation() {
        let adapter = LoraAdapter::new(512, 512, 8, 2.0);
        assert_eq!(adapter.rank, 8);
        assert_eq!(adapter.lora_a.shape(), &[8, 512]);
        assert_eq!(adapter.lora_b.shape(), &[512, 8]);
        assert_eq!(adapter.num_parameters(), 8 * 512 + 512 * 8);
    }

    #[test]
    fn test_lora_adapter_apply() {
        use crate::backend::cpu::CpuBackend;

        let adapter = LoraAdapter::new(4, 4, 2, 1.0);
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let backend = CpuBackend::new();

        let result = adapter.apply(&input, &backend);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[4]);
    }

    #[test]
    fn test_lora_adapters_collection() {
        let config = LoraConfig::default();
        let mut adapters = LoraAdapters::new(config);

        let adapter = LoraAdapter::new(512, 512, 8, 2.0);
        adapters.add_adapter("layer0.q_proj", adapter);

        assert_eq!(adapters.num_adapters(), 1);
        assert!(adapters.has_adapter("layer0.q_proj"));
        assert!(!adapters.has_adapter("layer0.k_proj"));

        adapters.disable();
        assert!(!adapters.has_adapter("layer0.q_proj"));

        adapters.enable();
        assert!(adapters.has_adapter("layer0.q_proj"));
    }
}
