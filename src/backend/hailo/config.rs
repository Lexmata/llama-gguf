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
    if let Some(base) = directories::BaseDirs::new() {
        base.cache_dir().join("llama-gguf").join("hailo")
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
