#![cfg(feature = "hailo")]

use llama_gguf::backend::hailo::{
    check_cache, model_config_hash, resolve_hef_path, HailoBackend, HailoConfig, HailoQuantization,
    HefManifest,
};
use llama_gguf::backend::{Backend, BackendError};
use llama_gguf::model::ModelConfig;
use llama_gguf::tensor::DType;
use std::path::PathBuf;
use tempfile::tempdir;

fn has_hailo_device() -> bool {
    llama_gguf::backend::hailo::check_device_available().is_ok()
}

#[test]
fn test_hailo_config_default() {
    let config = HailoConfig::default();
    assert!(!config.auto_compile);
    assert!(config.hef_dir.is_none());
    assert!(config.dfc_python.is_none());
    assert!(config.target_arch.is_none());
}

#[test]
fn test_hailo_quantization_default() {
    let q = HailoQuantization::default();
    assert!(matches!(q, HailoQuantization::Int8));
}

#[test]
fn test_hef_manifest_fields() {
    let manifest = HefManifest {
        model_config_hash: 12345,
        hidden_size: 768,
        num_heads: 12,
        num_kv_heads: 12,
        head_dim: 64,
        intermediate_size: 3072,
        num_layers: 12,
        target_arch: "hailo8l".to_string(),
    };
    assert_eq!(manifest.model_config_hash, 12345);
    assert_eq!(manifest.hidden_size, 768);
    assert_eq!(manifest.target_arch, "hailo8l");
}

#[test]
fn test_model_config_hash_deterministic() {
    let config = ModelConfig {
        hidden_size: 512,
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 64,
        intermediate_size: 2048,
        num_layers: 6,
        norm_eps: 1e-5,
        ..Default::default()
    };
    let h1 = model_config_hash(&config);
    let h2 = model_config_hash(&config);
    assert_eq!(h1, h2);
}

#[test]
fn test_model_config_hash_changes() {
    let base = ModelConfig {
        hidden_size: 512,
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 64,
        intermediate_size: 2048,
        num_layers: 6,
        norm_eps: 1e-5,
        ..Default::default()
    };
    let h_base = model_config_hash(&base);

    let mut diff_hidden = base.clone();
    diff_hidden.hidden_size = 768;
    assert_ne!(model_config_hash(&diff_hidden), h_base);

    let mut diff_layers = base.clone();
    diff_layers.num_layers = 8;
    assert_ne!(model_config_hash(&diff_layers), h_base);

    let mut diff_eps = base.clone();
    diff_eps.norm_eps = 1e-6;
    assert_ne!(model_config_hash(&diff_eps), h_base);
}

#[test]
fn test_resolve_hef_path() {
    let cache = PathBuf::from("/tmp/hef_cache");
    let config = HailoConfig {
        cache_dir: cache.clone(),
        hef_dir: None,
        ..Default::default()
    };

    let attn_0 = resolve_hef_path(&config, 0, "attn");
    assert_eq!(attn_0, PathBuf::from("/tmp/hef_cache/layer_00_attn.hef"));

    let ffn_1 = resolve_hef_path(&config, 1, "ffn");
    assert_eq!(ffn_1, PathBuf::from("/tmp/hef_cache/layer_01_ffn.hef"));

    let hef_dir = PathBuf::from("/custom/hef");
    let config_custom = HailoConfig {
        cache_dir: cache,
        hef_dir: Some(hef_dir.clone()),
        ..Default::default()
    };
    let attn_custom = resolve_hef_path(&config_custom, 5, "attn");
    assert_eq!(attn_custom, PathBuf::from("/custom/hef/layer_05_attn.hef"));
}

#[test]
fn test_check_cache_empty_dir() {
    let dir = tempdir().unwrap();
    let config = HailoConfig {
        cache_dir: dir.path().to_path_buf(),
        hef_dir: None,
        ..Default::default()
    };
    let model_config = ModelConfig {
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 64,
        intermediate_size: 1024,
        num_layers: 2,
        norm_eps: 1e-5,
        ..Default::default()
    };
    assert!(!check_cache(&config, &model_config));
}

#[test]
fn test_hailo_backend_name() {
    if !has_hailo_device() {
        eprintln!("Skipping: no Hailo device");
        return;
    }
    let backend = HailoBackend::with_config(HailoConfig::default()).unwrap();
    assert_eq!(backend.name(), "hailo");
}

#[test]
fn test_hailo_backend_ops_unsupported() {
    if !has_hailo_device() {
        eprintln!("Skipping: no Hailo device");
        return;
    }
    let backend = HailoBackend::with_config(HailoConfig::default()).unwrap();

    let result = backend.alloc(&[4, 8], DType::F32);
    assert!(matches!(result, Err(BackendError::Unsupported(_))));

    let tensor = llama_gguf::tensor::Tensor::zeros(vec![4, 8], DType::F32);
    let result = backend.copy_to(&tensor);
    assert!(matches!(result, Err(BackendError::Unsupported(_))));
}

#[test]
fn test_hailo_context_creation() {
    if !has_hailo_device() {
        eprintln!("Skipping: no Hailo device");
        return;
    }
    let ctx = llama_gguf::backend::hailo::HailoContext::new().unwrap();
    let info = ctx.device_info();
    assert!(info.num_devices > 0);
    assert!(!info.library_version.is_empty());
}
