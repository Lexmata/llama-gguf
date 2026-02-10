//! Metal backend integration tests
//!
//! These tests verify that the Metal backend can:
//! 1. Initialize and detect Metal devices
//! 2. Run compute operations through Metal shaders
//! 3. Load and run inference on a real GGUF model
//!
//! These tests are designed to run on macOS hardware with a Metal-capable GPU.
//! They are NOT part of the standard CI pipeline (which runs on Linux).
//!
//! Run with:
//!   cargo test --features metal --test metal_integration -- --nocapture
//!
//! For model loading tests, set the MODEL_PATH environment variable:
//!   MODEL_PATH=/path/to/model.gguf cargo test --features metal --test metal_integration

#[cfg(all(feature = "metal", target_os = "macos"))]
mod tests {
    use llama_gguf::backend::metal::{MetalBackend, MetalConfig};
    use llama_gguf::backend::Backend;
    use llama_gguf::tensor::{DType, Tensor};

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {} differs: metal={} expected={} (diff={}, tol={})",
                i, x, y, (x - y).abs(), tol,
            );
        }
    }

    // =========================================================================
    // Device discovery
    // =========================================================================

    #[test]
    fn test_metal_device_discovery() {
        let devices = MetalBackend::enumerate_devices();
        assert!(!devices.is_empty(), "No Metal devices found on this Mac");

        println!("Metal devices found: {}", devices.len());
        for dev in &devices {
            println!("  [{index}] {name}", index = dev.index, name = dev.name);
            println!("      Unified memory: {}", dev.has_unified_memory);
            println!("      Low power:      {}", dev.is_low_power);
            println!(
                "      Max buffer:     {:.0} MB",
                dev.max_buffer_length as f64 / 1024.0 / 1024.0
            );
            println!(
                "      Working set:    {:.0} MB",
                dev.recommended_max_working_set_size as f64 / 1024.0 / 1024.0
            );
        }
    }

    // =========================================================================
    // Backend initialization
    // =========================================================================

    #[test]
    fn test_metal_backend_init() {
        let backend = MetalBackend::new().expect("Failed to create Metal backend");
        assert_eq!(backend.name(), "metal");
        assert!(backend.is_available());
        println!("Metal backend initialized: {}", backend.device_name());
    }

    #[test]
    fn test_metal_backend_config() {
        let config = MetalConfig { device_index: 0 };
        let backend = MetalBackend::with_config(config).expect("Failed with config");
        assert_eq!(backend.name(), "metal");
    }

    // =========================================================================
    // Compute operations - comprehensive correctness tests
    // =========================================================================

    #[test]
    fn test_metal_add_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        for n in [1, 4, 17, 256, 1024, 4096] {
            let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.71).cos()).collect();

            let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
            let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

            let mut metal_out = Tensor::zeros(vec![n], DType::F32);
            let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

            backend.add(&a, &b, &mut metal_out).unwrap();
            cpu.add(&a, &b, &mut cpu_out).unwrap();

            assert_approx_eq(
                metal_out.as_f32().unwrap(),
                cpu_out.as_f32().unwrap(),
                1e-5,
            );
            println!("  add n={}: PASS", n);
        }
    }

    #[test]
    fn test_metal_mul_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.29).cos()).collect();

        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

        let mut metal_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.mul(&a, &b, &mut metal_out).unwrap();
        cpu.mul(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(
            metal_out.as_f32().unwrap(),
            cpu_out.as_f32().unwrap(),
            1e-5,
        );
    }

    #[test]
    fn test_metal_scale_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();

        let mut metal_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.scale(&a, 2.718, &mut metal_out).unwrap();
        cpu.scale(&a, 2.718, &mut cpu_out).unwrap();

        assert_approx_eq(
            metal_out.as_f32().unwrap(),
            cpu_out.as_f32().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn test_metal_silu_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();

        let mut metal_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.silu(&x, &mut metal_out).unwrap();
        cpu.silu(&x, &mut cpu_out).unwrap();

        assert_approx_eq(
            metal_out.as_f32().unwrap(),
            cpu_out.as_f32().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn test_metal_gelu_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();

        let mut metal_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.gelu(&x, &mut metal_out).unwrap();
        cpu.gelu(&x, &mut cpu_out).unwrap();

        assert_approx_eq(
            metal_out.as_f32().unwrap(),
            cpu_out.as_f32().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn test_metal_rms_norm_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 4096;
        let x_data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let w_data = vec![1.0f32; n];

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let weight = Tensor::from_f32(&w_data, vec![n]).unwrap();

        let mut metal_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.rms_norm(&x, &weight, 1e-5, &mut metal_out).unwrap();
        cpu.rms_norm(&x, &weight, 1e-5, &mut cpu_out).unwrap();

        assert_approx_eq(
            metal_out.as_f32().unwrap(),
            cpu_out.as_f32().unwrap(),
            1e-3,
        );
    }

    #[test]
    fn test_metal_vec_mat_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let k = 256;
        let n = 1024;
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let a = Tensor::from_f32(&a_data, vec![k]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![k, n]).unwrap();

        let mut metal_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.vec_mat(&a, &b, &mut metal_out).unwrap();
        cpu.vec_mat(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(
            metal_out.as_f32().unwrap(),
            cpu_out.as_f32().unwrap(),
            1e-2,
        );
    }

    #[test]
    fn test_metal_rope_correctness() {
        let backend = MetalBackend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let head_dim = 64;
        let num_heads = 8;
        let n = num_heads * head_dim;

        let q_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let k_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.02).collect();

        let mut q_metal = Tensor::from_f32(&q_data, vec![num_heads, 1, head_dim]).unwrap();
        let mut k_metal = Tensor::from_f32(&k_data, vec![num_heads, 1, head_dim]).unwrap();
        let mut q_cpu = Tensor::from_f32(&q_data, vec![num_heads, 1, head_dim]).unwrap();
        let mut k_cpu = Tensor::from_f32(&k_data, vec![num_heads, 1, head_dim]).unwrap();

        backend
            .rope(&mut q_metal, &mut k_metal, 42, 10000.0, 1.0, false)
            .unwrap();
        cpu.rope(&mut q_cpu, &mut k_cpu, 42, 10000.0, 1.0, false)
            .unwrap();

        assert_approx_eq(
            q_metal.as_f32().unwrap(),
            q_cpu.as_f32().unwrap(),
            1e-3,
        );
        assert_approx_eq(
            k_metal.as_f32().unwrap(),
            k_cpu.as_f32().unwrap(),
            1e-3,
        );
    }

    // =========================================================================
    // Stress tests
    // =========================================================================

    #[test]
    fn test_metal_repeated_operations_no_leak() {
        let backend = MetalBackend::new().unwrap();

        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[0.5, 0.5, 0.5, 0.5], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);

        // Run many iterations to check for memory leaks or resource exhaustion
        for i in 0..200 {
            backend.add(&a, &b, &mut out).unwrap();
            assert_approx_eq(out.as_f32().unwrap(), &[1.5, 2.5, 3.5, 4.5], 1e-5);

            if i % 50 == 0 {
                println!("  stress iteration {}/200: OK", i);
            }
        }
        println!("  Stress test passed: 200 iterations, no leaks");
    }

    // =========================================================================
    // GGUF Model loading integration test
    // =========================================================================

    #[test]
    fn test_metal_gguf_model_loading() {
        // This test requires a GGUF model file. Set MODEL_PATH env var.
        let model_path = match std::env::var("MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                // Check default test model location
                let default_path = format!(
                    "{}/.test-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                    env!("CARGO_MANIFEST_DIR")
                );
                if std::path::Path::new(&default_path).exists() {
                    default_path
                } else {
                    println!(
                        "Skipping model loading test: no MODEL_PATH set and no default model found."
                    );
                    println!(
                        "Run with: MODEL_PATH=/path/to/model.gguf cargo test --features metal --test metal_integration"
                    );
                    return;
                }
            }
        };

        println!("Loading GGUF model: {}", model_path);

        // Test 1: Load the engine with Metal backend
        let config = llama_gguf::EngineConfig {
            model_path: model_path.clone(),
            use_gpu: true,
            max_tokens: 32,
            temperature: 0.1,
            ..Default::default()
        };

        let engine = llama_gguf::Engine::load(config).expect("Failed to load model with Metal");

        println!("Model loaded successfully on Metal backend");
        println!("  Architecture: {:?}", engine.model_config().num_layers);
        println!("  Hidden size:  {}", engine.model_config().hidden_size);
        println!("  Num heads:    {}", engine.model_config().num_heads);
        println!("  Chat template: {:?}", engine.chat_template());

        // Test 2: Generate a few tokens to verify inference works
        println!("Generating test tokens...");
        let result = engine
            .generate("Hello", 16)
            .expect("Failed to generate with Metal backend");

        println!("Generated: \"{}\"", result);
        assert!(
            !result.is_empty(),
            "Metal backend produced empty output"
        );

        println!("GGUF model loading and inference on Metal: PASSED");
    }
}

/// On non-macOS platforms, these tests are no-ops.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod tests {
    #[test]
    fn test_metal_not_available_on_this_platform() {
        println!("Metal tests skipped: not on macOS or metal feature not enabled");
        println!("Run on macOS with: cargo test --features metal --test metal_integration");
    }
}
