//! DX12 backend integration tests
//!
//! These tests verify that the DX12 backend can:
//! 1. Initialize and detect DX12 devices
//! 2. Run compute operations through DX12 shaders
//! 3. Load and run inference on a real GGUF model
//!
//! These tests are designed to run on Windows hardware with a DX12-capable GPU.
//! WARP (Windows Advanced Rasterization Platform) is available on all Windows 10+
//! machines including CI runners.
//!
//! Run with:
//!   cargo test --features dx12 --test dx12_integration -- --nocapture
//!
//! For model loading tests, set the MODEL_PATH environment variable:
//!   MODEL_PATH=C:\path\to\model.gguf cargo test --features dx12 --test dx12_integration

#[cfg(all(feature = "dx12", target_os = "windows"))]
mod tests {
    use llama_gguf::backend::Backend;
    use llama_gguf::backend::dx12::{Dx12Backend, Dx12Config};
    use llama_gguf::tensor::{DType, Tensor};

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {} differs: dx12={} expected={} (diff={}, tol={})",
                i,
                x,
                y,
                (x - y).abs(),
                tol,
            );
        }
    }

    // =========================================================================
    // Device discovery
    // =========================================================================

    #[test]
    fn test_dx12_device_discovery() {
        let devices = Dx12Backend::enumerate_devices();
        assert!(
            !devices.is_empty(),
            "No DX12 devices found (not even WARP?)"
        );

        println!("DX12 devices found: {}", devices.len());
        for (i, dev) in devices.iter().enumerate() {
            println!("  [{}] {}", i, dev.name);
            println!("      Type:             {:?}", dev.device_type);
            println!(
                "      Dedicated VRAM:   {:.0} MB",
                dev.dedicated_video_memory as f64 / 1024.0 / 1024.0
            );
            println!(
                "      Shared memory:    {:.0} MB",
                dev.shared_system_memory as f64 / 1024.0 / 1024.0
            );
            println!("      Feature level:    {}", dev.feature_level);
        }
    }

    // =========================================================================
    // Backend initialization
    // =========================================================================

    #[test]
    fn test_dx12_backend_init() {
        let backend = Dx12Backend::new().expect("Failed to create DX12 backend");
        assert_eq!(backend.name(), "dx12");
        assert!(backend.is_available());
        println!("DX12 backend initialized: {}", backend.device_name());
    }

    #[test]
    fn test_dx12_backend_config() {
        let config = Dx12Config {
            device_index: 0,
            prefer_warp: false,
        };
        let backend = Dx12Backend::with_config(config).expect("Failed with config");
        assert_eq!(backend.name(), "dx12");
    }

    #[test]
    fn test_dx12_backend_warp() {
        // WARP should always be available on Windows 10+
        let config = Dx12Config {
            device_index: 0,
            prefer_warp: true,
        };
        match Dx12Backend::with_config(config) {
            Ok(backend) => {
                assert_eq!(backend.name(), "dx12");
                println!("WARP backend: {}", backend.device_name());
            }
            Err(e) => {
                println!("WARP not available: {} (expected on older Windows)", e);
            }
        }
    }

    // =========================================================================
    // Compute operations - comprehensive correctness tests
    // =========================================================================

    #[test]
    fn test_dx12_add_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        for n in [1, 4, 17, 256, 1024, 4096] {
            let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.71).cos()).collect();

            let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
            let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

            let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
            let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

            backend.add(&a, &b, &mut dx12_out).unwrap();
            cpu.add(&a, &b, &mut cpu_out).unwrap();

            assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-5);
            println!("  add n={}: PASS", n);
        }
    }

    #[test]
    fn test_dx12_mul_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.29).cos()).collect();

        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![n]).unwrap();

        let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.mul(&a, &b, &mut dx12_out).unwrap();
        cpu.mul(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-5);
    }

    #[test]
    fn test_dx12_scale_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let a = Tensor::from_f32(&a_data, vec![n]).unwrap();

        let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.scale(&a, 2.718, &mut dx12_out).unwrap();
        cpu.scale(&a, 2.718, &mut cpu_out).unwrap();

        assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
    }

    #[test]
    fn test_dx12_silu_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();

        let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.silu(&x, &mut dx12_out).unwrap();
        cpu.silu(&x, &mut cpu_out).unwrap();

        assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
    }

    #[test]
    fn test_dx12_gelu_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 2048;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();

        let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.gelu(&x, &mut dx12_out).unwrap();
        cpu.gelu(&x, &mut cpu_out).unwrap();

        assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-4);
    }

    #[test]
    fn test_dx12_rms_norm_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let n = 4096;
        let x_data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let w_data = vec![1.0f32; n];

        let x = Tensor::from_f32(&x_data, vec![n]).unwrap();
        let weight = Tensor::from_f32(&w_data, vec![n]).unwrap();

        let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.rms_norm(&x, &weight, 1e-5, &mut dx12_out).unwrap();
        cpu.rms_norm(&x, &weight, 1e-5, &mut cpu_out).unwrap();

        assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-3);
    }

    #[test]
    fn test_dx12_vec_mat_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let k = 256;
        let n = 1024;
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let a = Tensor::from_f32(&a_data, vec![k]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![k, n]).unwrap();

        let mut dx12_out = Tensor::zeros(vec![n], DType::F32);
        let mut cpu_out = Tensor::zeros(vec![n], DType::F32);

        backend.vec_mat(&a, &b, &mut dx12_out).unwrap();
        cpu.vec_mat(&a, &b, &mut cpu_out).unwrap();

        assert_approx_eq(dx12_out.as_f32().unwrap(), cpu_out.as_f32().unwrap(), 1e-2);
    }

    #[test]
    fn test_dx12_rope_correctness() {
        let backend = Dx12Backend::new().unwrap();
        let cpu = llama_gguf::backend::cpu::CpuBackend::new();

        let head_dim = 64;
        let num_heads = 8;
        let n = num_heads * head_dim;

        let q_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let k_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.02).collect();

        let mut q_dx12 = Tensor::from_f32(&q_data, vec![num_heads, 1, head_dim]).unwrap();
        let mut k_dx12 = Tensor::from_f32(&k_data, vec![num_heads, 1, head_dim]).unwrap();
        let mut q_cpu = Tensor::from_f32(&q_data, vec![num_heads, 1, head_dim]).unwrap();
        let mut k_cpu = Tensor::from_f32(&k_data, vec![num_heads, 1, head_dim]).unwrap();

        backend
            .rope(&mut q_dx12, &mut k_dx12, 42, 10000.0, 1.0, false)
            .unwrap();
        cpu.rope(&mut q_cpu, &mut k_cpu, 42, 10000.0, 1.0, false)
            .unwrap();

        assert_approx_eq(q_dx12.as_f32().unwrap(), q_cpu.as_f32().unwrap(), 1e-3);
        assert_approx_eq(k_dx12.as_f32().unwrap(), k_cpu.as_f32().unwrap(), 1e-3);
    }

    // =========================================================================
    // Stress tests
    // =========================================================================

    #[test]
    fn test_dx12_repeated_operations_no_leak() {
        let backend = Dx12Backend::new().unwrap();

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
    fn test_dx12_gguf_model_loading() {
        // This test requires a GGUF model file. Set MODEL_PATH env var.
        let model_path = match std::env::var("MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                // Check default test model location
                let default_path = format!(
                    "{}\\.test-models\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                    env!("CARGO_MANIFEST_DIR")
                );
                if std::path::Path::new(&default_path).exists() {
                    default_path
                } else {
                    println!(
                        "Skipping model loading test: no MODEL_PATH set and no default model found."
                    );
                    println!(
                        "Run with: MODEL_PATH=C:\\path\\to\\model.gguf cargo test --features dx12 --test dx12_integration"
                    );
                    return;
                }
            }
        };

        println!("Loading GGUF model: {}", model_path);

        // Test 1: Load the engine with DX12 backend
        let config = llama_gguf::EngineConfig {
            model_path: model_path.clone(),
            use_gpu: true,
            max_tokens: 32,
            temperature: 0.1,
            ..Default::default()
        };

        let engine = llama_gguf::Engine::load(config).expect("Failed to load model with DX12");

        println!("Model loaded successfully on DX12 backend");
        println!("  Architecture: {:?}", engine.model_config().num_layers);
        println!("  Hidden size:  {}", engine.model_config().hidden_size);
        println!("  Num heads:    {}", engine.model_config().num_heads);
        println!("  Chat template: {:?}", engine.chat_template());

        // Test 2: Generate a few tokens to verify inference works
        println!("Generating test tokens...");
        let result = engine
            .generate("Hello", 16)
            .expect("Failed to generate with DX12 backend");

        println!("Generated: \"{}\"", result);
        assert!(!result.is_empty(), "DX12 backend produced empty output");

        println!("GGUF model loading and inference on DX12: PASSED");
    }
}

/// On non-Windows platforms, these tests are no-ops.
#[cfg(not(all(feature = "dx12", target_os = "windows")))]
mod tests {
    #[test]
    fn test_dx12_not_available_on_this_platform() {
        println!("DX12 tests skipped: not on Windows or dx12 feature not enabled");
        println!("Run on Windows with: cargo test --features dx12 --test dx12_integration");
    }
}
