//! Performance benchmarks for llama-rs
//!
//! Run with: cargo bench

use bytemuck::Zeroable;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use llama_gguf::Backend;
use llama_gguf::backend::cpu::CpuBackend;
use llama_gguf::tensor::{DType, Tensor};
use llama_gguf::tensor::quant::{
    // Legacy blocks and dequant
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1,
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q5_1,
    dequantize_q8_0, dequantize_q8_1,
    // K-quant blocks and dequant
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K,
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_k,
    dequantize_q6_k, dequantize_q8_k,
    // IQ blocks and dequant
    BlockIQ1S, BlockIQ1M, BlockIQ2XXS, BlockIQ2XS, BlockIQ2S,
    BlockIQ3XXS, BlockIQ3S, BlockIQ4XS, BlockIQ4NL,
    dequantize_iq4_nl, dequantize_iq4_xs, dequantize_iq2_xxs, dequantize_iq2_xs,
    dequantize_iq2_s, dequantize_iq3_xxs, dequantize_iq3_s, dequantize_iq1_s, dequantize_iq1_m,
};

/// Benchmark tensor creation and basic operations
fn tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for size in [256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("zeros_f32", size), size, |b, &size| {
            b.iter(|| black_box(Tensor::zeros(vec![size], DType::F32)));
        });

        group.bench_with_input(BenchmarkId::new("from_f32", size), size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            b.iter(|| black_box(Tensor::from_f32(&data, vec![size])));
        });
    }

    group.finish();
}

/// Benchmark matrix-vector multiplication
fn matvec_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("matvec");

    // Common LLM dimensions
    for (m, n) in [(1024, 1024), (2048, 2048), (4096, 4096)].iter() {
        let flops = (*m * *n * 2) as u64; // multiply-add = 2 ops
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(
            BenchmarkId::new("f32", format!("{}x{}", m, n)),
            &(*m, *n),
            |b, &(m, n)| {
                let matrix = Tensor::zeros(vec![m, n], DType::F32);
                let vector = Tensor::zeros(vec![n], DType::F32);
                let mut output = Tensor::zeros(vec![m], DType::F32);
                b.iter(|| {
                    backend.matvec(&matrix, &vector, &mut output).unwrap();
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark matrix-matrix multiplication
fn matmul_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("matmul");

    for size in [128, 256, 512].iter() {
        let flops = (*size * *size * *size * 2) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(BenchmarkId::new("f32", size), size, |b, &size| {
            let a = Tensor::zeros(vec![size, size], DType::F32);
            let b_mat = Tensor::zeros(vec![size, size], DType::F32);
            let mut c = Tensor::zeros(vec![size, size], DType::F32);
            b.iter(|| {
                backend.matmul(&a, &b_mat, &mut c).unwrap();
                black_box(&c);
            });
        });
    }

    group.finish();
}

/// Benchmark softmax operation
fn softmax_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("softmax");

    // Typical vocab sizes
    for size in [32000, 50257, 128256].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("vocab", size), size, |b, &size| {
            let tensor = Tensor::from_f32(
                &(0..size).map(|i| (i as f32) / 1000.0).collect::<Vec<_>>(),
                vec![size],
            )
            .unwrap();
            let mut output = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.softmax(&tensor, &mut output).unwrap();
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Benchmark RMS normalization
fn rms_norm_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("rms_norm");

    for size in [2048, 4096, 8192].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("hidden_dim", size), size, |b, &size| {
            let input = Tensor::from_f32(
                &(0..size).map(|i| (i as f32) / 1000.0).collect::<Vec<_>>(),
                vec![size],
            )
            .unwrap();
            let weights = Tensor::from_f32(&vec![1.0f32; size], vec![size]).unwrap();
            let mut output = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend
                    .rms_norm(&input, &weights, 1e-5, &mut output)
                    .unwrap();
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Benchmark SiLU activation
fn silu_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("silu");

    for size in [4096, 11008, 14336].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("size", size), size, |b, &size| {
            let tensor = Tensor::from_f32(
                &(0..size)
                    .map(|i| ((i as f32) - (size as f32 / 2.0)) / 1000.0)
                    .collect::<Vec<_>>(),
                vec![size],
            )
            .unwrap();
            let mut output = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.silu(&tensor, &mut output).unwrap();
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Benchmark dequantization
fn dequant_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize");

    // Number of blocks
    for n_blocks in [256, 1024, 4096].iter() {
        let n_elements = n_blocks * 32; // Q4_0 has 32 elements per block
        group.throughput(Throughput::Elements(n_elements as u64));

        // Create Q4_0 blocks
        let q4_0_blocks: Vec<BlockQ4_0> = (0..*n_blocks)
            .map(|i| BlockQ4_0 {
                d: half::f16::from_f32(0.1 * (i as f32 + 1.0)),
                qs: [((i * 7) % 256) as u8; 16],
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("q4_0", n_blocks),
            &q4_0_blocks,
            |b, blocks| {
                b.iter(|| {
                    let mut output = [0.0f32; 32];
                    for block in blocks.iter() {
                        dequantize_q4_0(block, &mut output);
                        black_box(&output);
                    }
                });
            },
        );

        // Create Q8_0 blocks
        let q8_0_blocks: Vec<BlockQ8_0> = (0..*n_blocks)
            .map(|i| BlockQ8_0 {
                d: half::f16::from_f32(0.1 * (i as f32 + 1.0)),
                qs: std::array::from_fn(|j| ((i * 7 + j) % 256) as i8),
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("q8_0", n_blocks),
            &q8_0_blocks,
            |b, blocks| {
                b.iter(|| {
                    let mut output = [0.0f32; 32];
                    for block in blocks.iter() {
                        dequantize_q8_0(block, &mut output);
                        black_box(&output);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dot product (SIMD critical path)
fn dot_product_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements((*size * 2) as u64)); // mul + add

        group.bench_with_input(BenchmarkId::new("f32", size), size, |b, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32 / 1000.0).collect();
            let b_vec: Vec<f32> = (0..size).map(|i| (size - i) as f32 / 1000.0).collect();
            b.iter(|| {
                let result: f32 = a.iter().zip(b_vec.iter()).map(|(x, y)| x * y).sum();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise operations
fn elementwise_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("elementwise");

    for size in [4096, 16384, 65536].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("add", size), size, |b, &size| {
            let a = Tensor::from_f32(&vec![1.0f32; size], vec![size]).unwrap();
            let b_tensor = Tensor::from_f32(&vec![2.0f32; size], vec![size]).unwrap();
            let mut out = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.add(&a, &b_tensor, &mut out).unwrap();
                black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("mul", size), size, |b, &size| {
            let a = Tensor::from_f32(&vec![1.5f32; size], vec![size]).unwrap();
            let b_tensor = Tensor::from_f32(&vec![2.5f32; size], vec![size]).unwrap();
            let mut out = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.mul(&a, &b_tensor, &mut out).unwrap();
                black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("scale", size), size, |b, &size| {
            let a = Tensor::from_f32(&vec![1.0f32; size], vec![size]).unwrap();
            let mut out = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.scale(&a, 2.5, &mut out).unwrap();
                black_box(&out);
            });
        });
    }

    group.finish();
}

/// Benchmark legacy dequantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 - 32 elements per block)
fn dequant_legacy_benchmark(c: &mut Criterion) {
    const N_BLOCKS: usize = 1024;
    let mut group = c.benchmark_group("dequant_legacy");
    group.throughput(Throughput::Elements((N_BLOCKS * 32) as u64));

    let blocks_q4_0: Vec<BlockQ4_0> = (0..N_BLOCKS).map(|_| BlockQ4_0::zeroed()).collect();
    let blocks_q4_1: Vec<BlockQ4_1> = (0..N_BLOCKS).map(|_| BlockQ4_1::zeroed()).collect();
    let blocks_q5_0: Vec<BlockQ5_0> = (0..N_BLOCKS).map(|_| BlockQ5_0::zeroed()).collect();
    let blocks_q5_1: Vec<BlockQ5_1> = (0..N_BLOCKS).map(|_| BlockQ5_1::zeroed()).collect();
    let blocks_q8_0: Vec<BlockQ8_0> = (0..N_BLOCKS).map(|_| BlockQ8_0::zeroed()).collect();
    let blocks_q8_1: Vec<BlockQ8_1> = (0..N_BLOCKS).map(|_| BlockQ8_1::zeroed()).collect();

    group.bench_function("q4_0", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_q4_0.iter() {
                dequantize_q4_0(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q4_1", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_q4_1.iter() {
                dequantize_q4_1(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q5_0", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_q5_0.iter() {
                dequantize_q5_0(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q5_1", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_q5_1.iter() {
                dequantize_q5_1(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q8_0", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_q8_0.iter() {
                dequantize_q8_0(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q8_1", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_q8_1.iter() {
                dequantize_q8_1(block, &mut output);
                black_box(&output);
            }
        });
    });

    group.finish();
}

/// Benchmark K-quant dequantization (Q2K, Q3K, Q4K, Q5K, Q6K, Q8K - 256 elements per block)
fn dequant_kquant_benchmark(c: &mut Criterion) {
    const N_BLOCKS: usize = 1024;
    let mut group = c.benchmark_group("dequant_kquant");
    group.throughput(Throughput::Elements((N_BLOCKS * 256) as u64));

    let blocks_q2k: Vec<BlockQ2K> = (0..N_BLOCKS).map(|_| BlockQ2K::zeroed()).collect();
    let blocks_q3k: Vec<BlockQ3K> = (0..N_BLOCKS).map(|_| BlockQ3K::zeroed()).collect();
    let blocks_q4k: Vec<BlockQ4K> = (0..N_BLOCKS).map(|_| BlockQ4K::zeroed()).collect();
    let blocks_q5k: Vec<BlockQ5K> = (0..N_BLOCKS).map(|_| BlockQ5K::zeroed()).collect();
    let blocks_q6k: Vec<BlockQ6K> = (0..N_BLOCKS).map(|_| BlockQ6K::zeroed()).collect();
    let blocks_q8k: Vec<BlockQ8K> = (0..N_BLOCKS).map(|_| BlockQ8K::zeroed()).collect();

    group.bench_function("q2_k", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_q2k.iter() {
                dequantize_q2_k(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q3_k", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_q3k.iter() {
                dequantize_q3_k(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q4_k", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_q4k.iter() {
                dequantize_q4_k(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q5_k", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_q5k.iter() {
                dequantize_q5_k(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q6_k", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_q6k.iter() {
                dequantize_q6_k(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("q8_k", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_q8k.iter() {
                dequantize_q8_k(block, &mut output);
                black_box(&output);
            }
        });
    });

    group.finish();
}

/// Benchmark IQ dequantization (IQ4_NL=32, rest=256 elements per block)
fn dequant_iq_benchmark(c: &mut Criterion) {
    const N_BLOCKS: usize = 1024;
    let mut group = c.benchmark_group("dequant_iq");

    // IQ4_NL: 32 elements per block
    let blocks_iq4_nl: Vec<BlockIQ4NL> = (0..N_BLOCKS).map(|_| BlockIQ4NL::zeroed()).collect();
    group.throughput(Throughput::Elements((N_BLOCKS * 32) as u64));
    group.bench_function("iq4_nl", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 32];
            for block in blocks_iq4_nl.iter() {
                dequantize_iq4_nl(block, &mut output);
                black_box(&output);
            }
        });
    });

    // IQ formats with 256 elements per block
    group.throughput(Throughput::Elements((N_BLOCKS * 256) as u64));
    let blocks_iq4_xs: Vec<BlockIQ4XS> = (0..N_BLOCKS).map(|_| BlockIQ4XS::zeroed()).collect();
    let blocks_iq2_xxs: Vec<BlockIQ2XXS> = (0..N_BLOCKS).map(|_| BlockIQ2XXS::zeroed()).collect();
    let blocks_iq2_xs: Vec<BlockIQ2XS> = (0..N_BLOCKS).map(|_| BlockIQ2XS::zeroed()).collect();
    let blocks_iq2_s: Vec<BlockIQ2S> = (0..N_BLOCKS).map(|_| BlockIQ2S::zeroed()).collect();
    let blocks_iq3_xxs: Vec<BlockIQ3XXS> = (0..N_BLOCKS).map(|_| BlockIQ3XXS::zeroed()).collect();
    let blocks_iq3_s: Vec<BlockIQ3S> = (0..N_BLOCKS).map(|_| BlockIQ3S::zeroed()).collect();
    let blocks_iq1_s: Vec<BlockIQ1S> = (0..N_BLOCKS).map(|_| BlockIQ1S::zeroed()).collect();
    let blocks_iq1_m: Vec<BlockIQ1M> = (0..N_BLOCKS).map(|_| BlockIQ1M::zeroed()).collect();

    group.bench_function("iq4_xs", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq4_xs.iter() {
                dequantize_iq4_xs(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq2_xxs", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq2_xxs.iter() {
                dequantize_iq2_xxs(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq2_xs", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq2_xs.iter() {
                dequantize_iq2_xs(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq2_s", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq2_s.iter() {
                dequantize_iq2_s(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq3_xxs", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq3_xxs.iter() {
                dequantize_iq3_xxs(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq3_s", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq3_s.iter() {
                dequantize_iq3_s(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq1_s", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq1_s.iter() {
                dequantize_iq1_s(block, &mut output);
                black_box(&output);
            }
        });
    });
    group.bench_function("iq1_m", |b| {
        b.iter(|| {
            let mut output = [0.0f32; 256];
            for block in blocks_iq1_m.iter() {
                dequantize_iq1_m(block, &mut output);
                black_box(&output);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    tensor_creation,
    matvec_benchmark,
    matmul_benchmark,
    softmax_benchmark,
    rms_norm_benchmark,
    silu_benchmark,
    dequant_benchmark,
    dequant_legacy_benchmark,
    dequant_kquant_benchmark,
    dequant_iq_benchmark,
    dot_product_benchmark,
    elementwise_benchmark,
);
criterion_main!(benches);
