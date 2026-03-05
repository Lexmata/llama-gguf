//! Tests the full GGUF pipeline without external dependencies.
//! Builds synthetic GGUF files in-process, writes to tempfiles, loads them back,
//! and verifies metadata and tensor round-tripping.

use bytemuck;
use llama_gguf::gguf::{GgmlType, GgufWriter, TensorToWrite};
use llama_gguf::gguf::{MetadataArray, MetadataValue};
use llama_gguf::GgufFile;
use tempfile::NamedTempFile;

fn create_temp_gguf_path() -> NamedTempFile {
    NamedTempFile::new().expect("create temp file")
}

#[test]
fn test_gguf_roundtrip_metadata() {
    let temp = create_temp_gguf_path();
    let path = temp.path();

    {
        let mut writer = GgufWriter::create(path).expect("create writer");
        writer.add_string("general.architecture", "llama");
        writer.add_u32("llama.block_count", 2);
        writer.add_f32("test.float", 3.14);
        writer.add_bool("test.flag", true);
        writer.add_u64("test.big", 0x1234_5678_9abc_def0);
        writer.write().expect("write gguf");
    }

    let gguf = GgufFile::open(path).expect("open gguf");
    assert_eq!(gguf.data.get_string("general.architecture"), Some("llama"));
    assert_eq!(gguf.data.get_u32("llama.block_count"), Some(2));
    assert_eq!(gguf.data.get_f32("test.float"), Some(3.14));
    assert_eq!(gguf.data.get_bool("test.flag"), Some(true));
    assert_eq!(gguf.data.get_u64("test.big"), Some(0x1234_5678_9abc_def0));
}

#[test]
fn test_gguf_roundtrip_f32_tensor() {
    let temp = create_temp_gguf_path();
    let path = temp.path();

    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_bytes = bytemuck::cast_slice::<f32, u8>(&expected).to_vec();

    {
        let mut writer = GgufWriter::create(path).expect("create writer");
        writer.add_string("general.architecture", "llama");
        writer.add_tensor(TensorToWrite::new(
            "test.weight",
            vec![2, 4],
            GgmlType::F32,
            data_bytes,
        ));
        writer.write().expect("write gguf");
    }

    let gguf = GgufFile::open(path).expect("open gguf");
    let tensor_info = gguf.data.get_tensor("test.weight").expect("tensor exists");
    assert_eq!(tensor_info.dims, vec![2, 4]);
    assert_eq!(tensor_info.dtype, GgmlType::F32);

    let data = gguf.tensor_data("test.weight").expect("tensor data");
    let actual: &[f32] = bytemuck::cast_slice(data);
    assert_eq!(actual, expected.as_slice());
}

#[test]
fn test_gguf_roundtrip_multiple_tensors() {
    let temp = create_temp_gguf_path();
    let path = temp.path();

    let data_a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..8).map(|i| (i + 10) as f32).collect();

    {
        let mut writer = GgufWriter::create(path).expect("create writer");
        writer.add_string("general.architecture", "llama");
        writer.add_tensor(TensorToWrite::new(
            "tensor.a",
            vec![2, 3],
            GgmlType::F32,
            bytemuck::cast_slice::<f32, u8>(&data_a).to_vec(),
        ));
        writer.add_tensor(TensorToWrite::new(
            "tensor.b",
            vec![4, 2],
            GgmlType::F32,
            bytemuck::cast_slice::<f32, u8>(&data_b).to_vec(),
        ));
        writer.write().expect("write gguf");
    }

    let gguf = GgufFile::open(path).expect("open gguf");

    let info_a = gguf.data.get_tensor("tensor.a").expect("tensor a");
    assert_eq!(info_a.dims, vec![2, 3]);
    let read_a: &[f32] = bytemuck::cast_slice(gguf.tensor_data("tensor.a").unwrap());
    assert_eq!(read_a, data_a.as_slice());

    let info_b = gguf.data.get_tensor("tensor.b").expect("tensor b");
    assert_eq!(info_b.dims, vec![4, 2]);
    let read_b: &[f32] = bytemuck::cast_slice(gguf.tensor_data("tensor.b").unwrap());
    assert_eq!(read_b, data_b.as_slice());
}

#[test]
fn test_embedded_mini_model() {
    let temp = create_temp_gguf_path();
    let path = temp.path();

    const VOCAB_SIZE: usize = 5;
    const EMBED_LEN: usize = 64;
    const FFN_LEN: usize = 128;

    let tokens = ["<unk>", "<s>", "</s>", "hello", "world"];
    let scores: Vec<f32> = vec![0.0, 0.0, 0.0, -1.0, -2.0];

    {
        let mut writer = GgufWriter::create(path).expect("create writer");

        // LLaMA metadata
        writer.add_string("general.architecture", "llama");
        writer.add_u32("llama.embedding_length", EMBED_LEN as u32);
        writer.add_u32("llama.block_count", 1);
        writer.add_u32("llama.attention.head_count", 2);
        writer.add_u32("llama.attention.head_count_kv", 2);
        writer.add_u32("llama.feed_forward_length", FFN_LEN as u32);

        // Tokenizer metadata
        writer.add_string("tokenizer.ggml.model", "llama");
        writer.add_metadata(
            "tokenizer.ggml.tokens",
            MetadataValue::Array(MetadataArray {
                values: tokens
                    .iter()
                    .map(|t| MetadataValue::String(t.to_string()))
                    .collect(),
            }),
        );
        writer.add_metadata(
            "tokenizer.ggml.scores",
            MetadataValue::Array(MetadataArray {
                values: scores
                    .iter()
                    .map(|s| MetadataValue::Float32(*s))
                    .collect(),
            }),
        );

        // Helper to create F32 tensor bytes
        fn f32_tensor(dims: &[u64], fill: f32) -> Vec<u8> {
            let n: usize = dims.iter().map(|&d| d as usize).product();
            let data: Vec<f32> = vec![fill; n];
            bytemuck::cast_slice::<f32, u8>(&data).to_vec()
        }

        // Embedding and output
        writer.add_tensor(TensorToWrite::new(
            "token_embd.weight",
            vec![VOCAB_SIZE as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[VOCAB_SIZE as u64, EMBED_LEN as u64], 0.01),
        ));
        writer.add_tensor(TensorToWrite::new(
            "output.weight",
            vec![VOCAB_SIZE as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[VOCAB_SIZE as u64, EMBED_LEN as u64], 0.02),
        ));
        writer.add_tensor(TensorToWrite::new(
            "output_norm.weight",
            vec![EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64], 1.0),
        ));

        // Block 0 attention
        writer.add_tensor(TensorToWrite::new(
            "blk.0.attn_q.weight",
            vec![EMBED_LEN as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64, EMBED_LEN as u64], 0.1),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.attn_k.weight",
            vec![EMBED_LEN as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64, EMBED_LEN as u64], 0.11),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.attn_v.weight",
            vec![EMBED_LEN as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64, EMBED_LEN as u64], 0.12),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.attn_output.weight",
            vec![EMBED_LEN as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64, EMBED_LEN as u64], 0.13),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.attn_norm.weight",
            vec![EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64], 1.0),
        ));

        // Block 0 FFN
        writer.add_tensor(TensorToWrite::new(
            "blk.0.ffn_gate.weight",
            vec![FFN_LEN as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[FFN_LEN as u64, EMBED_LEN as u64], 0.2),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.ffn_up.weight",
            vec![FFN_LEN as u64, EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[FFN_LEN as u64, EMBED_LEN as u64], 0.21),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.ffn_down.weight",
            vec![EMBED_LEN as u64, FFN_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64, FFN_LEN as u64], 0.22),
        ));
        writer.add_tensor(TensorToWrite::new(
            "blk.0.ffn_norm.weight",
            vec![EMBED_LEN as u64],
            GgmlType::F32,
            f32_tensor(&[EMBED_LEN as u64], 1.0),
        ));

        writer.write().expect("write gguf");
    }

    let gguf = GgufFile::open(path).expect("open gguf");

    // Verify metadata
    assert_eq!(
        gguf.data.get_string("general.architecture"),
        Some("llama")
    );
    assert_eq!(gguf.data.get_u32("llama.embedding_length"), Some(64));
    assert_eq!(gguf.data.get_u32("llama.block_count"), Some(1));
    assert_eq!(gguf.data.get_u32("llama.attention.head_count"), Some(2));
    assert_eq!(gguf.data.get_u32("llama.attention.head_count_kv"), Some(2));
    assert_eq!(gguf.data.get_u32("llama.feed_forward_length"), Some(128));
    assert_eq!(gguf.data.get_string("tokenizer.ggml.model"), Some("llama"));

    // Verify token array
    let tokens_value = gguf.data.metadata.get("tokenizer.ggml.tokens").unwrap();
    if let MetadataValue::Array(arr) = tokens_value {
        assert_eq!(arr.values.len(), 5);
        for (i, v) in arr.values.iter().enumerate() {
            if let MetadataValue::String(s) = v {
                assert_eq!(s, tokens[i]);
            }
        }
    } else {
        panic!("expected token array");
    }

    // Verify scores array
    let scores_value = gguf.data.metadata.get("tokenizer.ggml.scores").unwrap();
    if let MetadataValue::Array(arr) = scores_value {
        assert_eq!(arr.values.len(), 5);
        for (i, v) in arr.values.iter().enumerate() {
            if let MetadataValue::Float32(f) = v {
                assert!((*f - scores[i]).abs() < 1e-6);
            }
        }
    } else {
        panic!("expected scores array");
    }

    // Verify tensor shapes
    let expected_tensors = [
        ("token_embd.weight", vec![5u64, 64]),
        ("output.weight", vec![5u64, 64]),
        ("output_norm.weight", vec![64]),
        ("blk.0.attn_q.weight", vec![64, 64]),
        ("blk.0.attn_k.weight", vec![64, 64]),
        ("blk.0.attn_v.weight", vec![64, 64]),
        ("blk.0.attn_output.weight", vec![64, 64]),
        ("blk.0.attn_norm.weight", vec![64]),
        ("blk.0.ffn_gate.weight", vec![128, 64]),
        ("blk.0.ffn_up.weight", vec![128, 64]),
        ("blk.0.ffn_down.weight", vec![64, 128]),
        ("blk.0.ffn_norm.weight", vec![64]),
    ];

    for (name, expected_dims) in expected_tensors {
        let info = gguf.data.get_tensor(name).expect(&format!("tensor {name}"));
        assert_eq!(info.dims, expected_dims, "tensor {name}");
        assert_eq!(info.dtype, GgmlType::F32);
        assert!(
            gguf.tensor_data(name).is_some(),
            "tensor data for {name}"
        );
    }
}
