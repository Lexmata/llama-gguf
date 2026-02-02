use std::io::Cursor;
use llama_rs::gguf::{GgufReader, GgufError, GGUF_MAGIC};

fn create_minimal_gguf_v3() -> Vec<u8> {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 3
    data.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count: 0
    data.extend_from_slice(&0u64.to_le_bytes());
    // Metadata count: 1
    data.extend_from_slice(&1u64.to_le_bytes());
    // Metadata: "general.architecture" = "llama"
    let key = b"general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    // Type: String (8)
    data.extend_from_slice(&8u32.to_le_bytes());
    let value = b"llama";
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value);
    data
}

fn create_minimal_gguf_v1() -> Vec<u8> {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 1
    data.extend_from_slice(&1u32.to_le_bytes());
    // Tensor count: 0 (u32 for v1)
    data.extend_from_slice(&0u32.to_le_bytes());
    // Metadata count: 1 (u32 for v1)
    data.extend_from_slice(&1u32.to_le_bytes());
    // Metadata: "test.key" = 42 (u32)
    let key = b"test.key";
    data.extend_from_slice(&(key.len() as u32).to_le_bytes());
    data.extend_from_slice(key);
    // Type: Uint32 (4)
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&42u32.to_le_bytes());
    data
}

#[test]
fn test_read_minimal_gguf() {
    let gguf_data = create_minimal_gguf_v3();
    let cursor = Cursor::new(gguf_data);
    let reader = GgufReader::new(cursor).unwrap();
    let data = reader.read().unwrap();
    assert_eq!(data.header.version, 3);
    assert_eq!(data.header.tensor_count, 0);
    assert_eq!(data.header.metadata_kv_count, 1);
    assert_eq!(data.get_string("general.architecture"), Some("llama"));
}

#[test]
fn test_read_gguf_v1() {
    let gguf_data = create_minimal_gguf_v1();
    let cursor = Cursor::new(gguf_data);
    let reader = GgufReader::new(cursor).unwrap();
    let data = reader.read().unwrap();
    assert_eq!(data.header.version, 1);
    assert_eq!(data.header.tensor_count, 0);
    assert_eq!(data.header.metadata_kv_count, 1);
    assert_eq!(data.get_u32("test.key"), Some(42));
}

#[test]
fn test_invalid_magic() {
    let bad_data = vec![0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00];
    let cursor = Cursor::new(bad_data);
    let result = GgufReader::new(cursor);
    assert!(matches!(result, Err(GgufError::InvalidMagic(0))));
}

#[test]
fn test_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&99u32.to_le_bytes());
    let cursor = Cursor::new(data);
    let result = GgufReader::new(cursor);
    assert!(matches!(result, Err(GgufError::UnsupportedVersion(99))));
}

#[test]
fn test_multiple_metadata_types() {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 3
    data.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count: 0
    data.extend_from_slice(&0u64.to_le_bytes());
    // Metadata count: 5
    data.extend_from_slice(&5u64.to_le_bytes());

    // Metadata 1: u8
    let key = b"test.u8";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&0u32.to_le_bytes()); // Type: Uint8
    data.push(255u8);

    // Metadata 2: i32
    let key = b"test.i32";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&5u32.to_le_bytes()); // Type: Int32
    data.extend_from_slice(&(-42i32).to_le_bytes());

    // Metadata 3: f32
    let key = b"test.f32";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&6u32.to_le_bytes()); // Type: Float32
    data.extend_from_slice(&3.14f32.to_le_bytes());

    // Metadata 4: bool
    let key = b"test.bool";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&7u32.to_le_bytes()); // Type: Bool
    data.push(1u8);

    // Metadata 5: u64
    let key = b"test.u64";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&10u32.to_le_bytes()); // Type: Uint64
    data.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());

    let cursor = Cursor::new(data);
    let reader = GgufReader::new(cursor).unwrap();
    let parsed = reader.read().unwrap();

    assert_eq!(parsed.header.metadata_kv_count, 5);
    assert_eq!(parsed.get_u64("test.u64"), Some(0xFFFF_FFFF_FFFF_FFFF));
    assert_eq!(parsed.get_f32("test.f32"), Some(3.14));
}

#[test]
fn test_unexpected_eof() {
    // Just magic, no version
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    let cursor = Cursor::new(data);
    let result = GgufReader::new(cursor);
    assert!(matches!(result, Err(GgufError::UnexpectedEof)));
}

#[test]
fn test_data_offset_alignment() {
    let gguf_data = create_minimal_gguf_v3();
    let cursor = Cursor::new(gguf_data);
    let reader = GgufReader::new(cursor).unwrap();
    let data = reader.read().unwrap();
    // Data offset should be aligned to 32 bytes (default alignment)
    assert_eq!(data.data_offset % 32, 0);
}
