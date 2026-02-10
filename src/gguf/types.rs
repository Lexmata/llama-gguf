//! GGUF parsed data types

use super::constants::GgmlType;
use std::collections::HashMap;

/// Metadata value variants
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Array of metadata values
#[derive(Debug, Clone)]
pub struct MetadataArray {
    pub values: Vec<MetadataValue>,
}

/// Information about a tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn data_size(&self) -> usize {
        let n_elements = self.n_elements() as usize;
        let block_size = self.dtype.block_size();
        let type_size = self.dtype.type_size();
        (n_elements / block_size) * type_size
    }
}

/// Parsed GGUF header information
#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// Complete parsed GGUF file
#[derive(Debug)]
pub struct GgufData {
    pub header: GgufHeader,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
}

impl GgufData {
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            MetadataValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            MetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            MetadataValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key)? {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}
