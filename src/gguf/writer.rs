//! GGUF file writer for model export and quantization
//!
//! This module provides functionality to write GGUF files, enabling:
//! - Model quantization and export
//! - Metadata modification
//! - Tensor conversion between formats

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use super::constants::{GgmlType, GgufMetadataValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};
use super::types::{MetadataArray, MetadataValue};
use super::GgufError;

/// GGUF file writer
pub struct GgufWriter<W: Write + Seek> {
    writer: BufWriter<W>,
    version: u32,
    alignment: usize,
    metadata: HashMap<String, MetadataValue>,
    tensors: Vec<TensorToWrite>,
    data_written: bool,
}

/// Tensor information for writing
#[derive(Debug, Clone)]
pub struct TensorToWrite {
    /// Tensor name
    pub name: String,
    /// Dimensions
    pub dims: Vec<u64>,
    /// Data type
    pub dtype: GgmlType,
    /// Raw tensor data
    pub data: Vec<u8>,
}

impl TensorToWrite {
    /// Create a new tensor to write
    pub fn new(name: impl Into<String>, dims: Vec<u64>, dtype: GgmlType, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            dims,
            dtype,
            data,
        }
    }

    /// Get the number of elements
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Get the data size in bytes
    pub fn data_size(&self) -> usize {
        self.data.len()
    }
}

impl GgufWriter<File> {
    /// Create a new GGUF writer for a file path
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::create(path)?;
        Ok(Self::new(file))
    }
}

impl<W: Write + Seek> GgufWriter<W> {
    /// Create a new GGUF writer
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::new(writer),
            version: 3,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            data_written: false,
        }
    }

    /// Set GGUF version (2 or 3)
    pub fn set_version(&mut self, version: u32) -> &mut Self {
        self.version = version;
        self
    }

    /// Set data alignment
    pub fn set_alignment(&mut self, alignment: usize) -> &mut Self {
        self.alignment = alignment;
        self
    }

    /// Add metadata value
    pub fn add_metadata(&mut self, key: impl Into<String>, value: MetadataValue) -> &mut Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add string metadata
    pub fn add_string(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.add_metadata(key, MetadataValue::String(value.into()))
    }

    /// Add u32 metadata
    pub fn add_u32(&mut self, key: impl Into<String>, value: u32) -> &mut Self {
        self.add_metadata(key, MetadataValue::Uint32(value))
    }

    /// Add u64 metadata
    pub fn add_u64(&mut self, key: impl Into<String>, value: u64) -> &mut Self {
        self.add_metadata(key, MetadataValue::Uint64(value))
    }

    /// Add f32 metadata
    pub fn add_f32(&mut self, key: impl Into<String>, value: f32) -> &mut Self {
        self.add_metadata(key, MetadataValue::Float32(value))
    }

    /// Add bool metadata
    pub fn add_bool(&mut self, key: impl Into<String>, value: bool) -> &mut Self {
        self.add_metadata(key, MetadataValue::Bool(value))
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, tensor: TensorToWrite) -> &mut Self {
        self.tensors.push(tensor);
        self
    }

    /// Write the GGUF file
    pub fn write(mut self) -> Result<(), GgufError> {
        if self.data_written {
            return Err(GgufError::InvalidData("Data already written".into()));
        }

        // Write header
        self.write_header()?;

        // Write metadata
        self.write_metadata()?;

        // Write tensor infos
        let tensor_offsets = self.write_tensor_infos()?;

        // Align to data section
        self.align_to(self.alignment)?;

        // Write tensor data
        self.write_tensor_data(&tensor_offsets)?;

        self.writer.flush()?;
        self.data_written = true;

        Ok(())
    }

    fn write_header(&mut self) -> Result<(), GgufError> {
        // Magic number
        self.writer.write_all(&GGUF_MAGIC.to_le_bytes())?;

        // Version
        self.writer.write_all(&self.version.to_le_bytes())?;

        // Tensor count
        let tensor_count = self.tensors.len() as u64;
        self.writer.write_all(&tensor_count.to_le_bytes())?;

        // Metadata count
        let metadata_count = self.metadata.len() as u64;
        self.writer.write_all(&metadata_count.to_le_bytes())?;

        Ok(())
    }

    fn write_metadata(&mut self) -> Result<(), GgufError> {
        // Sort keys for consistent output and clone values to avoid borrow issues
        let mut items: Vec<_> = self.metadata.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));

        for (key, value) in items {
            self.write_string(&key)?;
            self.write_metadata_value(&value)?;
        }

        Ok(())
    }

    fn write_metadata_value(&mut self, value: &MetadataValue) -> Result<(), GgufError> {
        match value {
            MetadataValue::Uint8(v) => {
                self.write_u32(GgufMetadataValueType::Uint8 as u32)?;
                self.writer.write_all(&[*v])?;
            }
            MetadataValue::Int8(v) => {
                self.write_u32(GgufMetadataValueType::Int8 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Uint16(v) => {
                self.write_u32(GgufMetadataValueType::Uint16 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Int16(v) => {
                self.write_u32(GgufMetadataValueType::Int16 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Uint32(v) => {
                self.write_u32(GgufMetadataValueType::Uint32 as u32)?;
                self.write_u32(*v)?;
            }
            MetadataValue::Int32(v) => {
                self.write_u32(GgufMetadataValueType::Int32 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Uint64(v) => {
                self.write_u32(GgufMetadataValueType::Uint64 as u32)?;
                self.write_u64(*v)?;
            }
            MetadataValue::Int64(v) => {
                self.write_u32(GgufMetadataValueType::Int64 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Float32(v) => {
                self.write_u32(GgufMetadataValueType::Float32 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Float64(v) => {
                self.write_u32(GgufMetadataValueType::Float64 as u32)?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Bool(v) => {
                self.write_u32(GgufMetadataValueType::Bool as u32)?;
                self.writer.write_all(&[if *v { 1 } else { 0 }])?;
            }
            MetadataValue::String(v) => {
                self.write_u32(GgufMetadataValueType::String as u32)?;
                self.write_string(v)?;
            }
            MetadataValue::Array(arr) => {
                self.write_u32(GgufMetadataValueType::Array as u32)?;
                self.write_array(arr)?;
            }
        }

        Ok(())
    }

    fn write_array(&mut self, arr: &MetadataArray) -> Result<(), GgufError> {
        // Infer element type from first value
        let elem_type = if let Some(first) = arr.values.first() {
            match first {
                MetadataValue::Uint8(_) => GgufMetadataValueType::Uint8,
                MetadataValue::Int8(_) => GgufMetadataValueType::Int8,
                MetadataValue::Uint16(_) => GgufMetadataValueType::Uint16,
                MetadataValue::Int16(_) => GgufMetadataValueType::Int16,
                MetadataValue::Uint32(_) => GgufMetadataValueType::Uint32,
                MetadataValue::Int32(_) => GgufMetadataValueType::Int32,
                MetadataValue::Uint64(_) => GgufMetadataValueType::Uint64,
                MetadataValue::Int64(_) => GgufMetadataValueType::Int64,
                MetadataValue::Float32(_) => GgufMetadataValueType::Float32,
                MetadataValue::Float64(_) => GgufMetadataValueType::Float64,
                MetadataValue::Bool(_) => GgufMetadataValueType::Bool,
                MetadataValue::String(_) => GgufMetadataValueType::String,
                MetadataValue::Array(_) => GgufMetadataValueType::Array,
            }
        } else {
            GgufMetadataValueType::Uint32 // Default for empty arrays
        };

        // Write element type
        self.write_u32(elem_type as u32)?;

        // Write count
        self.write_u64(arr.values.len() as u64)?;

        // Write values
        for value in &arr.values {
            // Write value without type prefix (array elements don't have individual type tags)
            self.write_metadata_value_raw(value)?;
        }

        Ok(())
    }

    fn write_metadata_value_raw(&mut self, value: &MetadataValue) -> Result<(), GgufError> {
        match value {
            MetadataValue::Uint8(v) => self.writer.write_all(&[*v])?,
            MetadataValue::Int8(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Uint16(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Int16(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Uint32(v) => self.write_u32(*v)?,
            MetadataValue::Int32(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Uint64(v) => self.write_u64(*v)?,
            MetadataValue::Int64(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Float32(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Float64(v) => self.writer.write_all(&v.to_le_bytes())?,
            MetadataValue::Bool(v) => self.writer.write_all(&[if *v { 1 } else { 0 }])?,
            MetadataValue::String(v) => self.write_string(v)?,
            MetadataValue::Array(_) => {
                return Err(GgufError::InvalidData("Nested arrays not supported".into()))
            }
        }
        Ok(())
    }

    fn write_tensor_infos(&mut self) -> Result<Vec<u64>, GgufError> {
        // Clone tensor info to avoid borrow issues
        let tensor_infos: Vec<_> = self.tensors.iter()
            .map(|t| (t.name.clone(), t.dims.clone(), t.dtype, t.data_size()))
            .collect();

        let mut offsets = Vec::with_capacity(tensor_infos.len());
        let mut current_offset = 0u64;
        let alignment = self.alignment as u64;

        for (name, dims, dtype, data_size) in tensor_infos {
            // Name
            self.write_string(&name)?;

            // Number of dimensions
            self.write_u32(dims.len() as u32)?;

            // Dimensions
            for dim in &dims {
                self.write_u64(*dim)?;
            }

            // Type
            self.write_u32(dtype as u32)?;

            // Offset (will be relative to data section start)
            self.write_u64(current_offset)?;

            offsets.push(current_offset);

            // Compute next offset with alignment
            let size = data_size as u64;
            current_offset += size;
            let remainder = current_offset % alignment;
            if remainder != 0 {
                current_offset += alignment - remainder;
            }
        }

        Ok(offsets)
    }

    fn write_tensor_data(&mut self, _offsets: &[u64]) -> Result<(), GgufError> {
        // Clone tensor data to avoid borrow issues
        let tensor_data: Vec<_> = self.tensors.iter()
            .map(|t| t.data.clone())
            .collect();
        let alignment = self.alignment;

        for data in tensor_data {
            self.writer.write_all(&data)?;

            // Align to next tensor
            self.align_to(alignment)?;
        }

        Ok(())
    }

    fn write_string(&mut self, s: &str) -> Result<(), GgufError> {
        let bytes = s.as_bytes();
        self.write_u64(bytes.len() as u64)?;
        self.writer.write_all(bytes)?;
        Ok(())
    }

    fn write_u32(&mut self, v: u32) -> Result<(), GgufError> {
        self.writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    fn write_u64(&mut self, v: u64) -> Result<(), GgufError> {
        self.writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    fn align_to(&mut self, alignment: usize) -> Result<(), GgufError> {
        let pos = self.writer.stream_position()? as usize;
        let remainder = pos % alignment;
        if remainder != 0 {
            let padding = alignment - remainder;
            for _ in 0..padding {
                self.writer.write_all(&[0])?;
            }
        }
        Ok(())
    }
}

/// Builder for creating GGUF files
pub struct GgufBuilder {
    version: u32,
    alignment: usize,
    metadata: HashMap<String, MetadataValue>,
    tensors: Vec<TensorToWrite>,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    /// Create a new GGUF builder
    pub fn new() -> Self {
        Self {
            version: 3,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            metadata: HashMap::new(),
            tensors: Vec::new(),
        }
    }

    /// Set GGUF version
    pub fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Set data alignment
    pub fn alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Add general architecture metadata
    pub fn architecture(mut self, arch: &str) -> Self {
        self.metadata
            .insert("general.architecture".to_string(), MetadataValue::String(arch.to_string()));
        self
    }

    /// Add model name
    pub fn name(mut self, name: &str) -> Self {
        self.metadata
            .insert("general.name".to_string(), MetadataValue::String(name.to_string()));
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: MetadataValue) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add a tensor
    pub fn tensor(mut self, tensor: TensorToWrite) -> Self {
        self.tensors.push(tensor);
        self
    }

    /// Write to a file
    pub fn write_to_file<P: AsRef<Path>>(self, path: P) -> Result<(), GgufError> {
        let mut writer = GgufWriter::create(path)?;
        writer.set_version(self.version);
        writer.set_alignment(self.alignment);

        for (key, value) in self.metadata {
            writer.add_metadata(key, value);
        }

        for tensor in self.tensors {
            writer.add_tensor(tensor);
        }

        writer.write()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_gguf_writer_basic() {
        let mut buffer = Cursor::new(Vec::new());

        {
            let mut writer = GgufWriter::new(&mut buffer);
            writer.add_string("general.architecture", "llama");
            writer.add_u32("llama.block_count", 32);

            let tensor = TensorToWrite::new(
                "test.weight",
                vec![4, 4],
                GgmlType::F32,
                vec![0u8; 64], // 16 f32s = 64 bytes
            );
            writer.add_tensor(tensor);

            writer.write().unwrap();
        }

        let data = buffer.into_inner();
        assert!(data.len() > 0);

        // Check magic number
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, GGUF_MAGIC);

        // Check version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(version, 3);
    }

    #[test]
    fn test_gguf_builder() {
        let builder = GgufBuilder::new()
            .version(3)
            .architecture("llama")
            .name("test-model")
            .metadata("test.key", MetadataValue::Uint32(42));

        assert!(builder.metadata.contains_key("general.architecture"));
        assert!(builder.metadata.contains_key("general.name"));
    }

    #[test]
    fn test_tensor_to_write() {
        let tensor = TensorToWrite::new(
            "layer.0.weight",
            vec![1024, 4096],
            GgmlType::Q4_0,
            vec![0u8; 1024 * 4096 / 2], // Q4_0 is 4 bits per element
        );

        assert_eq!(tensor.num_elements(), 1024 * 4096);
        assert_eq!(tensor.dims.len(), 2);
    }
}
