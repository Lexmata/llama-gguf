//! GGUF file reader implementation

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use super::constants::{
    GgmlType, GgufMetadataValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_V1,
    GGUF_VERSION_V2, GGUF_VERSION_V3,
};
use super::error::GgufError;
use super::types::{GgufData, GgufHeader, MetadataArray, MetadataValue, TensorInfo};

/// GGUF file reader
pub struct GgufReader<R> {
    reader: R,
    version: u32,
}

impl GgufReader<BufReader<File>> {
    /// Open a GGUF file from a path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::new(reader)
    }
}

impl<R: Read + Seek> GgufReader<R> {
    /// Create a new GGUF reader from any Read + Seek source
    pub fn new(mut reader: R) -> Result<Self, GgufError> {
        // Read and validate magic number
        let magic = Self::read_u32_static(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        // Read and validate version
        let version = Self::read_u32_static(&mut reader)?;
        if version != GGUF_VERSION_V1 && version != GGUF_VERSION_V2 && version != GGUF_VERSION_V3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        Ok(Self { reader, version })
    }

    /// Read the complete GGUF file and return parsed data
    pub fn read(mut self) -> Result<GgufData, GgufError> {
        // Read tensor and metadata counts based on version
        let (tensor_count, metadata_kv_count) = if self.version == GGUF_VERSION_V1 {
            // V1 uses u32 for counts
            let tensor_count = self.read_u32()? as u64;
            let metadata_kv_count = self.read_u32()? as u64;
            (tensor_count, metadata_kv_count)
        } else {
            // V2+ uses u64 for counts
            let tensor_count = self.read_u64()?;
            let metadata_kv_count = self.read_u64()?;
            (tensor_count, metadata_kv_count)
        };

        let header = GgufHeader {
            version: self.version,
            tensor_count,
            metadata_kv_count,
        };

        // Read metadata key-value pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;
            metadata.insert(key, value);
        }

        // Read tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = self.read_tensor_info()?;
            tensors.push(info);
        }

        // Get alignment from metadata or use default
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                MetadataValue::Uint32(a) => Some(*a as usize),
                MetadataValue::Uint64(a) => Some(*a as usize),
                _ => None,
            })
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Calculate data offset (aligned)
        let current_pos = self.reader.stream_position()?;
        let data_offset = align_offset(current_pos, alignment);

        Ok(GgufData {
            header,
            metadata,
            tensors,
            data_offset,
        })
    }

    // Private helper methods for reading primitives

    fn read_u32_static(reader: &mut R) -> Result<u32, GgufError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Self::read_u32_static(&mut self.reader)
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        // String length depends on version
        let len = if self.version == GGUF_VERSION_V1 {
            self.read_u32()? as usize
        } else {
            self.read_u64()? as usize
        };

        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof
            } else {
                GgufError::Io(e)
            }
        })?;

        String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let type_id = self.read_u32()?;
        let value_type =
            GgufMetadataValueType::try_from(type_id).map_err(GgufError::InvalidMetadataType)?;

        match value_type {
            GgufMetadataValueType::Uint8 => Ok(MetadataValue::Uint8(self.read_u8()?)),
            GgufMetadataValueType::Int8 => Ok(MetadataValue::Int8(self.read_i8()?)),
            GgufMetadataValueType::Uint16 => Ok(MetadataValue::Uint16(self.read_u16()?)),
            GgufMetadataValueType::Int16 => Ok(MetadataValue::Int16(self.read_i16()?)),
            GgufMetadataValueType::Uint32 => Ok(MetadataValue::Uint32(self.read_u32()?)),
            GgufMetadataValueType::Int32 => Ok(MetadataValue::Int32(self.read_i32()?)),
            GgufMetadataValueType::Float32 => Ok(MetadataValue::Float32(self.read_f32()?)),
            GgufMetadataValueType::Bool => Ok(MetadataValue::Bool(self.read_bool()?)),
            GgufMetadataValueType::String => Ok(MetadataValue::String(self.read_string()?)),
            GgufMetadataValueType::Uint64 => Ok(MetadataValue::Uint64(self.read_u64()?)),
            GgufMetadataValueType::Int64 => Ok(MetadataValue::Int64(self.read_i64()?)),
            GgufMetadataValueType::Float64 => Ok(MetadataValue::Float64(self.read_f64()?)),
            GgufMetadataValueType::Array => {
                let array = self.read_metadata_array()?;
                Ok(MetadataValue::Array(array))
            }
        }
    }

    fn read_metadata_array(&mut self) -> Result<MetadataArray, GgufError> {
        let element_type_id = self.read_u32()?;
        let element_type = GgufMetadataValueType::try_from(element_type_id)
            .map_err(GgufError::InvalidMetadataType)?;

        // Array length depends on version
        let len = if self.version == GGUF_VERSION_V1 {
            self.read_u32()? as usize
        } else {
            self.read_u64()? as usize
        };

        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let value = match element_type {
                GgufMetadataValueType::Uint8 => MetadataValue::Uint8(self.read_u8()?),
                GgufMetadataValueType::Int8 => MetadataValue::Int8(self.read_i8()?),
                GgufMetadataValueType::Uint16 => MetadataValue::Uint16(self.read_u16()?),
                GgufMetadataValueType::Int16 => MetadataValue::Int16(self.read_i16()?),
                GgufMetadataValueType::Uint32 => MetadataValue::Uint32(self.read_u32()?),
                GgufMetadataValueType::Int32 => MetadataValue::Int32(self.read_i32()?),
                GgufMetadataValueType::Float32 => MetadataValue::Float32(self.read_f32()?),
                GgufMetadataValueType::Bool => MetadataValue::Bool(self.read_bool()?),
                GgufMetadataValueType::String => MetadataValue::String(self.read_string()?),
                GgufMetadataValueType::Uint64 => MetadataValue::Uint64(self.read_u64()?),
                GgufMetadataValueType::Int64 => MetadataValue::Int64(self.read_i64()?),
                GgufMetadataValueType::Float64 => MetadataValue::Float64(self.read_f64()?),
                GgufMetadataValueType::Array => {
                    // Nested arrays
                    MetadataValue::Array(self.read_metadata_array()?)
                }
            };
            values.push(value);
        }

        Ok(MetadataArray { values })
    }

    fn read_tensor_info(&mut self) -> Result<TensorInfo, GgufError> {
        let name = self.read_string()?;

        let n_dims = self.read_u32()?;

        // Read dimensions (always u64 in v2+, u32 in v1)
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = if self.version == GGUF_VERSION_V1 {
                self.read_u32()? as u64
            } else {
                self.read_u64()?
            };
            dims.push(dim);
        }

        let dtype_id = self.read_u32()?;
        let dtype = GgmlType::try_from(dtype_id).map_err(GgufError::InvalidTensorType)?;

        let offset = self.read_u64()?;

        Ok(TensorInfo {
            name,
            n_dims,
            dims,
            dtype,
            offset,
        })
    }
}

/// Align an offset to the given alignment
fn align_offset(offset: u64, alignment: usize) -> u64 {
    let alignment = alignment as u64;
    (offset + alignment - 1) / alignment * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(100, 32), 128);
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
}
