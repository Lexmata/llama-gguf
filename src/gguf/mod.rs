//! GGUF file format parser

mod constants;
mod error;
mod reader;
mod types;

pub use constants::{GgmlType, GgufMetadataValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};
pub use error::GgufError;
pub use reader::GgufReader;
pub use types::{GgufData, GgufHeader, MetadataArray, MetadataValue, TensorInfo};

/// High-level GGUF file handle with memory-mapped tensor data
pub struct GgufFile {
    pub data: GgufData,
    mmap: memmap2::Mmap,
}

impl GgufFile {
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(&path)?;
        let data = GgufReader::open(&path)?.read()?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { data, mmap })
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.data.get_tensor(name)?;
        let start = (self.data.data_offset + info.offset) as usize;
        let end = start + info.data_size();
        Some(&self.mmap[start..end])
    }
}
