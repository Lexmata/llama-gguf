//! GGUF file format parser and writer

mod constants;
mod error;
mod reader;
mod types;
mod writer;

pub use constants::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GgmlType, GgufMetadataValueType};
pub use error::GgufError;
pub use reader::GgufReader;
pub use types::{GgufData, GgufHeader, MetadataArray, MetadataValue, TensorInfo};
pub use writer::{GgufBuilder, GgufWriter, TensorToWrite};

/// High-level GGUF file handle with memory-mapped tensor data
pub struct GgufFile {
    pub data: GgufData,
    mmap: memmap2::Mmap,
}

impl GgufFile {
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(&path)?;
        let data = GgufReader::open(&path)?.read()?;
        // SAFETY: memmap2::Mmap ensures safe memory access. The file handle is kept
        // alive for the lifetime of this struct. External modifications to the file
        // could cause undefined behavior, but this is documented in memmap2.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { data, mmap })
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.data.get_tensor(name)?;
        let start = (self.data.data_offset + info.offset) as usize;
        let end = start.checked_add(info.data_size())?;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }
}
