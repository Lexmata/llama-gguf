# Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundation layer - GGUF file parsing, tensor types with quantization, and basic CPU backend operations.

**Architecture:** Monolithic library with modules for gguf/, tensor/, and backend/. Memory-mapped file access for GGUF. Tensor owns data with dtype tracking. Backend trait defines operations, CPU backend implements with SIMD.

**Tech Stack:** Rust 2024, memmap2, bytemuck, half, rayon, thiserror, wide/pulp for SIMD

---

### Task 1: Project Setup and Dependencies

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/main.rs`
- Create: `src/lib.rs`

**Step 1: Update Cargo.toml with dependencies**

```toml
[package]
name = "llama-rs"
version = "0.1.0"
edition = "2024"

[lib]
name = "llama_rs"
path = "src/lib.rs"

[[bin]]
name = "llama-rs"
path = "src/main.rs"

[features]
default = ["cpu"]
cpu = []
cuda = ["dep:cudarc"]
vulkan = ["dep:ash", "dep:gpu-allocator"]
metal = ["dep:metal", "dep:objc"]
server = ["dep:axum", "dep:tokio", "dep:tower-http"]

[dependencies]
# Core
memmap2 = "0.9"
bytemuck = { version = "1.14", features = ["derive"] }
half = { version = "2.3", features = ["bytemuck"] }
rayon = "1.8"
thiserror = "1.0"
tracing = "0.1"

# CLI
clap = { version = "4.4", features = ["derive"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Optional: GPU backends
cudarc = { version = "0.11", optional = true }
ash = { version = "0.38", optional = true }
gpu-allocator = { version = "0.25", optional = true }
metal = { version = "0.28", optional = true }
objc = { version = "0.2", optional = true }

# Optional: Server
axum = { version = "0.7", optional = true }
tokio = { version = "1.35", features = ["full"], optional = true }
tower-http = { version = "0.5", features = ["cors"], optional = true }

[dev-dependencies]
criterion = "0.5"
tempfile = "3.10"

[profile.release]
lto = true
codegen-units = 1

[[bench]]
name = "quantization"
harness = false
```

**Step 2: Create src/lib.rs**

```rust
//! llama-rs: A Rust implementation of llama.cpp
//!
//! High-performance LLM inference engine with support for GGUF models.

pub mod gguf;
pub mod tensor;
pub mod backend;

pub use gguf::GgufFile;
pub use tensor::{Tensor, DType};
pub use backend::Backend;

/// Library-wide error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF error: {0}")]
    Gguf(#[from] gguf::GgufError),

    #[error("Tensor error: {0}")]
    Tensor(#[from] tensor::TensorError),

    #[error("Backend error: {0}")]
    Backend(#[from] backend::BackendError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Step 3: Update src/main.rs**

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llama-rs")]
#[command(about = "Rust implementation of llama.cpp", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show information about a GGUF model
    Info {
        /// Path to the GGUF model file
        model: String,
    },
    /// Run inference on a model
    Run {
        /// Path to the GGUF model file
        model: String,
        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { model } => {
            println!("Model info: {}", model);
            // TODO: Implement
        }
        Commands::Run { model, prompt } => {
            println!("Running model: {}", model);
            if let Some(p) = prompt {
                println!("Prompt: {}", p);
            }
            // TODO: Implement
        }
    }
}
```

**Step 4: Create module stubs**

Create `src/gguf/mod.rs`:
```rust
//! GGUF file format parser

mod error;

pub use error::GgufError;

/// Placeholder for GGUF file
pub struct GgufFile;
```

Create `src/gguf/error.rs`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("Invalid magic number")]
    InvalidMagic,
}
```

Create `src/tensor/mod.rs`:
```rust
//! Tensor types and operations

mod error;
mod dtype;

pub use error::TensorError;
pub use dtype::DType;

/// Placeholder for Tensor
pub struct Tensor;
```

Create `src/tensor/error.rs`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch")]
    ShapeMismatch,
}
```

Create `src/tensor/dtype.rs`:
```rust
/// Data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ3S,
    IQ4XS,
    IQ4NL,
}
```

Create `src/backend/mod.rs`:
```rust
//! Hardware backends

mod error;

pub use error::BackendError;

/// Placeholder for Backend trait
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
}
```

Create `src/backend/error.rs`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("Backend not available: {0}")]
    NotAvailable(String),
}
```

**Step 5: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: project setup with dependencies and module structure"
```

---

### Task 2: GGUF Constants and Types

**Files:**
- Create: `src/gguf/constants.rs`
- Create: `src/gguf/types.rs`
- Modify: `src/gguf/mod.rs`

**Step 1: Create constants matching GGUF spec**

Create `src/gguf/constants.rs`:
```rust
//! GGUF format constants

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// Supported GGUF versions
pub const GGUF_VERSION_V1: u32 = 1;
pub const GGUF_VERSION_V2: u32 = 2;
pub const GGUF_VERSION_V3: u32 = 3;

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Metadata value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufMetadataValueType {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(value),
        }
    }
}

/// GGML tensor types (used in GGUF tensor info)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // deprecated
    // Q4_3 = 5, // deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
}

impl TryFrom<u32> for GgmlType {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::BF16),
            _ => Err(value),
        }
    }
}

impl GgmlType {
    /// Block size for this type (number of elements per block)
    pub const fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 256,
            Self::Q5K => 256,
            Self::Q6K => 256,
            Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 256,
            Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ4XS | Self::IQ4NL => 256,
            Self::IQ1S => 256,
        }
    }

    /// Bytes per block for this type
    pub const fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,   // 2 (delta) + 16 (32 * 4 bits)
            Self::Q4_1 => 20,   // 2 (delta) + 2 (min) + 16
            Self::Q5_0 => 22,   // 2 + 4 + 16
            Self::Q5_1 => 24,   // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,   // 2 + 32
            Self::Q8_1 => 36,   // 2 + 2 + 32
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ4XS => 136,
            Self::IQ4NL => 132,
            Self::IQ1S => 50,
        }
    }
}
```

**Step 2: Create types for parsed GGUF data**

Create `src/gguf/types.rs`:
```rust
//! GGUF parsed data types

use std::collections::HashMap;
use super::constants::GgmlType;

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
    /// Total number of elements in this tensor
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Size in bytes of this tensor's data
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
    /// Get a string metadata value
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a u32 metadata value
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            MetadataValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a u64 metadata value
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            MetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a f32 metadata value
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            MetadataValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get tensor info by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}
```

**Step 3: Update mod.rs to export**

Update `src/gguf/mod.rs`:
```rust
//! GGUF file format parser

mod constants;
mod error;
mod types;

pub use constants::{GgmlType, GgufMetadataValueType, GGUF_MAGIC, GGUF_DEFAULT_ALIGNMENT};
pub use error::GgufError;
pub use types::{GgufData, GgufHeader, MetadataValue, MetadataArray, TensorInfo};

/// Placeholder for GGUF file handle
pub struct GgufFile;
```

**Step 4: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/gguf/constants.rs src/gguf/types.rs src/gguf/mod.rs
git commit -m "feat(gguf): add constants and types matching GGUF spec"
```

---

### Task 3: GGUF Reader Implementation

**Files:**
- Create: `src/gguf/reader.rs`
- Modify: `src/gguf/mod.rs`
- Modify: `src/gguf/error.rs`
- Create: `tests/gguf_reader_test.rs`

**Step 1: Expand error types**

Update `src/gguf/error.rs`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("Invalid magic number: expected 0x46554747, got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid metadata type: {0}")]
    InvalidMetadataType(u32),

    #[error("Invalid tensor type: {0}")]
    InvalidTensorType(u32),

    #[error("Invalid UTF-8 string")]
    InvalidUtf8,

    #[error("Unexpected end of file")]
    UnexpectedEof,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

**Step 2: Create reader implementation**

Create `src/gguf/reader.rs`:
```rust
//! GGUF file reader

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::constants::{GgmlType, GgufMetadataValueType, GGUF_MAGIC, GGUF_DEFAULT_ALIGNMENT};
use super::error::GgufError;
use super::types::{GgufData, GgufHeader, MetadataArray, MetadataValue, TensorInfo};

/// Reader for GGUF files
pub struct GgufReader<R> {
    reader: R,
    version: u32,
}

impl GgufReader<BufReader<File>> {
    /// Open a GGUF file from path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::new(reader)
    }
}

impl<R: Read + Seek> GgufReader<R> {
    /// Create a new reader from any Read + Seek source
    pub fn new(mut reader: R) -> Result<Self, GgufError> {
        // Read and validate magic
        let magic = Self::read_u32_raw(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        // Read version
        let version = Self::read_u32_raw(&mut reader)?;
        if version < 1 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        Ok(Self { reader, version })
    }

    /// Read all data from the GGUF file
    pub fn read(mut self) -> Result<GgufData, GgufError> {
        // Read header counts (format differs by version)
        let (tensor_count, metadata_kv_count) = if self.version == 1 {
            let tc = self.read_u32()? as u64;
            let mc = self.read_u32()? as u64;
            (tc, mc)
        } else {
            let tc = self.read_u64()?;
            let mc = self.read_u64()?;
            (tc, mc)
        };

        let header = GgufHeader {
            version: self.version,
            tensor_count,
            metadata_kv_count,
        };

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = self.read_tensor_info()?;
            tensors.push(info);
        }

        // Calculate data offset (aligned)
        let current_pos = self.reader.stream_position()? as usize;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                MetadataValue::Uint32(a) => Some(*a as usize),
                _ => None,
            })
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let data_offset = (current_pos + alignment - 1) / alignment * alignment;

        Ok(GgufData {
            header,
            metadata,
            tensors,
            data_offset: data_offset as u64,
        })
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let type_id = self.read_u32()?;
        let value_type = GgufMetadataValueType::try_from(type_id)
            .map_err(|_| GgufError::InvalidMetadataType(type_id))?;

        self.read_metadata_value_of_type(value_type)
    }

    fn read_metadata_value_of_type(
        &mut self,
        value_type: GgufMetadataValueType,
    ) -> Result<MetadataValue, GgufError> {
        Ok(match value_type {
            GgufMetadataValueType::Uint8 => MetadataValue::Uint8(self.read_u8()?),
            GgufMetadataValueType::Int8 => MetadataValue::Int8(self.read_i8()?),
            GgufMetadataValueType::Uint16 => MetadataValue::Uint16(self.read_u16()?),
            GgufMetadataValueType::Int16 => MetadataValue::Int16(self.read_i16()?),
            GgufMetadataValueType::Uint32 => MetadataValue::Uint32(self.read_u32()?),
            GgufMetadataValueType::Int32 => MetadataValue::Int32(self.read_i32()?),
            GgufMetadataValueType::Float32 => MetadataValue::Float32(self.read_f32()?),
            GgufMetadataValueType::Bool => MetadataValue::Bool(self.read_u8()? != 0),
            GgufMetadataValueType::String => MetadataValue::String(self.read_string()?),
            GgufMetadataValueType::Uint64 => MetadataValue::Uint64(self.read_u64()?),
            GgufMetadataValueType::Int64 => MetadataValue::Int64(self.read_i64()?),
            GgufMetadataValueType::Float64 => MetadataValue::Float64(self.read_f64()?),
            GgufMetadataValueType::Array => {
                let elem_type_id = self.read_u32()?;
                let elem_type = GgufMetadataValueType::try_from(elem_type_id)
                    .map_err(|_| GgufError::InvalidMetadataType(elem_type_id))?;

                let len = if self.version == 1 {
                    self.read_u32()? as u64
                } else {
                    self.read_u64()?
                };

                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.read_metadata_value_of_type(elem_type)?);
                }

                MetadataValue::Array(MetadataArray { values })
            }
        })
    }

    fn read_tensor_info(&mut self) -> Result<TensorInfo, GgufError> {
        let name = self.read_string()?;
        let n_dims = self.read_u32()?;

        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(self.read_u64()?);
        }

        let dtype_id = self.read_u32()?;
        let dtype = GgmlType::try_from(dtype_id)
            .map_err(|_| GgufError::InvalidTensorType(dtype_id))?;

        let offset = self.read_u64()?;

        Ok(TensorInfo {
            name,
            n_dims,
            dims,
            dtype,
            offset,
        })
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = if self.version == 1 {
            self.read_u32()? as u64
        } else {
            self.read_u64()?
        };

        let mut buf = vec![0u8; len as usize];
        self.reader.read_exact(&mut buf)?;

        String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
    }

    // Primitive readers
    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u32_raw(reader: &mut R) -> Result<u32, GgufError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}
```

**Step 3: Update mod.rs**

Update `src/gguf/mod.rs`:
```rust
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
    /// Open a GGUF file
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(&path)?;
        let data = GgufReader::open(&path)?.read()?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { data, mmap })
    }

    /// Get raw tensor data by name
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.data.get_tensor(name)?;
        let start = (self.data.data_offset + info.offset) as usize;
        let end = start + info.data_size();
        Some(&self.mmap[start..end])
    }
}
```

**Step 4: Write test for GGUF reader**

Create `tests/gguf_reader_test.rs`:
```rust
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
    data.extend_from_slice(&99u32.to_le_bytes()); // Bad version

    let cursor = Cursor::new(data);
    let result = GgufReader::new(cursor);
    assert!(matches!(result, Err(GgufError::UnsupportedVersion(99))));
}
```

**Step 5: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/gguf/reader.rs src/gguf/mod.rs src/gguf/error.rs tests/gguf_reader_test.rs
git commit -m "feat(gguf): implement GGUF reader with memory-mapped file support"
```

---

### Task 4: Tensor Core Types

**Files:**
- Create: `src/tensor/storage.rs`
- Create: `src/tensor/tensor.rs`
- Modify: `src/tensor/mod.rs`
- Modify: `src/tensor/dtype.rs`

**Step 1: Create tensor storage types**

Create `src/tensor/storage.rs`:
```rust
//! Tensor data storage

use std::sync::Arc;

/// Owned or borrowed tensor data
#[derive(Debug, Clone)]
pub enum TensorStorage {
    /// Owned data on CPU
    Owned(Arc<Vec<u8>>),
    /// View into external data (e.g., memory-mapped file)
    View { data: *const u8, len: usize },
}

// Safety: View data comes from memory-mapped files which are thread-safe for reads
unsafe impl Send for TensorStorage {}
unsafe impl Sync for TensorStorage {}

impl TensorStorage {
    /// Create owned storage from bytes
    pub fn owned(data: Vec<u8>) -> Self {
        Self::Owned(Arc::new(data))
    }

    /// Create a view into external data
    ///
    /// # Safety
    /// The data pointer must be valid for the lifetime of this storage
    /// and the data must not be mutated while views exist.
    pub unsafe fn view(data: *const u8, len: usize) -> Self {
        Self::View { data, len }
    }

    /// Get data as byte slice
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data.as_slice(),
            Self::View { data, len } => unsafe { std::slice::from_raw_parts(*data, *len) },
        }
    }

    /// Get mutable data (only works for owned storage)
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Owned(data) => Arc::get_mut(data).map(|v| v.as_mut_slice()),
            Self::View { .. } => None,
        }
    }

    /// Length in bytes
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::View { len, .. } => *len,
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Make owned copy
    pub fn to_owned(&self) -> Self {
        Self::Owned(Arc::new(self.as_bytes().to_vec()))
    }
}
```

**Step 2: Update dtype with conversion from GgmlType**

Update `src/tensor/dtype.rs`:
```rust
//! Tensor data types

use crate::gguf::GgmlType;

/// Data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    // Floating point
    F32,
    F16,
    BF16,
    F64,

    // Integer
    I8,
    I16,
    I32,
    I64,
    U8,

    // Basic quantization
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,

    // K-quants
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,

    // I-quants
    IQ1S,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4XS,
    IQ4NL,
}

impl DType {
    /// Number of elements per quantization block
    pub const fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ1S | Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 256,
            Self::IQ3XXS | Self::IQ3S | Self::IQ4XS | Self::IQ4NL => 256,
        }
    }

    /// Bytes per block
    pub const fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 | Self::U8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ1S => 50,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ4XS => 136,
            Self::IQ4NL => 132,
        }
    }

    /// Whether this is a quantized type
    pub const fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::U8
        )
    }

    /// Calculate byte size for n elements
    pub fn size_for_elements(&self, n_elements: usize) -> usize {
        let block_size = self.block_size();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * self.block_bytes()
    }
}

impl From<GgmlType> for DType {
    fn from(t: GgmlType) -> Self {
        match t {
            GgmlType::F32 => Self::F32,
            GgmlType::F16 => Self::F16,
            GgmlType::BF16 => Self::BF16,
            GgmlType::F64 => Self::F64,
            GgmlType::I8 => Self::I8,
            GgmlType::I16 => Self::I16,
            GgmlType::I32 => Self::I32,
            GgmlType::I64 => Self::I64,
            GgmlType::Q4_0 => Self::Q4_0,
            GgmlType::Q4_1 => Self::Q4_1,
            GgmlType::Q5_0 => Self::Q5_0,
            GgmlType::Q5_1 => Self::Q5_1,
            GgmlType::Q8_0 => Self::Q8_0,
            GgmlType::Q8_1 => Self::Q8_1,
            GgmlType::Q2K => Self::Q2K,
            GgmlType::Q3K => Self::Q3K,
            GgmlType::Q4K => Self::Q4K,
            GgmlType::Q5K => Self::Q5K,
            GgmlType::Q6K => Self::Q6K,
            GgmlType::Q8K => Self::Q8K,
            GgmlType::IQ1S => Self::IQ1S,
            GgmlType::IQ2XXS => Self::IQ2XXS,
            GgmlType::IQ2XS => Self::IQ2XS,
            GgmlType::IQ2S => Self::IQ2S,
            GgmlType::IQ3XXS => Self::IQ3XXS,
            GgmlType::IQ3S => Self::IQ3S,
            GgmlType::IQ4XS => Self::IQ4XS,
            GgmlType::IQ4NL => Self::IQ4NL,
        }
    }
}
```

**Step 3: Create tensor type**

Create `src/tensor/tensor.rs`:
```rust
//! Core tensor type

use super::dtype::DType;
use super::error::TensorError;
use super::storage::TensorStorage;

/// Multi-dimensional tensor
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: TensorStorage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    offset: usize,
}

impl Tensor {
    /// Create a new tensor with owned storage
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Result<Self, TensorError> {
        let expected_size = dtype.size_for_elements(shape.iter().product());
        if data.len() != expected_size {
            return Err(TensorError::SizeMismatch {
                expected: expected_size,
                got: data.len(),
            });
        }

        let strides = compute_strides(&shape);
        Ok(Self {
            storage: TensorStorage::owned(data),
            shape,
            strides,
            dtype,
            offset: 0,
        })
    }

    /// Create tensor from raw storage (e.g., memory-mapped data)
    ///
    /// # Safety
    /// The storage must contain valid data for the given dtype and shape.
    pub unsafe fn from_storage(
        storage: TensorStorage,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, TensorError> {
        let expected_size = dtype.size_for_elements(shape.iter().product());
        if storage.len() < expected_size {
            return Err(TensorError::SizeMismatch {
                expected: expected_size,
                got: storage.len(),
            });
        }

        let strides = compute_strides(&shape);
        Ok(Self {
            storage,
            shape,
            strides,
            dtype,
            offset: 0,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let size = dtype.size_for_elements(shape.iter().product());
        let data = vec![0u8; size];
        let strides = compute_strides(&shape);
        Self {
            storage: TensorStorage::owned(data),
            shape,
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Create tensor from f32 slice
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_elements: usize = shape.iter().product();
        if data.len() != expected_elements {
            return Err(TensorError::ShapeMismatch {
                expected: expected_elements,
                got: data.len(),
            });
        }

        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        Self::new(bytes, shape, DType::F32)
    }

    /// Shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Strides in elements (not bytes)
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Raw data as bytes
    pub fn data(&self) -> &[u8] {
        &self.storage.as_bytes()[self.offset..]
    }

    /// Mutable raw data (only for owned storage)
    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        self.storage.as_bytes_mut().map(|d| &mut d[self.offset..])
    }

    /// Interpret data as f32 slice (only valid for F32 dtype)
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(bytemuck::cast_slice(self.data()))
    }

    /// Interpret data as mutable f32 slice
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        self.data_mut().map(|d| bytemuck::cast_slice_mut(d))
    }

    /// Reshape tensor (must have same number of elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: self.numel(),
                got: new_numel,
            });
        }

        let strides = compute_strides(&new_shape);
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides,
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape);
        self.strides == expected
    }

    /// Make a contiguous copy if not already contiguous
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            // For now, just clone (proper implementation would reorder data)
            Self {
                storage: self.storage.to_owned(),
                shape: self.shape.clone(),
                strides: compute_strides(&self.shape),
                dtype: self.dtype,
                offset: 0,
            }
        }
    }
}

/// Compute strides for a contiguous tensor with given shape
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, vec![2, 3]).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(vec![4, 4], DType::F32);
        let data = tensor.as_f32().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_strides() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
    }
}
```

**Step 4: Update tensor mod.rs**

Update `src/tensor/mod.rs`:
```rust
//! Tensor types and operations

mod dtype;
mod error;
mod storage;
mod tensor;

pub use dtype::DType;
pub use error::TensorError;
pub use storage::TensorStorage;
pub use tensor::Tensor;
```

**Step 5: Update tensor error**

Update `src/tensor/error.rs`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("Size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("Invalid dtype for operation")]
    InvalidDType,

    #[error("Tensor is not contiguous")]
    NotContiguous,
}
```

**Step 6: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/tensor/
git commit -m "feat(tensor): implement core tensor type with storage and dtype"
```

---

### Task 5: Quantization Block Structures

**Files:**
- Create: `src/tensor/quant/mod.rs`
- Create: `src/tensor/quant/blocks.rs`
- Modify: `src/tensor/mod.rs`

**Step 1: Create quantization block structures**

Create directory and `src/tensor/quant/mod.rs`:
```rust
//! Quantization formats

mod blocks;

pub use blocks::*;
```

Create `src/tensor/quant/blocks.rs`:
```rust
//! Quantization block structures matching llama.cpp exactly

use bytemuck::{Pod, Zeroable};
use half::f16;

// =============================================================================
// Basic Quantization Blocks
// =============================================================================

/// Q4_0: 4-bit quantization, 32 elements per block
/// Block size: 18 bytes (2 + 16)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    /// Scale factor (f16)
    pub d: f16,
    /// Quantized values: 32 x 4-bit packed into 16 bytes
    pub qs: [u8; 16],
}

impl BlockQ4_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 18;
}

/// Q4_1: 4-bit quantization with min value, 32 elements per block
/// Block size: 20 bytes (2 + 2 + 16)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_1 {
    /// Scale factor (f16)
    pub d: f16,
    /// Minimum value (f16)
    pub m: f16,
    /// Quantized values
    pub qs: [u8; 16],
}

impl BlockQ4_1 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 20;
}

/// Q5_0: 5-bit quantization, 32 elements per block
/// Block size: 22 bytes (2 + 4 + 16)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_0 {
    /// Scale factor (f16)
    pub d: f16,
    /// High bits of quantized values (32 bits for 32 elements)
    pub qh: [u8; 4],
    /// Low 4 bits of quantized values
    pub qs: [u8; 16],
}

impl BlockQ5_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 22;
}

/// Q5_1: 5-bit quantization with min value, 32 elements per block
/// Block size: 24 bytes (2 + 2 + 4 + 16)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_1 {
    /// Scale factor (f16)
    pub d: f16,
    /// Minimum value (f16)
    pub m: f16,
    /// High bits
    pub qh: [u8; 4],
    /// Low 4 bits
    pub qs: [u8; 16],
}

impl BlockQ5_1 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 24;
}

/// Q8_0: 8-bit quantization, 32 elements per block
/// Block size: 34 bytes (2 + 32)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_0 {
    /// Scale factor (f16)
    pub d: f16,
    /// Quantized values (signed 8-bit)
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 34;
}

/// Q8_1: 8-bit quantization with sum, 32 elements per block
/// Block size: 36 bytes (4 + 32)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_1 {
    /// Scale factor (f32)
    pub d: f32,
    /// Quantized values
    pub qs: [i8; 32],
}

impl BlockQ8_1 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 36;
}

// =============================================================================
// K-Quants (256 elements per block)
// =============================================================================

/// Q2_K: 2-bit quantization
/// Block size: 84 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ2K {
    /// Scales and mins (quantized with 4 bits)
    pub scales: [u8; 16],
    /// Quantized values: 256 x 2-bit = 64 bytes
    pub qs: [u8; 64],
    /// Super-block scale (f16)
    pub d: f16,
    /// Super-block min (f16)
    pub dmin: f16,
}

impl BlockQ2K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 84;
}

/// Q3_K: 3-bit quantization
/// Block size: 110 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ3K {
    /// High bits of quantized values
    pub hmask: [u8; 32],
    /// Low 2 bits of quantized values
    pub qs: [u8; 64],
    /// Scales (6 bits each, packed)
    pub scales: [u8; 12],
    /// Super-block scale (f16)
    pub d: f16,
}

impl BlockQ3K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 110;
}

/// Q4_K: 4-bit quantization
/// Block size: 144 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4K {
    /// Super-block scale (f16)
    pub d: f16,
    /// Super-block min (f16)
    pub dmin: f16,
    /// Scales and mins (6 bits each)
    pub scales: [u8; 12],
    /// Quantized values
    pub qs: [u8; 128],
}

impl BlockQ4K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 144;
}

/// Q5_K: 5-bit quantization
/// Block size: 176 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5K {
    /// Super-block scale (f16)
    pub d: f16,
    /// Super-block min (f16)
    pub dmin: f16,
    /// Scales and mins
    pub scales: [u8; 12],
    /// High bits
    pub qh: [u8; 32],
    /// Low 4 bits
    pub qs: [u8; 128],
}

impl BlockQ5K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 176;
}

/// Q6_K: 6-bit quantization
/// Block size: 210 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ6K {
    /// Low 4 bits of quantized values
    pub ql: [u8; 128],
    /// High 2 bits of quantized values
    pub qh: [u8; 64],
    /// Scales (8-bit)
    pub scales: [i8; 16],
    /// Super-block scale (f16)
    pub d: f16,
}

impl BlockQ6K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 210;
}

/// Q8_K: 8-bit quantization (K-quant variant)
/// Block size: 292 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8K {
    /// Super-block scale (f32)
    pub d: f32,
    /// Quantized values
    pub qs: [i8; 256],
    /// Block sums for dot product
    pub bsums: [i16; 16],
}

impl BlockQ8K {
    pub const BLOCK_SIZE: usize = 256;
    pub const TYPE_SIZE: usize = 292;
}

// =============================================================================
// Size assertions to ensure layout matches llama.cpp
// =============================================================================

const _: () = {
    assert!(std::mem::size_of::<BlockQ4_0>() == BlockQ4_0::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ4_1>() == BlockQ4_1::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ5_0>() == BlockQ5_0::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ5_1>() == BlockQ5_1::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ8_0>() == BlockQ8_0::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ8_1>() == BlockQ8_1::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ2K>() == BlockQ2K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ3K>() == BlockQ3K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ4K>() == BlockQ4K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ5K>() == BlockQ5K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ6K>() == BlockQ6K::TYPE_SIZE);
    assert!(std::mem::size_of::<BlockQ8K>() == BlockQ8K::TYPE_SIZE);
};
```

**Step 2: Update tensor mod.rs to export quant**

Update `src/tensor/mod.rs`:
```rust
//! Tensor types and operations

mod dtype;
mod error;
pub mod quant;
mod storage;
mod tensor;

pub use dtype::DType;
pub use error::TensorError;
pub use storage::TensorStorage;
pub use tensor::Tensor;
```

**Step 3: Run tests**

Run: `cargo build`
Expected: Compiles successfully (compile-time size assertions verify layout)

**Step 4: Commit**

```bash
git add src/tensor/quant/
git commit -m "feat(tensor): add quantization block structures matching llama.cpp"
```

---

### Task 6: Basic Dequantization Functions

**Files:**
- Create: `src/tensor/quant/dequant.rs`
- Modify: `src/tensor/quant/mod.rs`
- Create: `tests/dequant_test.rs`

**Step 1: Implement dequantization for basic quant types**

Create `src/tensor/quant/dequant.rs`:
```rust
//! Dequantization functions

use super::blocks::*;
use half::f16;

/// Dequantize Q4_0 block to f32
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();

    for i in 0..16 {
        let byte = block.qs[i];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;

        output[i] = lo as f32 * d;
        output[i + 16] = hi as f32 * d;
    }
}

/// Dequantize Q4_1 block to f32
pub fn dequantize_q4_1(block: &BlockQ4_1, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let m = block.m.to_f32();

    for i in 0..16 {
        let byte = block.qs[i];
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;

        output[i] = lo * d + m;
        output[i + 16] = hi * d + m;
    }
}

/// Dequantize Q5_0 block to f32
pub fn dequantize_q5_0(block: &BlockQ5_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let qh = u32::from_le_bytes(block.qh);

    for i in 0..16 {
        let byte = block.qs[i];
        let lo4 = (byte & 0x0F) as i32;
        let hi4 = ((byte >> 4) & 0x0F) as i32;

        let lo5 = ((qh >> i) & 1) as i32;
        let hi5 = ((qh >> (i + 16)) & 1) as i32;

        let lo = (lo4 | (lo5 << 4)) - 16;
        let hi = (hi4 | (hi5 << 4)) - 16;

        output[i] = lo as f32 * d;
        output[i + 16] = hi as f32 * d;
    }
}

/// Dequantize Q5_1 block to f32
pub fn dequantize_q5_1(block: &BlockQ5_1, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let m = block.m.to_f32();
    let qh = u32::from_le_bytes(block.qh);

    for i in 0..16 {
        let byte = block.qs[i];
        let lo4 = (byte & 0x0F) as u32;
        let hi4 = ((byte >> 4) & 0x0F) as u32;

        let lo5 = (qh >> i) & 1;
        let hi5 = (qh >> (i + 16)) & 1;

        let lo = lo4 | (lo5 << 4);
        let hi = hi4 | (hi5 << 4);

        output[i] = lo as f32 * d + m;
        output[i + 16] = hi as f32 * d + m;
    }
}

/// Dequantize Q8_0 block to f32
pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();

    for i in 0..32 {
        output[i] = block.qs[i] as f32 * d;
    }
}

/// Quantize f32 values to Q4_0 block
pub fn quantize_q4_0(input: &[f32; 32]) -> BlockQ4_0 {
    // Find max absolute value
    let mut amax = 0.0f32;
    for &x in input.iter() {
        amax = amax.max(x.abs());
    }

    let d = amax / 7.0; // 4-bit signed range is -8 to 7
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];
    for i in 0..16 {
        let lo = ((input[i] * id).round() as i32).clamp(-8, 7) + 8;
        let hi = ((input[i + 16] * id).round() as i32).clamp(-8, 7) + 8;
        qs[i] = (lo as u8) | ((hi as u8) << 4);
    }

    BlockQ4_0 {
        d: f16::from_f32(d),
        qs,
    }
}

/// Quantize f32 values to Q8_0 block
pub fn quantize_q8_0(input: &[f32; 32]) -> BlockQ8_0 {
    // Find max absolute value
    let mut amax = 0.0f32;
    for &x in input.iter() {
        amax = amax.max(x.abs());
    }

    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (input[i] * id).round() as i8;
    }

    BlockQ8_0 {
        d: f16::from_f32(d),
        qs,
    }
}
```

**Step 2: Update mod.rs**

Update `src/tensor/quant/mod.rs`:
```rust
//! Quantization formats

mod blocks;
mod dequant;

pub use blocks::*;
pub use dequant::*;
```

**Step 3: Write dequantization tests**

Create `tests/dequant_test.rs`:
```rust
use llama_rs::tensor::quant::{
    dequantize_q4_0, dequantize_q8_0, quantize_q4_0, quantize_q8_0,
};

#[test]
fn test_q4_0_roundtrip() {
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);

    let block = quantize_q4_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_0(&block, &mut decoded);

    // Check values are close (quantization has some error)
    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!((o - d).abs() < 0.15, "original: {}, decoded: {}", o, d);
    }
}

#[test]
fn test_q8_0_roundtrip() {
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);

    let block = quantize_q8_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q8_0(&block, &mut decoded);

    // Q8 should be very close
    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!((o - d).abs() < 0.02, "original: {}, decoded: {}", o, d);
    }
}

#[test]
fn test_q4_0_zeros() {
    let original = [0.0f32; 32];
    let block = quantize_q4_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_0(&block, &mut decoded);

    for d in decoded.iter() {
        assert_eq!(*d, 0.0);
    }
}
```

**Step 4: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/tensor/quant/dequant.rs src/tensor/quant/mod.rs tests/dequant_test.rs
git commit -m "feat(tensor): implement basic dequantization (Q4_0, Q5_0, Q5_1, Q8_0)"
```

---

### Task 7: Backend Trait and CPU Backend Structure

**Files:**
- Modify: `src/backend/mod.rs`
- Modify: `src/backend/error.rs`
- Create: `src/backend/cpu/mod.rs`
- Create: `src/backend/cpu/ops.rs`

**Step 1: Define complete backend trait**

Update `src/backend/mod.rs`:
```rust
//! Hardware backends

mod error;
pub mod cpu;

pub use error::BackendError;

use crate::tensor::{DType, Tensor};

/// Result type for backend operations
pub type BackendResult<T> = Result<T, BackendError>;

/// Hardware backend trait for tensor operations
pub trait Backend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    // =========================================================================
    // Memory operations
    // =========================================================================

    /// Allocate tensor on this backend
    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor>;

    /// Copy tensor to this backend
    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor>;

    // =========================================================================
    // Element-wise operations
    // =========================================================================

    /// Element-wise addition: out = a + b
    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Element-wise multiplication: out = a * b
    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Scale tensor: out = a * scalar
    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Activation functions
    // =========================================================================

    /// SiLU activation: x * sigmoid(x)
    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// GELU activation
    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Softmax along last dimension
    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Normalization
    // =========================================================================

    /// RMS normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor)
        -> BackendResult<()>;

    // =========================================================================
    // Matrix operations
    // =========================================================================

    /// Matrix multiplication: out = a @ b
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Matrix-vector multiplication: out = a @ b where b is 1D
    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    // =========================================================================
    // Quantization
    // =========================================================================

    /// Dequantize tensor to f32
    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()>;

    /// Quantized matrix-vector multiply (for inference)
    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()>;
}

/// Get default backend (CPU)
pub fn default_backend() -> Box<dyn Backend> {
    Box::new(cpu::CpuBackend::new())
}
```

**Step 2: Expand error types**

Update `src/backend/error.rs`:
```rust
use crate::tensor::DType;

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("Backend not available: {0}")]
    NotAvailable(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDType(DType),

    #[error("Operation not supported: {0}")]
    Unsupported(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}
```

**Step 3: Create CPU backend structure**

Create `src/backend/cpu/mod.rs`:
```rust
//! CPU backend implementation

mod ops;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

/// CPU backend using SIMD operations
pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    /// Create new CPU backend
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Create with specific thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor> {
        Ok(Tensor::zeros(shape.to_vec(), dtype))
    }

    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor> {
        Ok(tensor.clone())
    }

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::add(a, b, out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::mul(a, b, out)
    }

    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
        ops::scale(a, scalar, out)
    }

    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::silu(x, out)
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::gelu(x, out)
    }

    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::softmax(x, out)
    }

    fn rms_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> BackendResult<()> {
        ops::rms_norm(x, weight, eps, out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::matmul(a, b, out)
    }

    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::matvec(a, b, out)
    }

    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::dequantize(src, out)
    }

    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        ops::matvec_q(a, b, out)
    }
}
```

**Step 4: Implement CPU operations**

Create `src/backend/cpu/ops.rs`:
```rust
//! CPU backend operations

use crate::backend::{BackendError, BackendResult};
use crate::tensor::quant::{dequantize_q4_0, dequantize_q8_0, BlockQ4_0, BlockQ8_0};
use crate::tensor::{DType, Tensor};
use rayon::prelude::*;

/// Element-wise addition
pub fn add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(a, b)?;
    check_same_shape(a, out)?;
    check_dtype(a, DType::F32)?;

    let a_data = a.as_f32().unwrap();
    let b_data = b.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    out_data
        .par_iter_mut()
        .zip(a_data.par_iter().zip(b_data.par_iter()))
        .for_each(|(o, (a, b))| *o = a + b);

    Ok(())
}

/// Element-wise multiplication
pub fn mul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(a, b)?;
    check_same_shape(a, out)?;
    check_dtype(a, DType::F32)?;

    let a_data = a.as_f32().unwrap();
    let b_data = b.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    out_data
        .par_iter_mut()
        .zip(a_data.par_iter().zip(b_data.par_iter()))
        .for_each(|(o, (a, b))| *o = a * b);

    Ok(())
}

/// Scale by scalar
pub fn scale(a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(a, out)?;
    check_dtype(a, DType::F32)?;

    let a_data = a.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    out_data
        .par_iter_mut()
        .zip(a_data.par_iter())
        .for_each(|(o, a)| *o = a * scalar);

    Ok(())
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;

    let x_data = x.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    out_data
        .par_iter_mut()
        .zip(x_data.par_iter())
        .for_each(|(o, &x)| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            *o = x * sigmoid;
        });

    Ok(())
}

/// GELU activation
pub fn gelu(x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;

    let x_data = x.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608;

    out_data
        .par_iter_mut()
        .zip(x_data.par_iter())
        .for_each(|(o, &x)| {
            let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
            *o = 0.5 * x * (1.0 + inner.tanh());
        });

    Ok(())
}

/// Softmax along last dimension
pub fn softmax(x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;

    let x_data = x.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    let last_dim = *x.shape().last().unwrap_or(&1);
    let n_rows = x.numel() / last_dim;

    // Process each row
    for row in 0..n_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_x = &x_data[start..end];
        let row_out = &mut out_data[start..end];

        // Find max for numerical stability
        let max = row_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for (o, &x) in row_out.iter_mut().zip(row_x.iter()) {
            *o = (x - max).exp();
            sum += *o;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for o in row_out.iter_mut() {
            *o *= inv_sum;
        }
    }

    Ok(())
}

/// RMS normalization
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) -> BackendResult<()> {
    check_same_shape(x, out)?;
    check_dtype(x, DType::F32)?;
    check_dtype(weight, DType::F32)?;

    let x_data = x.as_f32().unwrap();
    let w_data = weight.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    let hidden_size = *x.shape().last().unwrap_or(&1);
    let n_rows = x.numel() / hidden_size;

    if w_data.len() != hidden_size {
        return Err(BackendError::ShapeMismatch {
            expected: vec![hidden_size],
            got: weight.shape().to_vec(),
        });
    }

    for row in 0..n_rows {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let row_x = &x_data[start..end];
        let row_out = &mut out_data[start..end];

        // Compute RMS
        let ss: f32 = row_x.iter().map(|x| x * x).sum();
        let rms = (ss / hidden_size as f32 + eps).sqrt();
        let scale = 1.0 / rms;

        // Apply normalization and weight
        for ((o, &x), &w) in row_out.iter_mut().zip(row_x.iter()).zip(w_data.iter()) {
            *o = x * scale * w;
        }
    }

    Ok(())
}

/// Matrix multiplication (2D)
pub fn matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(a, DType::F32)?;
    check_dtype(b, DType::F32)?;
    check_dtype(out, DType::F32)?;

    // For now, simple 2D matmul
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(BackendError::InvalidArgument(
            "matmul requires 2D tensors".into(),
        ));
    }

    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);

    if k1 != k2 {
        return Err(BackendError::ShapeMismatch {
            expected: vec![m, k1],
            got: vec![k2, n],
        });
    }

    let a_data = a.as_f32().unwrap();
    let b_data = b.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    // Simple implementation (not optimized)
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..k1 {
                sum += a_data[i * k1 + k] * b_data[k * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }

    Ok(())
}

/// Matrix-vector multiplication
pub fn matvec(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(a, DType::F32)?;
    check_dtype(b, DType::F32)?;
    check_dtype(out, DType::F32)?;

    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(BackendError::InvalidArgument(
            "matvec requires 2D matrix and 1D vector".into(),
        ));
    }

    let (m, k) = (a.shape()[0], a.shape()[1]);
    if b.shape()[0] != k {
        return Err(BackendError::ShapeMismatch {
            expected: vec![k],
            got: b.shape().to_vec(),
        });
    }

    let a_data = a.as_f32().unwrap();
    let b_data = b.as_f32().unwrap();
    let out_data = out.as_f32_mut().unwrap();

    // Parallel over rows
    out_data.par_iter_mut().enumerate().for_each(|(i, o)| {
        let row_start = i * k;
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += a_data[row_start + j] * b_data[j];
        }
        *o = sum;
    });

    Ok(())
}

/// Dequantize tensor to f32
pub fn dequantize(src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    check_dtype(out, DType::F32)?;

    match src.dtype() {
        DType::Q4_0 => dequantize_blocks::<BlockQ4_0>(src, out, |block, output| {
            dequantize_q4_0(block, output);
        }),
        DType::Q8_0 => dequantize_blocks::<BlockQ8_0>(src, out, |block, output| {
            dequantize_q8_0(block, output);
        }),
        DType::F32 => {
            // Just copy
            let out_data = out.as_f32_mut().unwrap();
            out_data.copy_from_slice(src.as_f32().unwrap());
            Ok(())
        }
        dtype => Err(BackendError::UnsupportedDType(dtype)),
    }
}

fn dequantize_blocks<B: bytemuck::Pod>(
    src: &Tensor,
    out: &mut Tensor,
    dequant_fn: impl Fn(&B, &mut [f32; 32]) + Sync,
) -> BackendResult<()> {
    let blocks: &[B] = bytemuck::cast_slice(src.data());
    let out_data = out.as_f32_mut().unwrap();

    blocks.par_iter().enumerate().for_each(|(i, block)| {
        let mut tmp = [0.0f32; 32];
        dequant_fn(block, &mut tmp);
        let start = i * 32;
        out_data[start..start + 32].copy_from_slice(&tmp);
    });

    Ok(())
}

/// Quantized matrix-vector multiply
pub fn matvec_q(a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
    // For now, dequantize and use regular matvec
    // TODO: Implement fused quantized matmul for performance
    let mut a_f32 = Tensor::zeros(a.shape().to_vec(), DType::F32);
    dequantize(a, &mut a_f32)?;
    matvec(&a_f32, b, out)
}

// Helper functions

fn check_same_shape(a: &Tensor, b: &Tensor) -> BackendResult<()> {
    if a.shape() != b.shape() {
        return Err(BackendError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

fn check_dtype(t: &Tensor, expected: DType) -> BackendResult<()> {
    if t.dtype() != expected {
        return Err(BackendError::DTypeMismatch {
            expected,
            got: t.dtype(),
        });
    }
    Ok(())
}
```

**Step 5: Run tests**

Run: `cargo build`
Expected: Compiles successfully

**Step 6: Commit**

```bash
git add src/backend/
git commit -m "feat(backend): implement CPU backend with basic tensor operations"
```

---

### Task 8: Integration Test with CLI Info Command

**Files:**
- Modify: `src/main.rs`
- Modify: `src/lib.rs`
- Create: `tests/integration_test.rs`

**Step 1: Implement info command**

Update `src/main.rs`:
```rust
use clap::{Parser, Subcommand};
use llama_rs::gguf::GgufFile;

#[derive(Parser)]
#[command(name = "llama-rs")]
#[command(about = "Rust implementation of llama.cpp", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show information about a GGUF model
    Info {
        /// Path to the GGUF model file
        model: String,
    },
    /// Run inference on a model
    Run {
        /// Path to the GGUF model file
        model: String,
        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,
        /// Number of tokens to generate
        #[arg(short, long, default_value = "128")]
        n_predict: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { model } => {
            if let Err(e) = show_info(&model) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Run {
            model,
            prompt,
            n_predict,
        } => {
            println!("Running model: {}", model);
            if let Some(p) = prompt {
                println!("Prompt: {}", p);
            }
            println!("Tokens to generate: {}", n_predict);
            // TODO: Implement inference
            eprintln!("Inference not yet implemented");
        }
    }
}

fn show_info(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = GgufFile::open(path)?;
    let data = &file.data;

    println!("GGUF File: {}", path);
    println!("Version: {}", data.header.version);
    println!("Tensors: {}", data.header.tensor_count);
    println!("Metadata entries: {}", data.header.metadata_kv_count);
    println!();

    // Show key metadata
    if let Some(arch) = data.get_string("general.architecture") {
        println!("Architecture: {}", arch);
    }
    if let Some(name) = data.get_string("general.name") {
        println!("Name: {}", name);
    }

    // Model parameters
    let prefix = data
        .get_string("general.architecture")
        .unwrap_or("llama");

    if let Some(v) = data.get_u32(&format!("{}.context_length", prefix)) {
        println!("Context length: {}", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.embedding_length", prefix)) {
        println!("Embedding size: {}", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.block_count", prefix)) {
        println!("Layers: {}", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.attention.head_count", prefix)) {
        println!("Attention heads: {}", v);
    }
    if let Some(v) = data.get_u32(&format!("{}.attention.head_count_kv", prefix)) {
        println!("KV heads: {}", v);
    }

    println!();
    println!("Tensors:");
    for tensor in data.tensors.iter().take(10) {
        println!(
            "  {} [{:?}] {:?} @ {}",
            tensor.name,
            tensor.dims,
            tensor.dtype,
            tensor.offset
        );
    }
    if data.tensors.len() > 10 {
        println!("  ... and {} more", data.tensors.len() - 10);
    }

    Ok(())
}
```

**Step 2: Update lib.rs exports**

Update `src/lib.rs`:
```rust
//! llama-rs: A Rust implementation of llama.cpp
//!
//! High-performance LLM inference engine with support for GGUF models.

pub mod backend;
pub mod gguf;
pub mod tensor;

pub use backend::{default_backend, Backend, BackendError};
pub use gguf::{GgufData, GgufFile, GgufReader};
pub use tensor::{DType, Tensor, TensorError};

/// Library-wide error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF error: {0}")]
    Gguf(#[from] gguf::GgufError),

    #[error("Tensor error: {0}")]
    Tensor(#[from] tensor::TensorError),

    #[error("Backend error: {0}")]
    Backend(#[from] backend::BackendError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Step 3: Write integration test**

Create `tests/integration_test.rs`:
```rust
use llama_rs::backend::{default_backend, Backend};
use llama_rs::tensor::{DType, Tensor};

#[test]
fn test_backend_operations() {
    let backend = default_backend();
    assert_eq!(backend.name(), "cpu");
    assert!(backend.is_available());

    // Test allocation
    let tensor = backend.alloc(&[4, 4], DType::F32).unwrap();
    assert_eq!(tensor.shape(), &[4, 4]);
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn test_silu_activation() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.silu(&input, &mut output).unwrap();

    let result = output.as_f32().unwrap();
    // SiLU(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // SiLU(1) ≈ 0.731
    assert!((result[1] - 0.731).abs() < 0.01);
    // SiLU(-1) ≈ -0.269
    assert!((result[2] - (-0.269)).abs() < 0.01);
}

#[test]
fn test_softmax() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.softmax(&input, &mut output).unwrap();

    let result = output.as_f32().unwrap();
    // Sum should be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Values should be monotonically increasing
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
    assert!(result[2] < result[3]);
}

#[test]
fn test_rms_norm() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let weight = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

    let result = output.as_f32().unwrap();
    // After RMS norm with unit weights, values should be scaled
    // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.74
    // So output ≈ [0.37, 0.73, 1.10, 1.46]
    assert!((result[0] - 0.365).abs() < 0.01);
}
```

**Step 4: Run all tests**

Run: `cargo test`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/main.rs src/lib.rs tests/integration_test.rs
git commit -m "feat: implement CLI info command and add integration tests"
```

---

## Summary

Phase 1 implements the foundation:

| Task | Component | Description |
|------|-----------|-------------|
| 1 | Setup | Project structure, dependencies, module stubs |
| 2 | GGUF Types | Constants and types matching GGUF spec |
| 3 | GGUF Reader | Parse GGUF files with memory mapping |
| 4 | Tensor Core | Tensor type with storage and dtype |
| 5 | Quant Blocks | Block structures for all quantization formats |
| 6 | Dequantization | Basic dequant functions (Q4_0, Q5_x, Q8_0) |
| 7 | CPU Backend | Backend trait and CPU implementation |
| 8 | Integration | CLI info command and integration tests |

After Phase 1, you can:
- Load and inspect GGUF files
- Work with tensors (create, reshape, access data)
- Dequantize basic quantized tensors
- Run tensor operations on CPU

Phase 2 will add:
- Model architecture (LLaMA)
- KV cache
- Token generation loop
- Interactive inference
