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
