//! Error types for ONNX operations

/// Errors that can occur during ONNX file operations.
#[derive(thiserror::Error, Debug)]
pub enum OnnxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),

    #[error("Missing ONNX graph in model")]
    MissingGraph,

    #[error("Missing initializer tensor: {0}")]
    MissingTensor(String),

    #[error("Unsupported ONNX data type: {0}")]
    UnsupportedDataType(i32),

    #[error("Missing config file: {0}")]
    MissingConfig(String),

    #[error("Config parse error: {0}")]
    ConfigParse(String),

    #[error("Unsupported model type: {0}")]
    UnsupportedModelType(String),

    #[error("Tensor shape mismatch for {name}: expected {expected} elements, got {actual}")]
    ShapeMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("Model error: {0}")]
    Model(#[from] crate::model::ModelError),

    #[error("Tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),

    #[error("ONNX error: {0}")]
    Other(String),
}

pub type OnnxResult<T> = std::result::Result<T, OnnxError>;
