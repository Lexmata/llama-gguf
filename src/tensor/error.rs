#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch")]
    ShapeMismatch,
}
