#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("Invalid magic number")]
    InvalidMagic,
}
