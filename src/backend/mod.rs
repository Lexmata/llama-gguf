mod error;
pub use error::BackendError;
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
}
