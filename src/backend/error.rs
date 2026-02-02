#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("Backend not available: {0}")]
    NotAvailable(String),
}
