use crate::backend::BackendError;

pub fn check_device_available() -> Result<(), BackendError> {
    Err(BackendError::NotAvailable(
        "Hailo device check not yet implemented".to_string(),
    ))
}
