//! Tensor struct implementation

use super::dtype::DType;
use super::error::TensorError;
use super::storage::TensorStorage;

/// Compute strides from shape (row-major order)
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// A multi-dimensional tensor with typed storage
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: TensorStorage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    offset: usize,
    /// Optional name for GPU weight lookup
    name: Option<String>,
}

impl Tensor {
    /// Create a new tensor from raw data bytes with the given shape and dtype
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Result<Self, TensorError> {
        let numel: usize = shape.iter().product();
        let expected_size = dtype.size_for_elements(numel);

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
            name: None,
        })
    }

    /// Create a tensor from existing storage
    ///
    /// # Safety
    /// The storage must contain valid data for the given shape and dtype.
    /// The offset + size must not exceed the storage length.
    pub unsafe fn from_storage(
        storage: TensorStorage,
        shape: Vec<usize>,
        dtype: DType,
        offset: usize,
    ) -> Result<Self, TensorError> {
        let numel: usize = shape.iter().product();
        let required_size = dtype.size_for_elements(numel);

        if offset + required_size > storage.len() {
            return Err(TensorError::SizeMismatch {
                expected: offset + required_size,
                got: storage.len(),
            });
        }

        let strides = compute_strides(&shape);

        Ok(Self {
            storage,
            shape,
            strides,
            dtype,
            offset,
            name: None,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let size = dtype.size_for_elements(numel);
        let data = vec![0u8; size];
        let strides = compute_strides(&shape);

        Self {
            storage: TensorStorage::owned(data),
            shape,
            strides,
            dtype,
            offset: 0,
            name: None,
        }
    }

    /// Create a tensor from f32 data
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self, TensorError> {
        let numel: usize = shape.iter().product();

        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }

        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        Self::new(bytes, shape, DType::F32)
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the tensor name (for GPU weight lookup)
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the tensor name (for GPU weight lookup)
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Create a named tensor (builder pattern)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the raw byte data
    pub fn data(&self) -> &[u8] {
        let size = self.dtype.size_for_elements(self.numel());
        &self.storage.as_bytes()[self.offset..self.offset + size]
    }

    /// Get mutable access to the raw byte data
    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        let size = self.dtype.size_for_elements(self.numel());
        let offset = self.offset;
        self.storage
            .as_bytes_mut()
            .map(|bytes| &mut bytes[offset..offset + size])
    }

    /// Get the data as f32 slice (only valid for F32 dtype)
    pub fn as_f32(&self) -> Result<&[f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::InvalidDType);
        }
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }

        let data = self.data();
        // SAFETY: We verified dtype is F32 and data is contiguous
        let f32_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, self.numel()) };
        Ok(f32_slice)
    }

    /// Get mutable access to data as f32 slice (only valid for F32 dtype)
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::InvalidDType);
        }
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }

        let numel = self.numel();
        let data = self.data_mut().ok_or(TensorError::NotContiguous)?;
        // SAFETY: We verified dtype is F32 and data is contiguous
        let f32_slice =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, numel) };
        Ok(f32_slice)
    }

    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let expected_strides = compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Return a contiguous copy of this tensor if not already contiguous
    pub fn contiguous(&self) -> Result<Self, TensorError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        // For non-contiguous tensors, we need to copy data
        // This is only supported for non-quantized types
        if self.dtype.is_quantized() {
            return Err(TensorError::NotContiguous);
        }

        // Create a new contiguous tensor with the same data
        let new_storage = self.storage.to_owned();
        let new_strides = compute_strides(&self.shape);

        Ok(Self {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
            name: self.name.clone(),
        })
    }

    /// Reshape the tensor to a new shape
    ///
    /// Returns a new tensor with the same data but different shape.
    /// The new tensor has its own copy of the data to allow mutation.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(TensorError::ShapeMismatch {
                expected: old_numel,
                got: new_numel,
            });
        }

        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }

        let new_strides = compute_strides(&new_shape);

        // Create owned copy to allow mutation
        Ok(Self {
            storage: self.storage.to_owned(),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: 0, // Reset offset since we copied the data
            name: self.name.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        // Empty shape
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());

        // 1D
        assert_eq!(compute_strides(&[5]), vec![1]);

        // 2D (row-major)
        assert_eq!(compute_strides(&[3, 4]), vec![4, 1]);

        // 3D
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.strides(), &[3, 1]);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_tensor_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, vec![2, 3]).unwrap();

        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);

        let f32_data = t.as_f32().unwrap();
        assert_eq!(f32_data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_from_f32_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = Tensor::from_f32(&data, vec![2, 3]);
        assert!(result.is_err());

        match result {
            Err(TensorError::ShapeMismatch { expected, got }) => {
                assert_eq!(expected, 6);
                assert_eq!(got, 3);
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, vec![2, 3]).unwrap();

        let reshaped = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);

        let reshaped_1d = t.reshape(vec![6]).unwrap();
        assert_eq!(reshaped_1d.shape(), &[6]);
        assert_eq!(reshaped_1d.strides(), &[1]);
    }

    #[test]
    fn test_tensor_reshape_invalid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, vec![2, 3]).unwrap();

        let result = t.reshape(vec![2, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_as_f32_mut() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut t = Tensor::from_f32(&data, vec![2, 2]).unwrap();

        {
            let f32_data = t.as_f32_mut().unwrap();
            f32_data[0] = 10.0;
            f32_data[3] = 40.0;
        }

        let f32_data = t.as_f32().unwrap();
        assert_eq!(f32_data, &[10.0, 2.0, 3.0, 40.0]);
    }

    #[test]
    fn test_tensor_quantized_zeros() {
        let t = Tensor::zeros(vec![32], DType::Q4_0);
        assert_eq!(t.shape(), &[32]);
        assert_eq!(t.numel(), 32);
        assert_eq!(t.dtype(), DType::Q4_0);
        // Q4_0: 18 bytes per 32 elements
        assert_eq!(t.data().len(), 18);
    }

    #[test]
    fn test_tensor_is_contiguous() {
        let t = Tensor::zeros(vec![2, 3, 4], DType::F32);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_tensor_new_size_mismatch() {
        // F32 needs 24 bytes for 6 elements, but we provide 20
        let data = vec![0u8; 20];
        let result = Tensor::new(data, vec![2, 3], DType::F32);
        assert!(result.is_err());

        match result {
            Err(TensorError::SizeMismatch { expected, got }) => {
                assert_eq!(expected, 24);
                assert_eq!(got, 20);
            }
            _ => panic!("Expected SizeMismatch error"),
        }
    }

    #[test]
    fn test_tensor_as_f32_wrong_dtype() {
        let t = Tensor::zeros(vec![4], DType::F16);
        let result = t.as_f32();
        assert!(matches!(result, Err(TensorError::InvalidDType)));
    }
}
