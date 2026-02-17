//! Tensor serialization for gRPC transfer
//!
//! Converts between the crate's `Tensor` type and the protobuf `TensorData`
//! message for efficient network transfer of hidden states and model weights.

use crate::tensor::{DType, Tensor};
use super::proto::TensorData;
use super::DistributedError;

/// Encode a DType as a u32 for protobuf transfer.
/// The discriminant order matches the DType enum definition.
pub fn dtype_to_u32(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F64 => 3,
        DType::I8 => 4,
        DType::I16 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::U8 => 8,
        DType::Q4_0 => 9,
        DType::Q4_1 => 10,
        DType::Q5_0 => 11,
        DType::Q5_1 => 12,
        DType::Q8_0 => 13,
        DType::Q8_1 => 14,
        DType::Q2K => 15,
        DType::Q3K => 16,
        DType::Q4K => 17,
        DType::Q5K => 18,
        DType::Q6K => 19,
        DType::Q8K => 20,
        DType::IQ2XXS => 21,
        DType::IQ2XS => 22,
        DType::IQ2S => 23,
        DType::IQ3XXS => 24,
        DType::IQ3S => 25,
        DType::IQ4XS => 26,
        DType::IQ4NL => 27,
        DType::IQ1S => 28,
    }
}

/// Decode a u32 back to a DType.
pub fn u32_to_dtype(val: u32) -> Result<DType, DistributedError> {
    match val {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::F64),
        4 => Ok(DType::I8),
        5 => Ok(DType::I16),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        8 => Ok(DType::U8),
        9 => Ok(DType::Q4_0),
        10 => Ok(DType::Q4_1),
        11 => Ok(DType::Q5_0),
        12 => Ok(DType::Q5_1),
        13 => Ok(DType::Q8_0),
        14 => Ok(DType::Q8_1),
        15 => Ok(DType::Q2K),
        16 => Ok(DType::Q3K),
        17 => Ok(DType::Q4K),
        18 => Ok(DType::Q5K),
        19 => Ok(DType::Q6K),
        20 => Ok(DType::Q8K),
        21 => Ok(DType::IQ2XXS),
        22 => Ok(DType::IQ2XS),
        23 => Ok(DType::IQ2S),
        24 => Ok(DType::IQ3XXS),
        25 => Ok(DType::IQ3S),
        26 => Ok(DType::IQ4XS),
        27 => Ok(DType::IQ4NL),
        28 => Ok(DType::IQ1S),
        other => Err(DistributedError::TensorTransfer(
            format!("unknown dtype discriminant: {}", other),
        )),
    }
}

/// Serialize a `Tensor` into a protobuf `TensorData` message.
pub fn tensor_to_proto(tensor: &Tensor) -> TensorData {
    TensorData {
        shape: tensor.shape().iter().map(|&s| s as u64).collect(),
        dtype: dtype_to_u32(tensor.dtype()),
        data: tensor.data().to_vec(),
        name: tensor.name().unwrap_or("").to_string(),
    }
}

/// Deserialize a protobuf `TensorData` message into a `Tensor`.
pub fn tensor_from_proto(proto: &TensorData) -> Result<Tensor, DistributedError> {
    let dtype = u32_to_dtype(proto.dtype)?;
    let shape: Vec<usize> = proto.shape.iter().map(|&s| s as usize).collect();
    let data = proto.data.clone();

    let mut tensor = Tensor::new(data, shape, dtype).map_err(|e| {
        DistributedError::TensorTransfer(format!("failed to reconstruct tensor: {}", e))
    })?;

    if !proto.name.is_empty() {
        tensor.set_name(&proto.name);
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_roundtrip() {
        let dtypes = [
            DType::F32, DType::F16, DType::BF16, DType::F64,
            DType::I8, DType::I16, DType::I32, DType::I64, DType::U8,
            DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1,
            DType::Q8_0, DType::Q8_1,
            DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K,
            DType::IQ2XXS, DType::IQ2XS, DType::IQ2S,
            DType::IQ3XXS, DType::IQ3S, DType::IQ4XS, DType::IQ4NL, DType::IQ1S,
        ];

        for dt in dtypes {
            let encoded = dtype_to_u32(dt);
            let decoded = u32_to_dtype(encoded).unwrap();
            assert_eq!(dt, decoded, "roundtrip failed for {:?}", dt);
        }
    }

    #[test]
    fn test_f32_tensor_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, vec![2, 3]).unwrap();

        let proto = tensor_to_proto(&tensor);
        assert_eq!(proto.shape, vec![2u64, 3]);
        assert_eq!(proto.dtype, 0); // F32

        let restored = tensor_from_proto(&proto).unwrap();
        assert_eq!(restored.shape(), &[2, 3]);
        assert_eq!(restored.dtype(), DType::F32);
        assert_eq!(restored.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_named_tensor_roundtrip() {
        let data = vec![0.5f32; 8];
        let tensor = Tensor::from_f32(&data, vec![8])
            .unwrap()
            .with_name("test_weight");

        let proto = tensor_to_proto(&tensor);
        assert_eq!(proto.name, "test_weight");

        let restored = tensor_from_proto(&proto).unwrap();
        assert_eq!(restored.name(), Some("test_weight"));
    }

    #[test]
    fn test_zeros_tensor_roundtrip() {
        let tensor = Tensor::zeros(vec![4, 4], DType::F32);
        let proto = tensor_to_proto(&tensor);
        let restored = tensor_from_proto(&proto).unwrap();

        assert_eq!(restored.shape(), &[4, 4]);
        let vals = restored.as_f32().unwrap();
        assert!(vals.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_quantized_tensor_roundtrip() {
        // Q4_0: 18 bytes per 32 elements
        let raw = vec![0u8; 18];
        let tensor = Tensor::new(raw, vec![32], DType::Q4_0).unwrap();

        let proto = tensor_to_proto(&tensor);
        assert_eq!(proto.dtype, dtype_to_u32(DType::Q4_0));

        let restored = tensor_from_proto(&proto).unwrap();
        assert_eq!(restored.shape(), &[32]);
        assert_eq!(restored.dtype(), DType::Q4_0);
        assert_eq!(restored.data().len(), 18);
    }

    #[test]
    fn test_invalid_dtype() {
        assert!(u32_to_dtype(999).is_err());
    }
}
