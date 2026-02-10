//! ONNX file reader
//!
//! Parses `.onnx` protobuf files and provides access to model metadata,
//! graph structure, and initializer tensors (weights).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use prost::Message;

use crate::tensor::DType;

use super::error::{OnnxError, OnnxResult};
use super::proto;

/// Parsed ONNX model file
pub struct OnnxFile {
    /// The decoded protobuf model
    pub model: proto::ModelProto,
    /// Directory containing the ONNX file (for resolving external data)
    pub base_dir: PathBuf,
}

/// Metadata about an ONNX model
#[derive(Debug, Clone)]
pub struct OnnxMetadata {
    /// IR version
    pub ir_version: i64,
    /// Producer name (e.g., "pytorch", "optimum")
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Model domain
    pub domain: String,
    /// Model version
    pub model_version: i64,
    /// Documentation string
    pub doc_string: String,
    /// Opset imports (domain -> version)
    pub opset_imports: Vec<(String, i64)>,
}

/// Information about a tensor in the ONNX model
#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub dims: Vec<i64>,
    /// Data type
    pub data_type: i32,
    /// Total number of elements
    pub n_elements: usize,
}

impl OnnxFile {
    /// Open and parse an ONNX file
    pub fn open<P: AsRef<Path>>(path: P) -> OnnxResult<Self> {
        let path = path.as_ref();
        let base_dir = path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let data = std::fs::read(path)?;
        let model = proto::ModelProto::decode(data.as_slice())?;
        Ok(Self { model, base_dir })
    }

    /// Get model metadata
    pub fn metadata(&self) -> OnnxMetadata {
        let opset_imports = self
            .model
            .opset_import
            .iter()
            .map(|op| (op.domain.clone(), op.version))
            .collect();

        OnnxMetadata {
            ir_version: self.model.ir_version,
            producer_name: self.model.producer_name.clone(),
            producer_version: self.model.producer_version.clone(),
            domain: self.model.domain.clone(),
            model_version: self.model.model_version,
            doc_string: self.model.doc_string.clone(),
            opset_imports,
        }
    }

    /// Get the computation graph
    pub fn graph(&self) -> OnnxResult<&proto::GraphProto> {
        self.model.graph.as_ref().ok_or(OnnxError::MissingGraph)
    }

    /// Get all initializer tensors (weights) as a name-indexed map
    pub fn initializers(&self) -> OnnxResult<HashMap<&str, &proto::TensorProto>> {
        let graph = self.graph()?;
        let mut map = HashMap::new();
        for init in &graph.initializer {
            map.insert(init.name.as_str(), init);
        }
        Ok(map)
    }

    /// Get information about all initializer tensors
    pub fn tensor_infos(&self) -> OnnxResult<Vec<OnnxTensorInfo>> {
        let graph = self.graph()?;
        Ok(graph
            .initializer
            .iter()
            .map(|t| {
                let n_elements = t.dims.iter().map(|&d| d as usize).product::<usize>();
                OnnxTensorInfo {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    data_type: t.data_type,
                    n_elements,
                }
            })
            .collect())
    }

    /// Get a specific initializer tensor by name
    pub fn get_initializer(&self, name: &str) -> OnnxResult<&proto::TensorProto> {
        let graph = self.graph()?;
        graph
            .initializer
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| OnnxError::MissingTensor(name.to_string()))
    }

    /// Get graph node information (operators)
    pub fn nodes(&self) -> OnnxResult<&[proto::NodeProto]> {
        Ok(&self.graph()?.node)
    }

    /// Get graph input names and shapes
    pub fn inputs(&self) -> OnnxResult<Vec<(&str, Vec<i64>)>> {
        let graph = self.graph()?;
        Ok(graph
            .input
            .iter()
            .map(|inp| {
                let dims = inp
                    .r#type
                    .as_ref()
                    .and_then(|t| t.value.as_ref())
                    .and_then(|v| {
                        if let proto::type_proto::Value::TensorType(tt) = v {
                            tt.shape.as_ref().map(|s| {
                                s.dim
                                    .iter()
                                    .map(|d| match &d.value {
                                        Some(proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                                        _ => -1,
                                    })
                                    .collect()
                            })
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                (inp.name.as_str(), dims)
            })
            .collect())
    }

    /// Get graph output names
    pub fn outputs(&self) -> OnnxResult<Vec<&str>> {
        let graph = self.graph()?;
        Ok(graph.output.iter().map(|o| o.name.as_str()).collect())
    }
}

/// Extract raw tensor data from an ONNX TensorProto
///
/// ONNX tensors can store data in several ways:
/// - `raw_data`: raw bytes in little-endian format
/// - `float_data`: packed f32 values
/// - `int32_data`: packed i32 values (also used for f16/bf16 as bit patterns)
/// - `int64_data`: packed i64 values
/// - `double_data`: packed f64 values
/// - External data: stored in a separate file referenced by `external_data` field
pub fn extract_tensor_bytes(tensor: &proto::TensorProto, base_dir: &Path) -> OnnxResult<Vec<u8>> {
    // Check for external data first
    if tensor.data_location == 1 && !tensor.external_data.is_empty() {
        return extract_external_tensor_bytes(tensor, base_dir);
    }

    let data_type = tensor.data_type;

    // If raw_data is present, use it directly
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }

    // Otherwise, reconstruct from typed fields
    match data_type {
        // FLOAT (1)
        1 => {
            let mut bytes = Vec::with_capacity(tensor.float_data.len() * 4);
            for &val in &tensor.float_data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            Ok(bytes)
        }
        // FLOAT16 (10) - stored in int32_data as bit patterns
        10 => {
            let mut bytes = Vec::with_capacity(tensor.int32_data.len() * 2);
            for &val in &tensor.int32_data {
                bytes.extend_from_slice(&(val as u16).to_le_bytes());
            }
            Ok(bytes)
        }
        // BFLOAT16 (16) - stored in int32_data as bit patterns
        16 => {
            let mut bytes = Vec::with_capacity(tensor.int32_data.len() * 2);
            for &val in &tensor.int32_data {
                bytes.extend_from_slice(&(val as u16).to_le_bytes());
            }
            Ok(bytes)
        }
        // INT32 (6)
        6 => {
            let mut bytes = Vec::with_capacity(tensor.int32_data.len() * 4);
            for &val in &tensor.int32_data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            Ok(bytes)
        }
        // INT64 (7)
        7 => {
            let mut bytes = Vec::with_capacity(tensor.int64_data.len() * 8);
            for &val in &tensor.int64_data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            Ok(bytes)
        }
        // DOUBLE (11)
        11 => {
            let mut bytes = Vec::with_capacity(tensor.double_data.len() * 8);
            for &val in &tensor.double_data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            Ok(bytes)
        }
        // INT8 (3), UINT8 (2) - stored in int32_data
        2 | 3 => {
            let bytes: Vec<u8> = tensor.int32_data.iter().map(|&v| v as u8).collect();
            Ok(bytes)
        }
        _ => Err(OnnxError::UnsupportedDataType(data_type)),
    }
}

/// Extract tensor data from an external file
///
/// ONNX external data has key-value metadata:
/// - `location`: filename relative to the model file
/// - `offset`: byte offset into the file (default 0)
/// - `length`: number of bytes to read (default: rest of file from offset)
fn extract_external_tensor_bytes(
    tensor: &proto::TensorProto,
    base_dir: &Path,
) -> OnnxResult<Vec<u8>> {
    let mut location = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for kv in &tensor.external_data {
        match kv.key.as_str() {
            "location" => location = Some(kv.value.as_str()),
            "offset" => offset = kv.value.parse().unwrap_or(0),
            "length" => length = Some(kv.value.parse().unwrap_or(0)),
            _ => {}
        }
    }

    let filename = location.ok_or_else(|| {
        OnnxError::Other(format!(
            "External tensor '{}' missing 'location' field",
            tensor.name
        ))
    })?;

    let file_path = base_dir.join(filename);

    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(&file_path).map_err(|e| {
        OnnxError::Other(format!(
            "Failed to open external data file '{}': {}",
            file_path.display(),
            e
        ))
    })?;

    file.seek(SeekFrom::Start(offset)).map_err(|e| {
        OnnxError::Other(format!(
            "Failed to seek in external data file '{}': {}",
            file_path.display(),
            e
        ))
    })?;

    let data = if let Some(len) = length {
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf).map_err(|e| {
            OnnxError::Other(format!(
                "Failed to read {} bytes from external data file '{}': {}",
                len,
                file_path.display(),
                e
            ))
        })?;
        buf
    } else {
        // Read all remaining data from offset
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(|e| {
            OnnxError::Other(format!(
                "Failed to read external data file '{}': {}",
                file_path.display(),
                e
            ))
        })?;
        buf
    };

    Ok(data)
}

/// Convert an ONNX data type ID to our DType
pub fn onnx_dtype_to_dtype(onnx_type: i32) -> OnnxResult<DType> {
    match onnx_type {
        1 => Ok(DType::F32),     // FLOAT
        2 => Ok(DType::U8),      // UINT8
        3 => Ok(DType::I8),      // INT8
        5 => Ok(DType::I16),     // INT16
        6 => Ok(DType::I32),     // INT32
        7 => Ok(DType::I64),     // INT64
        10 => Ok(DType::F16),    // FLOAT16
        11 => Ok(DType::F64),    // DOUBLE
        16 => Ok(DType::BF16),   // BFLOAT16
        other => Err(OnnxError::UnsupportedDataType(other)),
    }
}

/// Build a mapping from opaque initializer names (like `onnx::MatMul_345`) to
/// logical HuggingFace names by tracing the computation graph.
///
/// The ONNX exporter (optimum/torch) often renames weight tensors to opaque names
/// during export. However, the graph node output names preserve the logical path
/// (e.g., `/model/layers.0/self_attn/q_proj/MatMul_output_0`).
///
/// This function:
/// 1. Scans `MatMul` nodes to find which initializer feeds each projection
/// 2. Scans `Identity` nodes to find weight aliases (shared weights)
/// 3. Returns a map: opaque_name -> logical_hf_name
pub fn trace_graph_tensor_names(onnx: &OnnxFile) -> OnnxResult<HashMap<String, String>> {
    let graph = onnx.graph()?;
    let init_names: std::collections::HashSet<&str> = graph
        .initializer
        .iter()
        .map(|i| i.name.as_str())
        .collect();

    let mut name_map: HashMap<String, String> = HashMap::new();

    // 1. Process Identity nodes first (weight aliases like norm sharing)
    for node in &graph.node {
        if node.op_type == "Identity" && !node.input.is_empty() && !node.output.is_empty() {
            let src = &node.input[0];
            let dst = &node.output[0];
            // If the source is an initializer and the destination has a recognizable name,
            // register the initializer as also being available under the alias
            if init_names.contains(src.as_str()) {
                name_map.insert(dst.clone(), src.clone());
            }
        }
    }

    // Build a reverse map from node outputs to the graph output names
    // This helps trace chains like: MatMul -> Cast -> logits
    let graph_output_names: std::collections::HashSet<&str> =
        graph.output.iter().map(|o| o.name.as_str()).collect();
    let mut output_aliases: HashMap<String, String> = HashMap::new();

    // Find Cast/Identity chains that lead to graph outputs (like logits)
    for node in &graph.node {
        if (node.op_type == "Cast" || node.op_type == "Identity")
            && !node.input.is_empty()
            && !node.output.is_empty()
        {
            let out = &node.output[0];
            if graph_output_names.contains(out.as_str()) {
                // This node's input feeds into a graph output
                output_aliases.insert(node.input[0].clone(), out.clone());
            }
        }
    }

    // 2. Process MatMul nodes to map opaque weight names to logical names
    for node in &graph.node {
        if node.op_type != "MatMul" || node.output.is_empty() {
            continue;
        }

        // Find the initializer input (if any)
        let weight_input = node
            .input
            .iter()
            .find(|inp| init_names.contains(inp.as_str()));

        let weight_name = match weight_input {
            Some(w) => w,
            None => continue, // This MatMul doesn't use an initializer (e.g., Q*K^T)
        };

        // If the weight already has a recognizable HF name, skip
        if !weight_name.starts_with("onnx::") {
            continue;
        }

        let output_name = &node.output[0];

        // Check if this MatMul's output leads to a graph output (via Cast chain)
        let effective_output = output_aliases
            .get(output_name.as_str())
            .map(|s| s.as_str())
            .unwrap_or(output_name.as_str());

        // Parse the output name to extract the logical HF path
        if let Some(hf_name) = matmul_output_to_hf_name(effective_output) {
            name_map.insert(weight_name.clone(), hf_name);
        }
    }

    Ok(name_map)
}

/// Convert a MatMul output name to its corresponding HuggingFace tensor name.
///
/// Examples:
/// - `/model/layers.0/self_attn/q_proj/MatMul_output_0` -> `model.layers.0.self_attn.q_proj.weight`
/// - `/model/layers.0/mlp/gate_proj/MatMul_output_0` -> `model.layers.0.mlp.gate_proj.weight`
/// - `/model/layers.0/self_attn/o_proj/MatMul_output_0` -> `model.layers.0.self_attn.o_proj.weight`
/// - `logits` -> `lm_head.weight`
fn matmul_output_to_hf_name(output_name: &str) -> Option<String> {
    // Special case: output logits -> lm_head
    if output_name == "logits" {
        return Some("lm_head.weight".to_string());
    }

    // Pattern: /model/layers.{i}/{component}/MatMul_output_0
    // or: /model/{component}/MatMul_output_0
    let path = output_name.strip_prefix('/')?;

    // Remove the /MatMul_output_N suffix
    let path = if let Some(idx) = path.rfind("/MatMul") {
        &path[..idx]
    } else {
        return None;
    };

    // Convert slashes to dots: model/layers.0/self_attn/q_proj -> model.layers.0.self_attn.q_proj
    let hf_name = path.replace('/', ".");

    // Append .weight
    Some(format!("{}.weight", hf_name))
}

/// Get a human-readable name for an ONNX data type
pub fn onnx_dtype_name(onnx_type: i32) -> &'static str {
    match onnx_type {
        0 => "UNDEFINED",
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        16 => "BFLOAT16",
        _ => "UNKNOWN",
    }
}
