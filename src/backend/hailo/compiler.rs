//! HEF auto-compile module: ONNX subgraph export and DFC invocation.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend::{Backend, BackendError, BackendResult};
use crate::model::layers::{AttentionLayer, FfnLayer, TransformerLayer};
use crate::model::ModelConfig;
use crate::tensor::{DType, Tensor};

use super::config::{HailoConfig, HefManifest};

use crate::backend::cpu::CpuBackend;

pub fn model_config_hash(config: &ModelConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.hidden_size.hash(&mut hasher);
    config.num_heads.hash(&mut hasher);
    config.num_kv_heads.hash(&mut hasher);
    config.head_dim.hash(&mut hasher);
    config.intermediate_size.hash(&mut hasher);
    config.num_layers.hash(&mut hasher);
    config.norm_eps.to_bits().hash(&mut hasher);
    hasher.finish()
}

pub fn resolve_hef_path(config: &HailoConfig, layer_idx: usize, kind: &str) -> PathBuf {
    let dir = config.hef_dir.as_ref().unwrap_or(&config.cache_dir);
    dir.join(format!("layer_{:02}_{}.hef", layer_idx, kind))
}

pub fn write_manifest(config: &HailoConfig, model_config: &ModelConfig) -> BackendResult<()> {
    let manifest = HefManifest {
        model_config_hash: model_config_hash(model_config),
        hidden_size: model_config.hidden_size,
        num_heads: model_config.num_heads,
        num_kv_heads: model_config.num_kv_heads,
        head_dim: model_config.head_dim,
        intermediate_size: model_config.intermediate_size,
        num_layers: model_config.num_layers,
        target_arch: config
            .target_arch
            .as_deref()
            .unwrap_or("hailo8l")
            .to_string(),
    };
    let manifest_path = config.cache_dir.join("manifest.json");
    fs::create_dir_all(&config.cache_dir).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to create cache dir: {}", e))
    })?;
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to serialize manifest: {}", e))
    })?;
    fs::write(&manifest_path, json).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to write manifest: {}", e))
    })?;
    Ok(())
}

pub fn check_cache(config: &HailoConfig, model_config: &ModelConfig) -> bool {
    let manifest_path = config.cache_dir.join("manifest.json");
    let Ok(contents) = fs::read_to_string(&manifest_path) else {
        return false;
    };
    let manifest: HefManifest = match serde_json::from_str(&contents) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if manifest.model_config_hash != model_config_hash(model_config) {
        return false;
    }
    let dir = config.hef_dir.as_ref().unwrap_or(&config.cache_dir);
    for layer_idx in 0..model_config.num_layers {
        let attn_path = dir.join(format!("layer_{:02}_attn.hef", layer_idx));
        let ffn_path = dir.join(format!("layer_{:02}_ffn.hef", layer_idx));
        if !attn_path.exists() || !ffn_path.exists() {
            return false;
        }
    }
    true
}

fn tensor_to_f32(t: &Tensor, cpu: &CpuBackend) -> BackendResult<Vec<f32>> {
    if t.dtype() == DType::F32 {
        return Ok(t.as_f32()?.to_vec());
    }
    let mut out = Tensor::zeros(t.shape().to_vec(), DType::F32);
    cpu.dequantize(t, &mut out)?;
    Ok(out.as_f32()?.to_vec())
}

fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> BackendResult<()> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","))
    };
    let header = format!("{{'descr':'<f4','fortran_order':False,'shape':{},}}\n", shape_str);
    let header_len = header.len();
    let padded_len: usize = 54;
    let padding = padded_len.saturating_sub(header_len);

    let mut file = fs::File::create(path)
        .map_err(|e| BackendError::OperationFailed(format!("Failed to create npy file: {}", e)))?;
    use std::io::Write;
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])
        .map_err(|e| BackendError::OperationFailed(format!("Failed to write npy: {}", e)))?;
    file.write_all(&[0x01, 0x00])
        .map_err(|e| BackendError::OperationFailed(format!("Failed to write npy: {}", e)))?;
    file.write_all(&(padded_len as u16).to_le_bytes())
        .map_err(|e| BackendError::OperationFailed(format!("Failed to write npy: {}", e)))?;
    file.write_all(header.as_bytes())
        .map_err(|e| BackendError::OperationFailed(format!("Failed to write npy: {}", e)))?;
    for _ in 0..padding {
        file.write_all(&[0x20])
            .map_err(|e| BackendError::OperationFailed(format!("Failed to write npy: {}", e)))?;
    }
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    file.write_all(bytes)
        .map_err(|e| BackendError::OperationFailed(format!("Failed to write npy: {}", e)))?;
    Ok(())
}

pub fn export_attn_onnx(
    layer: &TransformerLayer,
    layer_idx: usize,
    config: &ModelConfig,
    output_dir: &Path,
) -> BackendResult<PathBuf> {
    let attn = match &layer.attn_layer {
        AttentionLayer::FullAttention(a) => a,
        _ => {
            return Err(BackendError::Unsupported(
                "Hailo compiler requires FullAttention layers".into(),
            ))
        }
    };

    let cpu = CpuBackend::new();
    let hidden_size = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let q_size = num_heads * head_dim;
    let k_size = num_kv_heads * head_dim;
    let v_size = num_kv_heads * head_dim;

    fs::create_dir_all(output_dir).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to create output dir: {}", e))
    })?;

    let norm_weight = tensor_to_f32(layer.attn_norm.weight(), &cpu)?;
    let wq = tensor_to_f32(&attn.wq.weight, &cpu)?;
    let wk = tensor_to_f32(&attn.wk.weight, &cpu)?;
    let wv = tensor_to_f32(&attn.wv.weight, &cpu)?;

    let prefix = output_dir.join(format!("layer_{:02}_attn", layer_idx));
    write_npy_f32(&(prefix.with_extension("norm.npy")), &norm_weight, &[hidden_size])?;
    write_npy_f32(&(prefix.with_extension("wq.npy")), &wq, &[hidden_size, q_size])?;
    write_npy_f32(&(prefix.with_extension("wk.npy")), &wk, &[hidden_size, k_size])?;
    write_npy_f32(&(prefix.with_extension("wv.npy")), &wv, &[hidden_size, v_size])?;

    let onnx_path = prefix.with_extension("onnx");
    let script = format!(
        r#"
import numpy as np
import onnx
from onnx import helper, TensorProto

hidden_size = {}
num_heads = {}
num_kv_heads = {}
head_dim = {}
norm_eps = {}
q_size = num_heads * head_dim
k_size = num_kv_heads * head_dim
v_size = num_kv_heads * head_dim

norm_w = np.load("{}")
wq = np.load("{}")
wk = np.load("{}")
wv = np.load("{}")

input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [hidden_size])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [q_size + k_size + v_size])

norm_init = helper.make_tensor("norm_weight", TensorProto.FLOAT, [hidden_size], norm_w.flatten().tolist())
wq_init = helper.make_tensor("wq", TensorProto.FLOAT, [hidden_size, q_size], wq.flatten().tolist())
wk_init = helper.make_tensor("wk", TensorProto.FLOAT, [hidden_size, k_size], wk.flatten().tolist())
wv_init = helper.make_tensor("wv", TensorProto.FLOAT, [hidden_size, v_size], wv.flatten().tolist())

eps_init = helper.make_tensor("eps", TensorProto.FLOAT, [], [norm_eps])

mul_in = helper.make_node("Mul", ["input", "input"], ["x_sq"])
rms_mean = helper.make_node("ReduceMean", ["x_sq"], ["rms_in"], axes=[0], keepdims=0)
add_eps = helper.make_node("Add", ["rms_in", "eps"], ["rms_add"])
sqrt_node = helper.make_node("Sqrt", ["rms_add"], ["rms"])
div_node = helper.make_node("Div", ["input", "rms"], ["normed_div"])
norm_mul = helper.make_node("Mul", ["normed_div", "norm_weight"], ["normed"])
q_node = helper.make_node("Gemm", ["normed", "wq"], ["q"], transB=1)
k_node = helper.make_node("Gemm", ["normed", "wk"], ["k"], transB=1)
v_node = helper.make_node("Gemm", ["normed", "wv"], ["v"], transB=1)
concat_node = helper.make_node("Concat", ["q", "k", "v"], ["output"], axis=0)

graph = helper.make_graph(
    [mul_in, rms_mean, add_eps, sqrt_node, div_node, norm_mul, q_node, k_node, v_node, concat_node],
    "attn",
    [input],
    [output],
    initializer=[norm_init, wq_init, wk_init, wv_init, eps_init]
)

model = helper.make_model(graph)
onnx.save(model, "{}")
"#,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        config.norm_eps,
        prefix.with_extension("norm.npy").display(),
        prefix.with_extension("wq.npy").display(),
        prefix.with_extension("wk.npy").display(),
        prefix.with_extension("wv.npy").display(),
        onnx_path.display()
    );

    let script_path = prefix.with_extension("py");
    fs::write(&script_path, script.trim_start()).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to write export script: {}", e))
    })?;

    let status = Command::new("python3")
        .arg(script_path)
        .current_dir(output_dir)
        .status()
        .map_err(|e| {
            BackendError::OperationFailed(format!(
                "Failed to run ONNX export script (is Python with onnx/numpy installed?): {}",
                e
            ))
        })?;

    if !status.success() {
        return Err(BackendError::OperationFailed(
            "ONNX export script failed. Ensure Python 3 with onnx and numpy is installed."
                .into(),
        ));
    }

    Ok(onnx_path)
}

pub fn export_ffn_onnx(
    layer: &TransformerLayer,
    layer_idx: usize,
    config: &ModelConfig,
    output_dir: &Path,
) -> BackendResult<PathBuf> {
    let cpu = CpuBackend::new();
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let norm_eps = config.norm_eps;

    fs::create_dir_all(output_dir).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to create output dir: {}", e))
    })?;

    let (gate_weight, up_weight, down_weight, use_gelu) = match &layer.ffn_layer {
        FfnLayer::Dense(ffn) => (
            tensor_to_f32(&ffn.w_gate.weight, &cpu)?,
            tensor_to_f32(&ffn.w_up.weight, &cpu)?,
            tensor_to_f32(&ffn.w_down.weight, &cpu)?,
            false,
        ),
        FfnLayer::NoGate(ffn) => (
            vec![],
            tensor_to_f32(&ffn.w_up.weight, &cpu)?,
            tensor_to_f32(&ffn.w_down.weight, &cpu)?,
            ffn.use_gelu,
        ),
        _ => {
            return Err(BackendError::Unsupported(
                "Hailo compiler requires Dense or NoGate FFN layers".into(),
            ))
        }
    };

    let norm_weight = tensor_to_f32(layer.ffn_norm.weight(), &cpu)?;
    let prefix = output_dir.join(format!("layer_{:02}_ffn", layer_idx));
    write_npy_f32(&(prefix.with_extension("norm.npy")), &norm_weight, &[hidden_size])?;
    write_npy_f32(&(prefix.with_extension("down.npy")), &down_weight, &[intermediate_size, hidden_size])?;

    let has_gate = !gate_weight.is_empty();
    if has_gate {
        write_npy_f32(&(prefix.with_extension("gate.npy")), &gate_weight, &[hidden_size, intermediate_size])?;
        write_npy_f32(&(prefix.with_extension("up.npy")), &up_weight, &[hidden_size, intermediate_size])?;
    } else {
        write_npy_f32(&(prefix.with_extension("up.npy")), &up_weight, &[hidden_size, intermediate_size])?;
    }

    let onnx_path = prefix.with_extension("onnx");

    let script = if has_gate {
        format!(
            r#"
import numpy as np
import onnx
from onnx import helper, TensorProto

hidden_size = {}
intermediate_size = {}
norm_eps = {}

norm_w = np.load("{}")
gate_w = np.load("{}")
up_w = np.load("{}")
down_w = np.load("{}")

input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [hidden_size])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [hidden_size])

norm_init = helper.make_tensor("norm_weight", TensorProto.FLOAT, [hidden_size], norm_w.flatten().tolist())
gate_init = helper.make_tensor("gate", TensorProto.FLOAT, [hidden_size, intermediate_size], gate_w.flatten().tolist())
up_init = helper.make_tensor("up", TensorProto.FLOAT, [hidden_size, intermediate_size], up_w.flatten().tolist())
down_init = helper.make_tensor("down", TensorProto.FLOAT, [intermediate_size, hidden_size], down_w.flatten().tolist())
eps_init = helper.make_tensor("eps", TensorProto.FLOAT, [], [norm_eps])

mul_in = helper.make_node("Mul", ["input", "input"], ["x_sq"])
rms_mean = helper.make_node("ReduceMean", ["x_sq"], ["rms_in"], axes=[0], keepdims=0)
add_eps = helper.make_node("Add", ["rms_in", "eps"], ["rms_add"])
sqrt_node = helper.make_node("Sqrt", ["rms_add"], ["rms"])
div_node = helper.make_node("Div", ["input", "rms"], ["normed_div"])
norm_mul = helper.make_node("Mul", ["normed_div", "norm_weight"], ["normed"])
gate_node = helper.make_node("Gemm", ["normed", "gate"], ["gate_out"], transB=1)
up_node = helper.make_node("Gemm", ["normed", "up"], ["up_out"], transB=1)
silu_node = helper.make_node("SiLU", ["gate_out"], ["silu_out"])
mul_node = helper.make_node("Mul", ["silu_out", "up_out"], ["act_out"])
down_node = helper.make_node("Gemm", ["act_out", "down"], ["output"], transB=1)

graph = helper.make_graph(
    [mul_in, rms_mean, add_eps, sqrt_node, div_node, norm_mul, gate_node, up_node, silu_node, mul_node, down_node],
    "ffn",
    [input],
    [output],
    initializer=[norm_init, gate_init, up_init, down_init, eps_init]
)

model = helper.make_model(graph)
onnx.save(model, "{}")
"#,
            hidden_size,
            intermediate_size,
            norm_eps,
            prefix.with_extension("norm.npy").display(),
            prefix.with_extension("gate.npy").display(),
            prefix.with_extension("up.npy").display(),
            prefix.with_extension("down.npy").display(),
            onnx_path.display()
        )
    } else {
        let act_op = if use_gelu { "Gelu" } else { "SiLU" };
        format!(
            r#"
import numpy as np
import onnx
from onnx import helper, TensorProto

hidden_size = {}
intermediate_size = {}
norm_eps = {}

norm_w = np.load("{}")
up_w = np.load("{}")
down_w = np.load("{}")

input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [hidden_size])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [hidden_size])

norm_init = helper.make_tensor("norm_weight", TensorProto.FLOAT, [hidden_size], norm_w.flatten().tolist())
up_init = helper.make_tensor("up", TensorProto.FLOAT, [hidden_size, intermediate_size], up_w.flatten().tolist())
down_init = helper.make_tensor("down", TensorProto.FLOAT, [intermediate_size, hidden_size], down_w.flatten().tolist())
eps_init = helper.make_tensor("eps", TensorProto.FLOAT, [], [norm_eps])

mul_in = helper.make_node("Mul", ["input", "input"], ["x_sq"])
rms_mean = helper.make_node("ReduceMean", ["x_sq"], ["rms_in"], axes=[0], keepdims=0)
add_eps = helper.make_node("Add", ["rms_in", "eps"], ["rms_add"])
sqrt_node = helper.make_node("Sqrt", ["rms_add"], ["rms"])
div_node = helper.make_node("Div", ["input", "rms"], ["normed_div"])
norm_mul = helper.make_node("Mul", ["normed_div", "norm_weight"], ["normed"])
up_node = helper.make_node("Gemm", ["normed", "up"], ["up_out"], transB=1)
act_node = helper.make_node("{}", ["up_out"], ["act_out"])
down_node = helper.make_node("Gemm", ["act_out", "down"], ["output"], transB=1)

graph = helper.make_graph(
    [mul_in, rms_mean, add_eps, sqrt_node, div_node, norm_mul, up_node, act_node, down_node],
    "ffn",
    [input],
    [output],
    initializer=[norm_init, up_init, down_init, eps_init]
)

model = helper.make_model(graph)
onnx.save(model, "{}")
"#,
            hidden_size,
            intermediate_size,
            norm_eps,
            prefix.with_extension("norm.npy").display(),
            prefix.with_extension("up.npy").display(),
            prefix.with_extension("down.npy").display(),
            act_op,
            onnx_path.display()
        )
    };

    let script_path = prefix.with_extension("py");
    fs::write(&script_path, script.trim_start()).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to write export script: {}", e))
    })?;

    let status = Command::new("python3")
        .arg(script_path)
        .current_dir(output_dir)
        .status()
        .map_err(|e| {
            BackendError::OperationFailed(format!(
                "Failed to run ONNX export script (is Python with onnx/numpy installed?): {}",
                e
            ))
        })?;

    if !status.success() {
        return Err(BackendError::OperationFailed(
            "ONNX export script failed. Ensure Python 3 with onnx and numpy is installed.".into(),
        ));
    }

    Ok(onnx_path)
}

pub fn compile_hef(
    onnx_path: &Path,
    output_path: &Path,
    config: &HailoConfig,
) -> BackendResult<()> {
    let python = config
        .dfc_python
        .as_ref()
        .map(|p| p.as_os_str())
        .unwrap_or(std::ffi::OsStr::new("python3"));
    let arch = config
        .target_arch
        .as_deref()
        .unwrap_or("hailo8l");

    let script = format!(
        r#"
from hailo_sdk_client import ClientRunner
runner = ClientRunner(hw_arch='{}')
runner.translate_onnx_model('{}')
runner.compile()
runner.save_hef('{}')
"#,
        arch,
        onnx_path.display(),
        output_path.display()
    );

    let status = Command::new(python)
        .arg("-c")
        .arg(script.trim_start())
        .status()
        .map_err(|e| {
            BackendError::OperationFailed(format!(
                "Failed to run DFC (is hailo_sdk_client installed?): {}",
                e
            ))
        })?;

    if !status.success() {
        return Err(BackendError::OperationFailed(
            "DFC compilation failed. Ensure hailo_sdk_client is installed and ONNX model is compatible.".into(),
        ));
    }

    Ok(())
}

pub fn compile_layer_hefs(
    config: &HailoConfig,
    model_config: &ModelConfig,
    layer: &TransformerLayer,
    layer_idx: usize,
) -> BackendResult<(PathBuf, PathBuf)> {
    let output_dir = config.cache_dir.join("onnx");
    fs::create_dir_all(&output_dir).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to create ONNX output dir: {}", e))
    })?;

    let attn_onnx = export_attn_onnx(layer, layer_idx, model_config, &output_dir)?;
    let ffn_onnx = export_ffn_onnx(layer, layer_idx, model_config, &output_dir)?;

    let hef_dir = config.hef_dir.as_ref().unwrap_or(&config.cache_dir);
    fs::create_dir_all(hef_dir).map_err(|e| {
        BackendError::OperationFailed(format!("Failed to create HEF output dir: {}", e))
    })?;

    let attn_hef = resolve_hef_path(config, layer_idx, "attn");
    let ffn_hef = resolve_hef_path(config, layer_idx, "ffn");

    compile_hef(&attn_onnx, &attn_hef, config)?;
    compile_hef(&ffn_onnx, &ffn_hef, config)?;

    Ok((attn_hef, ffn_hef))
}

pub fn compile_all_layers(
    config: &HailoConfig,
    model_config: &ModelConfig,
    layers: &[TransformerLayer],
) -> BackendResult<()> {
    if check_cache(config, model_config) {
        tracing::info!("HEF cache valid, skipping compilation");
        return Ok(());
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        tracing::info!("Compiling HEF for layer {}", layer_idx);
        compile_layer_hefs(config, model_config, layer, layer_idx)?;
    }

    write_manifest(config, model_config)?;
    tracing::info!("HEF compilation complete");
    Ok(())
}
