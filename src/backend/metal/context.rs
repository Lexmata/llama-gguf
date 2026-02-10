//! Metal device initialization, pipeline management, and command submission.

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;

use crate::backend::BackendError;

/// Core Metal context holding device, command queue, and compute pipelines.
pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub pipelines: HashMap<String, ComputePipelineState>,
    device_name: String,
}

impl MetalContext {
    /// Initialize Metal: select device, create command queue, load pipelines.
    pub fn new(device_index: usize) -> Result<Self, BackendError> {
        // Get device
        let device = if device_index == 0 {
            Device::system_default().ok_or_else(|| {
                BackendError::InitializationFailed("No Metal device found".to_string())
            })?
        } else {
            let all_devices = Device::all();
            if device_index >= all_devices.len() {
                return Err(BackendError::InitializationFailed(format!(
                    "Metal device index {} out of range (found {} devices)",
                    device_index,
                    all_devices.len()
                )));
            }
            all_devices[device_index].clone()
        };

        let device_name = device.name().to_string();
        tracing::info!("Metal device: {}", device_name);

        if device.has_unified_memory() {
            tracing::info!("Unified memory architecture (Apple Silicon)");
        }

        // Create command queue
        let command_queue = device.new_command_queue();

        // Load compute pipelines from compiled Metal library
        let mut pipelines = HashMap::new();
        Self::load_pipelines(&device, &mut pipelines)?;

        tracing::info!(
            "Metal backend initialized: {} ({} pipelines)",
            device_name,
            pipelines.len()
        );

        Ok(Self {
            device,
            command_queue,
            pipelines,
            device_name,
        })
    }

    /// Get the device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Load Metal shaders and create compute pipelines.
    fn load_pipelines(
        device: &Device,
        pipelines: &mut HashMap<String, ComputePipelineState>,
    ) -> Result<(), BackendError> {
        // Load the compiled Metal library (.metallib)
        // At build time, all .metal shaders are compiled into a single .metallib
        let metallib_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));

        let library = device
            .new_library_with_data(metallib_bytes)
            .map_err(|e| {
                BackendError::InitializationFailed(format!("Failed to load Metal library: {}", e))
            })?;

        // Each shader function name maps to a pipeline
        let shader_functions = [
            ("add", "add_f32"),
            ("mul", "mul_f32"),
            ("scale", "scale_f32"),
            ("silu", "silu_f32"),
            ("gelu", "gelu_f32"),
            ("softmax_max", "softmax_max_f32"),
            ("softmax_exp", "softmax_exp_f32"),
            ("softmax_div", "softmax_div_f32"),
            ("rms_norm_sum", "rms_norm_sum_f32"),
            ("rms_norm_scale", "rms_norm_scale_f32"),
            ("vec_mat", "vec_mat_f32"),
            ("rope", "rope_f32"),
        ];

        for (pipeline_name, function_name) in &shader_functions {
            let function = library.get_function(function_name, None).map_err(|e| {
                BackendError::InitializationFailed(format!(
                    "Metal function '{}' not found: {}",
                    function_name, e
                ))
            })?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    BackendError::InitializationFailed(format!(
                        "Pipeline creation failed for '{}': {}",
                        pipeline_name, e
                    ))
                })?;

            pipelines.insert(pipeline_name.to_string(), pipeline);
        }

        Ok(())
    }

    // =========================================================================
    // Buffer management
    // =========================================================================

    /// Create a Metal buffer with data (shared storage for unified memory).
    pub fn create_buffer_with_data(&self, data: &[f32]) -> Result<Buffer, BackendError> {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        let buffer = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            // Use shared storage for Apple Silicon unified memory.
            // This avoids copies between CPU and GPU.
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Create an output Metal buffer.
    pub fn create_output_buffer(&self, num_floats: usize) -> Result<Buffer, BackendError> {
        let size = (num_floats * std::mem::size_of::<f32>()) as u64;
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);
        Ok(buffer)
    }

    /// Read float data from a Metal buffer.
    pub fn read_buffer(&self, buf: &Buffer, num_floats: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        let mut result = vec![0.0f32; num_floats];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), num_floats);
        }
        result
    }

    // =========================================================================
    // Command submission
    // =========================================================================

    /// Dispatch a compute shader with the given buffers and parameters.
    ///
    /// This creates a command buffer, encodes the compute command, commits, and
    /// waits for completion synchronously.
    pub fn dispatch(
        &self,
        pipeline_name: &str,
        buffers: &[&Buffer],
        param_bytes: Option<(&[u8], usize)>,
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) -> Result<(), BackendError> {
        let pipeline = self.pipelines.get(pipeline_name).ok_or_else(|| {
            BackendError::OperationFailed(format!("Pipeline not found: {}", pipeline_name))
        })?;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Bind buffers
        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }

        // Set inline parameter bytes if provided
        if let Some((bytes, buffer_index)) = param_bytes {
            encoder.set_bytes(
                buffer_index as u64,
                bytes.len() as u64,
                bytes.as_ptr() as *const _,
            );
        }

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Dispatch a compute shader using threadgroup-based dispatch (for reductions).
    pub fn dispatch_threadgroups(
        &self,
        pipeline_name: &str,
        buffers: &[&Buffer],
        param_bytes: Option<(&[u8], usize)>,
        threadgroup_count: MTLSize,
        threadgroup_size: MTLSize,
    ) -> Result<(), BackendError> {
        let pipeline = self.pipelines.get(pipeline_name).ok_or_else(|| {
            BackendError::OperationFailed(format!("Pipeline not found: {}", pipeline_name))
        })?;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }

        if let Some((bytes, buffer_index)) = param_bytes {
            encoder.set_bytes(
                buffer_index as u64,
                bytes.len() as u64,
                bytes.as_ptr() as *const _,
            );
        }

        encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Compute 1D grid and threadgroup sizes for a given element count.
    pub fn sizes_1d(n: usize) -> (MTLSize, MTLSize) {
        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(n as u64, 1, 1);
        (grid_size, threadgroup_size)
    }

    /// Compute threadgroup count for 1D dispatch with reduction.
    pub fn threadgroup_count_1d(n: usize, local_size: u64) -> MTLSize {
        let count = ((n as u64) + local_size - 1) / local_size;
        MTLSize::new(count, 1, 1)
    }
}
