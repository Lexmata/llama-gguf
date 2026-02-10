//! Vulkan device initialization, memory management, and command submission.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;
use std::mem::ManuallyDrop;

use crate::backend::BackendError;

/// Core Vulkan context holding device, queues, allocator, and pipelines.
///
/// The allocator is wrapped in `ManuallyDrop` to ensure it's dropped before
/// the Vulkan device is destroyed (gpu-allocator needs the device alive).
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub pipelines: HashMap<String, ComputePipeline>,
    pub descriptor_pool: vk::DescriptorPool,
}

/// A compiled compute pipeline with its layout and descriptor set layout.
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub num_bindings: u32,
    pub push_constant_size: u32,
}

/// A GPU buffer with its allocation.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: vk::DeviceSize,
}

impl VulkanContext {
    /// Initialize Vulkan: create instance, select device, create queues and allocator.
    pub fn new(device_index: usize, enable_validation: bool) -> Result<Self, BackendError> {
        unsafe { Self::init(device_index, enable_validation) }
    }

    unsafe fn init(device_index: usize, enable_validation: bool) -> Result<Self, BackendError> {
        // Load Vulkan runtime library
        let entry = ash::Entry::load()
            .map_err(|e| BackendError::InitializationFailed(format!("Failed to load Vulkan library: {}", e)))?;

        // Application info
        let app_name = c"llama-gguf";
        let engine_name = c"llama-gguf-vulkan";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 5, 2))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 5, 2))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        // Instance layers
        let mut layers = Vec::new();
        if enable_validation {
            let validation = c"VK_LAYER_KHRONOS_validation";
            // Check if validation layer is available
            if let Ok(available) = entry.enumerate_instance_layer_properties() {
                let has_validation = available.iter().any(|l| {
                    CStr::from_ptr(l.layer_name.as_ptr()) == validation
                });
                if has_validation {
                    layers.push(validation.as_ptr());
                }
            }
        }

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers);

        let instance = entry
            .create_instance(&instance_info, None)
            .map_err(|e| BackendError::InitializationFailed(format!("Vulkan instance creation failed: {}", e)))?;

        // Enumerate physical devices
        let physical_devices = instance
            .enumerate_physical_devices()
            .map_err(|e| BackendError::InitializationFailed(format!("Failed to enumerate GPUs: {}", e)))?;

        if physical_devices.is_empty() {
            return Err(BackendError::InitializationFailed(
                "No Vulkan-capable GPU found".to_string(),
            ));
        }

        if device_index >= physical_devices.len() {
            return Err(BackendError::InitializationFailed(format!(
                "Device index {} out of range (found {} devices)",
                device_index,
                physical_devices.len()
            )));
        }

        let physical_device = physical_devices[device_index];
        let device_properties = instance.get_physical_device_properties(physical_device);

        let device_name = CStr::from_ptr(device_properties.device_name.as_ptr())
            .to_string_lossy()
            .to_string();

        tracing::info!("Vulkan device: {}", device_name);

        // Find a compute queue family
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);

        let queue_family_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| {
                BackendError::InitializationFailed("No compute queue family found".to_string())
            })?;

        // Create logical device
        let queue_priority = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority);

        let queue_create_infos = [queue_create_info];

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features);

        let device = instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| BackendError::InitializationFailed(format!("Device creation failed: {}", e)))?;

        let compute_queue = device.get_device_queue(queue_family_index, 0);

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device
            .create_command_pool(&pool_info, None)
            .map_err(|e| BackendError::InitializationFailed(format!("Command pool creation failed: {}", e)))?;

        // Create memory allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| BackendError::InitializationFailed(format!("Allocator creation failed: {}", e)))?;

        // Create descriptor pool (large enough for many operations)
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4096,
        }];

        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1024)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let descriptor_pool = device
            .create_descriptor_pool(&desc_pool_info, None)
            .map_err(|e| BackendError::InitializationFailed(format!("Descriptor pool failed: {}", e)))?;

        // Load and compile compute pipelines
        let mut pipelines = HashMap::new();
        Self::load_pipelines(&device, &mut pipelines)?;

        tracing::info!(
            "Vulkan backend initialized: {} ({} pipelines)",
            device_name,
            pipelines.len()
        );

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family_index,
            command_pool,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            device_properties,
            pipelines,
            descriptor_pool,
        })
    }

    /// Load SPIR-V shaders and create compute pipelines.
    fn load_pipelines(
        device: &ash::Device,
        pipelines: &mut HashMap<String, ComputePipeline>,
    ) -> Result<(), BackendError> {
        // Each shader: (name, spir-v bytes, num_bindings, push_constant_size)
        let shader_defs: Vec<(&str, &[u8], u32, u32)> = vec![
            ("add", include_bytes!(concat!(env!("OUT_DIR"), "/add.spv")), 3, 4),
            ("mul", include_bytes!(concat!(env!("OUT_DIR"), "/mul.spv")), 3, 4),
            ("scale", include_bytes!(concat!(env!("OUT_DIR"), "/scale.spv")), 2, 8),
            ("silu", include_bytes!(concat!(env!("OUT_DIR"), "/silu.spv")), 2, 4),
            ("gelu", include_bytes!(concat!(env!("OUT_DIR"), "/gelu.spv")), 2, 4),
            ("softmax_max", include_bytes!(concat!(env!("OUT_DIR"), "/softmax_max.spv")), 2, 4),
            ("softmax_exp", include_bytes!(concat!(env!("OUT_DIR"), "/softmax_exp.spv")), 2, 8),
            ("softmax_div", include_bytes!(concat!(env!("OUT_DIR"), "/softmax_div.spv")), 1, 8),
            ("rms_norm_sum", include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_sum.spv")), 2, 4),
            ("rms_norm_scale", include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_scale.spv")), 3, 8),
            ("vec_mat", include_bytes!(concat!(env!("OUT_DIR"), "/vec_mat.spv")), 3, 8),
            ("rope", include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv")), 2, 28),
        ];

        for (name, spirv_bytes, num_bindings, push_constant_size) in shader_defs {
            let pipeline =
                Self::create_compute_pipeline(device, spirv_bytes, num_bindings, push_constant_size)?;
            pipelines.insert(name.to_string(), pipeline);
        }

        Ok(())
    }

    /// Create a single compute pipeline from SPIR-V bytes.
    fn create_compute_pipeline(
        device: &ash::Device,
        spirv_bytes: &[u8],
        num_bindings: u32,
        push_constant_size: u32,
    ) -> Result<ComputePipeline, BackendError> {
        unsafe {
            // Create shader module
            // SPIR-V must be aligned to 4 bytes
            let spirv_words: Vec<u32> = spirv_bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);

            let shader_module = device
                .create_shader_module(&shader_info, None)
                .map_err(|e| BackendError::InitializationFailed(format!("Shader module failed: {}", e)))?;

            // Descriptor set layout
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let desc_layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            let descriptor_set_layout = device
                .create_descriptor_set_layout(&desc_layout_info, None)
                .map_err(|e| BackendError::InitializationFailed(format!("Desc layout failed: {}", e)))?;

            // Pipeline layout
            let desc_layouts = [descriptor_set_layout];
            let push_constant_range = if push_constant_size > 0 {
                vec![vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: push_constant_size,
                }]
            } else {
                vec![]
            };

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&desc_layouts)
                .push_constant_ranges(&push_constant_range);

            let pipeline_layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| BackendError::InitializationFailed(format!("Pipeline layout failed: {}", e)))?;

            // Compute pipeline
            let entry_point = c"main";
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_point);

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipeline_infos = [pipeline_info];
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .map_err(|e| BackendError::InitializationFailed(format!("Pipeline creation failed: {:?}", e)))?[0];

            // Clean up shader module (no longer needed after pipeline creation)
            device.destroy_shader_module(shader_module, None);

            Ok(ComputePipeline {
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                num_bindings,
                push_constant_size,
            })
        }
    }

    // =========================================================================
    // Buffer management
    // =========================================================================

    /// Create a GPU storage buffer and upload data to it.
    pub fn create_buffer_with_data(&self, data: &[f32]) -> Result<GpuBuffer, BackendError> {
        let size = (data.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;
        self.create_buffer_with_bytes(bytemuck::cast_slice(data), size)
    }

    /// Create a GPU storage buffer from raw bytes.
    pub fn create_buffer_with_bytes(
        &self,
        data: &[u8],
        size: vk::DeviceSize,
    ) -> Result<GpuBuffer, BackendError> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| BackendError::AllocationFailed(format!("Buffer creation failed: {}", e)))?;

            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocation = self
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "gpu_buffer",
                    requirements,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| BackendError::AllocationFailed(format!("Allocation failed: {}", e)))?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| BackendError::AllocationFailed(format!("Bind memory failed: {}", e)))?;

            // Copy data
            if let Some(mapped) = allocation.mapped_ptr() {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    mapped.as_ptr() as *mut u8,
                    data.len(),
                );
            }

            Ok(GpuBuffer {
                buffer,
                allocation,
                size,
            })
        }
    }

    /// Create an output GPU buffer (for writing results).
    pub fn create_output_buffer(&self, num_floats: usize) -> Result<GpuBuffer, BackendError> {
        let size = (num_floats * std::mem::size_of::<f32>()) as vk::DeviceSize;

        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| BackendError::AllocationFailed(format!("Buffer creation failed: {}", e)))?;

            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocation = self
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "gpu_output",
                    requirements,
                    location: MemoryLocation::GpuToCpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| BackendError::AllocationFailed(format!("Allocation failed: {}", e)))?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| BackendError::AllocationFailed(format!("Bind memory failed: {}", e)))?;

            Ok(GpuBuffer {
                buffer,
                allocation,
                size,
            })
        }
    }

    /// Read float data back from a GPU buffer.
    pub fn read_buffer(&self, buf: &GpuBuffer) -> Result<Vec<f32>, BackendError> {
        if let Some(mapped) = buf.allocation.mapped_ptr() {
            let num_floats = buf.size as usize / std::mem::size_of::<f32>();
            let mut result = vec![0.0f32; num_floats];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped.as_ptr() as *const f32,
                    result.as_mut_ptr(),
                    num_floats,
                );
            }
            Ok(result)
        } else {
            Err(BackendError::OperationFailed(
                "Buffer not mapped for reading".to_string(),
            ))
        }
    }

    /// Free a GPU buffer.
    pub fn free_buffer(&self, buf: GpuBuffer) {
        unsafe {
            self.device.destroy_buffer(buf.buffer, None);
        }
        self.allocator.lock().unwrap().free(buf.allocation).ok();
    }

    // =========================================================================
    // Command submission
    // =========================================================================

    /// Dispatch a compute shader with the given buffers and push constants.
    pub fn dispatch(
        &self,
        pipeline_name: &str,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        workgroup_count: (u32, u32, u32),
    ) -> Result<(), BackendError> {
        let pipeline = self
            .pipelines
            .get(pipeline_name)
            .ok_or_else(|| BackendError::OperationFailed(format!("Pipeline not found: {}", pipeline_name)))?;

        unsafe {
            // Allocate descriptor set
            let layouts = [pipeline.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_set = self
                .device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| BackendError::OperationFailed(format!("Descriptor set alloc failed: {}", e)))?[0];

            // Update descriptor set with buffer bindings
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();

            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            self.device.update_descriptor_sets(&writes, &[]);

            // Allocate command buffer
            let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_buf = self
                .device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| BackendError::OperationFailed(format!("Command buffer alloc failed: {}", e)))?[0];

            // Record commands
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|e| BackendError::OperationFailed(format!("Begin command buffer failed: {}", e)))?;

            self.device
                .cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);

            self.device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            if !push_constants.is_empty() {
                self.device.cmd_push_constants(
                    cmd_buf,
                    pipeline.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            self.device
                .cmd_dispatch(cmd_buf, workgroup_count.0, workgroup_count.1, workgroup_count.2);

            // Memory barrier to ensure compute writes are visible
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);

            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );

            self.device
                .end_command_buffer(cmd_buf)
                .map_err(|e| BackendError::OperationFailed(format!("End command buffer failed: {}", e)))?;

            // Submit and wait
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

            let fence_info = vk::FenceCreateInfo::default();
            let fence = self
                .device
                .create_fence(&fence_info, None)
                .map_err(|e| BackendError::OperationFailed(format!("Fence creation failed: {}", e)))?;

            self.device
                .queue_submit(self.compute_queue, &[submit_info], fence)
                .map_err(|e| BackendError::OperationFailed(format!("Queue submit failed: {}", e)))?;

            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| BackendError::OperationFailed(format!("Fence wait failed: {}", e)))?;

            // Cleanup
            self.device.destroy_fence(fence, None);
            self.device
                .free_command_buffers(self.command_pool, &cmd_bufs);
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                .ok();
        }

        Ok(())
    }

    /// Compute workgroup count for 1D dispatch.
    pub fn workgroup_count_1d(n: usize, local_size: u32) -> (u32, u32, u32) {
        let count = ((n as u32) + local_size - 1) / local_size;
        (count, 1, 1)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Destroy pipelines
            for (_, pipeline) in self.pipelines.drain() {
                self.device.destroy_pipeline(pipeline.pipeline, None);
                self.device
                    .destroy_pipeline_layout(pipeline.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            }

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_command_pool(self.command_pool, None);

            // Drop the allocator BEFORE destroying the device.
            // gpu-allocator needs the device to still be alive to free its memory.
            ManuallyDrop::drop(&mut self.allocator);

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
