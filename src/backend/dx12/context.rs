//! DirectX 12 device initialization, pipeline management, buffer management, and command submission.

use std::collections::HashMap;
use std::sync::Mutex;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventA, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::backend::BackendError;

/// Core DX12 context holding device, command queue, pipelines, and synchronization primitives.
pub struct Dx12Context {
    pub device: ID3D12Device,
    pub command_queue: ID3D12CommandQueue,
    pub command_allocator: ID3D12CommandAllocator,
    pub pipelines: HashMap<String, ComputePipeline>,
    pub fence: ID3D12Fence,
    pub fence_value: Mutex<u64>,
    pub fence_event: HANDLE,
}

/// A compiled compute pipeline with root signature and PSO.
pub struct ComputePipeline {
    pub root_signature: ID3D12RootSignature,
    pub pipeline_state: ID3D12PipelineState,
    pub num_uav_params: u32,
    pub root_constant_count: u32,
}

/// A GPU buffer with upload, default (UAV), and optional readback heaps.
pub struct GpuBuffer {
    pub default_resource: ID3D12Resource,
    pub upload_resource: Option<ID3D12Resource>,
    pub readback_resource: Option<ID3D12Resource>,
    pub size: u64,
    pub num_elements: usize,
}

impl Dx12Context {
    /// Initialize DX12: create device, command queue, allocator, fence, and load pipelines.
    pub fn new(device_index: usize, prefer_warp: bool) -> Result<Self, BackendError> {
        unsafe { Self::init(device_index, prefer_warp) }
    }

    unsafe fn init(device_index: usize, prefer_warp: bool) -> Result<Self, BackendError> {
        // Create DXGI factory
        let factory: IDXGIFactory4 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| {
                BackendError::InitializationFailed(format!("CreateDXGIFactory2 failed: {}", e))
            })?;

        // Select adapter
        let adapter = if prefer_warp {
            factory.EnumWarpAdapter().map_err(|e| {
                BackendError::InitializationFailed(format!("WARP adapter not available: {}", e))
            })?
        } else {
            Self::select_adapter(&factory, device_index)?
        };

        // Create device
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).map_err(|e| {
            BackendError::InitializationFailed(format!("D3D12CreateDevice failed: {}", e))
        })?;
        let device = device.unwrap();

        // Create compute command queue
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let command_queue: ID3D12CommandQueue =
            device.CreateCommandQueue(&queue_desc).map_err(|e| {
                BackendError::InitializationFailed(format!("CreateCommandQueue failed: {}", e))
            })?;

        // Create command allocator
        let command_allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
            .map_err(|e| {
                BackendError::InitializationFailed(format!("CreateCommandAllocator failed: {}", e))
            })?;

        // Create fence for synchronization
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| {
            BackendError::InitializationFailed(format!("CreateFence failed: {}", e))
        })?;

        let fence_event = CreateEventA(None, false, false, None).map_err(|e| {
            BackendError::InitializationFailed(format!("CreateEvent failed: {}", e))
        })?;

        // Load compute pipelines
        let mut pipelines = HashMap::new();
        Self::load_pipelines(&device, &mut pipelines)?;

        let adapter_desc = {
            let adapter: IDXGIAdapter1 = adapter.cast().map_err(|e| {
                BackendError::InitializationFailed(format!("Adapter cast failed: {}", e))
            })?;
            adapter.GetDesc1().map_err(|e| {
                BackendError::InitializationFailed(format!("GetDesc1 failed: {}", e))
            })?
        };

        let device_name = String::from_utf16_lossy(
            &adapter_desc.Description[..adapter_desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(adapter_desc.Description.len())],
        );

        tracing::info!(
            "DX12 backend initialized: {} ({} pipelines)",
            device_name.trim(),
            pipelines.len()
        );

        Ok(Self {
            device,
            command_queue,
            command_allocator,
            pipelines,
            fence,
            fence_value: Mutex::new(0),
            fence_event,
        })
    }

    /// Select a non-WARP adapter by index.
    unsafe fn select_adapter(
        factory: &IDXGIFactory4,
        device_index: usize,
    ) -> Result<IDXGIAdapter1, BackendError> {
        let mut idx = 0u32;
        let mut found = Vec::new();

        loop {
            match factory.EnumAdapters1(idx) {
                Ok(adapter) => {
                    let desc = match adapter.GetDesc1() {
                        Ok(d) => d,
                        Err(_) => {
                            idx += 1;
                            continue;
                        }
                    };

                    // Skip software/WARP adapters
                    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0 {
                        found.push(adapter);
                    }
                    idx += 1;
                }
                Err(_) => break,
            }
        }

        if found.is_empty() {
            return Err(BackendError::InitializationFailed(
                "No DX12-capable GPU found".to_string(),
            ));
        }

        if device_index >= found.len() {
            return Err(BackendError::InitializationFailed(format!(
                "Device index {} out of range (found {} devices)",
                device_index,
                found.len()
            )));
        }

        Ok(found.into_iter().nth(device_index).unwrap())
    }

    /// Load CSO shaders and create compute pipelines.
    fn load_pipelines(
        device: &ID3D12Device,
        pipelines: &mut HashMap<String, ComputePipeline>,
    ) -> Result<(), BackendError> {
        // Each shader: (name, CSO bytes, num_uav_params, root_constant_count in 32-bit words)
        let shader_defs: Vec<(&str, &[u8], u32, u32)> = vec![
            (
                "add",
                include_bytes!(concat!(env!("OUT_DIR"), "/add.cso")),
                3,
                1,
            ),
            (
                "mul",
                include_bytes!(concat!(env!("OUT_DIR"), "/mul.cso")),
                3,
                1,
            ),
            (
                "scale",
                include_bytes!(concat!(env!("OUT_DIR"), "/scale.cso")),
                2,
                2,
            ),
            (
                "silu",
                include_bytes!(concat!(env!("OUT_DIR"), "/silu.cso")),
                2,
                1,
            ),
            (
                "gelu",
                include_bytes!(concat!(env!("OUT_DIR"), "/gelu.cso")),
                2,
                1,
            ),
            (
                "softmax_max",
                include_bytes!(concat!(env!("OUT_DIR"), "/softmax_max.cso")),
                2,
                1,
            ),
            (
                "softmax_exp",
                include_bytes!(concat!(env!("OUT_DIR"), "/softmax_exp.cso")),
                2,
                2,
            ),
            (
                "softmax_div",
                include_bytes!(concat!(env!("OUT_DIR"), "/softmax_div.cso")),
                1,
                2,
            ),
            (
                "rms_norm_sum",
                include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_sum.cso")),
                2,
                1,
            ),
            (
                "rms_norm_scale",
                include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_scale.cso")),
                3,
                2,
            ),
            (
                "vec_mat",
                include_bytes!(concat!(env!("OUT_DIR"), "/vec_mat.cso")),
                3,
                2,
            ),
            (
                "rope",
                include_bytes!(concat!(env!("OUT_DIR"), "/rope.cso")),
                2,
                7,
            ),
        ];

        for (name, cso_bytes, num_uav_params, root_constant_count) in shader_defs {
            let pipeline = unsafe {
                Self::create_compute_pipeline(
                    device,
                    cso_bytes,
                    num_uav_params,
                    root_constant_count,
                )?
            };
            pipelines.insert(name.to_string(), pipeline);
        }

        Ok(())
    }

    /// Create a single compute pipeline from CSO bytecode.
    ///
    /// Root signature layout:
    /// - Param 0: Root constants (32-bit values)
    /// - Params 1..N: Root UAV descriptors (one per buffer)
    unsafe fn create_compute_pipeline(
        device: &ID3D12Device,
        cso_bytes: &[u8],
        num_uav_params: u32,
        root_constant_count: u32,
    ) -> Result<ComputePipeline, BackendError> {
        // Build root parameters
        let mut root_params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();

        // Param 0: Root constants (cbuffer Params : register(b0))
        if root_constant_count > 0 {
            let mut param = D3D12_ROOT_PARAMETER::default();
            param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            param.Anonymous.Constants = D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: root_constant_count,
            };
            param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            root_params.push(param);
        }

        // Params 1..N: Root UAV descriptors (register(u0), u1, u2, ...)
        for i in 0..num_uav_params {
            let mut param = D3D12_ROOT_PARAMETER::default();
            param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            param.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: i,
                RegisterSpace: 0,
            };
            param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            root_params.push(param);
        }

        // Serialize root signature
        let root_sig_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: root_params.len() as u32,
            pParameters: if root_params.is_empty() {
                std::ptr::null()
            } else {
                root_params.as_ptr()
            },
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let mut signature_blob = None;
        let mut error_blob = None;

        D3D12SerializeRootSignature(
            &root_sig_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut signature_blob,
            Some(&mut error_blob),
        )
        .map_err(|e| {
            let msg = if let Some(ref err) = error_blob {
                let ptr = err.GetBufferPointer() as *const u8;
                let len = err.GetBufferSize();
                String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).to_string()
            } else {
                format!("{}", e)
            };
            BackendError::InitializationFailed(format!(
                "Root signature serialization failed: {}",
                msg
            ))
        })?;

        let signature_blob = signature_blob.unwrap();

        let root_signature: ID3D12RootSignature = device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    signature_blob.GetBufferPointer() as *const u8,
                    signature_blob.GetBufferSize(),
                ),
            )
            .map_err(|e| {
                BackendError::InitializationFailed(format!("CreateRootSignature failed: {}", e))
            })?;

        // Create pipeline state object
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cso_bytes.as_ptr() as *const _,
                BytecodeLength: cso_bytes.len(),
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        let pipeline_state: ID3D12PipelineState =
            device.CreateComputePipelineState(&pso_desc).map_err(|e| {
                BackendError::InitializationFailed(format!(
                    "CreateComputePipelineState failed: {}",
                    e
                ))
            })?;

        // Prevent the ManuallyDrop from dropping the cloned reference
        let _ = std::mem::ManuallyDrop::into_inner(pso_desc.pRootSignature);

        Ok(ComputePipeline {
            root_signature,
            pipeline_state,
            num_uav_params,
            root_constant_count,
        })
    }

    // =========================================================================
    // Buffer management
    // =========================================================================

    /// Create a GPU buffer and upload f32 data to it.
    pub fn create_buffer_with_data(&self, data: &[f32]) -> Result<GpuBuffer, BackendError> {
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let byte_data = bytemuck::cast_slice(data);

        unsafe {
            // Create upload heap
            let upload_resource = self.create_committed_resource(
                D3D12_HEAP_TYPE_UPLOAD,
                size,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;

            // Map and copy data to upload heap
            let mut mapped_ptr = std::ptr::null_mut();
            upload_resource
                .Map(0, None, Some(&mut mapped_ptr))
                .map_err(|e| BackendError::AllocationFailed(format!("Map upload failed: {}", e)))?;
            std::ptr::copy_nonoverlapping(
                byte_data.as_ptr(),
                mapped_ptr as *mut u8,
                byte_data.len(),
            );
            upload_resource.Unmap(0, None);

            // Create default heap (UAV)
            let default_resource = self.create_committed_resource(
                D3D12_HEAP_TYPE_DEFAULT,
                size,
                D3D12_RESOURCE_STATE_COMMON,
            )?;

            Ok(GpuBuffer {
                default_resource,
                upload_resource: Some(upload_resource),
                readback_resource: None,
                size,
                num_elements: data.len(),
            })
        }
    }

    /// Create an output GPU buffer (default heap + readback heap).
    pub fn create_output_buffer(&self, num_floats: usize) -> Result<GpuBuffer, BackendError> {
        let size = (num_floats * std::mem::size_of::<f32>()) as u64;

        unsafe {
            let default_resource = self.create_committed_resource(
                D3D12_HEAP_TYPE_DEFAULT,
                size,
                D3D12_RESOURCE_STATE_COMMON,
            )?;

            let readback_resource = self.create_committed_resource(
                D3D12_HEAP_TYPE_READBACK,
                size,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;

            Ok(GpuBuffer {
                default_resource,
                upload_resource: None,
                readback_resource: Some(readback_resource),
                size,
                num_elements: num_floats,
            })
        }
    }

    /// Read float data back from a GPU buffer's readback heap.
    pub fn read_buffer(&self, buf: &GpuBuffer) -> Result<Vec<f32>, BackendError> {
        let readback = buf.readback_resource.as_ref().ok_or_else(|| {
            BackendError::OperationFailed("Buffer has no readback resource".to_string())
        })?;

        unsafe {
            let mut mapped_ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut mapped_ptr)).map_err(|e| {
                BackendError::OperationFailed(format!("Map readback failed: {}", e))
            })?;

            let mut result = vec![0.0f32; buf.num_elements];
            std::ptr::copy_nonoverlapping(
                mapped_ptr as *const f32,
                result.as_mut_ptr(),
                buf.num_elements,
            );
            readback.Unmap(0, None);
            Ok(result)
        }
    }

    /// Helper to create a committed resource on a specific heap type.
    unsafe fn create_committed_resource(
        &self,
        heap_type: D3D12_HEAP_TYPE,
        size: u64,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, BackendError> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if heap_type == D3D12_HEAP_TYPE_DEFAULT {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };

        let mut resource: Option<ID3D12Resource> = None;
        self.device
            .CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                initial_state,
                None,
                &mut resource,
            )
            .map_err(|e| {
                BackendError::AllocationFailed(format!("CreateCommittedResource failed: {}", e))
            })?;

        resource.ok_or_else(|| BackendError::AllocationFailed("Resource was None".to_string()))
    }

    // =========================================================================
    // Command submission
    // =========================================================================

    /// Dispatch a compute shader with the given buffers and root constants.
    ///
    /// This is a synchronous operation: records commands, submits, and waits for completion.
    pub fn dispatch(
        &self,
        pipeline_name: &str,
        buffers: &[&GpuBuffer],
        root_constants: &[u32],
        workgroup_count: (u32, u32, u32),
    ) -> Result<(), BackendError> {
        let pipeline = self.pipelines.get(pipeline_name).ok_or_else(|| {
            BackendError::OperationFailed(format!("Pipeline not found: {}", pipeline_name))
        })?;

        unsafe {
            // Lock for mutable state (allocator reset, fence value)
            let mut fence_val = self.fence_value.lock().unwrap();

            // Reset command allocator
            self.command_allocator.Reset().map_err(|e| {
                BackendError::OperationFailed(format!("Reset allocator failed: {}", e))
            })?;

            // Create command list
            let command_list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    &self.command_allocator,
                    None,
                )
                .map_err(|e| {
                    BackendError::OperationFailed(format!("CreateCommandList failed: {}", e))
                })?;

            // Copy upload → default for buffers with upload resources
            for buf in buffers {
                if let Some(ref upload) = buf.upload_resource {
                    // Barrier: COMMON → COPY_DEST
                    let barrier = Self::transition_barrier(
                        &buf.default_resource,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    );
                    command_list.ResourceBarrier(&[barrier]);

                    command_list.CopyResource(&buf.default_resource, upload);

                    // Barrier: COPY_DEST → UNORDERED_ACCESS
                    let barrier = Self::transition_barrier(
                        &buf.default_resource,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    );
                    command_list.ResourceBarrier(&[barrier]);
                } else {
                    // Output buffer: COMMON → UNORDERED_ACCESS
                    let barrier = Self::transition_barrier(
                        &buf.default_resource,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    );
                    command_list.ResourceBarrier(&[barrier]);
                }
            }

            // Set pipeline state and root signature
            command_list.SetComputeRootSignature(&pipeline.root_signature);
            command_list.SetPipelineState(&pipeline.pipeline_state);

            // Set root constants (param 0)
            let mut param_idx = 0u32;
            if pipeline.root_constant_count > 0 {
                for (i, &val) in root_constants.iter().enumerate() {
                    command_list.SetComputeRoot32BitConstant(param_idx, val, i as u32);
                }
                param_idx += 1;
            }

            // Set root UAV descriptors (params 1..N)
            for buf in buffers {
                let gpu_addr = buf.default_resource.GetGPUVirtualAddress();
                command_list.SetComputeRootUnorderedAccessView(param_idx, gpu_addr);
                param_idx += 1;
            }

            // Dispatch
            command_list.Dispatch(workgroup_count.0, workgroup_count.1, workgroup_count.2);

            // UAV barrier for compute writes
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            };
            command_list.ResourceBarrier(&[uav_barrier]);

            // Copy default → readback for output buffers
            for buf in buffers {
                if buf.readback_resource.is_some() {
                    // Barrier: UAV → COPY_SOURCE
                    let barrier = Self::transition_barrier(
                        &buf.default_resource,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    );
                    command_list.ResourceBarrier(&[barrier]);

                    command_list.CopyResource(
                        buf.readback_resource.as_ref().unwrap(),
                        &buf.default_resource,
                    );

                    // Barrier: COPY_SOURCE → COMMON
                    let barrier = Self::transition_barrier(
                        &buf.default_resource,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                    command_list.ResourceBarrier(&[barrier]);
                } else {
                    // Input buffer: UAV → COMMON
                    let barrier = Self::transition_barrier(
                        &buf.default_resource,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                    command_list.ResourceBarrier(&[barrier]);
                }
            }

            // Close command list
            command_list.Close().map_err(|e| {
                BackendError::OperationFailed(format!("Close command list failed: {}", e))
            })?;

            // Execute
            let cmd_lists: [Option<ID3D12CommandList>; 1] = [Some(command_list.cast().unwrap())];
            self.command_queue.ExecuteCommandLists(&cmd_lists);

            // Signal and wait on fence
            *fence_val += 1;
            self.command_queue
                .Signal(&self.fence, *fence_val)
                .map_err(|e| {
                    BackendError::OperationFailed(format!("Signal fence failed: {}", e))
                })?;

            self.fence
                .SetEventOnCompletion(*fence_val, self.fence_event)
                .map_err(|e| {
                    BackendError::OperationFailed(format!("SetEventOnCompletion failed: {}", e))
                })?;

            WaitForSingleObject(self.fence_event, INFINITE);
        }

        Ok(())
    }

    /// Create a resource transition barrier.
    unsafe fn transition_barrier(
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    /// Compute workgroup count for 1D dispatch.
    pub fn workgroup_count_1d(n: usize, local_size: u32) -> (u32, u32, u32) {
        let count = ((n as u32) + local_size - 1) / local_size;
        (count, 1, 1)
    }
}

impl Drop for Dx12Context {
    fn drop(&mut self) {
        unsafe {
            // Wait for GPU idle
            let fence_val = *self.fence_value.lock().unwrap() + 1;
            let _ = self.command_queue.Signal(&self.fence, fence_val);
            let _ = self.fence.SetEventOnCompletion(fence_val, self.fence_event);
            WaitForSingleObject(self.fence_event, INFINITE);

            CloseHandle(self.fence_event).ok();
        }
    }
}

// Safety: DX12 COM objects are thread-safe (they use internal reference counting),
// and mutable state is protected by Mutex.
unsafe impl Send for Dx12Context {}
unsafe impl Sync for Dx12Context {}
