use crate::backend::{BackendError, BackendResult};
use hailort_sys::constants::HAILO_MAX_NETWORK_GROUPS;
use hailort_sys::enums::{HAILO_D2H_STREAM, HAILO_FORMAT_ORDER_AUTO, HAILO_FORMAT_TYPE_AUTO, HAILO_H2D_STREAM};
use hailort_sys::ffi::{
    hailo_activate_network_group, hailo_configure_vdevice, hailo_create_hef_file, hailo_create_input_vstreams,
    hailo_create_output_vstreams, hailo_create_vdevice, hailo_deactivate_network_group, hailo_get_default_vstream_params,
    hailo_get_library_version, hailo_get_status_message, hailo_hef_get_network_group_infos,
    hailo_hef_get_vstream_infos, hailo_init_configure_params_by_vdevice, hailo_init_vdevice_params,
    hailo_input_vstream_write, hailo_output_vstream_read, hailo_release_hef, hailo_release_input_vstream,
    hailo_release_network_group, hailo_release_vdevice, hailo_scan_devices,
};
use hailort_sys::handles::{
    hailo_activated_network_group, hailo_configured_network_group, hailo_hef, hailo_input_vstream,
    hailo_output_vstream, hailo_vdevice,
};
use hailort_sys::status::HAILO_SUCCESS;
use hailort_sys::types::{
    hailo_activate_network_group_params_t, hailo_configure_params_t, hailo_input_vstream_params_by_name_t,
    hailo_network_group_info_t, hailo_output_vstream_params_by_name_t, hailo_vdevice_params_t,
    hailo_version_t, hailo_vstream_info_t,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use tracing;

pub fn check_status(status: hailort_sys::status::hailo_status, context: &str) -> BackendResult<()> {
    if status == HAILO_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let ptr = hailo_get_status_message(status);
        if ptr.is_null() {
            format!("Hailo error {} in {}", status, context)
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(BackendError::OperationFailed(format!("{}: {}", context, msg)))
}

pub fn check_device_available() -> BackendResult<()> {
    let mut count: usize = 0;
    let status = unsafe {
        hailo_scan_devices(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut count as *mut usize,
        )
    };
    check_status(status, "hailo_scan_devices")?;
    if count == 0 {
        return Err(BackendError::NotAvailable(
            "No Hailo devices found".to_string(),
        ));
    }
    tracing::info!("Hailo device check: {} device(s) available", count);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct HailoDeviceInfo {
    pub num_devices: usize,
    pub library_version: String,
}

pub struct HefHandle {
    hef: hailo_hef,
    configured_network_group: hailo_configured_network_group,
    activated_network_group: hailo_activated_network_group,
    input_vstreams: Vec<hailo_input_vstream>,
    output_vstreams: Vec<hailo_output_vstream>,
}

impl HefHandle {
    pub fn write_input(&self, data: &[u8]) -> BackendResult<()> {
        self.write_input_idx(0, data)
    }

    pub fn read_output(&self, buf: &mut [u8]) -> BackendResult<()> {
        self.read_output_idx(0, buf)
    }

    pub fn write_input_idx(&self, idx: usize, data: &[u8]) -> BackendResult<()> {
        let vstream = self
            .input_vstreams
            .get(idx)
            .copied()
            .ok_or_else(|| BackendError::InvalidArgument(format!("Invalid input vstream index {}", idx)))?;
        let status = unsafe { hailo_input_vstream_write(vstream, data.as_ptr() as *const _, data.len()) };
        check_status(status, "hailo_input_vstream_write")
    }

    pub fn read_output_idx(&self, idx: usize, buf: &mut [u8]) -> BackendResult<()> {
        let vstream = self
            .output_vstreams
            .get(idx)
            .copied()
            .ok_or_else(|| BackendError::InvalidArgument(format!("Invalid output vstream index {}", idx)))?;
        let status = unsafe { hailo_output_vstream_read(vstream, buf.as_mut_ptr() as *mut _, buf.len()) };
        check_status(status, "hailo_output_vstream_read")
    }
}

impl Drop for HefHandle {
    fn drop(&mut self) {
        for &vstream in &self.output_vstreams {
            let status = unsafe { hailort_sys::ffi::hailo_release_output_vstream(vstream) };
            if status != HAILO_SUCCESS {
                let msg = unsafe {
                    let ptr = hailo_get_status_message(status);
                    if ptr.is_null() {
                        format!("status {}", status)
                    } else {
                        CStr::from_ptr(ptr).to_string_lossy().into_owned()
                    }
                };
                tracing::warn!("hailo_release_output_vstream failed: {}", msg);
            }
        }
        for &vstream in &self.input_vstreams {
            let status = unsafe { hailo_release_input_vstream(vstream) };
            if status != HAILO_SUCCESS {
                let msg = unsafe {
                    let ptr = hailo_get_status_message(status);
                    if ptr.is_null() {
                        format!("status {}", status)
                    } else {
                        CStr::from_ptr(ptr).to_string_lossy().into_owned()
                    }
                };
                tracing::warn!("hailo_release_input_vstream failed: {}", msg);
            }
        }
        let status = unsafe { hailo_deactivate_network_group(self.activated_network_group) };
        if status != HAILO_SUCCESS {
            let msg = unsafe {
                let ptr = hailo_get_status_message(status);
                if ptr.is_null() {
                    format!("status {}", status)
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            tracing::warn!("hailo_deactivate_network_group failed: {}", msg);
        }
        let status = unsafe { hailo_release_network_group(self.configured_network_group) };
        if status != HAILO_SUCCESS {
            let msg = unsafe {
                let ptr = hailo_get_status_message(status);
                if ptr.is_null() {
                    format!("status {}", status)
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            tracing::warn!("hailo_release_network_group failed: {}", msg);
        }
        let status = unsafe { hailo_release_hef(self.hef) };
        if status != HAILO_SUCCESS {
            let msg = unsafe {
                let ptr = hailo_get_status_message(status);
                if ptr.is_null() {
                    format!("status {}", status)
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            tracing::warn!("hailo_release_hef failed: {}", msg);
        }
    }
}

pub struct HailoContext {
    vdevice: hailo_vdevice,
    device_info: HailoDeviceInfo,
}

impl HailoContext {
    pub fn new() -> BackendResult<Self> {
        check_device_available()?;

        let mut vdevice_params: hailo_vdevice_params_t = unsafe { std::mem::zeroed() };
        unsafe {
            hailo_init_vdevice_params(&mut vdevice_params as *mut _);
        }

        let mut vdevice: hailo_vdevice = std::ptr::null_mut();
        let status = unsafe {
            hailo_create_vdevice(&mut vdevice_params as *mut _, &mut vdevice as *mut _)
        };
        check_status(status, "hailo_create_vdevice")?;

        let mut version: hailo_version_t = unsafe { std::mem::zeroed() };
        let status = unsafe { hailo_get_library_version(&mut version as *mut _) };
        check_status(status, "hailo_get_library_version")?;

        let library_version = format!(
            "{}.{}.{}",
            version.major, version.minor, version.revision
        );

        let mut count: usize = 0;
        let status = unsafe {
            hailo_scan_devices(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count as *mut usize,
            )
        };
        check_status(status, "hailo_scan_devices")?;

        let device_info = HailoDeviceInfo {
            num_devices: count,
            library_version,
        };

        Ok(Self {
            vdevice,
            device_info,
        })
    }

    pub fn load_hef(&self, path: &Path) -> BackendResult<HefHandle> {
        let path_cstr = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| BackendError::InvalidArgument(format!("Invalid path: {}", e)))?;

        let mut hef: hailo_hef = std::ptr::null_mut();
        let status = unsafe { hailo_create_hef_file(&mut hef as *mut _, path_cstr.as_ptr()) };
        check_status(status, "hailo_create_hef_file")?;

        let mut ng_count: usize = 0;
        let status = unsafe {
            hailo_hef_get_network_group_infos(
                hef,
                std::ptr::null_mut(),
                &mut ng_count as *mut usize,
            )
        };
        check_status(status, "hailo_hef_get_network_group_infos (count)")?;

        if ng_count == 0 {
            unsafe { hailo_release_hef(hef) };
            return Err(BackendError::OperationFailed(
                "HEF has no network groups".to_string(),
            ));
        }

        let mut ng_infos: Vec<hailo_network_group_info_t> =
            vec![unsafe { std::mem::zeroed() }; ng_count];
        let status = unsafe {
            hailo_hef_get_network_group_infos(
                hef,
                ng_infos.as_mut_ptr(),
                &mut ng_count as *mut usize,
            )
        };
        check_status(status, "hailo_hef_get_network_group_infos")?;

        let ng_name = ng_infos[0].name;
        let ng_name_cstr = c_char_array_to_cstr(&ng_name);

        let mut configure_params: hailo_configure_params_t = unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailo_init_configure_params_by_vdevice(self.vdevice, hef, &mut configure_params as *mut _)
        };
        check_status(status, "hailo_init_configure_params_by_vdevice")?;

        let mut network_groups: [hailo_configured_network_group; 1] =
            [std::ptr::null_mut(); 1];
        let mut ng_out_count: usize = HAILO_MAX_NETWORK_GROUPS;
        let status = unsafe {
            hailo_configure_vdevice(
                self.vdevice,
                hef,
                &mut configure_params as *mut _,
                network_groups.as_mut_ptr(),
                &mut ng_out_count as *mut usize,
            )
        };
        check_status(status, "hailo_configure_vdevice")?;

        let configured_network_group = network_groups[0];
        if configured_network_group.is_null() {
            unsafe { hailo_release_hef(hef) };
            return Err(BackendError::OperationFailed(
                "hailo_configure_vdevice returned null network group".to_string(),
            ));
        }

        let mut activate_params: hailo_activate_network_group_params_t = unsafe { std::mem::zeroed() };
        let mut activated: hailo_activated_network_group = std::ptr::null_mut();
        let status = unsafe {
            hailo_activate_network_group(
                configured_network_group,
                &mut activate_params as *mut _,
                &mut activated as *mut _,
            )
        };
        check_status(status, "hailo_activate_network_group")?;

        if activated.is_null() {
            unsafe { hailo_release_network_group(configured_network_group) };
            unsafe { hailo_release_hef(hef) };
            return Err(BackendError::OperationFailed(
                "hailo_activate_network_group returned null".to_string(),
            ));
        }

        let mut input_count: usize = 0;
        let status = unsafe {
            hailo_hef_get_vstream_infos(
                hef,
                ng_name_cstr.as_ptr(),
                std::ptr::null_mut(),
                &mut input_count as *mut usize,
            )
        };
        check_status(status, "hailo_hef_get_vstream_infos (input count)")?;

        let mut all_infos: Vec<hailo_vstream_info_t> = (0..input_count)
            .map(|_| unsafe { std::mem::zeroed() })
            .collect();
        let status = unsafe {
            hailo_hef_get_vstream_infos(
                hef,
                ng_name_cstr.as_ptr(),
                all_infos.as_mut_ptr(),
                &mut input_count as *mut usize,
            )
        };
        check_status(status, "hailo_hef_get_vstream_infos")?;

        let input_indices: Vec<usize> = all_infos
            .iter()
            .enumerate()
            .filter(|(_, i)| i.direction == HAILO_H2D_STREAM)
            .map(|(idx, _)| idx)
            .collect();
        let output_indices: Vec<usize> = all_infos
            .iter()
            .enumerate()
            .filter(|(_, i)| i.direction == HAILO_D2H_STREAM)
            .map(|(idx, _)| idx)
            .collect();

        let user_format = hailort_sys::types::hailo_format_t {
            type_: HAILO_FORMAT_TYPE_AUTO,
            order: HAILO_FORMAT_ORDER_AUTO,
            flags: hailort_sys::enums::HAILO_FORMAT_FLAGS_NONE,
        };

        let mut input_params: Vec<hailo_input_vstream_params_by_name_t> =
            Vec::with_capacity(input_indices.len());
        for &idx in &input_indices {
            let info = &all_infos[idx];
            let mut params: hailort_sys::types::hailo_vstream_params_t = unsafe { std::mem::zeroed() };
            let status = unsafe {
                hailo_get_default_vstream_params(
                    info as *const _,
                    user_format,
                    HAILO_H2D_STREAM,
                    &mut params as *mut _,
                )
            };
            check_status(status, "hailo_get_default_vstream_params (input)")?;
            let mut name_buf = [0i8; 128];
            let copy_len = info.name.len().min(name_buf.len());
            name_buf[..copy_len].copy_from_slice(&info.name[..copy_len]);
            input_params.push(hailo_input_vstream_params_by_name_t {
                name: name_buf,
                params,
            });
        }

        let mut output_params: Vec<hailo_output_vstream_params_by_name_t> =
            Vec::with_capacity(output_indices.len());
        for &idx in &output_indices {
            let info = &all_infos[idx];
            let mut params: hailort_sys::types::hailo_vstream_params_t = unsafe { std::mem::zeroed() };
            let status = unsafe {
                hailo_get_default_vstream_params(
                    info as *const _,
                    user_format,
                    HAILO_D2H_STREAM,
                    &mut params as *mut _,
                )
            };
            check_status(status, "hailo_get_default_vstream_params (output)")?;
            let mut name_buf = [0i8; 128];
            let copy_len = info.name.len().min(name_buf.len());
            name_buf[..copy_len].copy_from_slice(&info.name[..copy_len]);
            output_params.push(hailo_output_vstream_params_by_name_t {
                name: name_buf,
                params,
            });
        }

        let mut input_vstreams: Vec<hailo_input_vstream> =
            vec![std::ptr::null_mut(); input_params.len()];
        if !input_params.is_empty() {
            let status = unsafe {
                hailo_create_input_vstreams(
                    configured_network_group,
                    input_params.as_ptr(),
                    input_params.len(),
                    input_vstreams.as_mut_ptr(),
                )
            };
            check_status(status, "hailo_create_input_vstreams")?;
        }

        let mut output_vstreams: Vec<hailo_output_vstream> =
            vec![std::ptr::null_mut(); output_params.len()];
        if !output_params.is_empty() {
            let status = unsafe {
                hailo_create_output_vstreams(
                    configured_network_group,
                    output_params.as_ptr(),
                    output_params.len(),
                    output_vstreams.as_mut_ptr(),
                )
            };
            check_status(status, "hailo_create_output_vstreams")?;
        }

        Ok(HefHandle {
            hef,
            configured_network_group,
            activated_network_group: activated,
            input_vstreams,
            output_vstreams,
        })
    }

    pub fn device_info(&self) -> &HailoDeviceInfo {
        &self.device_info
    }
}

fn c_char_array_to_cstr(arr: &[c_char; 128]) -> CString {
    let len = arr.iter().position(|&c| c == 0).unwrap_or(arr.len());
    let slice = &arr[..len];
    let bytes: Vec<u8> = slice.iter().map(|&c| c as u8).collect();
    CString::new(bytes).unwrap_or_else(|_| CString::new("").unwrap())
}

impl Drop for HailoContext {
    fn drop(&mut self) {
        let status = unsafe { hailo_release_vdevice(self.vdevice) };
        if status != HAILO_SUCCESS {
            let msg = unsafe {
                let ptr = hailo_get_status_message(status);
                if ptr.is_null() {
                    format!("status {}", status)
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            tracing::warn!("hailo_release_vdevice failed: {}", msg);
        }
    }
}
