use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Compile ONNX protobuf when the onnx feature is enabled
    if env::var("CARGO_FEATURE_ONNX").is_ok() {
        compile_onnx_proto();
    }

    // Compile distributed gRPC proto when the distributed feature is enabled
    if env::var("CARGO_FEATURE_DISTRIBUTED").is_ok() {
        compile_distributed_proto();
    }

    // Compile Metal shaders when the metal feature is enabled (macOS only)
    #[cfg(target_os = "macos")]
    if env::var("CARGO_FEATURE_METAL").is_ok() {
        compile_metal_shaders();
    }

    // Compile Vulkan shaders when the vulkan feature is enabled
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        compile_vulkan_shaders();
    }

    // Compile DX12 shaders when the dx12 feature is enabled (Windows only)
    #[cfg(target_os = "windows")]
    if env::var("CARGO_FEATURE_DX12").is_ok() {
        compile_dx12_shaders();
    }
}

// =============================================================================
// Vulkan shader compilation
// =============================================================================

fn compile_vulkan_shaders() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("src/backend/vulkan/shaders");

    // Find shader compiler
    let compiler = find_shader_compiler();

    // List of shaders to compile
    let shaders = [
        "add",
        "mul",
        "scale",
        "silu",
        "gelu",
        "softmax_max",
        "softmax_exp",
        "softmax_div",
        "rms_norm_sum",
        "rms_norm_scale",
        "vec_mat",
        "rope",
    ];

    for shader_name in &shaders {
        let input = shader_dir.join(format!("{}.comp", shader_name));
        let output = out_dir.join(format!("{}.spv", shader_name));

        // Tell cargo to rerun if shader source changes
        println!("cargo:rerun-if-changed={}", input.display());

        if let Some(ref compiler) = compiler {
            compile_shader(compiler, &input, &output, shader_name);
        } else {
            // Try to use pre-compiled SPIR-V from source tree
            let precompiled = shader_dir.join(format!("{}.spv", shader_name));
            if precompiled.exists() {
                std::fs::copy(&precompiled, &output).unwrap_or_else(|e| {
                    panic!("Failed to copy pre-compiled shader {}: {}", shader_name, e);
                });
                eprintln!(
                    "cargo:warning=Using pre-compiled SPIR-V for {} (no shader compiler found)",
                    shader_name
                );
            } else {
                panic!(
                    "Cannot compile Vulkan shaders: no shader compiler (glslc or glslangValidator) \
                     found and no pre-compiled .spv files available.\n\
                     Install the Vulkan SDK (https://vulkan.lunarg.com/sdk/home) or \
                     pre-compile shaders with: glslc -o {}.spv {}.comp",
                    shader_name, shader_name
                );
            }
        }
    }
}

fn find_shader_compiler() -> Option<ShaderCompiler> {
    // Try glslc first (from Vulkan SDK or system)
    if let Ok(output) = Command::new("glslc").arg("--version").output()
        && output.status.success()
    {
        return Some(ShaderCompiler::Glslc);
    }

    // Try glslangValidator
    if let Ok(output) = Command::new("glslangValidator").arg("--version").output()
        && output.status.success()
    {
        return Some(ShaderCompiler::GlslangValidator);
    }

    None
}

enum ShaderCompiler {
    Glslc,
    GlslangValidator,
}

fn compile_shader(compiler: &ShaderCompiler, input: &Path, output: &Path, name: &str) {
    let status = match compiler {
        ShaderCompiler::Glslc => Command::new("glslc")
            .arg("--target-env=vulkan1.2")
            .arg("-O")
            .arg("-o")
            .arg(output)
            .arg(input)
            .status(),
        ShaderCompiler::GlslangValidator => Command::new("glslangValidator")
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.2")
            .arg("-o")
            .arg(output)
            .arg(input)
            .status(),
    };

    match status {
        Ok(s) if s.success() => {
            eprintln!("Compiled shader: {}", name);
        }
        Ok(s) => {
            panic!(
                "Shader compilation failed for {}: exit code {:?}",
                name,
                s.code()
            );
        }
        Err(e) => {
            panic!("Failed to run shader compiler for {}: {}", name, e);
        }
    }
}

// =============================================================================
// Metal shader compilation (macOS only)
// =============================================================================

#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("src/backend/metal/shaders");

    let metal_shaders = [
        "add",
        "mul",
        "scale",
        "silu",
        "gelu",
        "softmax_max",
        "softmax_exp",
        "softmax_div",
        "rms_norm_sum",
        "rms_norm_scale",
        "vec_mat",
        "rope",
    ];

    // Compile each .metal -> .air (Apple Intermediate Representation)
    let mut air_files = Vec::new();
    for shader_name in &metal_shaders {
        let input = shader_dir.join(format!("{}.metal", shader_name));
        let air_output = out_dir.join(format!("{}.air", shader_name));

        println!("cargo:rerun-if-changed={}", input.display());

        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal"])
            .arg("-c")
            .arg("-O2")
            .arg("-o")
            .arg(&air_output)
            .arg(&input)
            .status();

        match status {
            Ok(s) if s.success() => {
                eprintln!("Compiled Metal shader: {}", shader_name);
                air_files.push(air_output);
            }
            Ok(s) => {
                // Try pre-compiled metallib
                let precompiled = shader_dir.join("shaders.metallib");
                if precompiled.exists() {
                    eprintln!(
                        "cargo:warning=Metal shader compilation failed for {}, using pre-compiled metallib",
                        shader_name
                    );
                    std::fs::copy(&precompiled, out_dir.join("shaders.metallib"))
                        .expect("Failed to copy pre-compiled metallib");
                    return;
                }
                panic!(
                    "Metal shader compilation failed for {}: exit code {:?}\n\
                     Install Xcode Command Line Tools: xcode-select --install",
                    shader_name,
                    s.code()
                );
            }
            Err(e) => {
                // xcrun not available -- try pre-compiled
                let precompiled = shader_dir.join("shaders.metallib");
                if precompiled.exists() {
                    eprintln!("cargo:warning=xcrun not found, using pre-compiled Metal library");
                    std::fs::copy(&precompiled, out_dir.join("shaders.metallib"))
                        .expect("Failed to copy pre-compiled metallib");
                    return;
                }
                panic!(
                    "Cannot compile Metal shaders: xcrun not found ({})\n\
                     Install Xcode Command Line Tools: xcode-select --install",
                    e
                );
            }
        }
    }

    // Link all .air files into a single .metallib
    let metallib_output = out_dir.join("shaders.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    cmd.arg("-o").arg(&metallib_output);
    for air_file in &air_files {
        cmd.arg(air_file);
    }

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => {
            eprintln!(
                "Linked Metal library: shaders.metallib ({} shaders)",
                air_files.len()
            );
        }
        Ok(s) => {
            panic!("Metal library linking failed: exit code {:?}", s.code());
        }
        Err(e) => {
            panic!("Failed to run metallib linker: {}", e);
        }
    }
}

// =============================================================================
// DX12 HLSL shader compilation (Windows only)
// =============================================================================

#[cfg(target_os = "windows")]
fn compile_dx12_shaders() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("src/backend/dx12/shaders");

    let shaders = [
        "add",
        "mul",
        "scale",
        "silu",
        "gelu",
        "softmax_max",
        "softmax_exp",
        "softmax_div",
        "rms_norm_sum",
        "rms_norm_scale",
        "vec_mat",
        "rope",
    ];

    let compiler = find_dxc_compiler();

    for shader_name in &shaders {
        let input = shader_dir.join(format!("{}.hlsl", shader_name));
        let output = out_dir.join(format!("{}.cso", shader_name));

        println!("cargo:rerun-if-changed={}", input.display());

        if let Some(ref dxc) = compiler {
            let status = Command::new(dxc)
                .arg("/T")
                .arg("cs_6_0")
                .arg("/O2")
                .arg("/Fo")
                .arg(&output)
                .arg(&input)
                .status();

            match status {
                Ok(s) if s.success() => {
                    eprintln!("Compiled DX12 shader: {}", shader_name);
                }
                Ok(s) => {
                    panic!(
                        "DX12 shader compilation failed for {}: exit code {:?}",
                        shader_name,
                        s.code()
                    );
                }
                Err(e) => {
                    panic!("Failed to run dxc for {}: {}", shader_name, e);
                }
            }
        } else {
            // Try pre-compiled CSO from source tree
            let precompiled = shader_dir.join(format!("{}.cso", shader_name));
            if precompiled.exists() {
                std::fs::copy(&precompiled, &output).unwrap_or_else(|e| {
                    panic!("Failed to copy pre-compiled shader {}: {}", shader_name, e);
                });
                eprintln!(
                    "cargo:warning=Using pre-compiled CSO for {} (dxc not found)",
                    shader_name
                );
            } else {
                panic!(
                    "Cannot compile DX12 shaders: dxc.exe not found and no pre-compiled \
                     .cso files available.\n\
                     Install the Windows SDK or DirectX Shader Compiler, or \
                     pre-compile shaders with: dxc /T cs_6_0 /O2 /Fo {}.cso {}.hlsl",
                    shader_name, shader_name
                );
            }
        }
    }
}

#[cfg(target_os = "windows")]
fn find_dxc_compiler() -> Option<PathBuf> {
    // Try dxc from PATH
    if let Ok(output) = Command::new("dxc").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("dxc"));
        }
    }

    // Try Windows SDK paths
    let program_files =
        env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());

    let sdk_base = Path::new(&program_files).join("Windows Kits\\10\\bin");
    if sdk_base.is_dir() {
        // Look for the newest SDK version
        if let Ok(entries) = std::fs::read_dir(&sdk_base) {
            let mut versions: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .map(|e| e.path())
                .collect();
            versions.sort();
            versions.reverse();

            for ver_dir in versions {
                let dxc_path = ver_dir.join("x64\\dxc.exe");
                if dxc_path.exists() {
                    return Some(dxc_path);
                }
            }
        }
    }

    None
}

// =============================================================================
// ONNX protobuf compilation
// =============================================================================

fn compile_onnx_proto() {
    let proto_path = Path::new("proto/onnx.proto3");
    if !proto_path.exists() {
        panic!("ONNX proto file not found at proto/onnx.proto3");
    }

    println!("cargo:rerun-if-changed=proto/onnx.proto3");

    prost_build::Config::new()
        .out_dir(PathBuf::from(env::var("OUT_DIR").unwrap()))
        .compile_protos(&[proto_path], &[Path::new("proto")])
        .expect("Failed to compile ONNX protobuf");
}

// =============================================================================
// Distributed gRPC protobuf compilation
// =============================================================================

fn compile_distributed_proto() {
    let proto_path = Path::new("proto/distributed.proto");
    if !proto_path.exists() {
        panic!("Distributed proto file not found at proto/distributed.proto");
    }

    println!("cargo:rerun-if-changed=proto/distributed.proto");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(PathBuf::from(env::var("OUT_DIR").unwrap()))
        .compile_protos(&[proto_path], &[Path::new("proto")])
        .expect("Failed to compile distributed protobuf");
}
