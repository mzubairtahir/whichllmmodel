// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const CREATE_NO_WINDOW: u32 = 0x08000000;

fn get_app_data_dir() -> PathBuf {
    if let Some(mut local_app) = dirs::data_local_dir() {
        let app_dir = local_app.join("whichllmmodel");
        let _ = std::fs::create_dir_all(&app_dir);
        return app_dir;
    }
    std::env::temp_dir().join("whichllmmodel")
}

fn get_bin_dir() -> PathBuf {
    // 1. First check if bundled inside current exe directory or resources
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let direct_bench = exe_dir.join("llama-bench.exe");
            if direct_bench.is_file() {
                return exe_dir.to_path_buf();
            }
            let sub_bin = exe_dir.join("bin");
            if sub_bin.join("llama-bench.exe").is_file() {
                return sub_bin;
            }
            let res_bin = exe_dir.join("resources").join("bin");
            if res_bin.join("llama-bench.exe").is_file() {
                return res_bin;
            }
        }
    }
    // 2. Development fallback
    let dev_bin = PathBuf::from(r"e:\llama.cpp study\whichllmmodel\src-tauri\bin");
    if dev_bin.join("llama-bench.exe").is_file() {
        return dev_bin;
    }
    // 3. Fallback to upstream build
    PathBuf::from(r"e:\llama.cpp study\llama.cpp\build\bin\Release")
}

#[repr(C)]
struct MEMORYSTATUSEX {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MEMORYSTATUSEX) -> i32;
}

fn refresh_live_ram_ffi(hw: &mut HardwareInfo) {
    unsafe {
        let mut stat: MEMORYSTATUSEX = std::mem::zeroed();
        stat.dw_length = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut stat) != 0 {
            let total_mib = (stat.ull_total_phys as f64) / (1024.0 * 1024.0);
            let free_mib = (stat.ull_avail_phys as f64) / (1024.0 * 1024.0);
            hw.ram_total_mib = (total_mib * 10.0).round() / 10.0;
            hw.ram_free_mib = (free_mib * 10.0).round() / 10.0;
            hw.ram_used_mib = ((total_mib - free_mib).max(0.0) * 10.0).round() / 10.0;
            hw.ram_load_pct = stat.dw_memory_load;
            hw.gpu_shared_ram_mib = ((total_mib * 0.5) * 10.0).round() / 10.0;
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Default)]
struct HardwareInfo {
    cpu_name: String,
    cpu_threads: u32,
    cpu_mhz: u32,
    ram_total_mib: f64,
    ram_free_mib: f64,
    ram_used_mib: f64,
    ram_load_pct: u32,
    ram_installed_gb: u32,
    ram_channels: u32,
    ram_speed_mts: u32,
    ram_bandwidth_gbs: f64,
    #[serde(default)]
    measured_ram_gbs: f64,
    #[serde(default)]
    effective_ram_gbs: f64,
    gpu_name: String,
    gpu_is_discrete: bool,
    gpu_vram_total_mib: f64,
    gpu_vram_free_mib: f64,
    gpu_shared_ram_mib: f64,
    gpu_bandwidth_gbs: f64,
    #[serde(default)]
    measured_gpu_gbs: f64,
    #[serde(default)]
    effective_gpu_gbs: f64,
}

#[derive(Deserialize, Serialize, Clone)]
struct LocalModelEntry {
    name: String,
    source: String,
    model_path: String,
    mmproj_path: String,
    arch: String,
    context_length: u32,
    is_moe: bool,
    expert_count: u32,
    expert_used_count: u32,
    model_logical_mib: f64,
    model_disk_mib: f64,
    mmproj_logical_mib: f64,
    mmproj_disk_mib: f64,
    is_sparse_skeleton: bool,
    #[serde(default)]
    bookmarked: bool,
    #[serde(default)]
    category: String,
}

#[derive(Deserialize, Serialize, Clone, Default)]
struct DownloadStatus {
    #[serde(default)]
    active: bool,
    #[serde(default)]
    status: String,
    #[serde(default)]
    filename: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    downloaded_mib: f64,
    #[serde(default)]
    total_mib: f64,
    #[serde(default)]
    percent: f64,
    #[serde(default)]
    speed_mbs: f64,
    #[serde(default)]
    output_path: String,
    error: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Default)]
struct ModelUserMeta {
    #[serde(default)]
    bookmarked: bool,
    #[serde(default)]
    category: String,
}

#[derive(Deserialize, Clone)]
struct HelperResponse {
    ok: bool,
    error: Option<String>,
    skeleton_path: Option<String>,
    resolved_url: Option<String>,
    #[serde(default)]
    is_local: Option<bool>,
    #[serde(default)]
    prep_time_ms: Option<f64>,
    arch: Option<String>,
    arch_supported: Option<bool>,
    supported_arch_count: Option<usize>,
    context_length: Option<u32>,
    is_moe: Option<bool>,
    expert_count: Option<u32>,
    expert_used_count: Option<u32>,
    mmproj_skeleton_path: Option<String>,
    hardware: Option<HardwareInfo>,
    local_models: Option<Vec<LocalModelEntry>>,
}

#[derive(Clone)]
struct CachedVision {
    has_mmproj: bool,
    mmproj_name: String,
    mmproj_projector_type: String,
    vram_vision_model_mib: f64,
    ram_vision_model_mib: f64,
    vram_vision_compute_mib: f64,
    ram_vision_compute_mib: f64,
    extra_lines: Vec<String>,
}

static HELPER_CACHE: OnceLock<Mutex<HashMap<(String, String), HelperResponse>>> = OnceLock::new();
static VISION_CACHE: OnceLock<Mutex<HashMap<(String, String), CachedVision>>> = OnceLock::new();
static HW_CACHE: OnceLock<Mutex<Option<HardwareInfo>>> = OnceLock::new();
static DOWNLOAD_CHILD: OnceLock<Mutex<Option<std::process::Child>>> = OnceLock::new();

#[derive(Deserialize)]
struct IsolatedVisionBuf {
    device: String,
    model_mib: f64,
    compute_mib: f64,
}

#[derive(Serialize)]
struct BenchResult {
    model_name: String,
    skeleton_path: String,
    resolved_url: String,
    arch: String,
    arch_supported: bool,
    supported_arch_count: usize,
    native_context_length: u32,
    is_moe: bool,
    expert_count: u32,
    expert_used_count: u32,
    has_mmproj: bool,
    mmproj_name: String,
    mmproj_projector_type: String,
    vram_vision_model_mib: f64,
    ram_vision_model_mib: f64,
    vram_vision_compute_mib: f64,
    ram_vision_compute_mib: f64,
    context_size: u32,
    ngl: i32,
    ncmoe: u32,
    device: String,
    vram_model_mib: f64,
    ram_model_mib: f64,
    vram_kv_mib: f64,
    ram_kv_mib: f64,
    vram_rs_mib: f64,
    ram_rs_mib: f64,
    vram_compute_mib: f64,
    ram_compute_mib: f64,
    output_buffer_mib: f64,
    total_vram_mib: f64,
    total_ram_mib: f64,
    elapsed_ms: u64,
    prep_time_ms: f64,
    bench_time_ms: f64,
    is_local_model: bool,
    raw_lines: Vec<String>,
    hardware: Option<HardwareInfo>,
}

fn run_analyze_blocking(
    url_or_path: String,
    mmproj_url_or_path: Option<String>,
    context_size: u32,
    ngl: i32,
    ncmoe: Option<u32>,
    device: String,
) -> Result<BenchResult, String> {
    let start_time = Instant::now();
    let ncmoe_val = ncmoe.unwrap_or(0);
    let mmproj_arg = mmproj_url_or_path.unwrap_or_default().trim().to_string();

    let bin_dir = get_bin_dir();
    let helper_py = bin_dir.join("desktop_helper.py");
    let bench_exe = bin_dir.join("llama-bench.exe");
    let isolated_exe = bin_dir.join("isolated-reserve.exe");

    if !bench_exe.exists() {
        return Err(format!("llama-bench.exe not found at {:?}", bench_exe));
    }

    // Step 1: Resolve HF URL (or local path), create GGUF skeleton(s), and verify llama.cpp architecture support (cached in memory after 1st run)
    let cache_map = HELPER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = (url_or_path.clone(), mmproj_arg.clone());
    let cached_entry = cache_map.lock().unwrap().get(&cache_key).cloned();

    let mut helper_data = if let Some(cached) = cached_entry {
        cached
    } else {
        let helper_out = Command::new("py")
            .args(["-3.11", helper_py.to_str().unwrap(), &url_or_path, &mmproj_arg])
            .current_dir(&bin_dir)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map_err(|e| format!("Failed to invoke desktop_helper.py: {}", e))?;

        let helper_stdout = String::from_utf8_lossy(&helper_out.stdout);
        let helper_stderr = String::from_utf8_lossy(&helper_out.stderr);

        let mut parsed_helper: Option<HelperResponse> = None;
        for line in helper_stdout.lines().rev() {
            if let Some(json_str) = line.strip_prefix("JSON_RESULT:") {
                if let Ok(resp) = serde_json::from_str::<HelperResponse>(json_str) {
                    parsed_helper = Some(resp);
                    break;
                }
            }
        }

        let data = match parsed_helper {
            Some(d) => d,
            None => {
                return Err(format!(
                    "Failed to resolve GGUF / create skeleton:\n{}\n{}",
                    helper_stdout, helper_stderr
                ));
            }
        };

        if data.ok {
            cache_map.lock().unwrap().insert(cache_key, data.clone());
            if let Some(ref hw) = data.hardware {
                *HW_CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(hw.clone());
            }
        }
        data
    };

    if let Some(ref mut hw) = helper_data.hardware {
        refresh_live_ram_ffi(hw);
    }

    if !helper_data.ok {
        return Err(helper_data
            .error
            .unwrap_or_else(|| "Unknown error preparing GGUF skeleton".to_string()));
    }

    let target_gguf = helper_data.skeleton_path.unwrap_or_default();
    let resolved_url = helper_data.resolved_url.unwrap_or_else(|| url_or_path.clone());
    let is_local_model = helper_data.is_local.unwrap_or_else(|| Path::new(&url_or_path).exists());
    let prep_time_ms = helper_data.prep_time_ms.unwrap_or(0.0);
    let arch = helper_data.arch.unwrap_or_else(|| "unknown".to_string());
    let arch_supported = helper_data.arch_supported.unwrap_or(false);
    let supported_arch_count = helper_data.supported_arch_count.unwrap_or(0);
    let native_context_length = helper_data.context_length.unwrap_or(4096);
    let is_moe = helper_data.is_moe.unwrap_or(false);
    let expert_count = helper_data.expert_count.unwrap_or(0);
    let expert_used_count = helper_data.expert_used_count.unwrap_or(if is_moe { 4 } else { 0 });
    let mmproj_skeleton_path = helper_data.mmproj_skeleton_path.unwrap_or_default();
    let hardware = helper_data.hardware;

    // Gate check: Is this architecture implemented in llama.cpp?
    if !arch_supported {
        return Err(format!(
            "Architecture '{}' is NOT implemented in this llama.cpp build (checked {} registered architectures in src/llama-arch.cpp).",
            arch, supported_arch_count
        ));
    }

    // Step 2: Execute llama-bench.exe with -r 0 (Zero-Weight Null-Backend reservation)
    let bench_start = Instant::now();
    let output = Command::new(&bench_exe)
        .args([
            "-m",
            &target_gguf,
            "-v",
            "-r",
            "0",
            "--no-warmup",
            "-p",
            &context_size.to_string(),
            "-n",
            "0",
            "-ub",
            "512",
            "-b",
            "512",
            "-dev",
            &device,
            "-ngl",
            &ngl.to_string(),
            "-ncmoe",
            &ncmoe_val.to_string(),
            "-fa",
            "on",
            "--repack",
            "0",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to execute llama-bench: {}", e))?;

    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    if combined.contains("unknown model architecture") {
        return Err(format!(
            "llama-bench rejected architecture '{}': unknown model architecture in binary.",
            arch
        ));
    }

    // Step 3: Parse exact buffer lines directly from llama-bench output
    let re_buf = Regex::new(
        r"(load_tensors|llama_context|llama_kv_cache|llama_memory_recurrent|sched_reserve):\s+(\S+)\s+(model|output|KV|RS|compute) buffer size\s*=\s*([\d\.]+)\s*MiB",
    )
    .unwrap();

    let mut vram_model_mib = 0.0;
    let mut ram_model_mib = 0.0;
    let mut vram_kv_mib = 0.0;
    let mut ram_kv_mib = 0.0;
    let mut vram_rs_mib = 0.0;
    let mut ram_rs_mib = 0.0;
    let mut vram_compute_mib = 0.0;
    let mut ram_compute_mib = 0.0;
    let mut output_buffer_mib = 0.0;
    let mut raw_lines = Vec::new();

    for line in combined.lines() {
        if line.contains("buffer size =") {
            raw_lines.push(line.trim().to_string());
        }
        if let Some(caps) = re_buf.captures(line) {
            let backend = &caps[2];
            let kind = &caps[3];
            let val: f64 = caps[4].parse().unwrap_or(0.0);
            let is_gpu = backend.starts_with("CUDA0")
                || backend.starts_with("Metal")
                || backend.starts_with("ROCm")
                || backend.starts_with("Vulkan");

            match kind {
                "model" => {
                    if is_gpu {
                        vram_model_mib += val;
                    } else {
                        ram_model_mib += val;
                    }
                }
                "KV" => {
                    if is_gpu {
                        vram_kv_mib += val;
                    } else {
                        ram_kv_mib += val;
                    }
                }
                "RS" => {
                    if is_gpu {
                        vram_rs_mib += val;
                    } else {
                        ram_rs_mib += val;
                    }
                }
                "compute" => {
                    if is_gpu {
                        vram_compute_mib = val;
                    } else {
                        ram_compute_mib = val;
                    }
                }
                "output" => {
                    output_buffer_mib += val;
                }
                _ => {}
            }
        }
    }

    if raw_lines.is_empty() {
        return Err(format!(
            "No buffer lines returned by llama-bench (arch='{}'). Raw log:\n{}",
            arch,
            combined.chars().take(1200).collect::<String>()
        ));
    }

    // Step 4: If mmproj Vision Projector is provided/detected, run isolated-reserve.exe (clip_init)
    let mut has_mmproj = false;
    let mut mmproj_name = String::new();
    let mut mmproj_projector_type = String::new();
    let mut vram_vision_model_mib = 0.0;
    let mut ram_vision_model_mib = 0.0;
    let mut vram_vision_compute_mib = 0.0;
    let mut ram_vision_compute_mib = 0.0;

    if !mmproj_skeleton_path.is_empty() && isolated_exe.exists() {
        let iso_dev = if device.to_lowercase().contains("metal") {
            "metal"
        } else if ngl == 0 {
            "cpu"
        } else {
            "cuda"
        };

        let vcache_map = VISION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let vcache_key = (mmproj_skeleton_path.clone(), iso_dev.to_string());
        let cached_vis = vcache_map.lock().unwrap().get(&vcache_key).cloned();

        if let Some(cv) = cached_vis {
            has_mmproj = cv.has_mmproj;
            mmproj_name = cv.mmproj_name;
            mmproj_projector_type = cv.mmproj_projector_type;
            vram_vision_model_mib = cv.vram_vision_model_mib;
            ram_vision_model_mib = cv.ram_vision_model_mib;
            vram_vision_compute_mib = cv.vram_vision_compute_mib;
            ram_vision_compute_mib = cv.ram_vision_compute_mib;
            raw_lines.extend(cv.extra_lines);
        } else if let Ok(iso_out) = Command::new(&isolated_exe)
            .args([
                "-m",
                &target_gguf,
                "-mmproj",
                &mmproj_skeleton_path,
                "-ngl",
                &ngl.to_string(),
                "-ctx",
                "4096",
                "-dev",
                iso_dev,
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
        {
            let iso_stdout = String::from_utf8_lossy(&iso_out.stdout);
            let iso_stderr = String::from_utf8_lossy(&iso_out.stderr);
            let iso_all = format!("{}\n{}", iso_stdout, iso_stderr);

            let re_vis_compute = Regex::new(
                r"reserve_compute_meta:\s+(\S+)\s+compute buffer size\s*=\s*([\d\.]+)\s*MiB",
            )
            .unwrap();
            let re_vis_model = Regex::new(r"load_hparams:\s+model size:\s*([\d\.]+)\s*MiB").unwrap();

            let mut clip_using_gpu = ngl != 0;
            let mut log_model_mib = 0.0;
            let mut extra_lines = Vec::new();

            for line in iso_all.lines() {
                if line.contains("CLIP using") {
                    clip_using_gpu = line.contains("CUDA")
                        || line.contains("Metal")
                        || line.contains("ROCm")
                        || line.contains("Vulkan");
                    extra_lines.push(format!("[mmproj] {}", line.trim()));
                }
                if line.contains("reserve_compute_meta:") || line.contains("load_hparams: model size:") {
                    extra_lines.push(format!("[mmproj] {}", line.trim()));
                }
                if line.contains("load_hparams: projector:") {
                    if let Some(p) = line.split("projector:").nth(1) {
                        mmproj_projector_type = p.trim().to_string();
                    }
                }
                if let Some(caps) = re_vis_model.captures(line) {
                    log_model_mib = caps[1].parse().unwrap_or(0.0);
                }
                if let Some(caps) = re_vis_compute.captures(line) {
                    let dev_name = &caps[1];
                    let val: f64 = caps[2].parse().unwrap_or(0.0);
                    let is_gpu_dev = dev_name.starts_with("CUDA")
                        || dev_name.starts_with("Metal")
                        || dev_name.starts_with("ROCm")
                        || dev_name.starts_with("Vulkan");
                    if is_gpu_dev {
                        vram_vision_compute_mib = val;
                    } else {
                        ram_vision_compute_mib = val;
                    }
                }
            }

            if log_model_mib > 0.0 {
                has_mmproj = true;
                mmproj_name = Path::new(&mmproj_skeleton_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "mmproj".to_string());
                if clip_using_gpu {
                    vram_vision_model_mib = log_model_mib;
                } else {
                    ram_vision_model_mib = log_model_mib;
                }
            }

            // Also check the "vision_buffers": [...] JSON array if present
            if let Some(vb_key_idx) = iso_stdout.find("\"vision_buffers\"") {
                if let Some(arr_start) = iso_stdout[vb_key_idx..].find('[') {
                    let slice_from_arr = &iso_stdout[vb_key_idx + arr_start..];
                    if let Some(arr_end) = slice_from_arr.find(']') {
                        let arr_json = &slice_from_arr[..=arr_end];
                        if let Ok(vbufs) = serde_json::from_str::<Vec<IsolatedVisionBuf>>(arr_json) {
                            has_mmproj = true;
                            mmproj_name = Path::new(&mmproj_skeleton_path)
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_else(|| "mmproj".to_string());
                            vram_vision_model_mib = 0.0;
                            vram_vision_compute_mib = 0.0;
                            ram_vision_model_mib = 0.0;
                            ram_vision_compute_mib = 0.0;
                            for vb in vbufs {
                                let is_gpu_dev = vb.device.starts_with("CUDA")
                                    || vb.device.starts_with("Metal")
                                    || vb.device.starts_with("ROCm")
                                    || vb.device.starts_with("Vulkan");
                                if is_gpu_dev {
                                    vram_vision_model_mib += vb.model_mib;
                                    vram_vision_compute_mib += vb.compute_mib;
                                } else {
                                    ram_vision_model_mib += vb.model_mib;
                                    ram_vision_compute_mib += vb.compute_mib;
                                }
                            }
                        }
                    }
                }
            }

            raw_lines.extend(extra_lines.clone());
            if has_mmproj {
                vcache_map.lock().unwrap().insert(
                    vcache_key,
                    CachedVision {
                        has_mmproj,
                        mmproj_name: mmproj_name.clone(),
                        mmproj_projector_type: mmproj_projector_type.clone(),
                        vram_vision_model_mib,
                        ram_vision_model_mib,
                        vram_vision_compute_mib,
                        ram_vision_compute_mib,
                        extra_lines,
                    },
                );
            }
        }
    }

    let total_vram_mib = vram_model_mib
        + vram_kv_mib
        + vram_rs_mib
        + vram_compute_mib
        + vram_vision_model_mib
        + vram_vision_compute_mib;
    let total_ram_mib = ram_model_mib
        + ram_kv_mib
        + ram_rs_mib
        + ram_compute_mib
        + output_buffer_mib
        + ram_vision_model_mib
        + ram_vision_compute_mib;

    let model_name = Path::new(&target_gguf)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Model".to_string());

    Ok(BenchResult {
        model_name,
        skeleton_path: target_gguf,
        resolved_url,
        arch,
        arch_supported,
        supported_arch_count,
        native_context_length,
        is_moe,
        expert_count,
        expert_used_count,
        has_mmproj,
        mmproj_name,
        mmproj_projector_type,
        vram_vision_model_mib,
        ram_vision_model_mib,
        vram_vision_compute_mib,
        ram_vision_compute_mib,
        context_size,
        ngl,
        ncmoe: ncmoe_val,
        device,
        vram_model_mib,
        ram_model_mib,
        vram_kv_mib,
        ram_kv_mib,
        vram_rs_mib,
        ram_rs_mib,
        vram_compute_mib,
        ram_compute_mib,
        output_buffer_mib,
        total_vram_mib,
        total_ram_mib,
        elapsed_ms: start_time.elapsed().as_millis() as u64,
        prep_time_ms,
        bench_time_ms: bench_start.elapsed().as_millis() as f64,
        is_local_model,
        raw_lines,
        hardware,
    })
}

fn run_detect_hardware_blocking() -> Result<HardwareInfo, String> {
    if let Some(mut cached_hw) = HW_CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap().clone() {
        refresh_live_ram_ffi(&mut cached_hw);
        return Ok(cached_hw);
    }

    let bin_dir = get_bin_dir();
    let helper_py = bin_dir.join("desktop_helper.py");
    let helper_out = Command::new("py")
        .args(["-3.11", helper_py.to_str().unwrap(), "--detect-hw"])
        .current_dir(&bin_dir)
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to invoke hardware detector: {}", e))?;

    let helper_stdout = String::from_utf8_lossy(&helper_out.stdout);
    for line in helper_stdout.lines().rev() {
        if let Some(json_str) = line.strip_prefix("JSON_RESULT:") {
            if let Ok(resp) = serde_json::from_str::<HelperResponse>(json_str) {
                if let Some(mut hw) = resp.hardware {
                    refresh_live_ram_ffi(&mut hw);
                    *HW_CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(hw.clone());
                    return Ok(hw);
                }
            }
        }
    }
    Err("Failed to parse hardware info".to_string())
}

#[tauri::command]
async fn detect_hardware() -> Result<HardwareInfo, String> {
    tauri::async_runtime::spawn_blocking(run_detect_hardware_blocking)
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

fn run_scan_local_blocking(extra_dir: Option<String>) -> Result<Vec<LocalModelEntry>, String> {
    let bin_dir = get_bin_dir();
    let helper_py = bin_dir.join("desktop_helper.py");
    let extra_arg = extra_dir.unwrap_or_default().trim().to_string();
    let helper_out = Command::new("py")
        .args(["-3.11", helper_py.to_str().unwrap(), "--scan-local", &extra_arg])
        .current_dir(&bin_dir)
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to invoke local model scanner: {}", e))?;

    let helper_stdout = String::from_utf8_lossy(&helper_out.stdout);
    for line in helper_stdout.lines().rev() {
        if let Some(json_str) = line.strip_prefix("JSON_RESULT:") {
            if let Ok(resp) = serde_json::from_str::<HelperResponse>(json_str) {
                if let Some(models) = resp.local_models {
                    return Ok(models);
                }
            }
        }
    }
    Err("Failed to parse local models scan output".to_string())
}

#[tauri::command]
async fn scan_local_models(extra_dir: Option<String>) -> Result<Vec<LocalModelEntry>, String> {
    tauri::async_runtime::spawn_blocking(move || run_scan_local_blocking(extra_dir))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

#[tauri::command]
async fn update_model_meta(model_path: String, bookmarked: bool, category: String) -> Result<(), String> {
    let meta_path = get_app_data_dir().join("local_models_meta.json");
    let mut map: HashMap<String, ModelUserMeta> = if meta_path.is_file() {
        std::fs::read_to_string(&meta_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    } else {
        HashMap::new()
    };

    let norm_key = model_path.to_lowercase();
    map.retain(|k, _| k.to_lowercase() != norm_key);
    map.insert(
        model_path,
        ModelUserMeta {
            bookmarked,
            category,
        },
    );

    let json_str = serde_json::to_string_pretty(&map).map_err(|e| e.to_string())?;
    std::fs::write(&meta_path, json_str).map_err(|e| format!("Failed to save metadata: {}", e))?;
    Ok(())
}

#[tauri::command]
async fn delete_local_model(model_path: String) -> Result<String, String> {
    let target = PathBuf::from(&model_path);
    if !target.is_file() {
        return Err(format!("Model file not found at {}", model_path));
    }

    let mut deleted_files = 0usize;

    // Check if this is shard 1 of a multi-shard GGUF (-00001-of-0000N.gguf)
    if let Some(fname) = target.file_name().and_then(|s| s.to_str()) {
        let re_shard = Regex::new(r"^(.*)-00001-of-(\d{5})\.gguf$").unwrap();
        if let Some(caps) = re_shard.captures(fname) {
            let prefix = &caps[1];
            let count_str = &caps[2];
            let total_shards: usize = count_str.parse().unwrap_or(1);
            if let Some(parent) = target.parent() {
                for idx in 1..=total_shards {
                    let sib_name = format!("{}-{:05}-of-{}.gguf", prefix, idx, count_str);
                    let sib_path = parent.join(sib_name);
                    if sib_path.is_file() {
                        std::fs::remove_file(&sib_path)
                            .map_err(|e| format!("Failed to delete shard {:?}: {}", sib_path, e))?;
                        deleted_files += 1;
                    }
                }
            }
        }
    }

    if deleted_files == 0 && target.is_file() {
        std::fs::remove_file(&target)
            .map_err(|e| format!("Failed to delete {}: {}", model_path, e))?;
        deleted_files = 1;
    }

    // If inside skeletons/<subfolder>/ and subfolder is now empty, remove empty folder
    if let Some(parent) = target.parent() {
        if parent.to_string_lossy().to_lowercase().contains("skeletons") {
            let _ = std::fs::remove_dir(parent);
        }
    }

    // Remove from local_models_meta.json if present
    let meta_path = get_app_data_dir().join("local_models_meta.json");
    if meta_path.is_file() {
        if let Ok(mut map) = std::fs::read_to_string(&meta_path)
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, ModelUserMeta>>(&s).ok())
            .ok_or(())
        {
            let norm_key = model_path.to_lowercase();
            map.retain(|k, _| k.to_lowercase() != norm_key);
            if let Ok(json_str) = serde_json::to_string_pretty(&map) {
                let _ = std::fs::write(&meta_path, json_str);
            }
        }
    }

    if let Some(cache) = HELPER_CACHE.get() {
        cache.lock().unwrap().clear();
    }

    Ok(format!("Deleted {} file(s)", deleted_files))
}

#[tauri::command]
async fn start_model_download(url_or_repo: String) -> Result<String, String> {
    let target = url_or_repo.trim().to_string();
    if target.is_empty() {
        return Err("Please enter a Hugging Face repo or GGUF URL to download.".to_string());
    }

    let child_slot = DOWNLOAD_CHILD.get_or_init(|| Mutex::new(None));
    {
        let mut guard = child_slot.lock().unwrap();
        if let Some(mut existing) = guard.take() {
            let _ = existing.kill();
            let _ = existing.wait();
        }
    }

    let bin_dir = get_bin_dir();
    let helper_py = bin_dir.join("desktop_helper.py");
    let status_path = get_app_data_dir().join("download_status.json");

    let init_status = DownloadStatus {
        active: true,
        status: "resolving".to_string(),
        filename: target.clone(),
        url: target.clone(),
        downloaded_mib: 0.0,
        total_mib: 0.0,
        percent: 0.0,
        speed_mbs: 0.0,
        output_path: String::new(),
        error: None,
    };
    if let Ok(js) = serde_json::to_string(&init_status) {
        let _ = std::fs::write(&status_path, js);
    }

    let child = Command::new("py")
        .args([
            "-3.11",
            helper_py.to_str().unwrap(),
            "--download-full",
            &target,
            status_path.to_str().unwrap(),
        ])
        .current_dir(&bin_dir)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|e| format!("Failed to start downloader process: {}", e))?;

    *child_slot.lock().unwrap() = Some(child);
    Ok("Download started".to_string())
}

#[tauri::command]
async fn get_download_status() -> Result<DownloadStatus, String> {
    let status_path = get_app_data_dir().join("download_status.json");
    let mut status: DownloadStatus = if status_path.is_file() {
        std::fs::read_to_string(&status_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    } else {
        DownloadStatus::default()
    };

    if let Some(child_slot) = DOWNLOAD_CHILD.get() {
        let mut guard = child_slot.lock().unwrap();
        if let Some(child) = guard.as_mut() {
            if let Ok(Some(_exit)) = child.try_wait() {
                *guard = None;
                // Re-read status after process exit
                if status_path.is_file() {
                    if let Some(updated) = std::fs::read_to_string(&status_path)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                    {
                        status = updated;
                    }
                }
                if status.active {
                    status.active = false;
                    if status.status == "downloading" || status.status == "resolving" {
                        status.status = "stopped".to_string();
                    }
                }
            }
        }
    }

    Ok(status)
}

#[tauri::command]
async fn cancel_model_download() -> Result<(), String> {
    if let Some(child_slot) = DOWNLOAD_CHILD.get() {
        let mut guard = child_slot.lock().unwrap();
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    let status_path = get_app_data_dir().join("download_status.json");
    if status_path.is_file() {
        if let Some(mut st) = std::fs::read_to_string(&status_path)
            .ok()
            .and_then(|s| serde_json::from_str::<DownloadStatus>(&s).ok())
        {
            st.active = false;
            st.status = "cancelled".to_string();
            if let Ok(js) = serde_json::to_string(&st) {
                let _ = std::fs::write(&status_path, js);
            }
        }
    }
    Ok(())
}

#[tauri::command]
async fn analyze_model(
    url_or_path: String,
    mmproj_url_or_path: Option<String>,
    context_size: u32,
    ngl: i32,
    ncmoe: Option<u32>,
    device: String,
) -> Result<BenchResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_analyze_blocking(url_or_path, mmproj_url_or_path, context_size, ngl, ncmoe, device)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

fn clean_session_skeletons() {
    let temp_dir = std::env::temp_dir().join("whichllmmodel").join("session_skeletons");
    if temp_dir.is_dir() {
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

fn main() {
    // Clean any leftover ephemeral skeletons from previous sessions on startup
    clean_session_skeletons();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            analyze_model,
            detect_hardware,
            scan_local_models,
            update_model_meta,
            delete_local_model,
            start_model_download,
            get_download_status,
            cancel_model_download
        ])
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                // Clean ephemeral skeletons on exit
                clean_session_skeletons();
            }
        });
}
