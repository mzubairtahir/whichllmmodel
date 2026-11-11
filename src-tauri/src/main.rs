// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod downloader;
mod gguf;
mod hf;
mod hw;
mod scanner;

use downloader::DownloadStatus;
use hw::HardwareInfo;
use regex::Regex;
use scanner::LocalModelEntry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const CREATE_NO_WINDOW: u32 = 0x08000000;

fn get_app_data_dir() -> PathBuf {
    if let Some(local_app) = dirs::data_local_dir() {
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

#[derive(Deserialize, Serialize, Clone, Default)]
struct ModelUserMeta {
    #[serde(default)]
    bookmarked: bool,
    #[serde(default)]
    category: String,
}

#[derive(Clone, Debug)]
struct HelperResponse {
    ok: bool,
    error: Option<String>,
    skeleton_path: Option<String>,
    resolved_url: Option<String>,
    is_local: Option<bool>,
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
static DOWNLOAD_CANCEL: OnceLock<Arc<AtomicBool>> = OnceLock::new();

fn resolve_local_gguf_path(input_path: &str) -> Result<PathBuf, String> {
    let p = Path::new(input_path);
    if p.is_file() {
        return Ok(p.canonicalize().unwrap_or_else(|_| p.to_path_buf()));
    }
    if p.is_dir() {
        if let Ok(entries) = std::fs::read_dir(p) {
            let mut ggufs: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|f| {
                    f.is_file()
                        && f.extension()
                            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                            .unwrap_or(false)
                })
                .collect();
            ggufs.sort();
            if let Some(first) = ggufs.into_iter().next() {
                return Ok(first.canonicalize().unwrap_or_else(|_| first));
            }
        }
    }
    Err(format!("Could not find a valid .gguf file at: {}", input_path))
}

fn get_llama_cpp_supported_archs() -> HashSet<String> {
    let mut supported = HashSet::new();
    for arch in &[
        "llama", "falcon", "grok", "gpt2", "gptj", "gptneox", "mpt", "baichuan",
        "starcoder", "refact", "bert", "nomic-bert", "jina-bert-v2", "bloom", "stablelm",
        "qwen", "qwen2", "qwen2moe", "qwen2vl", "phi2", "phi3", "plamo", "codeshell",
        "orion", "internlm2", "minicpm", "minicpm3", "gemma", "gemma2", "gemma3", "starcoder2",
        "mamba", "xverse", "command-r", "dbrx", "olmo", "olmo2", "openelm", "arctic",
        "deepseek", "deepseek2", "chatglm", "bitnet", "t5", "t5encoder", "jais", "nemotron",
        "exaone", "rwkv6", "granite", "granite-moe", "chameleon", "wavtokenizer", "whisper",
    ] {
        supported.insert(arch.to_string());
    }

    let bin_dir = get_bin_dir();
    let arch_cpp_candidates = [
        bin_dir.join("..").join("llama.cpp").join("src").join("llama-arch.cpp"),
        PathBuf::from(r"e:\llama.cpp study\llama.cpp\src\llama-arch.cpp"),
    ];
    for cand in &arch_cpp_candidates {
        if cand.is_file() {
            if let Ok(content) = std::fs::read_to_string(cand) {
                let re = Regex::new(r#"\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}"#).unwrap();
                for cap in re.captures_iter(&content) {
                    let name = cap[1].trim();
                    if !name.is_empty() && name != "(unknown)" {
                        supported.insert(name.to_string());
                    }
                }
                break;
            }
        }
    }
    supported
}

fn prepare_model_native(url_or_path: &str, mmproj_arg: &str) -> Result<HelperResponse, String> {
    let t0 = Instant::now();
    let is_already_local = Path::new(url_or_path).exists();
    let session_dir = std::env::temp_dir().join("whichllmmodel").join("session_skeletons");
    let _ = std::fs::create_dir_all(&session_dir);

    // 1. Resolve / download main model skeleton
    let (skeleton_path_buf, resolved_url) = if is_already_local {
        let local_path = resolve_local_gguf_path(url_or_path)?;
        (local_path, url_or_path.to_string())
    } else {
        let res_url = hf::resolve_target_to_gguf_url(url_or_path)?;
        if Path::new(&res_url).exists() {
            let local_path = resolve_local_gguf_path(&res_url)?;
            (local_path, res_url)
        } else {
            let skel = gguf::create_model_skeleton(&res_url, &session_dir)?;
            (skel, res_url)
        }
    };

    // 2. Inspect GGUF metadata
    let meta = gguf::inspect_file(&skeleton_path_buf)?;
    let arch = meta.arch;
    let context_length = meta.context_length;
    let is_moe = meta.is_moe;
    let expert_count = meta.expert_count;
    let expert_used_count = meta.expert_used_count;

    // 3. Architecture check
    let supported = get_llama_cpp_supported_archs();
    let arch_supported = supported.contains(&arch) || supported.is_empty();
    let supported_arch_count = supported.len();

    // 4. Resolve or auto-detect optional Vision Projector (mmproj)
    let mut mmproj_skeleton_path = String::new();
    let clean_mm = mmproj_arg.trim();
    if !clean_mm.is_empty() {
        if Path::new(clean_mm).exists() {
            if let Ok(p) = resolve_local_gguf_path(clean_mm) {
                mmproj_skeleton_path = p.to_string_lossy().to_string();
            }
        } else {
            let mm_url = if clean_mm.ends_with(".gguf")
                && (clean_mm.starts_with("http://") || clean_mm.starts_with("https://"))
            {
                clean_mm.replace("/blob/", "/resolve/")
            } else if let Some(auto) = hf::find_mmproj_in_hf_repo(clean_mm) {
                auto
            } else {
                hf::resolve_target_to_gguf_url(clean_mm).unwrap_or_default()
            };

            if !mm_url.is_empty() {
                if let Ok(skel) = gguf::create_model_skeleton(&mm_url, &session_dir) {
                    mmproj_skeleton_path = skel.to_string_lossy().to_string();
                }
            }
        }
    } else if !is_already_local {
        // Auto-detect vision projector if HF repo contains mmproj-*.gguf
        if let Some(auto_mm) = hf::find_mmproj_in_hf_repo(url_or_path) {
            if let Ok(skel) = gguf::create_model_skeleton(&auto_mm, &session_dir) {
                mmproj_skeleton_path = skel.to_string_lossy().to_string();
            }
        }
    }

    let prep_time_ms = ((t0.elapsed().as_secs_f64() * 1000.0) * 10.0).round() / 10.0;
    let hw = hw::detect_system_hardware();

    Ok(HelperResponse {
        ok: true,
        error: None,
        skeleton_path: Some(skeleton_path_buf.to_string_lossy().to_string()),
        resolved_url: Some(resolved_url),
        is_local: Some(is_already_local),
        prep_time_ms: Some(prep_time_ms),
        arch: Some(arch),
        arch_supported: Some(arch_supported),
        supported_arch_count: Some(supported_arch_count),
        context_length: Some(context_length),
        is_moe: Some(is_moe),
        expert_count: Some(expert_count),
        expert_used_count: Some(expert_used_count),
        mmproj_skeleton_path: Some(mmproj_skeleton_path),
        hardware: Some(hw),
    })
}

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
    let bench_exe = bin_dir.join("llama-bench.exe");
    let isolated_exe = bin_dir.join("isolated-reserve.exe");

    if !bench_exe.exists() {
        return Err(format!("llama-bench.exe not found at {:?}", bench_exe));
    }

    // Step 1: Native zero-download GGUF preparation (cached in memory after 1st run)
    let cache_map = HELPER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = (url_or_path.clone(), mmproj_arg.clone());
    let cached_entry = cache_map.lock().unwrap().get(&cache_key).cloned();

    let mut helper_data = if let Some(cached) = cached_entry {
        cached
    } else {
        let data = prepare_model_native(&url_or_path, &mmproj_arg)?;
        cache_map.lock().unwrap().insert(cache_key, data.clone());
        if let Some(ref hw) = data.hardware {
            *HW_CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(hw.clone());
        }
        data
    };

    if let Some(ref mut hw) = helper_data.hardware {
        hw::refresh_live_ram_ffi(hw);
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
    // Note: llama-bench uses -p <prompt_tokens> to test context size (it does NOT accept -c)
    let bench_start = Instant::now();
    let mut bench_args = vec![
        "-m".to_string(),
        target_gguf.clone(),
        "-r".to_string(),
        "0".to_string(),
        "-p".to_string(),
        context_size.to_string(),
        "-n".to_string(),
        "0".to_string(),
        "--no-warmup".to_string(),
        "-v".to_string(),
    ];

    let target_dev = if device.trim().is_empty() {
        "CUDA0".to_string()
    } else {
        device.clone()
    };
    bench_args.push("-dev".to_string());
    bench_args.push(target_dev);

    if device.to_uppercase() == "CPU" {
        bench_args.push("-ngl".to_string());
        bench_args.push("0".to_string());
    } else {
        bench_args.push("-ngl".to_string());
        bench_args.push(ngl.to_string());
    }

    if ncmoe_val > 0 {
        bench_args.push("-ncmoe".to_string());
        bench_args.push(ncmoe_val.to_string());
    }

    let output = Command::new(&bench_exe)
        .args(&bench_args)
        .current_dir(&bin_dir)
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to execute llama-bench: {}", e))?;

    let bench_time_ms = ((bench_start.elapsed().as_secs_f64() * 1000.0) * 10.0).round() / 10.0;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Step 3: Exact Memory Extraction from llama.cpp logs
    let combined_logs = format!("{}\n{}", stdout, stderr);
    let raw_lines: Vec<String> = combined_logs
        .lines()
        .filter(|l| {
            l.contains("load_tensors:")
                || l.contains("init_kv:")
                || l.contains("init_rs:")
                || l.contains("init_compute:")
                || l.contains("print_device_info:")
                || l.contains("failed")
                || l.contains("error")
        })
        .map(|s| s.to_string())
        .collect();

    let mut vram_model_mib = 0.0;
    let mut ram_model_mib = 0.0;
    let mut vram_kv_mib = 0.0;
    let mut ram_kv_mib = 0.0;
    let mut vram_rs_mib = 0.0;
    let mut ram_rs_mib = 0.0;
    let mut vram_compute_mib = 0.0;
    let mut ram_compute_mib = 0.0;
    let mut output_buffer_mib = 0.0;

    let re_model = Regex::new(r"model size\s*=\s*([0-9.]+)\s*MiB").unwrap();
    let re_model_dev = Regex::new(r"([A-Za-z0-9_-]+)\s+model buffer size\s*=\s*([0-9.]+)\s*MiB").unwrap();
    let re_kv_split = Regex::new(r"([A-Za-z0-9_-]+)\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB").unwrap();
    let re_rs_split = Regex::new(r"([A-Za-z0-9_-]+)\s+recurrent state buffer size\s*=\s*([0-9.]+)\s*MiB").unwrap();
    let re_compute = Regex::new(r"([A-Za-z0-9_-]+)\s+compute buffer size\s*=\s*([0-9.]+)\s*MiB").unwrap();
    let re_out_buf = Regex::new(r"output buffer size\s*=\s*([0-9.]+)\s*MiB").unwrap();

    let is_gpu = |dev: &str| {
        let d = dev.to_lowercase();
        d.contains("cuda") || d.contains("metal") || d.contains("vulkan") || d.contains("kompute") || d.contains("sycl")
    };

    let mut total_model_size_mib = 0.0;
    for line in combined_logs.lines() {
        if let Some(caps) = re_model.captures(line) {
            total_model_size_mib = caps[1].parse::<f64>().unwrap_or(0.0);
        }
        for caps in re_model_dev.captures_iter(line) {
            let dev = &caps[1];
            let val = caps[2].parse::<f64>().unwrap_or(0.0);
            if is_gpu(dev) {
                vram_model_mib += val;
            } else {
                ram_model_mib += val;
            }
        }
        for caps in re_kv_split.captures_iter(line) {
            let dev = &caps[1];
            let val = caps[2].parse::<f64>().unwrap_or(0.0);
            if is_gpu(dev) {
                vram_kv_mib += val;
            } else {
                ram_kv_mib += val;
            }
        }
        for caps in re_rs_split.captures_iter(line) {
            let dev = &caps[1];
            let val = caps[2].parse::<f64>().unwrap_or(0.0);
            if is_gpu(dev) {
                vram_rs_mib += val;
            } else {
                ram_rs_mib += val;
            }
        }
        for caps in re_compute.captures_iter(line) {
            let dev = &caps[1];
            let val = caps[2].parse::<f64>().unwrap_or(0.0);
            if is_gpu(dev) {
                vram_compute_mib += val;
            } else {
                ram_compute_mib += val;
            }
        }
        if let Some(caps) = re_out_buf.captures(line) {
            output_buffer_mib = caps[1].parse::<f64>().unwrap_or(0.0);
        }
    }

    if (vram_model_mib + ram_model_mib) == 0.0 && total_model_size_mib > 0.0 {
        if ngl > 0 {
            vram_model_mib = total_model_size_mib;
        } else {
            ram_model_mib = total_model_size_mib;
        }
    }

    // Step 4: Optional Vision Projector (mmproj) Isolated Reservation
    let mut has_mmproj = false;
    let mut mmproj_name = String::new();
    let mut mmproj_projector_type = String::new();
    let mut vram_vision_model_mib = 0.0;
    let mut ram_vision_model_mib = 0.0;
    let mut vram_vision_compute_mib = 0.0;
    let mut ram_vision_compute_mib = 0.0;

    if !mmproj_skeleton_path.is_empty() && Path::new(&mmproj_skeleton_path).is_file() {
        let vision_cache = VISION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let v_key = (mmproj_skeleton_path.clone(), device.clone());
        let cached_v = vision_cache.lock().unwrap().get(&v_key).cloned();

        let v_data = if let Some(cv) = cached_v {
            cv
        } else {
            let mut cv = CachedVision {
                has_mmproj: true,
                mmproj_name: Path::new(&mmproj_skeleton_path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                mmproj_projector_type: "clip_vision_encoder".to_string(),
                vram_vision_model_mib: 0.0,
                ram_vision_model_mib: 0.0,
                vram_vision_compute_mib: 0.0,
                ram_vision_compute_mib: 0.0,
                extra_lines: Vec::new(),
            };

            if isolated_exe.is_file() {
                if let Ok(iso_out) = Command::new(&isolated_exe)
                    .args(["--mmproj", &mmproj_skeleton_path, "--device", &device])
                    .current_dir(&bin_dir)
                    .creation_flags(CREATE_NO_WINDOW)
                    .output()
                {
                    let iso_stdout = String::from_utf8_lossy(&iso_out.stdout);
                    for line in iso_stdout.lines().rev() {
                        if let Some(json_str) = line.strip_prefix("VISION_RESERVE_JSON:") {
                            if let Ok(bufs) = serde_json::from_str::<Vec<IsolatedVisionBuf>>(json_str) {
                                for b in bufs {
                                    if is_gpu(&b.device) {
                                        cv.vram_vision_model_mib += b.model_mib;
                                        cv.vram_vision_compute_mib += b.compute_mib;
                                    } else {
                                        cv.ram_vision_model_mib += b.model_mib;
                                        cv.ram_vision_compute_mib += b.compute_mib;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // Fallback estimation from file size if isolated tool is unavailable
            if cv.vram_vision_model_mib == 0.0 && cv.ram_vision_model_mib == 0.0 {
                let sz = std::fs::metadata(&mmproj_skeleton_path).map(|m| m.len()).unwrap_or(0);
                let mm_mib = ((sz as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
                if ngl > 0 {
                    cv.vram_vision_model_mib = mm_mib;
                    cv.vram_vision_compute_mib = 128.0;
                } else {
                    cv.ram_vision_model_mib = mm_mib;
                    cv.ram_vision_compute_mib = 128.0;
                }
            }

            vision_cache.lock().unwrap().insert(v_key, cv.clone());
            cv
        };

        has_mmproj = v_data.has_mmproj;
        mmproj_name = v_data.mmproj_name;
        mmproj_projector_type = v_data.mmproj_projector_type;
        vram_vision_model_mib = v_data.vram_vision_model_mib;
        ram_vision_model_mib = v_data.ram_vision_model_mib;
        vram_vision_compute_mib = v_data.vram_vision_compute_mib;
        ram_vision_compute_mib = v_data.ram_vision_compute_mib;
    }

    let total_vram_mib = vram_model_mib
        + vram_kv_mib
        + vram_rs_mib
        + vram_compute_mib
        + output_buffer_mib
        + vram_vision_model_mib
        + vram_vision_compute_mib;

    let total_ram_mib =
        ram_model_mib + ram_kv_mib + ram_rs_mib + ram_compute_mib + ram_vision_model_mib + ram_vision_compute_mib;

    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    let model_name = Path::new(&resolved_url)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| resolved_url.clone());

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
        elapsed_ms,
        prep_time_ms,
        bench_time_ms,
        is_local_model,
        raw_lines,
        hardware,
    })
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

fn run_detect_hardware_blocking() -> Result<HardwareInfo, String> {
    if let Some(mut cached_hw) = HW_CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap().clone() {
        hw::refresh_live_ram_ffi(&mut cached_hw);
        return Ok(cached_hw);
    }
    let hw = hw::detect_system_hardware();
    *HW_CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(hw.clone());
    Ok(hw)
}

#[tauri::command]
async fn detect_hardware() -> Result<HardwareInfo, String> {
    tauri::async_runtime::spawn_blocking(run_detect_hardware_blocking)
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

fn run_scan_local_blocking(extra_dir: Option<String>) -> Result<Vec<LocalModelEntry>, String> {
    Ok(scanner::scan_local_models(extra_dir))
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

    let cancel_flag = DOWNLOAD_CANCEL.get_or_init(|| Arc::new(AtomicBool::new(false)));
    cancel_flag.store(false, Ordering::Relaxed);
    let cancel_clone = cancel_flag.clone();

    let status_path = get_app_data_dir().join("download_status.json");

    downloader::write_download_status(
        &status_path,
        &DownloadStatus {
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
        },
    );

    std::thread::spawn(move || {
        let _ = downloader::download_full_gguf_sync(&target, &status_path, cancel_clone);
    });

    Ok("Download started".to_string())
}

#[tauri::command]
async fn get_download_status() -> Result<DownloadStatus, String> {
    let status_path = get_app_data_dir().join("download_status.json");
    if status_path.is_file() {
        if let Ok(content) = std::fs::read_to_string(&status_path) {
            if let Ok(status) = serde_json::from_str::<DownloadStatus>(&content) {
                return Ok(status);
            }
        }
    }
    Ok(DownloadStatus::default())
}

#[tauri::command]
async fn cancel_model_download() -> Result<String, String> {
    if let Some(flag) = DOWNLOAD_CANCEL.get() {
        flag.store(true, Ordering::Relaxed);
    }
    let status_path = get_app_data_dir().join("download_status.json");
    let mut current = DownloadStatus::default();
    if status_path.is_file() {
        if let Ok(content) = std::fs::read_to_string(&status_path) {
            if let Ok(s) = serde_json::from_str::<DownloadStatus>(&content) {
                current = s;
            }
        }
    }
    current.active = false;
    current.status = "cancelled".to_string();
    current.error = Some("Download cancelled by user".to_string());
    downloader::write_download_status(&status_path, &current);
    Ok("Download cancelled".to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection_native() {
        let hw = hw::detect_system_hardware();
        assert!(!hw.cpu_name.is_empty(), "CPU name should not be empty");
        assert!(hw.ram_total_mib > 0.0, "Total RAM should be greater than 0");
        assert!(hw.ram_channels >= 1, "RAM channels should be at least 1");
        assert!(hw.measured_ram_gbs > 0.0, "Measured RAM bandwidth should be > 0");
        println!(
            "Detected CPU: {} ({} threads, {} MHz)",
            hw.cpu_name, hw.cpu_threads, hw.cpu_mhz
        );
        println!(
            "RAM: {:.1} MiB total, {:.1} MiB free, {} channels @ {} MT/s ({:.1} GB/s measured)",
            hw.ram_total_mib, hw.ram_free_mib, hw.ram_channels, hw.ram_speed_mts, hw.measured_ram_gbs
        );
        println!(
            "GPU: {} (discrete: {}, {:.1} MiB VRAM)",
            hw.gpu_name, hw.gpu_is_discrete, hw.gpu_vram_total_mib
        );
    }

    #[test]
    fn test_hf_repo_parser() {
        assert_eq!(
            hf::parse_repo_id("unsloth/DeepSeek-R1-GGUF"),
            Some("unsloth/DeepSeek-R1-GGUF".to_string())
        );
        assert_eq!(
            hf::parse_repo_id("https://huggingface.co/unsloth/DeepSeek-R1-GGUF"),
            Some("unsloth/DeepSeek-R1-GGUF".to_string())
        );
        assert_eq!(
            hf::parse_repo_id("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main"),
            Some("Qwen/Qwen2.5-7B-Instruct-GGUF".to_string())
        );
    }

    #[test]
    fn test_llama_cpp_arch_support() {
        let archs = get_llama_cpp_supported_archs();
        assert!(archs.contains("llama"));
        assert!(archs.contains("qwen2"));
        assert!(archs.contains("deepseek2"));
        assert!(archs.contains("gemma2"));
        assert!(archs.len() >= 40);
    }

    #[test]
    fn test_local_model_scanner() {
        let models = scanner::scan_local_models(None);
        println!("Scanned local models count: {}", models.len());
        for m in models.iter().take(3) {
            println!("- [{}] {} (arch: {}, size: {:.1} MiB)", m.source, m.name, m.arch, m.model_logical_mib);
        }
    }
}
