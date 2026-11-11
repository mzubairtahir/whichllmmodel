//! Local GGUF model scanner across Ollama, LM Studio, Hugging Face Hub, and custom directories.

use crate::gguf::{self, GgufMeta};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LocalModelEntry {
    pub name: String,
    pub source: String,
    pub model_path: String,
    pub mmproj_path: String,
    pub arch: String,
    pub context_length: u32,
    pub is_moe: bool,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub model_logical_mib: f64,
    pub model_disk_mib: f64,
    pub mmproj_logical_mib: f64,
    pub mmproj_disk_mib: f64,
    pub is_sparse_skeleton: bool,
    pub bookmarked: bool,
    pub category: String,
}

#[derive(Deserialize, Serialize, Default)]
struct UserMeta {
    #[serde(default)]
    bookmarked: bool,
    #[serde(default)]
    category: Option<String>,
}

#[link(name = "kernel32")]
extern "system" {
    fn GetCompressedFileSizeW(lp_file_name: *const u16, lp_file_size_high: *mut u32) -> u32;
}

/// Retrieves the actual compressed or sparse allocation size on Windows NTFS.
pub fn get_actual_disk_size_bytes(path: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut high: u32 = 0;
    unsafe {
        let low = GetCompressedFileSizeW(wide.as_ptr(), &mut high);
        if low != 0xFFFF_FFFF {
            return ((high as u64) << 32) | (low as u64);
        }
    }
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn load_saved_meta() -> HashMap<String, UserMeta> {
    if let Some(local_app) = dirs::data_local_dir() {
        let meta_path = local_app.join("whichllmmodel").join("local_models_meta.json");
        if meta_path.is_file() {
            if let Ok(content) = std::fs::read_to_string(meta_path) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, UserMeta>>(&content) {
                    return map
                        .into_iter()
                        .map(|(k, v)| (k.to_lowercase(), v))
                        .collect();
                }
            }
        }
    }
    HashMap::new()
}

fn auto_detect_category(name: &str, mmproj_path: &str, logical_mib: f64) -> String {
    let low = name.to_lowercase();
    if !mmproj_path.is_empty()
        || [
            "-vl", "vision", "minicpm-v", "omni", "llava", "pixtral", "qwen2-vl", "qwen2.5-vl",
            "qwen3-vl", "gemma-3",
        ]
        .iter()
        .any(|k| low.contains(k))
    {
        return "Vision".to_string();
    }
    if ["-r1", "deepseek-r1", "thinking", "qwq", "reason", "o1-", "marco-o1"]
        .iter()
        .any(|k| low.contains(k))
    {
        return "Reasoning".to_string();
    }
    if [
        "code", "coder", "starcoder", "codestral", "devstral", "flint", "qwen2.5-coder",
    ]
    .iter()
    .any(|k| low.contains(k))
    {
        return "Coding".to_string();
    }
    if logical_mib > 0.0 && logical_mib <= 1600.0 {
        return "Fast / Edge".to_string();
    }
    "General".to_string()
}

pub fn scan_local_models(extra_dir: Option<String>) -> Vec<LocalModelEntry> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let mut discovered: Vec<LocalModelEntry> = Vec::new();
    let mut seen_paths: HashSet<String> = HashSet::new();
    let saved_meta = load_saved_meta();

    let mut add_entry = |name: String, source: &str, model_path: &Path, mmproj_path: &Path| {
        let abs_model = match model_path.canonicalize() {
            Ok(p) => p,
            Err(_) => model_path.to_path_buf(),
        };
        let norm_key = abs_model.to_string_lossy().to_lowercase();
        if seen_paths.contains(&norm_key) || !abs_model.is_file() {
            return;
        }
        seen_paths.insert(norm_key.clone());

        let mut logical_bytes = std::fs::metadata(&abs_model).map(|m| m.len()).unwrap_or(0);
        let mut disk_bytes = get_actual_disk_size_bytes(&abs_model);

        // Check if shard 1 of multi-shard GGUF (-00001-of-0000N.gguf)
        let file_name = abs_model.file_name().unwrap_or_default().to_string_lossy();
        if let Some(caps) = regex::Regex::new(r"-00001-of-(\d{5})\.gguf$")
            .unwrap()
            .captures(&file_name)
        {
            if let Ok(total_shards) = caps[1].parse::<usize>() {
                for s_idx in 2..=total_shards {
                    let sib_name = format!("-{:05}-of-{}.gguf", s_idx, &caps[1]);
                    let sib_file = regex::Regex::new(r"-00001-of-\d{5}\.gguf$")
                        .unwrap()
                        .replace(&file_name, &sib_name);
                    let sib_path = abs_model.with_file_name(sib_file.as_ref());
                    if sib_path.is_file() {
                        logical_bytes += std::fs::metadata(&sib_path).map(|m| m.len()).unwrap_or(0);
                        disk_bytes += get_actual_disk_size_bytes(&sib_path);
                    }
                }
            }
        }

        let has_mmproj = mmproj_path.is_file();
        let mmproj_logical_bytes = if has_mmproj {
            std::fs::metadata(mmproj_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };
        let mmproj_disk_bytes = if has_mmproj {
            get_actual_disk_size_bytes(mmproj_path)
        } else {
            0
        };

        let meta = gguf::inspect_file(&abs_model).unwrap_or(GgufMeta {
            arch: "unknown".to_string(),
            context_length: 4096,
            is_moe: false,
            expert_count: 0,
            expert_used_count: 0,
        });

        let logical_mib = ((logical_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
        let disk_mib = ((disk_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
        let is_sparse_skeleton = (disk_bytes < (logical_bytes / 4)) && (logical_bytes > 100 * 1024 * 1024);

        let user_meta = saved_meta.get(&norm_key);
        let bookmarked = user_meta.map(|u| u.bookmarked).unwrap_or(false);
        let category = user_meta
            .and_then(|u| u.category.clone())
            .unwrap_or_else(|| auto_detect_category(&name, mmproj_path.to_string_lossy().as_ref(), logical_mib));

        discovered.push(LocalModelEntry {
            name,
            source: source.to_string(),
            model_path: abs_model.to_string_lossy().to_string(),
            mmproj_path: if has_mmproj {
                mmproj_path.to_string_lossy().to_string()
            } else {
                String::new()
            },
            arch: meta.arch,
            context_length: meta.context_length,
            is_moe: meta.is_moe,
            expert_count: meta.expert_count,
            expert_used_count: meta.expert_used_count,
            model_logical_mib: logical_mib,
            model_disk_mib: disk_mib,
            mmproj_logical_mib: ((mmproj_logical_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
            mmproj_disk_mib: ((mmproj_disk_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
            is_sparse_skeleton,
            bookmarked,
            category,
        });
    };

    // 1. Ollama (~/.ollama/models)
    let ollama_models = home.join(".ollama").join("models");
    let blobs_dir = ollama_models.join("blobs");
    if ollama_models.is_dir() && blobs_dir.is_dir() {
        for man_sub in &["manifests", "manifests-v2"] {
            let man_dir = ollama_models.join(man_sub);
            if !man_dir.is_dir() {
                continue;
            }
            if let Ok(entries) = walk_dir_all(&man_dir) {
                for fpath in entries {
                    if !fpath.is_file() {
                        continue;
                    }
                    if let Ok(content) = std::fs::read_to_string(&fpath) {
                        if let Ok(man_json) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(layers) = man_json.get("layers").and_then(|l| l.as_array()) {
                                let mut model_blob = PathBuf::new();
                                let mut proj_blob = PathBuf::new();
                                for lyr in layers {
                                    let mtype = lyr.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
                                    let digest = lyr.get("digest").and_then(|d| d.as_str()).unwrap_or("");
                                    let clean_digest = digest.replace(':', "-");
                                    let bpath = blobs_dir.join(clean_digest);
                                    if mtype.contains("image.model") && bpath.is_file() {
                                        model_blob = bpath;
                                    } else if mtype.contains("image.projector") && bpath.is_file() {
                                        proj_blob = bpath;
                                    }
                                }
                                if model_blob.is_file() && gguf::is_gguf_file(&model_blob) {
                                    let rel = fpath
                                        .strip_prefix(&man_dir)
                                        .unwrap_or(&fpath)
                                        .to_string_lossy()
                                        .replace('\\', "/");
                                    let parts: Vec<&str> = rel
                                        .split('/')
                                        .filter(|p| !["registry.ollama.ai", "ollama.com", "library"].contains(p))
                                        .collect();
                                    let tag_name = if parts.len() >= 3 {
                                        format!("{}/{}:{}", parts[parts.len() - 3], parts[parts.len() - 2], parts[parts.len() - 1])
                                    } else if parts.len() >= 2 {
                                        format!("{}:{}", parts[parts.len() - 2], parts[parts.len() - 1])
                                    } else {
                                        rel
                                    };
                                    add_entry(tag_name, "Ollama", &model_blob, &proj_blob);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Hugging Face Hub Cache (~/.cache/huggingface/hub/models--*)
    let hf_hub = home.join(".cache").join("huggingface").join("hub");
    if hf_hub.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&hf_hub) {
            for entry in entries.flatten() {
                let p = entry.path();
                let fname = p.file_name().unwrap_or_default().to_string_lossy();
                if p.is_dir() && fname.starts_with("models--") {
                    let repo_label = fname["models--".len()..].replace("--", "/");
                    if let Ok(all_files) = walk_dir_all(&p) {
                        let ggufs: Vec<PathBuf> = all_files
                            .into_iter()
                            .filter(|f| f.is_file() && f.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false))
                            .collect();
                        let mmprojs: Vec<&PathBuf> = ggufs
                            .iter()
                            .filter(|g| g.file_name().unwrap_or_default().to_string_lossy().to_lowercase().contains("mmproj"))
                            .collect();
                        let mains: Vec<&PathBuf> = ggufs
                            .iter()
                            .filter(|g| !g.file_name().unwrap_or_default().to_string_lossy().to_lowercase().contains("mmproj"))
                            .collect();
                        let paired_mm = mmprojs.first().cloned();

                        for mp in mains {
                            let gname = mp.file_name().unwrap_or_default().to_string_lossy();
                            if regex::Regex::new(r"-0000[2-9]-of-\d{5}\.gguf$").unwrap().is_match(&gname) {
                                continue;
                            }
                            add_entry(
                                format!("{} ({})", repo_label, gname),
                                "HuggingFace Cache",
                                mp,
                                paired_mm.unwrap_or(&PathBuf::new()),
                            );
                        }
                    }
                }
            }
        }
    }

    // 3. LM Studio (~/.lmstudio/models and ~/.cache/lm-studio/models)
    for lms_dir in &[
        home.join(".lmstudio").join("models"),
        home.join(".cache").join("lm-studio").join("models"),
    ] {
        if lms_dir.is_dir() {
            if let Ok(all_files) = walk_dir_all(lms_dir) {
                scan_directory_ggufs(&all_files, "LM Studio", &mut add_entry);
            }
        }
    }

    // 4. Downloaded models (%LOCALAPPDATA%\whichllmmodel\downloaded_models)
    if let Some(local_app) = dirs::data_local_dir() {
        let dl_dir = local_app.join("whichllmmodel").join("downloaded_models");
        if dl_dir.is_dir() {
            if let Ok(all_files) = walk_dir_all(&dl_dir) {
                scan_directory_ggufs(&all_files, "Downloaded", &mut add_entry);
            }
        }
    }

    // 5. Custom Folder
    if let Some(extra) = extra_dir {
        let trimmed = extra.trim();
        if !trimmed.is_empty() {
            let p = Path::new(trimmed);
            if p.is_dir() {
                if let Ok(all_files) = walk_dir_all(p) {
                    scan_directory_ggufs(&all_files, "Custom Folder", &mut add_entry);
                }
            }
        }
    }

    discovered
}

fn scan_directory_ggufs<F>(all_files: &[PathBuf], source_label: &str, add_entry: &mut F)
where
    F: FnMut(String, &str, &Path, &Path),
{
    let ggufs: Vec<&PathBuf> = all_files
        .iter()
        .filter(|f| f.is_file() && f.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false))
        .collect();

    let mmprojs: Vec<&PathBuf> = ggufs
        .iter()
        .filter(|g| g.file_name().unwrap_or_default().to_string_lossy().to_lowercase().contains("mmproj"))
        .cloned()
        .collect();

    let mains: Vec<&PathBuf> = ggufs
        .iter()
        .filter(|g| !g.file_name().unwrap_or_default().to_string_lossy().to_lowercase().contains("mmproj"))
        .cloned()
        .collect();

    let paired_mm = mmprojs.first().cloned();
    for mp in mains {
        let gname = mp.file_name().unwrap_or_default().to_string_lossy();
        if regex::Regex::new(r"-0000[2-9]-of-\d{5}\.gguf$").unwrap().is_match(&gname) {
            continue;
        }
        add_entry(
            gname.to_string(),
            source_label,
            mp,
            paired_mm.unwrap_or(&PathBuf::new()),
        );
    }
}

fn walk_dir_all(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Ok(mut sub) = walk_dir_all(&path) {
                    files.append(&mut sub);
                }
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}

