//! Hugging Face API interactions and GGUF URL resolution.

use serde::Deserialize;
use std::path::Path;

pub type R<T> = Result<T, String>;

#[derive(Deserialize, Debug)]
struct HfSibling {
    rfilename: String,
}

#[derive(Deserialize, Debug)]
struct HfModelInfo {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Parses an input string (either "org/repo", "https://huggingface.co/org/repo/...", etc.)
/// and extracts the canonical Hugging Face "org/repo" identifier.
pub fn parse_repo_id(input: &str) -> Option<String> {
    let s = input.trim();
    if Path::new(s).exists() {
        return None;
    }
    let clean = s.trim_end_matches('/');

    if let Some(pos) = clean.find("huggingface.co/") {
        let part = &clean[pos + "huggingface.co/".len()..];
        let tokens: Vec<&str> = part.split('/').filter(|t| !t.is_empty()).collect();
        if tokens.len() >= 2 {
            return Some(format!("{}/{}", tokens[0], tokens[1]));
        }
    } else if clean.contains('/') && !clean.ends_with(".gguf") {
        let tokens: Vec<&str> = clean.split('/').filter(|t| !t.is_empty()).collect();
        if tokens.len() == 2 {
            return Some(format!("{}/{}", tokens[0], tokens[1]));
        }
    }
    None
}

/// Queries Hugging Face API for model info.
fn fetch_repo_info(repo_id: &str) -> R<HfModelInfo> {
    let api_url = format!("https://huggingface.co/api/models/{}", repo_id);
    let mut req = ureq::get(&api_url)
        .set("User-Agent", "whichllmmodel/1.0")
        .timeout(std::time::Duration::from_secs(15));

    if let Some(tok) = hf_token() {
        req = req.set("Authorization", &format!("Bearer {}", tok));
    }

    let resp = req.call().map_err(|e| match e {
        ureq::Error::Status(code, _) => {
            format!("Hugging Face API returned HTTP {} for repo '{}'", code, repo_id)
        }
        ureq::Error::Transport(t) => {
            format!("Network error reaching huggingface.co: {}", t)
        }
    })?;

    resp.into_json::<HfModelInfo>()
        .map_err(|e| format!("Failed to parse Hugging Face API response: {}", e))
}

/// Quant priority ordering (from most desirable to least)
const QUANT_PREFERENCES: &[&str] = &[
    "UD-Q4_K_XL",
    "UD-Q4_K_L",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_K",
    "Q4_0",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_0",
    "Q8_0",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
    "BF16",
    "F16",
];

/// Resolves any input target (local path, direct URL, or HF repo) to a direct GGUF resolve URL or local path.
pub fn resolve_target_to_gguf_url(target: &str) -> R<String> {
    let s = target.trim();
    if Path::new(s).exists() {
        return Ok(s.to_string());
    }

    // Direct HTTP(S) link to a .gguf file
    if s.ends_with(".gguf") && (s.starts_with("http://") || s.starts_with("https://")) {
        return Ok(s.replace("/blob/", "/resolve/"));
    }

    let repo_id = parse_repo_id(s)
        .ok_or_else(|| format!("Could not parse Hugging Face repo ID or valid URL from: {}", s))?;

    let info = fetch_repo_info(&repo_id)?;
    let gguf_files: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|f| f.to_lowercase().ends_with(".gguf") && !f.to_lowercase().contains("mmproj"))
        .collect();

    if gguf_files.is_empty() {
        return Err(format!("No .gguf files found in Hugging Face repository '{}'", repo_id));
    }

    // Check if any file matches our quant preference priority
    let mut chosen: Option<String> = None;
    for pref in QUANT_PREFERENCES {
        for file in &gguf_files {
            let upper = file.to_uppercase();
            if upper.contains(pref) {
                // If it's a sharded model, pick shard 1
                if upper.contains("-OF-") && !upper.contains("-00001-OF-") {
                    continue;
                }
                chosen = Some(file.clone());
                break;
            }
        }
        if chosen.is_some() {
            break;
        }
    }

    // If no preference matched, pick the first valid non-shard2+ file
    let final_filename = chosen.unwrap_or_else(|| {
        gguf_files
            .iter()
            .find(|f| !regex::Regex::new(r"-0000[2-9]-of-\d{5}\.gguf$").unwrap().is_match(f))
            .cloned()
            .unwrap_or_else(|| gguf_files[0].clone())
    });

    Ok(format!("https://huggingface.co/{}/resolve/main/{}", repo_id, final_filename))
}

/// Checks if a Hugging Face repository contains an `mmproj-*.gguf` file and returns its resolve URL.
pub fn find_mmproj_in_hf_repo(url_or_id: &str) -> Option<String> {
    let repo_id = parse_repo_id(url_or_id)?;
    let info = fetch_repo_info(&repo_id).ok()?;

    let mmproj_files: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|f| f.to_lowercase().ends_with(".gguf") && f.to_lowercase().contains("mmproj"))
        .collect();

    if mmproj_files.is_empty() {
        return None;
    }

    // Prefer f16, bf16, f32, q8_0
    for pref in &["f16", "bf16", "f32", "q8_0"] {
        if let Some(m) = mmproj_files.iter().find(|f| f.to_lowercase().contains(pref)) {
            return Some(format!("https://huggingface.co/{}/resolve/main/{}", repo_id, m));
        }
    }

    Some(format!("https://huggingface.co/{}/resolve/main/{}", repo_id, mmproj_files[0]))
}

