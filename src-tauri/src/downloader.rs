//! Native streaming HTTP downloader with HTTP Range resumption and atomic status updates.

use crate::hf;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DownloadStatus {
    pub active: bool,
    pub status: String,
    pub filename: String,
    pub url: String,
    pub downloaded_mib: f64,
    pub total_mib: f64,
    pub percent: f64,
    pub speed_mbs: f64,
    pub output_path: String,
    pub error: Option<String>,
}

impl Default for DownloadStatus {
    fn default() -> Self {
        Self {
            active: false,
            status: "idle".to_string(),
            filename: String::new(),
            url: String::new(),
            downloaded_mib: 0.0,
            total_mib: 0.0,
            percent: 0.0,
            speed_mbs: 0.0,
            output_path: String::new(),
            error: None,
        }
    }
}

pub fn write_download_status(status_path: &Path, status: &DownloadStatus) {
    if let Ok(content) = serde_json::to_string(status) {
        let tmp = status_path.with_extension("tmp");
        if std::fs::write(&tmp, content).is_ok() {
            let _ = std::fs::rename(&tmp, status_path);
        }
    }
}

pub fn download_full_gguf_sync(
    target: &str,
    status_path: &Path,
    cancel_flag: Arc<AtomicBool>,
) -> Result<PathBuf, String> {
    let local_app = dirs::data_local_dir().ok_or("Cannot locate local app data directory")?;
    let dl_dir = local_app.join("whichllmmodel").join("downloaded_models");
    let _ = std::fs::create_dir_all(&dl_dir);

    write_download_status(
        status_path,
        &DownloadStatus {
            active: true,
            status: "resolving".to_string(),
            filename: target.to_string(),
            url: target.to_string(),
            downloaded_mib: 0.0,
            total_mib: 0.0,
            percent: 0.0,
            speed_mbs: 0.0,
            output_path: String::new(),
            error: None,
        },
    );

    let resolved_url = hf::resolve_target_to_gguf_url(target)?;

    // If target was already a local file
    let resolved_local = Path::new(&resolved_url);
    if resolved_local.is_file() {
        let sz = std::fs::metadata(resolved_local).map(|m| m.len()).unwrap_or(0);
        let mib = ((sz as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
        let final_status = DownloadStatus {
            active: false,
            status: "completed".to_string(),
            filename: resolved_local.file_name().unwrap_or_default().to_string_lossy().to_string(),
            url: resolved_url.clone(),
            downloaded_mib: mib,
            total_mib: mib,
            percent: 100.0,
            speed_mbs: 0.0,
            output_path: resolved_local.to_string_lossy().to_string(),
            error: None,
        };
        write_download_status(status_path, &final_status);
        return Ok(resolved_local.to_path_buf());
    }

    let url_clean = resolved_url.split('?').next().unwrap_or(&resolved_url);
    let mut filename = url_clean.rsplit('/').next().unwrap_or("model.gguf").to_string();
    if !filename.to_lowercase().ends_with(".gguf") {
        filename.push_str(".gguf");
    }

    let output_path = dl_dir.join(&filename);
    let part_path = output_path.with_extension("gguf.part");

    let mut resume_bytes: u64 = 0;
    if part_path.is_file() {
        resume_bytes = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    }

    let mut req = ureq::get(&resolved_url)
        .set("User-Agent", "whichllmmodel/1.0")
        .timeout(std::time::Duration::from_secs(30));

    if let Ok(tok) = std::env::var("HF_TOKEN") {
        req = req.set("Authorization", &format!("Bearer {}", tok.trim()));
    }

    if resume_bytes > 0 {
        req = req.set("Range", &format!("bytes={}-", resume_bytes));
    }

    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(416, _)) => {
            // Already fully downloaded into .part
            let _ = std::fs::rename(&part_path, &output_path);
            let sz = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
            let mib = ((sz as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
            write_download_status(
                status_path,
                &DownloadStatus {
                    active: false,
                    status: "completed".to_string(),
                    filename,
                    url: resolved_url,
                    downloaded_mib: mib,
                    total_mib: mib,
                    percent: 100.0,
                    speed_mbs: 0.0,
                    output_path: output_path.to_string_lossy().to_string(),
                    error: None,
                },
            );
            return Ok(output_path);
        }
        Err(e) => {
            let err_str = match &e {
                ureq::Error::Status(c, _) => format!("HTTP {} when downloading {}", c, resolved_url),
                ureq::Error::Transport(t) => format!("Network error: {}", t),
            };
            write_download_status(
                status_path,
                &DownloadStatus {
                    active: false,
                    status: "error".to_string(),
                    filename: filename.clone(),
                    url: resolved_url.clone(),
                    downloaded_mib: 0.0,
                    total_mib: 0.0,
                    percent: 0.0,
                    speed_mbs: 0.0,
                    output_path: String::new(),
                    error: Some(err_str.clone()),
                },
            );
            return Err(err_str);
        }
    };

    let status_code = resp.status();
    if status_code != 200 && status_code != 206 {
        let err_str = format!("HTTP {} when downloading {}", status_code, resolved_url);
        write_download_status(
            status_path,
            &DownloadStatus {
                active: false,
                status: "error".to_string(),
                filename,
                url: resolved_url,
                downloaded_mib: 0.0,
                total_mib: 0.0,
                percent: 0.0,
                speed_mbs: 0.0,
                output_path: String::new(),
                error: Some(err_str.clone()),
            },
        );
        return Err(err_str);
    }

    if status_code == 200 {
        resume_bytes = 0;
    }

    let content_len: u64 = resp
        .header("Content-Length")
        .and_then(|h| h.parse::<u64>().ok())
        .unwrap_or(0);
    let total_bytes = resume_bytes + content_len;
    let mut downloaded_bytes = resume_bytes;

    let mut file: File = if status_code == 206 && resume_bytes > 0 {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .map_err(|e| format!("Failed to open .part file for append: {}", e))?
    } else {
        File::create(&part_path).map_err(|e| format!("Failed to create .part file: {}", e))?
    };

    let mut reader = resp.into_reader();
    let mut buffer = [0u8; 1024 * 1024]; // 1 MiB chunk
    let start_t = Instant::now();
    let mut last_report_t = Instant::now();
    let mut bytes_since_start: u64 = 0;

    loop {
        if cancel_flag.load(Ordering::Relaxed) {
            write_download_status(
                status_path,
                &DownloadStatus {
                    active: false,
                    status: "cancelled".to_string(),
                    filename,
                    url: resolved_url,
                    downloaded_mib: ((downloaded_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
                    total_mib: ((total_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
                    percent: if total_bytes > 0 {
                        ((downloaded_bytes as f64 / total_bytes as f64 * 100.0) * 10.0).round() / 10.0
                    } else {
                        0.0
                    },
                    speed_mbs: 0.0,
                    output_path: String::new(),
                    error: Some("Download cancelled by user".to_string()),
                },
            );
            return Err("Download cancelled by user".to_string());
        }

        let n = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                let err_str = format!("Stream read error: {}", e);
                write_download_status(
                    status_path,
                    &DownloadStatus {
                        active: false,
                        status: "error".to_string(),
                        filename,
                        url: resolved_url,
                        downloaded_mib: ((downloaded_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
                        total_mib: ((total_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
                        percent: 0.0,
                        speed_mbs: 0.0,
                        output_path: String::new(),
                        error: Some(err_str.clone()),
                    },
                );
                return Err(err_str);
            }
        };

        file.write_all(&buffer[..n])
            .map_err(|e| format!("Failed to write chunk to disk: {}", e))?;

        downloaded_bytes += n as u64;
        bytes_since_start += n as u64;

        if last_report_t.elapsed().as_millis() >= 250 {
            let elapsed_sec = start_t.elapsed().as_secs_f64().max(0.001);
            let speed_mbs = ((bytes_since_start as f64 / (1024.0 * 1024.0)) / elapsed_sec * 10.0).round() / 10.0;
            let pct = if total_bytes > 0 {
                ((downloaded_bytes as f64 / total_bytes as f64 * 100.0) * 10.0).round() / 10.0
            } else {
                0.0
            };

            write_download_status(
                status_path,
                &DownloadStatus {
                    active: true,
                    status: "downloading".to_string(),
                    filename: filename.clone(),
                    url: resolved_url.clone(),
                    downloaded_mib: ((downloaded_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
                    total_mib: ((total_bytes as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0,
                    percent: pct.min(99.9),
                    speed_mbs,
                    output_path: output_path.to_string_lossy().to_string(),
                    error: None,
                },
            );
            last_report_t = Instant::now();
        }
    }

    drop(file);
    let _ = std::fs::rename(&part_path, &output_path);

    let final_sz = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
    let final_mib = ((final_sz as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;

    write_download_status(
        status_path,
        &DownloadStatus {
            active: false,
            status: "completed".to_string(),
            filename,
            url: resolved_url,
            downloaded_mib: final_mib,
            total_mib: final_mib,
            percent: 100.0,
            speed_mbs: 0.0,
            output_path: output_path.to_string_lossy().to_string(),
            error: None,
        },
    );

    Ok(output_path)
}

