use serde::{Deserialize, Serialize};
use std::time::Instant;

const RUNPOD_GRAPHQL_URL: &str = "https://api.runpod.io/graphql";
const LLAMA_SERVER_CUDA_IMAGE: &str = "ghcr.io/ggml-org/llama.cpp:server-cuda";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPodGpuType {
    pub id: String,
    pub display_name: String,
    pub memory_gb: f64,
    pub secure_price: Option<f64>,
    pub community_price: Option<f64>,
    pub lowest_price: Option<f64>,
    pub stock_status: String,
    pub is_live_pricing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPodPodInfo {
    pub id: String,
    pub name: String,
    pub desired_status: String,
    pub gpu_type: String,
    pub gpu_count: u32,
    pub cost_per_hr: f64,
    pub uptime_seconds: u64,
    pub endpoint_url: String,
    pub health_status: String,
    pub gpu_util_pct: Option<f64>,
    pub vram_util_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPodDeployRequest {
    pub api_key: String,
    pub gpu_type_id: String,
    pub cloud_type: String,
    pub hf_repo: String,
    pub hf_file: String,
    pub total_ctx: u32,
    pub parallel_slots: u32,
    pub kv_quant: String,
    pub include_vision: bool,
    pub detected_mmproj: bool,
    pub container_disk_gb: u32,
    pub hf_token: Option<String>,
    pub pod_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPodDeployResponse {
    pub pod_id: String,
    pub endpoint_url: String,
    pub server_api_key: String,
    pub docker_args: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPodTestResult {
    pub ok: bool,
    pub reply: String,
    pub latency_ms: u128,
}

pub fn fallback_runpod_gpus() -> Vec<RunPodGpuType> {
    vec![
        RunPodGpuType {
            id: "NVIDIA GeForce RTX 4090".to_string(),
            display_name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            secure_price: Some(0.59),
            community_price: Some(0.34),
            lowest_price: Some(0.34),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA RTX A5000".to_string(),
            display_name: "RTX A5000".to_string(),
            memory_gb: 24.0,
            secure_price: Some(0.27),
            community_price: Some(0.16),
            lowest_price: Some(0.16),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA L4".to_string(),
            display_name: "L4".to_string(),
            memory_gb: 24.0,
            secure_price: Some(0.43),
            community_price: Some(0.39),
            lowest_price: Some(0.39),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA A40".to_string(),
            display_name: "A40".to_string(),
            memory_gb: 48.0,
            secure_price: Some(0.40),
            community_price: Some(0.35),
            lowest_price: Some(0.35),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA RTX A6000".to_string(),
            display_name: "RTX A6000".to_string(),
            memory_gb: 48.0,
            secure_price: Some(0.49),
            community_price: Some(0.33),
            lowest_price: Some(0.33),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA L40S".to_string(),
            display_name: "L40S".to_string(),
            memory_gb: 48.0,
            secure_price: Some(0.86),
            community_price: Some(0.79),
            lowest_price: Some(0.79),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA A100 80GB PCIe".to_string(),
            display_name: "A100 PCIe".to_string(),
            memory_gb: 80.0,
            secure_price: Some(1.39),
            community_price: Some(1.19),
            lowest_price: Some(1.19),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA A100-SXM4-80GB".to_string(),
            display_name: "A100 SXM".to_string(),
            memory_gb: 80.0,
            secure_price: Some(1.49),
            community_price: Some(1.39),
            lowest_price: Some(1.39),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA H100 80GB HBM3".to_string(),
            display_name: "H100 SXM".to_string(),
            memory_gb: 80.0,
            secure_price: Some(2.69),
            community_price: Some(2.39),
            lowest_price: Some(2.39),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
        RunPodGpuType {
            id: "NVIDIA H200".to_string(),
            display_name: "H200 SXM".to_string(),
            memory_gb: 141.0,
            secure_price: Some(3.59),
            community_price: None,
            lowest_price: Some(3.59),
            stock_status: "Estimated".to_string(),
            is_live_pricing: false,
        },
    ]
}

pub fn build_llama_server_args(
    hf_repo: &str,
    hf_file: &str,
    total_ctx: u32,
    parallel_slots: u32,
    kv_quant: &str,
    include_vision: bool,
    detected_mmproj: bool,
    server_api_key: &str,
) -> String {
    let mut parts = vec![
        format!("-hf {}", hf_repo.trim()),
        format!("-hff \"{}\"", hf_file.trim()),
    ];
    if detected_mmproj && !include_vision {
        parts.push("--no-mmproj".to_string());
    }
    parts.push(format!("-c {}", total_ctx.max(512)));
    parts.push(format!("-np {}", parallel_slots.max(1)));
    parts.push("-ngl 999".to_string());
    if !kv_quant.trim().is_empty() && kv_quant.trim() != "f16" {
        let q = kv_quant.trim();
        parts.push(format!("-ctk {}", q));
        parts.push(format!("-ctv {}", q));
    }
    parts.push("--host 0.0.0.0".to_string());
    parts.push("--port 8000".to_string());
    parts.push("-v".to_string());
    parts.push("--log-file /tmp/llama.log".to_string());
    parts.push(format!("--api-key {}", server_api_key));
    parts.join(" ")
}

pub fn calculate_container_disk_gb(model_mib: f64, mmproj_mib: f64) -> u32 {
    let total_gib = (model_mib + mmproj_mib) / 1024.0;
    let needed = (total_gib * 1.15).ceil() as u32 + 15;
    needed.max(25)
}

fn generate_server_api_key() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(123456789);
    let pid = std::process::id() as u128;
    let mixed = nanos ^ (pid.wrapping_mul(0x9e3779b97f4a7c15));
    format!("wlm-{:016x}{:08x}", mixed as u64, (mixed >> 64) as u32)
}

fn post_runpod_graphql(
    api_key: Option<&str>,
    query: &str,
    variables: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let body = serde_json::json!({
        "query": query,
        "variables": variables
    });
    let url = match api_key.map(str::trim).filter(|s| !s.is_empty()) {
        Some(k) => format!("{}?api_key={}", RUNPOD_GRAPHQL_URL, k),
        None => RUNPOD_GRAPHQL_URL.to_string(),
    };
    let mut req = ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(15));
    if let Some(k) = api_key.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.set("Authorization", &format!("Bearer {}", k));
    }

    let resp = req
        .send_json(body)
        .map_err(|e| format!("RunPod API request failed: {}", e))?;
    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("Invalid RunPod JSON response: {}", e))?;

    if let Some(errors) = json.get("errors").and_then(|e| e.as_array()) {
        if !errors.is_empty() {
            let msg = errors[0]
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown RunPod GraphQL error");
            return Err(msg.to_string());
        }
    }
    Ok(json)
}

pub fn fetch_runpod_gpus(api_key: Option<String>) -> Result<Vec<RunPodGpuType>, String> {
    let query = r#"
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                securePrice
                communityPrice
                lowestPrice(input: { gpuCount: 1 }) {
                    uninterruptablePrice
                    stockStatus
                }
            }
        }
    "#;

    let key_ref = api_key.as_deref();
    let json = match post_runpod_graphql(key_ref, query, serde_json::json!({})) {
        Ok(v) => v,
        Err(_) => return Ok(fallback_runpod_gpus()),
    };

    let Some(arr) = json
        .get("data")
        .and_then(|d| d.get("gpuTypes"))
        .and_then(|g| g.as_array())
    else {
        return Ok(fallback_runpod_gpus());
    };

    let mut out = Vec::new();
    for item in arr {
        let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let display_name = item
            .get("displayName")
            .and_then(|v| v.as_str())
            .unwrap_or(&id)
            .to_string();
        let memory_gb = item.get("memoryInGb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if id.is_empty() || memory_gb <= 0.0 {
            continue;
        }
        let secure_price = item.get("securePrice").and_then(|v| v.as_f64()).filter(|p| *p > 0.0);
        let community_price = item
            .get("communityPrice")
            .and_then(|v| v.as_f64())
            .filter(|p| *p > 0.0);
        let lowest_obj = item.get("lowestPrice");
        let uninterruptable = lowest_obj
            .and_then(|l| l.get("uninterruptablePrice"))
            .and_then(|v| v.as_f64())
            .filter(|p| *p > 0.0);
        let stock_status = lowest_obj
            .and_then(|l| l.get("stockStatus"))
            .and_then(|v| v.as_str())
            .unwrap_or("Available")
            .to_string();

        let lowest_price = uninterruptable.or_else(|| match (secure_price, community_price) {
            (Some(s), Some(c)) => Some(s.min(c)),
            (Some(s), None) => Some(s),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        });

        if lowest_price.is_none() && secure_price.is_none() && community_price.is_none() {
            continue;
        }

        out.push(RunPodGpuType {
            id,
            display_name,
            memory_gb,
            secure_price,
            community_price,
            lowest_price,
            stock_status,
            is_live_pricing: true,
        });
    }

    if out.is_empty() {
        return Ok(fallback_runpod_gpus());
    }

    out.sort_by(|a, b| {
        a.memory_gb
            .partial_cmp(&b.memory_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.lowest_price
                    .unwrap_or(999.0)
                    .partial_cmp(&b.lowest_price.unwrap_or(999.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    Ok(out)
}

pub fn deploy_runpod_pod(req: RunPodDeployRequest) -> Result<RunPodDeployResponse, String> {
    if req.api_key.trim().is_empty() {
        return Err("RunPod API key is required to deploy a pod.".to_string());
    }
    if req.hf_repo.trim().is_empty() || req.hf_file.trim().is_empty() {
        return Err("Hugging Face repo and GGUF filename are required.".to_string());
    }

    let server_api_key = generate_server_api_key();
    let docker_args = build_llama_server_args(
        &req.hf_repo,
        &req.hf_file,
        req.total_ctx,
        req.parallel_slots,
        &req.kv_quant,
        req.include_vision,
        req.detected_mmproj,
        &server_api_key,
    );

    let pod_name = req
        .pod_name
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            let short = req
                .hf_repo
                .split('/')
                .last()
                .unwrap_or("llm")
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .take(24)
                .collect::<String>();
            format!("wlm-{}", short)
        });

    let mut env = Vec::new();
    if let Some(tok) = req.hf_token.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        env.push(serde_json::json!({ "key": "HF_TOKEN", "value": tok }));
    }

    let cloud_type = match req.cloud_type.to_uppercase().as_str() {
        "SECURE" => "SECURE",
        "COMMUNITY" => "COMMUNITY",
        _ => "ALL",
    };

    let mutation = r#"
        mutation DeployPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                desiredStatus
                imageName
                machine {
                    gpuDisplayName
                }
            }
        }
    "#;

    let variables = serde_json::json!({
        "input": {
            "cloudType": cloud_type,
            "gpuCount": 1,
            "gpuTypeId": req.gpu_type_id,
            "name": pod_name,
            "imageName": LLAMA_SERVER_CUDA_IMAGE,
            "dockerArgs": docker_args,
            "ports": "8000/http",
            "containerDiskInGb": req.container_disk_gb.max(25),
            "volumeInGb": 0,
            "startSsh": true,
            "env": env
        }
    });

    let json = post_runpod_graphql(Some(&req.api_key), mutation, variables)?;
    let pod_id = json
        .get("data")
        .and_then(|d| d.get("podFindAndDeployOnDemand"))
        .and_then(|p| p.get("id"))
        .and_then(|id| id.as_str())
        .ok_or_else(|| "RunPod did not return a pod ID".to_string())?
        .to_string();

    Ok(RunPodDeployResponse {
        endpoint_url: format!("https://{}-8000.proxy.runpod.net/v1", pod_id),
        pod_id,
        server_api_key,
        docker_args,
    })
}

fn check_llama_server_health(pod_id: &str) -> String {
    let url = format!("https://{}-8000.proxy.runpod.net/health", pod_id);
    match ureq::get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .call()
    {
        Ok(resp) if resp.status() == 200 => "ready".to_string(),
        Ok(_) => "downloading".to_string(),
        Err(ureq::Error::Status(503, _)) => "downloading".to_string(),
        Err(_) => "downloading".to_string(),
    }
}

pub fn list_runpod_pods(api_key: String) -> Result<Vec<RunPodPodInfo>, String> {
    if api_key.trim().is_empty() {
        return Ok(Vec::new());
    }
    let query = r#"
        query MyPods {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    gpuCount
                    costPerHr
                    machine {
                        gpuDisplayName
                    }
                    runtime {
                        uptimeInSeconds
                        gpus {
                            gpuUtilPercent
                            memoryUtilPercent
                        }
                    }
                }
            }
        }
    "#;
    let json = post_runpod_graphql(Some(&api_key), query, serde_json::json!({}))?;
    let Some(pods) = json
        .get("data")
        .and_then(|d| d.get("myself"))
        .and_then(|m| m.get("pods"))
        .and_then(|p| p.as_array())
    else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for p in pods {
        let id = p.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if id.is_empty() {
            continue;
        }
        let name = p.get("name").and_then(|v| v.as_str()).unwrap_or(&id).to_string();
        let desired_status = p
            .get("desiredStatus")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let gpu_type = p
            .get("machine")
            .and_then(|m| m.get("gpuDisplayName"))
            .and_then(|v| v.as_str())
            .unwrap_or("GPU")
            .to_string();
        let gpu_count = p.get("gpuCount").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let cost_per_hr = p.get("costPerHr").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let runtime_obj = p.get("runtime");
        let uptime_seconds = runtime_obj
            .and_then(|r| r.get("uptimeInSeconds"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let first_gpu = runtime_obj
            .and_then(|r| r.get("gpus"))
            .and_then(|g| g.as_array())
            .and_then(|arr| arr.first());
        let gpu_util_pct = first_gpu
            .and_then(|g| g.get("gpuUtilPercent"))
            .and_then(|v| v.as_f64());
        let vram_util_pct = first_gpu
            .and_then(|g| g.get("memoryUtilPercent"))
            .and_then(|v| v.as_f64());

        let health_status = if desired_status != "RUNNING" {
            desired_status.to_lowercase()
        } else if runtime_obj.is_none() || runtime_obj.map(|r| r.is_null()).unwrap_or(true) {
            "provisioning".to_string()
        } else {
            check_llama_server_health(&id)
        };

        out.push(RunPodPodInfo {
            endpoint_url: format!("https://{}-8000.proxy.runpod.net/v1", id),
            id,
            name,
            desired_status,
            gpu_type,
            gpu_count,
            cost_per_hr,
            uptime_seconds,
            health_status,
            gpu_util_pct,
            vram_util_pct,
        });
    }
    Ok(out)
}

pub fn stop_runpod_pod(api_key: String, pod_id: String) -> Result<String, String> {
    let mutation = r#"
        mutation StopPod($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
    "#;
    post_runpod_graphql(
        Some(&api_key),
        mutation,
        serde_json::json!({ "input": { "podId": pod_id } }),
    )?;
    Ok("Stopped".to_string())
}

pub fn terminate_runpod_pod(api_key: String, pod_id: String) -> Result<String, String> {
    let mutation = r#"
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
    "#;
    post_runpod_graphql(
        Some(&api_key),
        mutation,
        serde_json::json!({ "input": { "podId": pod_id } }),
    )?;
    Ok("Terminated".to_string())
}

pub fn test_runpod_endpoint(
    endpoint_url: String,
    server_api_key: String,
) -> Result<RunPodTestResult, String> {
    let url = format!("{}/chat/completions", endpoint_url.trim_end_matches('/'));
    let t0 = Instant::now();
    let mut req = ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(30));
    if !server_api_key.trim().is_empty() {
        req = req.set("Authorization", &format!("Bearer {}", server_api_key.trim()));
    }
    let body = serde_json::json!({
        "messages": [{ "role": "user", "content": "Say hello in 5 words." }],
        "max_tokens": 128,
        "temperature": 0.2
    });
    let resp = req
        .send_json(body)
        .map_err(|e| format!("Endpoint test failed: {}", e))?;
    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("Invalid response JSON: {}", e))?;
    let msg = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));
    let content = msg
        .and_then(|m| m.get("content"))
        .and_then(|s| s.as_str())
        .map(str::trim)
        .unwrap_or("");
    let reasoning = msg
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|s| s.as_str())
        .map(str::trim)
        .unwrap_or("");
    let reply = if !content.is_empty() {
        content.to_string()
    } else if !reasoning.is_empty() {
        format!("[thinking] {}", reasoning)
    } else {
        "(empty reply)".to_string()
    };

    Ok(RunPodTestResult {
        ok: true,
        reply,
        latency_ms: t0.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_llama_server_args() {
        let args = build_llama_server_args(
            "unsloth/Qwen3.5-35B-A3B-GGUF",
            "Qwen3.5-35B-A3B-Q4_K_M.gguf",
            32768,
            4,
            "q8_0",
            false,
            true,
            "wlm-test123",
        );
        assert!(args.contains("-hf unsloth/Qwen3.5-35B-A3B-GGUF"));
        assert!(args.contains("-hff \"Qwen3.5-35B-A3B-Q4_K_M.gguf\""));
        assert!(args.contains("--no-mmproj"));
        assert!(args.contains("-c 32768"));
        assert!(args.contains("-np 4"));
        assert!(args.contains("-ctk q8_0 -ctv q8_0"));
        assert!(args.contains("--port 8000 -v --log-file /tmp/llama.log --api-key wlm-test123"));
    }

    #[test]
    fn test_calculate_container_disk_gb() {
        assert_eq!(calculate_container_disk_gb(1024.0, 0.0), 25);
        assert_eq!(calculate_container_disk_gb(40960.0, 0.0), 61);
    }
}

