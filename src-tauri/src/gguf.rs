//! Native GGUF parsing and zero-weight "skeleton" creation.
//!
//! A skeleton is a faithful copy of a GGUF file's header (magic, KV metadata, tensor
//! infos) followed by a sparse, zero-filled data region of the exact original size.
//! llama.cpp's null-backend can load it and report byte-accurate buffer sizes without
//! the weights ever being downloaded.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

pub type R<T> = Result<T, String>;

/// Result of reading the hyper-parameters we care about out of a GGUF header.
#[derive(Clone, Debug)]
pub struct GgufMeta {
    pub arch: String,
    pub context_length: u32,
    pub is_moe: bool,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

/// Minimal byte source so the same parser works on local files and remote Range reads.
pub trait ByteSrc {
    fn read(&mut self, n: usize) -> R<Vec<u8>>;
    fn skip(&mut self, n: u64) -> R<()>;

    fn read_u32(&mut self) -> R<u32> {
        let b = self.read(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> R<u64> {
        let b = self.read(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn read_string(&mut self) -> R<String> {
        let len = self.read_u64()?;
        if len > 64 * 1024 * 1024 {
            return Err("GGUF string length is implausibly large (corrupt header?)".to_string());
        }
        let b = self.read(len as usize)?;
        Ok(String::from_utf8_lossy(&b).to_string())
    }
}

/// GGUF scalar value type sizes (STRING=8 and ARRAY=9 are variable).
fn scalar_size(vtype: u32) -> Option<u64> {
    match vtype {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

fn skip_value<S: ByteSrc>(s: &mut S, vtype: u32) -> R<()> {
    if let Some(sz) = scalar_size(vtype) {
        return s.skip(sz);
    }
    match vtype {
        8 => {
            let len = s.read_u64()?;
            s.skip(len)
        }
        9 => {
            let atype = s.read_u32()?;
            let n = s.read_u64()?;
            if let Some(sz) = scalar_size(atype) {
                s.skip(sz * n)
            } else if atype == 8 {
                for _ in 0..n {
                    let l = s.read_u64()?;
                    s.skip(l)?;
                }
                Ok(())
            } else {
                for _ in 0..n {
                    skip_value(s, atype)?;
                }
                Ok(())
            }
        }
        other => Err(format!("Unknown GGUF value type {}", other)),
    }
}

/// Reads an integer-typed scalar value; returns None (after consuming it) for other types.
fn read_int_value<S: ByteSrc>(s: &mut S, vtype: u32) -> R<Option<u64>> {
    Ok(match vtype {
        0 | 1 | 7 => Some(s.read(1)?[0] as u64),
        2 | 3 => {
            let b = s.read(2)?;
            Some(u16::from_le_bytes([b[0], b[1]]) as u64)
        }
        4 | 5 => Some(s.read_u32()? as u64),
        10 | 11 => Some(s.read_u64()?),
        _ => {
            skip_value(s, vtype)?;
            None
        }
    })
}

struct FileSrc {
    r: BufReader<File>,
}

impl ByteSrc for FileSrc {
    fn read(&mut self, n: usize) -> R<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.r.read_exact(&mut buf).map_err(|e| format!("GGUF read error: {}", e))?;
        Ok(buf)
    }
    fn skip(&mut self, n: u64) -> R<()> {
        self.r
            .seek_relative(n as i64)
            .map_err(|e| format!("GGUF seek error: {}", e))
    }
}

/// Reads architecture / context length / MoE parameters from a local GGUF file without
/// touching the (huge) tokenizer arrays.
pub fn inspect_file(path: &Path) -> R<GgufMeta> {
    let f = File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut s = FileSrc { r: BufReader::new(f) };

    if s.read(4)? != b"GGUF" {
        return Err(format!("Not a valid GGUF file: {}", path.display()));
    }
    let _version = s.read_u32()?;
    let _n_tensors = s.read_u64()?;
    let n_kv = s.read_u64()?;

    let mut arch = String::from("unknown");
    let mut ints: Vec<(String, u64)> = Vec::new();

    for _ in 0..n_kv {
        let key = s.read_string()?;
        let vtype = s.read_u32()?;

        if key == "general.architecture" {
            if vtype == 8 {
                let v = s.read_string()?;
                let v = v.trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string();
                if !v.is_empty() {
                    arch = v;
                }
            } else {
                skip_value(&mut s, vtype)?;
            }
        } else if key.ends_with(".context_length")
            || key.ends_with(".expert_count")
            || key.ends_with(".expert_used_count")
            || key.ends_with(".leading_dense_block_count")
        {
            if let Some(v) = read_int_value(&mut s, vtype)? {
                ints.push((key, v));
            }
        } else if matches!(
            key.as_str(),
            "tokenizer.ggml.tokens" | "tokenizer.ggml.merges" | "tokenizer.ggml.token_type"
        ) && arch != "unknown"
        {
            // All general.* and <arch>.* hyper-parameters precede the vocabulary in GGUF.
            break;
        } else {
            skip_value(&mut s, vtype)?;
        }
    }

    let get = |k: &str| ints.iter().find(|(key, _)| key == k).map(|(_, v)| *v);
    let context_length = get(&format!("{}.context_length", arch))
        .or_else(|| get("general.context_length"))
        .unwrap_or(4096) as u32;

    let mut is_moe = false;
    let mut expert_count = 0u32;
    let mut expert_used_count = 0u32;
    for (k, v) in &ints {
        if k.ends_with(".expert_count") && *v > 0 {
            is_moe = true;
            expert_count = *v as u32;
        } else if k.ends_with(".expert_used_count") && *v > 0 {
            expert_used_count = *v as u32;
        }
    }

    Ok(GgufMeta { arch, context_length, is_moe, expert_count, expert_used_count })
}

pub fn is_gguf_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.to_lowercase().ends_with(".gguf"))
        .unwrap_or(false)
    {
        return true;
    }
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == b"GGUF")
        .unwrap_or(false)
}

/// (block size, bytes per block) per ggml_type id — from gguf-py `GGML_QUANT_SIZES`
/// in the llama.cpp tree this app is built against (QK_K = 256).
fn quant_size(ggml_type: u32) -> Option<(u64, u64)> {
    Some(match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        40 => (64, 36),   // NVFP4
        41 => (128, 18),  // Q1_0
        42 => (64, 18),   // Q2_0
        _ => return None,
    })
}

fn pad(n: u64, align: u64) -> u64 {
    n.div_ceil(align) * align
}

// ---------------------------------------------------------------------------------
// Remote header reader (HTTP Range) and skeleton writer
// ---------------------------------------------------------------------------------

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|t| !t.is_empty())
}

pub fn friendly_net_error(url: &str, e: &ureq::Error) -> String {
    let host = url.split('/').nth(2).unwrap_or("the server");
    match e {
        ureq::Error::Status(401, _) | ureq::Error::Status(403, _) => format!(
            "Access denied by {} (HTTP {}). This repository may be gated or private — set the HF_TOKEN environment variable to a Hugging Face token that has access.",
            host,
            if matches!(e, ureq::Error::Status(401, _)) { 401 } else { 403 }
        ),
        ureq::Error::Status(404, _) => format!("Not found (HTTP 404): {}", url),
        ureq::Error::Status(code, _) => format!("{} returned HTTP {} for {}", host, code, url),
        ureq::Error::Transport(t) => match t.kind() {
            ureq::ErrorKind::Dns => format!(
                "Can't reach {} — check your internet connection (DNS lookup failed).",
                host
            ),
            ureq::ErrorKind::ConnectionFailed => format!(
                "Can't connect to {} — check your internet connection, VPN or proxy.",
                host
            ),
            _ => format!("Network error talking to {}: {}", host, t),
        },
    }
}

fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .user_agent("whichllmmodel/1.0")
        .timeout_connect(std::time::Duration::from_secs(10))
        .timeout_read(std::time::Duration::from_secs(60))
        .build()
}

/// GET with the HF token attached when present.
pub fn http_get(url: &str, range: Option<(u64, u64)>) -> Result<ureq::Response, ureq::Error> {
    let mut req = http_agent().get(url);
    if let Some(t) = hf_token() {
        req = req.set("Authorization", &format!("Bearer {}", t));
    }
    if let Some((a, b)) = range {
        req = req.set("Range", &format!("bytes={}-{}", a, b));
    }
    req.call()
}

/// GET with retries on transient failures only (never on 4xx).
pub fn http_get_retry(url: &str, range: Option<(u64, u64)>) -> R<ureq::Response> {
    let mut last = String::new();
    for attempt in 0..3u32 {
        match http_get(url, range) {
            Ok(r) => return Ok(r),
            Err(e) => {
                last = friendly_net_error(url, &e);
                let retryable = match &e {
                    ureq::Error::Status(code, _) => *code >= 500,
                    ureq::Error::Transport(t) => !matches!(t.kind(), ureq::ErrorKind::Dns),
                };
                if !retryable {
                    return Err(last);
                }
                std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
            }
        }
    }
    Err(last)
}

struct RemoteSrc {
    url: String,
    buf: Vec<u8>,
    pos: usize,
}

const CHUNK: usize = 4 * 1024 * 1024;

impl RemoteSrc {
    fn ensure(&mut self, upto: usize) -> R<()> {
        while self.buf.len() < upto {
            let need = (upto - self.buf.len()).max(CHUNK) as u64;
            let start = self.buf.len() as u64;
            let resp = http_get_retry(&self.url, Some((start, start + need - 1)))?;
            let mut data = Vec::new();
            resp.into_reader()
                .take(need)
                .read_to_end(&mut data)
                .map_err(|e| format!("Network read failed: {}", e))?;
            if data.is_empty() {
                return Err("Unexpected end of remote GGUF file while reading its header".to_string());
            }
            self.buf.extend_from_slice(&data);
        }
        Ok(())
    }
}

impl ByteSrc for RemoteSrc {
    fn read(&mut self, n: usize) -> R<Vec<u8>> {
        self.ensure(self.pos + n)?;
        let out = self.buf[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(out)
    }
    fn skip(&mut self, n: u64) -> R<()> {
        let n = n as usize;
        self.ensure(self.pos + n)?;
        self.pos += n;
        Ok(())
    }
}

#[cfg(windows)]
fn mark_sparse(f: &File) {
    use std::os::windows::io::AsRawHandle;
    #[link(name = "kernel32")]
    extern "system" {
        fn DeviceIoControl(
            h: *mut std::ffi::c_void,
            code: u32,
            inb: *mut std::ffi::c_void,
            insz: u32,
            outb: *mut std::ffi::c_void,
            outsz: u32,
            ret: *mut u32,
            ov: *mut std::ffi::c_void,
        ) -> i32;
    }
    const FSCTL_SET_SPARSE: u32 = 0x000900C4;
    let mut ret: u32 = 0;
    unsafe {
        DeviceIoControl(
            f.as_raw_handle() as *mut _,
            FSCTL_SET_SPARSE,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            &mut ret,
            std::ptr::null_mut(),
        );
    }
}

#[cfg(not(windows))]
fn mark_sparse(_f: &File) {}

/// Downloads only the header of a remote GGUF (one shard) and writes a sparse skeleton.
/// Reuses an existing valid skeleton on disk.
pub fn create_skeleton_for_url(url: &str, output_path: &Path) -> R<()> {
    if output_path.is_file() && std::fs::metadata(output_path).map(|m| m.len() > 0).unwrap_or(false) {
        if inspect_file(output_path).is_ok() {
            return Ok(());
        }
    }

    let url = url.replace("/blob/", "/resolve/");
    let mut s = RemoteSrc { url: url.clone(), buf: Vec::new(), pos: 0 };

    if s.read(4)? != b"GGUF" {
        return Err(format!("Not a GGUF file (bad magic): {}", url));
    }
    let version = s.read_u32()?;
    if version < 2 {
        return Err(format!("Unsupported GGUF version {} (need v2 or v3)", version));
    }
    let n_tensors = s.read_u64()?;
    let n_kv = s.read_u64()?;

    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = s.read_string()?;
        let vtype = s.read_u32()?;
        if key == "general.alignment" {
            if let Some(v) = read_int_value(&mut s, vtype)? {
                if v > 0 {
                    alignment = v;
                }
            }
        } else {
            skip_value(&mut s, vtype)?;
        }
    }

    // Tensor infos: we need name/dims/type/offset only to compute the virtual file size.
    let mut data_end: u64 = 0;
    for _ in 0..n_tensors {
        let _name = s.read_string()?;
        let n_dims = s.read_u32()?;
        let mut n_elems: u64 = 1;
        for _ in 0..n_dims {
            n_elems = n_elems.saturating_mul(s.read_u64()?);
        }
        let ggml_type = s.read_u32()?;
        let offset = s.read_u64()?;
        let (blck, tsize) =
            quant_size(ggml_type).ok_or_else(|| format!("Unknown ggml tensor type id {}", ggml_type))?;
        let nbytes = (n_elems / blck) * tsize;
        data_end = data_end.max(offset + nbytes);
    }

    let header_len = s.pos as u64;
    let data_start = pad(header_len, alignment);
    let total_size = data_start + pad(data_end, alignment);

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    let mut f = File::create(output_path).map_err(|e| format!("Cannot create {}: {}", output_path.display(), e))?;
    mark_sparse(&f);
    f.write_all(&s.buf[..s.pos]).map_err(|e| e.to_string())?;
    f.write_all(&vec![0u8; (data_start - header_len) as usize]).map_err(|e| e.to_string())?;
    // Extending the length of a sparse file allocates no disk blocks.
    f.set_len(total_size).map_err(|e| e.to_string())?;
    Ok(())
}

/// Creates skeleton(s) for `primary_url`. For multi-shard models (-00001-of-0000N.gguf)
/// every shard gets its own skeleton inside a dedicated sub-folder with the original
/// file names so llama.cpp's split loader works. Returns the path of shard 1 / the file.
pub fn create_model_skeleton(primary_url: &str, target_dir: &Path) -> R<std::path::PathBuf> {
    let primary_url = primary_url.replace("/blob/", "/resolve/");
    let fname = primary_url
        .split('?')
        .next()
        .unwrap_or("")
        .rsplit('/')
        .next()
        .unwrap_or("model.gguf")
        .to_string();

    let shard_re = regex::Regex::new(r"-(\d{5})-of-(\d{5})\.gguf$").unwrap();
    if let Some(caps) = shard_re.captures(&fname) {
        let total: u32 = caps[2].parse().unwrap_or(1);
        let prefix = &fname[..caps.get(0).unwrap().start()];
        let sub = target_dir.join(prefix);
        std::fs::create_dir_all(&sub).map_err(|e| e.to_string())?;
        let first = sub.join(&fname);
        create_skeleton_for_url(&primary_url, &first)?;
        for i in 2..=total {
            let rep = format!("-{:05}-of-{}.gguf", i, &caps[2]);
            let sib_name = shard_re.replace(&fname, rep.as_str()).to_string();
            let sib_url = shard_re.replace(&primary_url, rep.as_str()).to_string();
            create_skeleton_for_url(&sib_url, &sub.join(sib_name))?;
        }
        Ok(first)
    } else {
        let out_name = if fname.to_lowercase().ends_with(".gguf") { fname } else { format!("{}.gguf", fname) };
        let out = target_dir.join(out_name);
        create_skeleton_for_url(&primary_url, &out)?;
        Ok(out)
    }
}

