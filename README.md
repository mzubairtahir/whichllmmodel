# whichllmmodel

> **Zero-download memory & speed hardware simulator for llama.cpp.**  
> Know if an LLM will fit your GPU or RAM before downloading 20 GB of weights.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Tauri](https://img.shields.io/badge/Tauri-v2-24C8DB?logo=tauri&logoColor=white)
![llama.cpp](https://img.shields.io/badge/powered%20by-llama.cpp-orange.svg)

---

## ⚡ What is whichllmmodel?

Downloading large models (10 GB – 70 GB) just to find out they cause Out-Of-Memory (`OOM`) or crawl at 1 token per second wastes hours of time and bandwidth.

`whichllmmodel` connects directly to upstream `llama.cpp` using **zero-weight null-backend reservation**. It downloads only the first few megabytes of header metadata via HTTP Range requests and computes **100% byte-accurate memory requirements** in under two seconds.

### Key Features:
* 🎯 **Byte-Accurate Memory Prediction:** Uses `llama-bench` and `isolated-reserve` to get exact allocations for:
  * Model Weights (GPU VRAM vs Host RAM offloading splits with `-ngl`)
  * KV Cache at exact context sizes (`-c 4096`, `8192`, `32768`, etc.)
  * Recurrent State (`RS`) buffers for hybrid architectures (e.g. Qwen 3.5/3.6, LFM)
  * Peak Compute Buffers (`ggml-gallocr`)
  * Vision Projectors (`--mmproj`) for Multimodal / VLMs (CLIP weights + compute staging)
* ⚡ **Streaming Bandwidth Speed Estimator:**
  * Runs a 16-pass memory streaming benchmark on your actual system hardware.
  * Estimates exact tokens/second generation speed ($\pm 15\%$ uncertainty tier) without guessing.
* 📦 **My Models (Local Model Manager):**
  * Auto-discovers installed GGUF models across **Ollama**, **LM Studio**, and **Hugging Face cache**.
  * Shows exact physical disk usage (accounting for Windows NTFS sparse files & multi-part shards).
  * Tag, categorize (`Coding`, `General`, `Reasoning`, `Vision`, `Fast / Edge`), and delete models with one click.
* 🧹 **Ephemeral Session Skeletons:**
  * Remote Hugging Face model checks are stored in the OS temp directory and automatically purged on exit.
* ☀️ **Clean Light / Dark Mode.**

---

## 🚀 Getting Started

### Download Pre-built Release
Download the latest Windows installer or portable `.exe` from the [Releases](https://github.com/your-username/whichllmmodel/releases) page.

### Build from Source

#### Prerequisites:
1. **Node.js** (v18+)
2. **Rust** (`cargo` 1.75+)
3. Visual Studio C++ Build Tools (Windows)

#### Steps:
```bash
git clone https://github.com/your-username/whichllmmodel.git
cd whichllmmodel

# Install frontend dependencies
npm install

# Run in development mode
npm run tauri dev

# Build standalone release package
npm run tauri build
```

---

## 🛠️ Architecture

`whichllmmodel` does not approximate or guess memory formulas in JavaScript. Instead, it pairs a modern React + Tailwind UI with a Rust backend that drives upstream `llama.cpp` binaries (`llama-bench` & `isolated-reserve`) compiled with the Null-Backend (`Null-CUDA` & `Null-Metal`).

```
whichllmmodel/
├── src/                <-- React UI (Memory check, hardware detector, library)
├── src-tauri/
│   ├── bin/            <-- Bundled llama.cpp null-backend binaries & DLLs
│   ├── src/main.rs     <-- Rust backend FFI & process orchestrator
│   └── tauri.conf.json <-- Tauri v2 configuration
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Built with [Tauri v2](https://tauri.app/) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

