# whichllmmodel

>**Plan and deploy a private OpenAI-compatible AI server on the cloud in minutes.**  
> Simulate exact VRAM for your model, context, and concurrent users (`-np`), pick the cheapest cloud GPU guaranteed to fit, and deploy in one click.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Tauri](https://img.shields.io/badge/Tauri-v2-24C8DB?logo=tauri&logoColor=white)
![llama.cpp](https://img.shields.io/badge/powered%20by-llama.cpp-orange.svg)

---

## What is whichllmmodel?

The goal of this project is to make deploying and running your own **private AI server in the cloud** effortless—no DevOps, SSH debugging, or GPU guesswork.

While plenty of tools exist for running small models on a local PC, spinning up a **private, multi-user cloud AI API** still forces you to guess which GPU to rent, overpay for oversized VRAM, or hit Out-Of-Memory (`OOM`) crashes after waiting 15 minutes for weights to download.

`whichllmmodel` solves this using upstream `llama.cpp` **zero-weight null-backend reservation**. It fetches only GGUF header metadata via HTTP Range requests, computes **100% byte-accurate VRAM requirements** for your exact context length and concurrency (`-np`), matches the cheapest fitting cloud GPU, and launches a ready-to-use OpenAI-compatible `llama-server` endpoint in one click.

### Key Features:
* ☁️ **1-Click Private AI Server Deployment (`Deploy`):**
  * Simulate exact VRAM for your **context per user** and **concurrent users (`-np`)** before renting any GPU.
  * Rank live cloud GPUs (starting with **RunPod**) to highlight the **Cheapest Guaranteed Fit** without OOMs or overpaying.
  * Review final pod settings & disk size, deploy `llama-server` in one click, and get a private HTTPS OpenAI-compatible `/v1` endpoint + Bearer key with live readiness polling and Stop/Terminate controls.
* 🎯 **Byte-Accurate Memory Prediction:** Uses `isolated-reserve` (upstream `llama.cpp`) to compute exact allocations for:
  * Model Weights (GPU VRAM vs Host RAM offloading splits with `-ngl`)
  * KV Cache at exact context sizes (`-c 4096`, `8192`, `32768`, `131072`, etc.) and KV quant (`f16`, `q8_0`, `q4_0`)
  * Recurrent State (`RS`) buffers scaled by concurrent users (`-np`) for hybrid architectures (e.g. Qwen 3.5/3.6, LFM)
  * Peak Compute Buffers (`ggml-gallocr`)
  * Vision Projectors (`--mmproj`) for Multimodal / VLMs



---

## 🚀 Getting Started

### Download Pre-built Release
Download the latest Windows installer or portable `.exe` from the [Releases](https://github.com/mzubairtahir/whichllmmodel/releases) page.

### Build from Source

#### Prerequisites:
1. **Node.js** (v18+)
2. **Rust** (`cargo` 1.75+)
3. Visual Studio C++ Build Tools (Windows)

#### Steps:
```bash
git clone https://github.com/mzubairtahir/whichllmmodel.git
cd whichllmmodel

# Install frontend dependencies
npm install

# Run in development mode
npm run tauri dev

# Build standalone release package
npm run tauri build
```

---

## Methodology

`whichllmmodel` does not approximate or guess memory formulas in JavaScript. Instead, it pairs a modern React + Tailwind UI with a Rust backend that drives upstream `llama.cpp` binaries (`llama-bench` & `isolated-reserve`) compiled with the Null-Backend (`Null-CUDA` & `Null-Metal`), to calculate actual numbers.


---

## 🤝 Contributing

Contributions are welcome! Please open an issue for any new ideas, feature requests, or bug fixes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Built with [Tauri v2](https://tauri.app/) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

