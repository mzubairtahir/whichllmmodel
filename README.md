# whichllmmodel (wllm)

> **Official open-source Node.js CLI client for [whichllmmodel.com](https://www.whichllmmodel.com)**  
> Hardware-aware local LLM recommendation engine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Node: >=18](https://img.shields.io/badge/node->=18.0.0-green.svg)](https://nodejs.org)
---

## Features

1. **Zero Friction**: Zero login, zero signups, zero API keys. Run and get instant recommendations.
2. **Pure Node.js Hardware Detection**: No heavy native C++ build dependencies.
   - **NVIDIA GPU**: Detects GPU name, **Total VRAM**, and **Usable/Free VRAM**.
   - **Apple Silicon (M-series)**: Detects chip name, **Total Unified Memory**, and applies the macOS **75% VRAM allocation ceiling** for usable graphics memory.
   - **System RAM**: Captures **Total System RAM** and **Usable/Free RAM** across Windows, macOS, and Linux.

---

## 📦 Installation & Executables

### Direct Run via npx
```bash
npx whichllmmodel
# or
npx wllm
```

### Global Installation
```bash
npm install -g whichllmmodel
```

Executable aliases:
- `whichllmmodel`
- `wllm`
- `wlm`

---

## 🛠️ Commands

The MVP provides two focused commands:

### 1. `profile`
Inspects and displays your system hardware, including GPU/chip name, architecture type (Unified Memory vs Discrete GPU vs CPU/RAM), and total vs usable memory breakdown.

```bash
wllm profile
```

Example output:
```text
────────────────────────────────────────────────────────────────────
  HARDWARE PROFILE
────────────────────────────────────────────────────────────────────
  Hardware Name:    Intel(R) HD Graphics 620
  CPU Platform:     Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz (win32 x64)
  Architecture:     CPU / System RAM Acceleration (No discrete GPU detected)
  Integrated GPU:  Intel(R) HD Graphics 620
  System RAM:       5.95 GB usable / 15.88 GB total

  ℹ Models will execute using CPU threads and system RAM.
────────────────────────────────────────────────────────────────────
```

---

### 2. `recommend` (Default)
Recommends the **top 3 models** that best match your hardware and workload. Running `wllm` without subcommands defaults directly to `recommend`.

```bash
wllm recommend [options]
# or simply
wllm [options]
```

#### Options:
- `-c, --context <size>`: Context window size (e.g. `8k`, `16k`, `32k`, `128k`). **Defaults to `32k`**.
- `--cpu-offload`: Enable CPU RAM offloading for models exceeding VRAM on discrete GPUs. (**Default: `true`**).  
  *(Note: CPU offloading is automatically disabled on Unified Memory systems since memory is shared natively).*
- `--no-cpu-offload`: Strictly enforce that the model and KV cache must fit within GPU VRAM.
- `-s, --sort <mode>`: Sorting strategy (**Default: `largest-vram`**).

#### Sorting Options:
| Option | Description |
| :--- | :--- |
| **`largest-vram`** (Default) | Maximizes memory fit without running out of memory. |
| **`coding`** | Ranks models by highest code generation and refactoring scores. |
| **`high-params`** | Ranks by total parameter size (e.g. 70B > 32B > 14B > 8B). |
| **`high-quant`** | Prioritizes models that can run at higher quantization precision (`FP16`, `Q8_0`, `Q5_K_M`). |

#### Examples:

```bash
# Default: Top 3 models by VRAM utilization
wllm

# Top coding models
wllm --sort coding

# Highest parameter models
wllm --sort high-params

# Highest quantization quality
wllm --sort high-quant

# Strict GPU VRAM fit with 16k context
wllm -c 16k --no-cpu-offload
```

---

## 📄 License

MIT © [whichllmmodel.com](https://www.whichllmmodel.com)
