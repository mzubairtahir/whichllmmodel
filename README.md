# whichllmmodel

> **Official Hardware & Memory Profiler for [whichllmmodel.com](https://www.whichllmmodel.com)**  
> Inspect your local hardware memory and bridge directly into the whichllmmodel Local Finder.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Node: >=18](https://img.shields.io/badge/node->=18.0.0-green.svg)](https://nodejs.org)


---

## 📦 Quick Start

### Run Instantly via `npx` (No Installation Required)
```bash
npx whichllmmodel
```

### Global Installation
```bash
npm install -g whichllmmodel
```

---

## 🖥️ Terminal Output Example

```text
whichllmmodel • Hardware & Memory Profiler

Hardware Detected:
  • Device:         NVIDIA GeForce RTX 3060
  • CPU:            Intel(R) Core(TM) i7-12700K (win32 x64)
  • Architecture:   Discrete GPU

Memory Breakdown:
  Physical Installed Memory:
    • GPU VRAM:     12.0 GB installed
    • System RAM:   32.0 GB installed

  Live Available Memory (Ready for LLMs):
    • GPU VRAM:     10.4 GB available (87% free)
    • System RAM:   22.1 GB available (69% free)

Explore all compatible models online:
https://www.whichllmmodel.com/app/text/local?vram=10.4&ram=22.1&memory_mode=available
```

---

## 🛠️ CLI Options

| Flag | Description | Example |
| :--- | :--- | :--- |
| **`-o, --open`** | Automatically opens the pre-filled Local Finder URL directly in your default browser. | `whichllmmodel --open` |

```bash
# Profile and immediately open WhichLLM Local Finder in browser
whichllmmodel -o
```

---

## 📄 License

MIT © [whichllmmodel.com](https://www.whichllmmodel.com)
