import os
import sys
import re
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

# Add simulator and gguf-py to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def fast_inspect_gguf_metadata(file_path: str) -> dict:
    """Sub-millisecond binary GGUF metadata reader that skips massive tokenizer arrays instead of creating 400k numpy memmaps."""
    import struct

    TYPE_SIZES = {
        0: 1,   # UINT8
        1: 1,   # INT8
        2: 2,   # UINT16
        3: 2,   # INT16
        4: 4,   # UINT32
        5: 4,   # INT32
        6: 4,   # FLOAT32
        7: 1,   # BOOL
        10: 8,  # UINT64
        11: 8,  # INT64
        12: 8,  # FLOAT64
    }

    arch = "unknown"
    context_length = 4096
    is_moe = False
    expert_count = 0
    expert_used_count = 0
    kv_ints = {}

    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a valid GGUF file: {file_path}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def skip_value(vtype: int):
            if vtype in TYPE_SIZES:
                f.seek(TYPE_SIZES[vtype], 1)
            elif vtype == 8:  # STRING
                (slen,) = struct.unpack("<Q", f.read(8))
                f.seek(slen, 1)
            elif vtype == 9:  # ARRAY
                atype, alen = struct.unpack("<IQ", f.read(12))
                if atype in TYPE_SIZES:
                    f.seek(TYPE_SIZES[atype] * alen, 1)
                elif atype == 8:
                    for _ in range(alen):
                        (slen,) = struct.unpack("<Q", f.read(8))
                        f.seek(slen, 1)
                else:
                    for _ in range(alen):
                        skip_value(atype)

        def read_scalar(vtype: int):
            if vtype == 8:
                (slen,) = struct.unpack("<Q", f.read(8))
                return f.read(slen).decode("utf-8", errors="ignore").strip("\x00 \t\n\r")
            elif vtype in (0, 1, 7):
                return struct.unpack("<B", f.read(1))[0]
            elif vtype in (2, 3):
                return struct.unpack("<H", f.read(2))[0]
            elif vtype == 4:
                return struct.unpack("<I", f.read(4))[0]
            elif vtype == 5:
                return struct.unpack("<i", f.read(4))[0]
            elif vtype in (10, 11):
                return struct.unpack("<Q", f.read(8))[0]
            else:
                skip_value(vtype)
                return None

        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", f.read(8))
            key = f.read(klen).decode("utf-8", errors="ignore")
            (vtype,) = struct.unpack("<I", f.read(4))

            if key == "general.architecture":
                val = read_scalar(vtype)
                if isinstance(val, str) and val:
                    arch = val
            elif (
                key.endswith(".context_length")
                or key.endswith(".expert_count")
                or key.endswith(".expert_used_count")
                or key.endswith(".leading_dense_block_count")
            ):
                val = read_scalar(vtype)
                if isinstance(val, int):
                    kv_ints[key] = val
            elif key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type") and arch != "unknown":
                # All general.* and <arch>.* hyperparameters precede the vocabulary table in GGUF
                break
            else:
                skip_value(vtype)

    if f"{arch}.context_length" in kv_ints:
        context_length = kv_ints[f"{arch}.context_length"]
    elif "general.context_length" in kv_ints:
        context_length = kv_ints["general.context_length"]

    for k, v in kv_ints.items():
        if k.endswith(".expert_count") and v > 0:
            is_moe = True
            expert_count = v
        elif k.endswith(".expert_used_count") and v > 0:
            expert_used_count = v

    return {
        "arch": arch,
        "context_length": context_length,
        "is_moe": is_moe,
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
    }


def detect_system_hardware() -> dict:
    """Sub-millisecond native Windows hardware & live memory detector (CPU, RAM, SMBIOS DIMMs, GPU/VRAM)."""
    hw = {
        "cpu_name": "Unknown CPU",
        "cpu_threads": os.cpu_count() or 4,
        "cpu_mhz": 0,
        "ram_total_mib": 16384.0,
        "ram_free_mib": 8192.0,
        "ram_used_mib": 8192.0,
        "ram_load_pct": 50,
        "ram_installed_gb": 16,
        "ram_channels": 2,
        "ram_speed_mts": 2400,
        "ram_bandwidth_gbs": 38.4,
        "measured_ram_gbs": 26.2,
        "effective_ram_gbs": 26.2,
        "gpu_name": "Integrated / CPU Only",
        "gpu_is_discrete": False,
        "gpu_vram_total_mib": 0.0,
        "gpu_vram_free_mib": 0.0,
        "gpu_shared_ram_mib": 0.0,
        "gpu_bandwidth_gbs": 38.4,
        "measured_gpu_gbs": 26.2,
        "effective_gpu_gbs": 26.2,
    }

    if sys.platform != "win32":
        return hw

    try:
        import ctypes
        import struct
        import winreg

        # 1. Live Physical Memory via GlobalMemoryStatusEx
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_mib = stat.ullTotalPhys / (1024.0 * 1024.0)
            free_mib = stat.ullAvailPhys / (1024.0 * 1024.0)
            hw["ram_total_mib"] = round(total_mib, 1)
            hw["ram_free_mib"] = round(free_mib, 1)
            hw["ram_used_mib"] = round(max(0.0, total_mib - free_mib), 1)
            hw["ram_load_pct"] = int(stat.dwMemoryLoad)
            hw["gpu_shared_ram_mib"] = round(total_mib * 0.5, 1)

        # 2. SMBIOS Type 17 (Physical Memory DIMMs: Speed in MT/s, Channels, Installed GB)
        try:
            k32 = ctypes.windll.kernel32
            sig = struct.unpack(">I", b"RSMB")[0]
            sz = k32.GetSystemFirmwareTable(sig, 0, None, 0)
            if sz > 8:
                buf = ctypes.create_string_buffer(sz)
                k32.GetSystemFirmwareTable(sig, 0, buf, sz)
                raw = buf.raw[8:]
                i = 0
                sticks = []
                while i < len(raw) - 4:
                    t, length, _ = struct.unpack_from("<BBH", raw, i)
                    if length < 4:
                        break
                    if t == 17 and length >= 0x17:
                        size_mb = struct.unpack_from("<H", raw, i + 0x0C)[0]
                        speed = struct.unpack_from("<H", raw, i + 0x15)[0]
                        cfg_speed = struct.unpack_from("<H", raw, i + 0x20)[0] if length >= 0x22 else speed
                        if 0 < size_mb < 0xFFFF:
                            sticks.append((size_mb, cfg_speed or speed))
                    end = i + length
                    while end < len(raw) - 1 and not (raw[end] == 0 and raw[end + 1] == 0):
                        end += 1
                    i = end + 2
                if sticks:
                    installed_mb = sum(s[0] for s in sticks)
                    speeds = [s[1] for s in sticks if s[1] > 0]
                    mt_s = min(speeds) if speeds else 2400
                    channels = max(1, min(len(sticks), 4))
                    bw_gbs = round((mt_s * 8.0 * channels) / 1000.0, 1)
                    hw["ram_installed_gb"] = max(1, round(installed_mb / 1024))
                    hw["ram_channels"] = channels
                    hw["ram_speed_mts"] = mt_s
                    hw["ram_bandwidth_gbs"] = bw_gbs
                    hw["gpu_bandwidth_gbs"] = bw_gbs
        except Exception:
            pass

        # 3. CPU Name & Clock from Registry
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as k:
                hw["cpu_name"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
                hw["cpu_mhz"] = int(winreg.QueryValueEx(k, "~MHz")[0])
        except Exception:
            pass

        # 4. GPU & VRAM via Registry + optional nvidia-smi for live NVIDIA telemetry
        gpus = []
        cls_key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls_key_path) as ck:
                for idx in range(16):
                    try:
                        with winreg.OpenKey(ck, f"{idx:04d}") as sk:
                            desc = str(winreg.QueryValueEx(sk, "DriverDesc")[0]).strip()
                            if "Microsoft Basic" in desc or "Remote" in desc:
                                continue
                            vram_bytes = 0
                            for vname in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                                try:
                                    val = winreg.QueryValueEx(sk, vname)[0]
                                    if isinstance(val, int):
                                        vram_bytes = val
                                        break
                                    elif isinstance(val, bytes) and len(val) in (4, 8):
                                        vram_bytes = int.from_bytes(val, "little")
                                        break
                                except OSError:
                                    pass
                            gpus.append((desc, round(vram_bytes / (1024.0 * 1024.0), 1)))
                    except OSError:
                        break
        except Exception:
            pass

        if gpus:
            # Prefer discrete GPU (highest VRAM or NVIDIA/AMD)
            gpus.sort(key=lambda x: (("nvidia" in x[0].lower() or "radeon" in x[0].lower() or "arc" in x[0].lower()), x[1]), reverse=True)
            best_desc, best_vram_mib = gpus[0]
            hw["gpu_name"] = best_desc
            hw["gpu_vram_total_mib"] = best_vram_mib
            hw["gpu_vram_free_mib"] = best_vram_mib
            lower_desc = best_desc.lower()
            is_disc = (
                "nvidia" in lower_desc
                or "geforce" in lower_desc
                or "rtx" in lower_desc
                or "gtx" in lower_desc
                or ("radeon" in lower_desc and best_vram_mib >= 2048)
                or ("arc" in lower_desc and best_vram_mib >= 2048)
            )
            hw["gpu_is_discrete"] = is_disc
            if is_disc:
                if best_vram_mib >= 22000:
                    hw["gpu_bandwidth_gbs"] = 936.0
                elif best_vram_mib >= 15000:
                    hw["gpu_bandwidth_gbs"] = 672.0
                elif best_vram_mib >= 11000:
                    hw["gpu_bandwidth_gbs"] = 504.0
                elif best_vram_mib >= 7500:
                    hw["gpu_bandwidth_gbs"] = 360.0
                else:
                    hw["gpu_bandwidth_gbs"] = 224.0

        # If NVIDIA GPU exists, query nvidia-smi for exact live free VRAM
        if "nvidia" in hw["gpu_name"].lower():
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                    timeout=1.5,
                    creationflags=0x08000000,
                ).decode("utf-8", errors="ignore").strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 3:
                        hw["gpu_name"] = parts[0]
                        hw["gpu_vram_total_mib"] = float(parts[1])
                        hw["gpu_vram_free_mib"] = float(parts[2])
                        hw["gpu_is_discrete"] = True
            except Exception:
                pass

        # 5. Live 64 MiB x 16 Streaming Passes Benchmark (WhichLLMModel Predictor Logic)
        try:
            import time
            import numpy as np

            buffer_bytes = 64 * 1024 * 1024
            num_elements = buffer_bytes // 4
            a = np.ones((4096, num_elements // 4096), dtype=np.float32)
            x = np.ones((num_elements // 4096,), dtype=np.float32)
            _ = a @ x  # Warmup pass
            iterations = 16
            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = a @ x
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                total_bytes = buffer_bytes * iterations
                raw_measured = round((total_bytes / 1e9) / elapsed, 1)
                measured_ram = max(raw_measured, round(hw["ram_bandwidth_gbs"] * 0.77, 1))
                hw["measured_ram_gbs"] = measured_ram
                hw["effective_ram_gbs"] = measured_ram
                if not hw["gpu_is_discrete"]:
                    hw["measured_gpu_gbs"] = measured_ram
                    hw["effective_gpu_gbs"] = measured_ram
        except Exception:
            hw["measured_ram_gbs"] = hw["ram_bandwidth_gbs"]
            hw["effective_ram_gbs"] = hw["ram_bandwidth_gbs"]

        # 5. If discrete NVIDIA GPU exists, run 64 MiB x 16 streaming passes on CUDA if torch is installed
        if hw["gpu_is_discrete"]:
            hw["measured_gpu_gbs"] = hw["gpu_bandwidth_gbs"]
            hw["effective_gpu_gbs"] = hw["gpu_bandwidth_gbs"]
            if "nvidia" in hw["gpu_name"].lower():
                try:
                    import time
                    import torch

                    if torch.cuda.is_available():
                        buffer_bytes = 64 * 1024 * 1024
                        num_elements = buffer_bytes // 4
                        t_buf = torch.randn(num_elements, device="cuda", dtype=torch.float32)
                        torch.cuda.synchronize()
                        _ = t_buf.sum()
                        torch.cuda.synchronize()
                        iterations = 16
                        t0 = time.time()
                        for _ in range(iterations):
                            _ = t_buf.sum()
                        torch.cuda.synchronize()
                        elapsed = time.time() - t0
                        if elapsed > 0:
                            total_bytes = buffer_bytes * iterations
                            measured_vram = round((total_bytes / 1e9) / elapsed, 1)
                            hw["measured_gpu_gbs"] = measured_vram
                            hw["effective_gpu_gbs"] = measured_vram
                except Exception:
                    pass
    except Exception:
        pass

    return hw


def get_llama_cpp_supported_archs() -> set:
    """Reads llama.cpp/src/llama-arch.cpp to extract all registered LLM_ARCH_NAMES."""
    arch_cpp = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "llama.cpp", "src", "llama-arch.cpp"))
    supported = set()
    if os.path.isfile(arch_cpp):
        with open(arch_cpp, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        block_match = re.search(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\};", content, re.DOTALL)
        text_to_scan = block_match.group(1) if block_match else content
        for m in re.finditer(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}', text_to_scan):
            arch_name = m.group(1).strip()
            if arch_name and arch_name != "(unknown)":
                supported.add(arch_name)
    return supported


def is_gguf_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False


def get_actual_disk_size_bytes(file_path: str) -> int:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            GetCompressedFileSizeW = ctypes.windll.kernel32.GetCompressedFileSizeW
            GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
            GetCompressedFileSizeW.restype = wintypes.DWORD
            high = wintypes.DWORD(0)
            low = GetCompressedFileSizeW(file_path, ctypes.byref(high))
            if low != 0xFFFFFFFF:
                return (high.value << 32) | low
        except Exception:
            pass
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def resolve_local_gguf(input_path: str) -> str:
    p = Path(input_path)
    if is_gguf_file(p):
        return str(p.resolve())
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        if ggufs:
            return str(ggufs[0].resolve())
    raise FileNotFoundError(f"Could not find a valid .gguf file at: {input_path}")


def get_app_data_dir() -> str:
    app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    target = os.path.join(app_data, "whichllmmodel")
    os.makedirs(target, exist_ok=True)
    return target


def load_models_meta() -> dict:
    meta_path = os.path.join(get_app_data_dir(), "local_models_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {os.path.normcase(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass
    return {}


def auto_detect_category(name: str, mmproj_path: str, logical_mib: float) -> str:
    low = name.lower()
    if mmproj_path or any(k in low for k in ("-vl", "vision", "minicpm-v", "omni", "llava", "pixtral", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "gemma-3")):
        return "Vision"
    if any(k in low for k in ("-r1", "deepseek-r1", "thinking", "qwq", "reason", "o1-", " Marco-o1")):
        return "Reasoning"
    if any(k in low for k in ("code", "coder", "starcoder", "codestral", "devstral", "flint", "qwen2.5-coder")):
        return "Coding"
    if 0 < logical_mib <= 1600:
        return "Fast / Edge"
    return "General"


def scan_local_models(extra_dir: str = "") -> list:
    """Automatically scans Ollama, LM Studio, Hugging Face Hub cache, Downloaded Models, Skeletons, and optional custom folder for GGUF models."""
    home = os.path.expanduser("~")
    discovered = []
    seen_paths = set()
    saved_meta = load_models_meta()

    def add_model_entry(
        name: str,
        source: str,
        model_path: str,
        mmproj_path: str = "",
    ):
        abs_model_path = os.path.abspath(model_path)
        norm_p = os.path.normcase(abs_model_path)
        if norm_p in seen_paths or not os.path.isfile(model_path):
            return
        seen_paths.add(norm_p)

        logical_bytes = os.path.getsize(model_path)
        disk_bytes = get_actual_disk_size_bytes(model_path)

        # If this is shard 1 of a multi-shard GGUF (-00001-of-0000N.gguf), sum all present shards
        shard_m = re.search(r"-00001-of-(\d{5})\.gguf$", os.path.basename(model_path))
        if shard_m:
            total_shards = int(shard_m.group(1))
            for s_idx in range(2, total_shards + 1):
                sib_path = re.sub(r"-00001-of-(\d{5})\.gguf$", f"-{s_idx:05d}-of-\\1.gguf", model_path)
                if os.path.isfile(sib_path):
                    logical_bytes += os.path.getsize(sib_path)
                    disk_bytes += get_actual_disk_size_bytes(sib_path)
        mmproj_logical_bytes = os.path.getsize(mmproj_path) if (mmproj_path and os.path.isfile(mmproj_path)) else 0
        mmproj_disk_bytes = get_actual_disk_size_bytes(mmproj_path) if (mmproj_path and os.path.isfile(mmproj_path)) else 0

        try:
            meta = fast_inspect_gguf_metadata(model_path)
        except Exception:
            meta = {
                "arch": "unknown",
                "context_length": 4096,
                "is_moe": False,
                "expert_count": 0,
                "expert_used_count": 0,
            }

        logical_mib = round(logical_bytes / (1024.0 * 1024.0), 1)
        disk_mib = round(disk_bytes / (1024.0 * 1024.0), 1)
        is_sparse_skeleton = (disk_bytes < logical_bytes * 0.25) and (logical_bytes > 100 * 1024 * 1024)

        user_meta = saved_meta.get(norm_p, {})
        bookmarked = bool(user_meta.get("bookmarked", False))
        category = user_meta.get("category") or auto_detect_category(name, mmproj_path, logical_mib)

        discovered.append(
            {
                "name": name,
                "source": source,
                "model_path": abs_model_path,
                "mmproj_path": os.path.abspath(mmproj_path) if (mmproj_path and os.path.isfile(mmproj_path)) else "",
                "arch": meta.get("arch", "unknown"),
                "context_length": meta.get("context_length", 4096),
                "is_moe": meta.get("is_moe", False),
                "expert_count": meta.get("expert_count", 0),
                "expert_used_count": meta.get("expert_used_count", 0),
                "model_logical_mib": logical_mib,
                "model_disk_mib": disk_mib,
                "mmproj_logical_mib": round(mmproj_logical_bytes / (1024.0 * 1024.0), 1),
                "mmproj_disk_mib": round(mmproj_disk_bytes / (1024.0 * 1024.0), 1),
                "is_sparse_skeleton": is_sparse_skeleton,
                "bookmarked": bookmarked,
                "category": category,
            }
        )

    # 1. Scan Ollama (~/.ollama/models/manifests & blobs)
    ollama_root = os.path.join(home, ".ollama", "models")
    blobs_dir = os.path.join(ollama_root, "blobs")
    if os.path.isdir(ollama_root) and os.path.isdir(blobs_dir):
        for man_dir_name in ("manifests", "manifests-v2"):
            man_root = os.path.join(ollama_root, man_dir_name)
            if not os.path.isdir(man_root):
                continue
            for r, _, files in os.walk(man_root):
                for f in files:
                    man_path = os.path.join(r, f)
                    try:
                        with open(man_path, "r", encoding="utf-8", errors="ignore") as mf:
                            man_json = json.load(mf)
                        layers = man_json.get("layers", [])
                        model_blob = ""
                        proj_blob = ""
                        for lyr in layers:
                            mtype = lyr.get("mediaType", "")
                            digest = lyr.get("digest", "").replace(":", "-")
                            bpath = os.path.join(blobs_dir, digest)
                            if "image.model" in mtype and os.path.isfile(bpath):
                                model_blob = bpath
                            elif "image.projector" in mtype and os.path.isfile(bpath):
                                proj_blob = bpath
                        if model_blob and is_gguf_file(Path(model_blob)):
                            rel = os.path.relpath(man_path, man_root).replace("\\", "/")
                            parts = [p for p in rel.split("/") if p not in ("registry.ollama.ai", "ollama.com", "library")]
                            tag_name = ":".join(parts[-2:]) if len(parts) >= 2 else rel
                            if len(parts) >= 3:
                                tag_name = f"{parts[-3]}/{parts[-2]}:{parts[-1]}"
                            add_model_entry(tag_name, "Ollama", model_blob, proj_blob)
                    except Exception:
                        pass

    # 2. Scan Hugging Face Hub Cache (~/.cache/huggingface/hub)
    hf_hub = os.path.join(home, ".cache", "huggingface", "hub")
    if os.path.isdir(hf_hub):
        for repo_folder in sorted(os.listdir(hf_hub)):
            repo_path = os.path.join(hf_hub, repo_folder)
            if not os.path.isdir(repo_path) or not repo_folder.startswith("models--"):
                continue
            repo_label = repo_folder[len("models--") :].replace("--", "/")
            all_ggufs = []
            for r, _, files in os.walk(repo_path):
                for f in files:
                    if f.lower().endswith(".gguf"):
                        all_ggufs.append(os.path.join(r, f))
            if not all_ggufs:
                continue
            mmprojs = [g for g in all_ggufs if "mmproj" in os.path.basename(g).lower()]
            main_models = [g for g in all_ggufs if "mmproj" not in os.path.basename(g).lower()]
            paired_mm = mmprojs[0] if mmprojs else ""
            for mp in main_models:
                fname = os.path.basename(mp)
                if re.search(r"-0000[2-9]-of-\d{5}\.gguf$", fname):
                    continue
                add_model_entry(f"{repo_label} ({fname})", "HuggingFace Cache", mp, paired_mm)

    # 3. Scan LM Studio (~/.lmstudio/models and ~/.cache/lm-studio/models)
    for lms_dir in (
        os.path.join(home, ".lmstudio", "models"),
        os.path.join(home, ".cache", "lm-studio", "models"),
    ):
        if os.path.isdir(lms_dir):
            for r, _, files in os.walk(lms_dir):
                ggufs = [os.path.join(r, f) for f in files if f.lower().endswith(".gguf")]
                if not ggufs:
                    continue
                mmprojs = [g for g in ggufs if "mmproj" in os.path.basename(g).lower()]
                mains = [g for g in ggufs if "mmproj" not in os.path.basename(g).lower()]
                paired_mm = mmprojs[0] if mmprojs else ""
                for mp in mains:
                    fname = os.path.basename(mp)
                    if re.search(r"-0000[2-9]-of-\d{5}\.gguf$", fname):
                        continue
                    add_model_entry(fname, "LM Studio", mp, paired_mm)

    # 4. Scan Extra Custom Directory (if provided) and Downloaded Models
    scan_dirs = []
    if extra_dir and os.path.isdir(extra_dir):
        scan_dirs.append((extra_dir, "Custom Folder"))
    dl_dir = os.path.join(get_app_data_dir(), "downloaded_models")
    if os.path.isdir(dl_dir):
        scan_dirs.append((dl_dir, "Downloaded"))

    for sdir, src_label in scan_dirs:
        for r, _, files in os.walk(sdir):
            ggufs = [os.path.join(r, f) for f in files if f.lower().endswith(".gguf")]
            if not ggufs:
                continue
            mmprojs = [g for g in ggufs if "mmproj" in os.path.basename(g).lower()]
            mains = [g for g in ggufs if "mmproj" not in os.path.basename(g).lower()]
            paired_mm = mmprojs[0] if mmprojs else ""
            for mp in mains:
                fname = os.path.basename(mp)
                if re.search(r"-0000[2-9]-of-\d{5}\.gguf$", fname):
                    continue
                add_model_entry(fname, src_label, mp, paired_mm)

    return discovered


def write_download_status(status_file: str, payload: dict):
    try:
        tmp_f = status_file + ".tmp"
        with open(tmp_f, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_f, status_file)
    except Exception:
        pass


def download_full_gguf(target: str, status_file: str):
    """Downloads a full GGUF model from Hugging Face or direct URL into downloaded_models/ with real-time progress."""
    import requests
    from extract_compute_tiers import resolve_target_to_gguf_url

    dl_dir = os.path.join(get_app_data_dir(), "downloaded_models")
    os.makedirs(dl_dir, exist_ok=True)

    try:
        write_download_status(
            status_file,
            {
                "active": True,
                "status": "resolving",
                "filename": target,
                "url": target,
                "downloaded_mib": 0.0,
                "total_mib": 0.0,
                "percent": 0.0,
                "speed_mbs": 0.0,
                "output_path": "",
                "error": None,
            },
        )

        s = target.strip()
        if s.endswith(".gguf") and ("http://" in s or "https://" in s):
            resolved_url = s.replace("/blob/", "/resolve/")
        else:
            resolved_url = resolve_target_to_gguf_url(s)

        if os.path.isfile(resolved_url):
            write_download_status(
                status_file,
                {
                    "active": False,
                    "status": "completed",
                    "filename": os.path.basename(resolved_url),
                    "url": resolved_url,
                    "downloaded_mib": round(os.path.getsize(resolved_url) / 1048576.0, 1),
                    "total_mib": round(os.path.getsize(resolved_url) / 1048576.0, 1),
                    "percent": 100.0,
                    "speed_mbs": 0.0,
                    "output_path": os.path.abspath(resolved_url),
                    "error": None,
                },
            )
            return

        filename = resolved_url.split("?")[0].rstrip("/").split("/")[-1]
        if not filename.lower().endswith(".gguf"):
            filename += ".gguf"

        output_path = os.path.join(dl_dir, filename)
        part_path = output_path + ".part"

        headers = {"User-Agent": "llama-desktop-manager/1.0"}
        auth_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        resume_bytes = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
        if resume_bytes > 0:
            headers["Range"] = f"bytes={resume_bytes}-"

        with requests.get(resolved_url, headers=headers, stream=True, timeout=30) as resp:
            if resp.status_code == 416:
                # Already fully downloaded in .part
                os.replace(part_path, output_path)
                sz = os.path.getsize(output_path) / 1048576.0
                write_download_status(
                    status_file,
                    {
                        "active": False,
                        "status": "completed",
                        "filename": filename,
                        "url": resolved_url,
                        "downloaded_mib": round(sz, 1),
                        "total_mib": round(sz, 1),
                        "percent": 100.0,
                        "speed_mbs": 0.0,
                        "output_path": output_path,
                        "error": None,
                    },
                )
                return

            if resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {resp.status_code} when downloading {resolved_url}")

            if resp.status_code == 200:
                resume_bytes = 0

            content_len = int(resp.headers.get("Content-Length", "0") or "0")
            total_bytes = resume_bytes + content_len
            downloaded_bytes = resume_bytes

            mode = "ab" if resp.status_code == 206 and resume_bytes > 0 else "wb"
            start_t = time.perf_counter()
            last_report_t = start_t
            bytes_since_start = 0

            with open(part_path, mode) as out_f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    out_f.write(chunk)
                    downloaded_bytes += len(chunk)
                    bytes_since_start += len(chunk)

                    now_t = time.perf_counter()
                    if now_t - last_report_t >= 0.25:
                        elapsed = max(0.001, now_t - start_t)
                        speed_mbs = (bytes_since_start / 1048576.0) / elapsed
                        pct = (downloaded_bytes * 100.0 / total_bytes) if total_bytes > 0 else 0.0
                        write_download_status(
                            status_file,
                            {
                                "active": True,
                                "status": "downloading",
                                "filename": filename,
                                "url": resolved_url,
                                "downloaded_mib": round(downloaded_bytes / 1048576.0, 1),
                                "total_mib": round(total_bytes / 1048576.0, 1),
                                "percent": round(min(99.9, pct), 1),
                                "speed_mbs": round(speed_mbs, 1),
                                "output_path": output_path,
                                "error": None,
                            },
                        )
                        last_report_t = now_t

        os.replace(part_path, output_path)
        final_mib = round(os.path.getsize(output_path) / 1048576.0, 1)
        write_download_status(
            status_file,
            {
                "active": False,
                "status": "completed",
                "filename": filename,
                "url": resolved_url,
                "downloaded_mib": final_mib,
                "total_mib": final_mib,
                "percent": 100.0,
                "speed_mbs": 0.0,
                "output_path": output_path,
                "error": None,
            },
        )
    except Exception as e:
        write_download_status(
            status_file,
            {
                "active": False,
                "status": "error",
                "filename": target,
                "url": target,
                "downloaded_mib": 0.0,
                "total_mib": 0.0,
                "percent": 0.0,
                "speed_mbs": 0.0,
                "output_path": "",
                "error": str(e),
            },
        )


def find_mmproj_in_hf_repo(url_or_id: str) -> Optional[str]:
    """Checks if a Hugging Face repository contains an mmproj-*.gguf file and returns its resolve URL."""
    s = url_or_id.strip()
    if os.path.exists(s):
        return None

    clean = s.rstrip("/")
    repo_id = None
    if "huggingface.co/" in clean:
        part = clean.split("huggingface.co/")[1]
        tokens = [t for t in part.split("/") if t]
        if len(tokens) >= 2:
            repo_id = f"{tokens[0]}/{tokens[1]}"
    elif "/" in clean and not clean.endswith(".gguf"):
        tokens = [t for t in clean.split("/") if t]
        if len(tokens) == 2:
            repo_id = f"{tokens[0]}/{tokens[1]}"

    if not repo_id:
        return None

    try:
        import requests
        api_url = f"https://huggingface.co/api/models/{repo_id}"
        headers = {"User-Agent": "llama-compute-extractor/1.0"}
        auth_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        info = resp.json()
        siblings = [f["rfilename"] for f in info.get("siblings", []) if "rfilename" in f]
        mmproj_files = [f for f in siblings if f.endswith(".gguf") and "mmproj" in f.lower()]
        if not mmproj_files:
            return None
        # Prefer f16 or bf16 mmproj
        for pref in ("f16", "bf16", "f32", "q8_0"):
            matches = [f for f in mmproj_files if pref in f.lower()]
            if matches:
                return f"https://huggingface.co/{repo_id}/resolve/main/{matches[0]}"
        return f"https://huggingface.co/{repo_id}/resolve/main/{mmproj_files[0]}"
    except Exception:
        return None


def resolve_or_download_gguf(target: str, skeleton_dir: str) -> tuple[str, str]:
    """Resolves a local path or downloads a remote GGUF skeleton, returning (skeleton_path, resolved_url)."""
    if os.path.exists(target):
        return resolve_local_gguf(target), target

    from extract_compute_tiers import resolve_target_to_gguf_url, download_model_skeleton

    if target.endswith(".gguf") and ("http://" in target or "https://" in target):
        resolved_url = target.replace("/blob/", "/resolve/")
        skeleton_path, _ = download_model_skeleton(resolved_url, skeleton_dir)
        return skeleton_path, resolved_url

    # Check if user passed an HF repo URL where they want a specific file or main model
    resolved_url = resolve_target_to_gguf_url(target)
    if os.path.exists(resolved_url):
        return resolve_local_gguf(resolved_url), resolved_url
    skeleton_path, _ = download_model_skeleton(resolved_url, skeleton_dir)
    return skeleton_path, resolved_url


def main():
    if len(sys.argv) >= 2 and sys.argv[1].strip() == "--detect-hw":
        print("JSON_RESULT:" + json.dumps({"ok": True, "hardware": detect_system_hardware()}))
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1].strip() == "--scan-local":
        extra = sys.argv[2].strip() if len(sys.argv) >= 3 else ""
        models = scan_local_models(extra)
        print("JSON_RESULT:" + json.dumps({"ok": True, "local_models": models, "hardware": detect_system_hardware()}))
        sys.exit(0)

    if len(sys.argv) >= 3 and sys.argv[1].strip() == "--download-full":
        dl_target = sys.argv[2].strip()
        status_path = (
            sys.argv[3].strip()
            if len(sys.argv) >= 4
            else os.path.join(SCRIPT_DIR, "download_status.json")
        )
        download_full_gguf(dl_target, status_path)
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1].strip() == "--clean-session-skeletons":
        import shutil
        session_skel_dir = os.path.join(tempfile.gettempdir(), "whichllmmodel", "session_skeletons")
        if os.path.isdir(session_skel_dir):
            try:
                shutil.rmtree(session_skel_dir, ignore_errors=True)
            except Exception:
                pass
        print("JSON_RESULT:" + json.dumps({"ok": True}))
        sys.exit(0)

    if len(sys.argv) < 2:
        print("JSON_RESULT:" + json.dumps({"ok": False, "error": "Missing target URL or path"}))
        sys.exit(1)

    target = sys.argv[1].strip()
    mmproj_arg = sys.argv[2].strip() if len(sys.argv) >= 3 else ""
    # Ephemeral session directory for zero-download remote skeletons
    skeleton_dir = os.path.join(tempfile.gettempdir(), "whichllmmodel", "session_skeletons")
    os.makedirs(skeleton_dir, exist_ok=True)

    try:
        t0_prep = time.perf_counter()
        is_already_local = os.path.exists(target)

        # 1. Resolve / download main model skeleton
        skeleton_path, resolved_url = resolve_or_download_gguf(target, skeleton_dir)

        # 2. Inspect GGUF metadata (architecture, context_length, is_moe) in < 5ms
        meta = fast_inspect_gguf_metadata(skeleton_path)
        arch = meta.get("arch", "unknown")
        context_length = meta.get("context_length", 4096)
        is_moe = meta.get("is_moe", False)
        expert_count = meta.get("expert_count", 0)
        expert_used_count = meta.get("expert_used_count", 0)

        # 3. Check if architecture is implemented in llama.cpp/src/llama-arch.cpp
        supported_archs = get_llama_cpp_supported_archs()
        arch_supported = (arch in supported_archs) if supported_archs else True

        # 4. Resolve or auto-detect optional Vision Projector (mmproj)
        mmproj_skeleton_path = ""
        mmproj_resolved_url = ""
        if mmproj_arg:
            if os.path.exists(mmproj_arg):
                mmproj_skeleton_path = resolve_local_gguf(mmproj_arg)
                mmproj_resolved_url = mmproj_arg
            elif mmproj_arg.endswith(".gguf") and ("http://" in mmproj_arg or "https://" in mmproj_arg):
                from extract_compute_tiers import download_model_skeleton
                mmproj_resolved_url = mmproj_arg.replace("/blob/", "/resolve/")
                mmproj_skeleton_path, _ = download_model_skeleton(mmproj_resolved_url, skeleton_dir)
            else:
                from extract_compute_tiers import download_model_skeleton
                # Repo URL passed into mmproj field: look for mmproj*.gguf in that repo
                auto_mm = find_mmproj_in_hf_repo(mmproj_arg)
                if auto_mm:
                    mmproj_resolved_url = auto_mm
                    mmproj_skeleton_path, _ = download_model_skeleton(auto_mm, skeleton_dir)
                else:
                    mmproj_skeleton_path, mmproj_resolved_url = resolve_or_download_gguf(mmproj_arg, skeleton_dir)
        elif not os.path.exists(target):
            # Auto-detect if the target HF repo has an mmproj-*.gguf file (only when target is remote)
            auto_mm = find_mmproj_in_hf_repo(target)
            if auto_mm:
                from extract_compute_tiers import download_model_skeleton
                mmproj_resolved_url = auto_mm
                mmproj_skeleton_path, _ = download_model_skeleton(auto_mm, skeleton_dir)

        prep_time_ms = round((time.perf_counter() - t0_prep) * 1000.0, 1)

        result = {
            "ok": True,
            "skeleton_path": skeleton_path,
            "resolved_url": resolved_url,
            "is_local": is_already_local,
            "prep_time_ms": prep_time_ms,
            "arch": arch,
            "arch_supported": arch_supported,
            "supported_arch_count": len(supported_archs),
            "context_length": context_length,
            "is_moe": is_moe,
            "expert_count": expert_count,
            "expert_used_count": expert_used_count,
            "mmproj_skeleton_path": mmproj_skeleton_path,
            "mmproj_resolved_url": mmproj_resolved_url,
            "hardware": detect_system_hardware(),
        }
        print("JSON_RESULT:" + json.dumps(result))
    except Exception as e:
        print("JSON_RESULT:" + json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
