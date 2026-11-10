#!/usr/bin/env python3
"""Automated Compute Buffer Tier Pipeline for whichllmmodel.com.

Takes Hugging Face GGUF URL(s), creates zero-weight sparse skeleton GGUF(s),
detects whether the model is MoE or Dense, runs upstream llama-bench
(supporting both CUDA on Windows/Linux and Metal on macOS),
and outputs the exact 3-tier compute buffer JSON block.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure package root is discoverable
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from download_skeleton import download_and_create_skeleton, download_model_skeleton

DEFAULT_SKELETON_DIR = os.path.join(_script_dir, "skeletons")
DEFAULT_DUMP_DIR = os.path.join(_script_dir, "dumps")


def find_llama_bench_exe() -> str:
    """Locates llama-bench (or llama-bench.exe) across Windows, macOS, and Linux."""
    ext = ".exe" if sys.platform == "win32" else ""
    candidates = [
        os.path.join(_script_dir, "..", "llama.cpp", "build", "bin", "Release", f"llama-bench{ext}"),
        os.path.join(_script_dir, "..", "llama.cpp", "build", "bin", f"llama-bench{ext}"),
        os.path.abspath(f"E:/llama.cpp study/llama.cpp/build/bin/Release/llama-bench{ext}"),
        os.path.abspath(f"llama-bench{ext}"),
        f"llama-bench{ext}",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    which_bin = shutil.which(f"llama-bench{ext}")
    if which_bin:
        return which_bin
    raise FileNotFoundError(f"Could not find llama-bench{ext} in llama.cpp/build/bin/")


def read_targets_from_file(file_path: str) -> List[str]:
    """Reads repository URLs or IDs from a text or JSON file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    targets: List[str] = []
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        targets.append(item)
                    elif isinstance(item, dict) and "huggingfaceUrl" in item:
                        targets.append(item["huggingfaceUrl"])
                    elif isinstance(item, dict) and "id" in item:
                        targets.append(item["id"])
            elif isinstance(data, dict):
                targets.extend(list(data.keys()))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    targets.append(s)
    return targets


def resolve_true_context_window(target_str: str, fallback_ctx: int) -> int:
    """Finds advertised context_window from models_v2.json if available, else falls back to GGUF header."""
    candidate_paths = [
        os.path.abspath(os.path.join(_script_dir, "..", "..", "whichllmmodel", "which-model", "src", "data", "models_v2.json")),
        r"E:\whichllmmodel\which-model\src\data\models_v2.json",
        os.path.abspath(os.path.join(_script_dir, "models_v2.json")),
    ]
    models_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if models_path:
        try:
            with open(models_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            t_norm = target_str.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
            for m in data:
                m_name = m.get("name", "")
                m_hf = m.get("huggingfaceUrl", "")
                leaf = m_hf.split("/")[-1]
                n_name = m_name.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
                n_leaf = leaf.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
                if (n_leaf and n_leaf in t_norm) or (n_name and n_name in t_norm):
                    cw = m.get("context_window")
                    if cw and int(cw) > fallback_ctx:
                        return int(cw)
        except Exception:
            pass
    return fallback_ctx


def resolve_target_to_gguf_url(url_or_id: str, token: Optional[str] = None) -> str:
    """Resolves a raw model URL, repo ID, or direct GGUF link to a concrete GGUF download URL."""
    s = url_or_id.strip()

    # Local file path
    if os.path.isfile(s):
        return os.path.abspath(s)

    # Direct GGUF link
    if s.endswith(".gguf"):
        return s.replace("/blob/", "/resolve/")

    # Parse repo_id and optional subpath/branch from URL
    clean = s.rstrip("/")
    subpath = ""
    if "huggingface.co/" in clean:
        part = clean.split("huggingface.co/")[1]
        tokens = [t for t in part.split("/") if t]
        if len(tokens) >= 2:
            repo_id = f"{tokens[0]}/{tokens[1]}"
            rest = tokens[2:]
            rest = [t for t in rest if t not in ("raw", "tree", "blob", "main")]
            if rest:
                subpath = "/".join(rest)
        else:
            repo_id = part
    else:
        repo_id = clean

    # Query Hugging Face API to find files in repository
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    headers = {"User-Agent": "llama-compute-extractor/1.0"}
    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        import requests
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code == 401:
            raise RuntimeError(f"HTTP error: 401 Unauthorized for {api_url}. Gated repository requires HF_TOKEN.")
        resp.raise_for_status()
        info = resp.json()
        siblings = [f["rfilename"] for f in info.get("siblings", []) if "rfilename" in f]
    except Exception as e:
        raise RuntimeError(f"Failed to query Hugging Face API for {repo_id}: {e}")

    # Filter files ending with .gguf
    gguf_files = [f for f in siblings if f.endswith(".gguf")]
    if subpath:
        filtered = [f for f in gguf_files if f.startswith(subpath.strip("/")) or subpath.strip("/") in f]
        if filtered:
            gguf_files = filtered

    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in repository {repo_id} (subpath: '{subpath}')")

    # Priority ranking for standard quantization types
    priority_order = [
        "UD-Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
        "Q5_K_M", "Q5_0", "Q3_K_M", "Q3_K_L", "Q2_K",
        "Q6_K", "Q8_0", "BF16", "F16"
    ]

    selected_file: Optional[str] = None
    for p in priority_order:
        matches = [f for f in gguf_files if p.lower() in f.lower()]
        if matches:
            shard_1 = [m for m in matches if "-00001-of-" in m]
            selected_file = shard_1[0] if shard_1 else matches[0]
            break

    if not selected_file:
        shard_1 = [f for f in gguf_files if "-00001-of-" in f]
        selected_file = shard_1[0] if shard_1 else gguf_files[0]

    resolved_url = f"https://huggingface.co/{repo_id}/resolve/main/{selected_file}"
    print(f"[*] Auto-resolved '{url_or_id}' -> '{repo_id}/{selected_file}'")
    return resolved_url


def inspect_gguf_metadata(file_path: str) -> Dict[str, Any]:
    """Reads GGUF metadata directly using gguf-py to find architecture, context length, and MoE parameters."""
    sys.path.insert(0, os.path.join(_script_dir, "..", "llama.cpp", "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(file_path)

    def get_field_val(f):
        if not f or not getattr(f, "parts", None) or len(f.parts) == 0:
            return None
        val_part = f.parts[-1]
        vtype = f.types[0] if (getattr(f, "types", None) and len(f.types) > 0) else None
        if vtype == gguf.GGUFValueType.STRING:
            return bytes(val_part).decode("utf-8", errors="ignore").strip("\x00 \t\n\r")
        elif hasattr(val_part, "__getitem__") and len(val_part) > 0:
            return int(val_part[0])
        elif hasattr(val_part, "item"):
            return int(val_part.item())
        try:
            return int(val_part)
        except Exception:
            return val_part

    arch_field = reader.fields.get("general.architecture")
    if not arch_field:
        raise ValueError(f"Could not find general.architecture in {file_path}")
    arch = get_field_val(arch_field)
    if not arch or not isinstance(arch, str):
        arch = bytes(arch_field.parts[-1]).decode("utf-8", errors="ignore").strip("\x00 \t\n\r")

    # Context length
    ctx_field = reader.fields.get(f"{arch}.context_length") or reader.fields.get("general.context_length")
    raw_ctx = get_field_val(ctx_field)
    context_length = int(raw_ctx) if raw_ctx is not None else 4096

    # MoE detection
    is_moe = False
    expert_count = 0
    for k, f in reader.fields.items():
        if k.endswith(".expert_count") or k.endswith(".leading_dense_block_count"):
            val = get_field_val(f)
            if val is not None and int(val) > 0:
                is_moe = True
                expert_count = int(val)
                break

    if not is_moe:
        for t in reader.tensors:
            if any(exp_term in t.name for exp_term in ("exps", "expert", "ffn_gate_exp", "ffn_down_exp", "ffn_up_exp")):
                is_moe = True
                break

    return {
        "arch": arch,
        "context_length": context_length,
        "is_moe": is_moe,
        "expert_count": expert_count,
    }


def run_bench_for_ctx(
    bench_exe: str,
    model_path: str,
    ctx: int,
    is_moe: bool = False,
    ubatch: int = 512,
    is_mac: bool = False,
    dump_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Runs llama-bench for a specific context size to measure full and partial compute buffers on CUDA or Metal."""
    gpu_dev = "Metal0" if is_mac else "CUDA0"

    # 1. Full GPU Offload
    cmd_full = [
        bench_exe,
        "-m", model_path,
        "-v", "-r", "0",
        "--no-warmup",
        "-p", str(ctx),
        "-n", "0",
        "-ub", str(ubatch),
        "-b", str(ubatch),
        "-dev", gpu_dev,
        "-fa", "on",
        "--repack", "0",
    ]
    res_full = subprocess.run(cmd_full, capture_output=True, text=True, errors="replace")
    out_full = res_full.stderr + res_full.stdout

    # Universal regex matching CUDA0, Metal0, Metal, ROCm0, Vulkan0
    m_gpu_full = re.search(r"(?:CUDA0|Metal0|Metal|ROCm0|Vulkan0)\s+compute buffer size\s*=\s*([\d\.]+)\s*MiB", out_full)
    m_host_full = re.search(r"(?:CUDA_Host|CPU)\s+compute buffer size\s*=\s*([\d\.]+)\s*MiB", out_full)

    gpu_metal_or_cuda_full = round(float(m_gpu_full.group(1))) if m_gpu_full else 0
    host_cpu_full = round(float(m_host_full.group(1))) if m_host_full else 0

    # 2. Partial Offload (MoE uses ncmoe=10; Dense uses ngl=10)
    cmd_part = [
        bench_exe,
        "-m", model_path,
        "-v", "-r", "0",
        "--no-warmup",
        "-p", str(ctx),
        "-n", "0",
        "-ub", str(ubatch),
        "-b", str(ubatch),
        "-dev", gpu_dev,
        "-fa", "on",
        "--repack", "0",
    ]

    if is_moe:
        cmd_part.extend(["-ncmoe", "10"])
    else:
        cmd_part.extend(["-ngl", "10"])

    res_part = subprocess.run(cmd_part, capture_output=True, text=True, errors="replace")
    out_part = res_part.stderr + res_part.stdout

    m_gpu_part = re.search(r"(?:CUDA0|Metal0|Metal|ROCm0|Vulkan0)\s+compute buffer size\s*=\s*([\d\.]+)\s*MiB", out_part)
    m_host_part = re.search(r"(?:CUDA_Host|CPU)\s+compute buffer size\s*=\s*([\d\.]+)\s*MiB", out_part)

    gpu_metal_or_cuda_part = round(float(m_gpu_part.group(1))) if m_gpu_part else gpu_metal_or_cuda_full
    host_cpu_part = round(float(m_host_part.group(1))) if m_host_part else host_cpu_full

    if is_mac:
        # On Apple Silicon Unified Memory, both Metal and CPU buffers reside in the same
        # physical RAM pool. Pre-sum them so the single unified bar is byte-accurate.
        gpu_vram_full = gpu_metal_or_cuda_full + host_cpu_full
        gpu_vram_partial = gpu_metal_or_cuda_part + host_cpu_part
        cpu_ram = 0
    else:
        # On PC (CUDA/ROCm), discrete GPU VRAM and System RAM are separate physical pools
        gpu_vram_full = gpu_metal_or_cuda_full
        gpu_vram_partial = gpu_metal_or_cuda_part
        cpu_ram = host_cpu_full

    # Optionally persist raw logs
    if dump_dir:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            mname = Path(model_path).stem.replace("_skeleton", "")
            prefix = "mac_" if is_mac else ""
            with open(os.path.join(dump_dir, f"{prefix}bench_{mname}_ctx{ctx}_full.txt"), "w", encoding="utf-8") as f:
                f.write(out_full)
            with open(os.path.join(dump_dir, f"{prefix}bench_{mname}_ctx{ctx}_part.txt"), "w", encoding="utf-8") as f:
                f.write(out_part)
        except Exception as e:
            print(f"    [!] Warning: could not write bench dump: {e}")

    return {
        "gpu_vram_full": gpu_vram_full,
        "gpu_vram_partial": gpu_vram_partial,
        "cpu_ram": cpu_ram,
    }


def generate_tiers_for_model(
    url_or_path: str,
    output_skeleton_dir: Optional[str] = None,
    dump_dir: Optional[str] = None,
    keep_skeleton: bool = False,
    ubatch: int = 512,
    is_mac: bool = False,
) -> Dict[str, Any]:
    """Downloads skeleton (if URL), runs llama-bench, and returns compute buffer tiers."""
    bench_exe = find_llama_bench_exe()

    # Step 1: Ensure URL is resolved
    resolved_url = url_or_path
    if not os.path.isfile(url_or_path) and not url_or_path.endswith(".gguf"):
        resolved_url = resolve_target_to_gguf_url(url_or_path)

    # Step 2: Ensure skeleton GGUF exists
    is_remote = resolved_url.startswith("http://") or resolved_url.startswith("https://")
    skeleton_path = ""
    cleanup_target = ""
    if is_remote:
        target_dir = output_skeleton_dir or DEFAULT_SKELETON_DIR
        os.makedirs(target_dir, exist_ok=True)
        print(f"[*] Downloading remote GGUF skeleton(s) for: {resolved_url}")
        skeleton_path, cleanup_target = download_model_skeleton(resolved_url, target_dir)
    else:
        skeleton_path = os.path.abspath(resolved_url)
        cleanup_target = skeleton_path
        if not os.path.isfile(skeleton_path):
            raise FileNotFoundError(f"Local file not found: {skeleton_path}")

    try:
        # Step 3: Read metadata and resolve true advertised context window from models_v2.json
        meta = inspect_gguf_metadata(skeleton_path)
        arch = meta["arch"]
        is_moe = meta["is_moe"]
        header_ctx = meta["context_length"]
        true_ctx = resolve_true_context_window(url_or_path, header_ctx)

        device_name = "Metal (Apple Silicon)" if is_mac else "CUDA (NVIDIA)"
        print(f"[*] Model Architecture: {arch} | Device: {device_name} | Type: {'MoE (-ncmoe)' if is_moe else 'Dense (-ngl)'}")
        print(f"[*] Header Context: {header_ctx} tokens | True Runtime Context: {true_ctx} tokens")

        # Define context tiers (8K, 64K, max_ctx)
        tier_3_key = str(round(true_ctx / 1000) if true_ctx > 65536 else round(true_ctx / 1024))
        tier_ctxs = [
            ("8", min(8192, true_ctx)),
            ("64", min(65536, true_ctx)),
        ]
        if true_ctx > 65536 or tier_3_key not in ("8", "64"):
            tier_ctxs.append((tier_3_key, true_ctx))

        tiers_dict: Dict[str, Dict[str, int]] = {}

        for key, ctx in tier_ctxs:
            print(f"  [>] Benchmarking tier '{key}' at ctx={ctx} (ubatch={ubatch})...", end=" ", flush=True)
            tier_data = run_bench_for_ctx(
                bench_exe,
                skeleton_path,
                ctx=ctx,
                is_moe=is_moe,
                ubatch=ubatch,
                is_mac=is_mac,
                dump_dir=dump_dir,
            )
            tiers_dict[key] = tier_data
            print(f"[OK] Full: {tier_data['gpu_vram_full']}M, Partial: {tier_data['gpu_vram_partial']}M, CPU: {tier_data['cpu_ram']}M")

        res_obj = {
            "source_target": url_or_path,
            "resolved_gguf_url": resolved_url,
            "architecture": arch,
            "is_moe": is_moe,
            "max_context": true_ctx,
            "device": "metal" if is_mac else "cuda",
            "compute_buffer_tiers_mib": tiers_dict,
        }

        # Also store explicitly under mac keys if in Mac mode
        if is_mac:
            mac_key = "compute_buffer_ncpumoe_tiers_mib_mac" if is_moe else "compute_buffer_tiers_mib_mac"
            res_obj[mac_key] = tiers_dict

        return res_obj
    finally:
        # Step 4: Cleanup temporary skeleton
        if not keep_skeleton and is_remote and cleanup_target and os.path.exists(cleanup_target):
            try:
                if os.path.isdir(cleanup_target):
                    shutil.rmtree(cleanup_target, ignore_errors=True)
                elif os.path.isfile(cleanup_target):
                    os.remove(cleanup_target)
                print(f"[*] Cleaned up temporary skeleton: {cleanup_target}")
            except Exception as e:
                print(f"    [!] Warning: could not delete skeleton: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated GGUF Skeleton & Compute Buffer Tier Generator (CUDA + Apple Silicon Metal)"
    )
    parser.add_argument(
        "urls_or_paths",
        nargs="*",
        help="Hugging Face GGUF URLs, repo URLs/IDs, text files (e.g. repos.txt), or local skeleton GGUFs",
    )
    parser.add_argument("-f", "--file", help="Path to text or JSON file containing model URLs/IDs (e.g. repos.txt)")
    parser.add_argument("--skeleton-dir", default=DEFAULT_SKELETON_DIR, help="Directory to store skeleton GGUF files")
    parser.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR, help="Directory to store raw bench logs and alloc dumps")
    parser.add_argument("-ub", "--ubatch", type=int, default=512, help="Micro-batch size (default: 512)")
    parser.add_argument("--keep-skeleton", action="store_true", help="Keep downloaded skeleton GGUF on disk")
    parser.add_argument(
        "--mac",
        action="store_true",
        default=(sys.platform == "darwin"),
        help="Run benchmarks using Apple Silicon Metal backend (default: auto-detected on macOS)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to save/load incremental progress JSON (default: results_mac.json if --mac else results.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-benchmarking targets even if they exist in output JSON",
    )

    args = parser.parse_args()

    targets: List[str] = []
    if args.file:
        targets.extend(read_targets_from_file(args.file))

    for item in args.urls_or_paths:
        if os.path.isfile(item) and (item.endswith(".txt") or item.endswith(".json")):
            targets.extend(read_targets_from_file(item))
        else:
            targets.append(item)

    if not targets:
        parser.print_help()
        sys.exit(1)

    output_file = args.output
    if not output_file:
        output_file = "results_mac.json" if args.mac else "results.json"

    results: Dict[str, Any] = {}

    if os.path.isfile(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            completed_count = sum(1 for v in results.values() if isinstance(v, dict) and "compute_buffer_tiers_mib" in v)
            print(f"[*] Loaded existing progress from {output_file} ({completed_count} model(s) already completed)")
        except Exception as e:
            print(f"    [!] Warning: could not parse existing {output_file}: {e}")

    total_targets = len(targets)
    for idx, target in enumerate(targets, 1):
        if not args.force and target in results and isinstance(results[target], dict) and "compute_buffer_tiers_mib" in results[target]:
            existing_tiers = results[target].get("compute_buffer_tiers_mib", {})
            if any(isinstance(t, dict) and t.get("gpu_vram_full", 0) > 0 for t in existing_tiers.values()):
                print(f"\n[{idx}/{total_targets}] Skipping already-completed target: {target}")
                continue

        print(f"\n=======================================================")
        print(f"[{idx}/{total_targets}] Processing ({'Metal/Mac' if args.mac else 'CUDA'}): {target}")
        print(f"=======================================================")
        try:
            res = generate_tiers_for_model(
                target,
                output_skeleton_dir=args.skeleton_dir,
                dump_dir=args.dump_dir,
                keep_skeleton=args.keep_skeleton,
                ubatch=args.ubatch,
                is_mac=args.mac,
            )
            results[target] = res

            target_display_key = "compute_buffer_ncpumoe_tiers_mib_mac" if (args.mac and res["is_moe"]) else (
                "compute_buffer_tiers_mib_mac" if args.mac else (
                    "compute_buffer_ncpumoe_tiers_mib" if res["is_moe"] else "compute_buffer_tiers_mib"
                )
            )
            print(f"\n[+] Ready-to-Paste JSON for '{target}':")
            print(json.dumps({target_display_key: res["compute_buffer_tiers_mib"]}, indent=2))
        except Exception as e:
            print(f"[!] Error processing {target}: {e}")
            results[target] = {"error": str(e)}

        try:
            temp_out = f"{output_file}.tmp"
            with open(temp_out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            os.replace(temp_out, output_file)
            print(f"[*] Progress saved on the fly to {output_file} ({idx}/{total_targets} processed)")
        except Exception as e:
            print(f"    [!] Warning: could not save incremental progress: {e}")

    print(f"\n=======================================================")
    print(f"ALL MODELS COMPLETED ({'Apple Metal' if args.mac else 'CUDA'})")
    print(f"=======================================================")
    summary = {t: r.get("compute_buffer_tiers_mib") for t, r in results.items() if isinstance(r, dict) and "compute_buffer_tiers_mib" in r}
    print(json.dumps(summary, indent=2))

    summary_file = os.path.splitext(output_file)[0] + "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[*] Full results: {output_file}")
    print(f"[*] Clean summary for models_v2.json: {summary_file}")


if __name__ == "__main__":
    main()
