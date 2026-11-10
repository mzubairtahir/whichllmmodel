import os
import sys
import struct
import argparse
import re
import shutil
from pathlib import Path
import requests

# Add gguf-py to sys.path
sys.path.insert(0, "E:/llama.cpp study/llama.cpp/gguf-py")
import gguf
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

class ChunkReader:
    def __init__(self, url, token=None):
        self.url = url
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.buffer = bytearray()
        self.chunk_size = 4 * 1024 * 1024  # 4 MB chunks
        self.fetched = 0
        self.pos = 0
        self.session = requests.Session()

    def fetch_more(self, needed):
        to_fetch = max(needed, self.chunk_size)
        headers = {'Range': f'bytes={self.fetched}-{self.fetched + to_fetch - 1}'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        # Retry up to 5 times with exponential backoff on network hiccups or CDN timeouts
        last_err = None
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers=headers, stream=True, allow_redirects=True, timeout=(10, 60))
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP error: {r.status_code} for {self.url}")
                data = r.content
                self.buffer.extend(data)
                self.fetched += len(data)
                return
            except Exception as e:
                last_err = e
                if attempt < 4:
                    import time
                    wait_sec = 2 * (2 ** attempt)  # 2s, 4s, 8s, 16s
                    print(f"\n    [network retry {attempt+1}/5] {e} - waiting {wait_sec}s...", flush=True)
                    time.sleep(wait_sec)
        raise RuntimeError(f"Failed to fetch chunk after 5 attempts from {self.url}: {last_err}")

    def read_bytes(self, n):
        needed = (self.pos + n) - len(self.buffer)
        if needed > 0:
            self.fetch_more(needed)
        res = bytes(self.buffer[self.pos : self.pos + n])
        self.pos += n
        return res

    def read_str(self):
        slen = struct.unpack('<Q', self.read_bytes(8))[0]
        sbytes = self.read_bytes(slen)
        return sbytes.decode('utf-8', errors='ignore')

    def read_val(self, vtype):
        if vtype == GGUFValueType.UINT8:
            return struct.unpack('<B', self.read_bytes(1))[0]
        elif vtype == GGUFValueType.INT8:
            return struct.unpack('<b', self.read_bytes(1))[0]
        elif vtype == GGUFValueType.UINT16:
            return struct.unpack('<H', self.read_bytes(2))[0]
        elif vtype == GGUFValueType.INT16:
            return struct.unpack('<h', self.read_bytes(2))[0]
        elif vtype == GGUFValueType.UINT32:
            return struct.unpack('<I', self.read_bytes(4))[0]
        elif vtype == GGUFValueType.INT32:
            return struct.unpack('<i', self.read_bytes(4))[0]
        elif vtype == GGUFValueType.FLOAT32:
            return struct.unpack('<f', self.read_bytes(4))[0]
        elif vtype == GGUFValueType.BOOL:
            return struct.unpack('<?', self.read_bytes(1))[0]
        elif vtype == GGUFValueType.STRING:
            return self.read_str()
        elif vtype == GGUFValueType.UINT64:
            return struct.unpack('<Q', self.read_bytes(8))[0]
        elif vtype == GGUFValueType.INT64:
            return struct.unpack('<q', self.read_bytes(8))[0]
        elif vtype == GGUFValueType.FLOAT64:
            return struct.unpack('<d', self.read_bytes(8))[0]
        elif vtype == GGUFValueType.ARRAY:
            sub_type = GGUFValueType(struct.unpack('<I', self.read_bytes(4))[0])
            arr_len = struct.unpack('<Q', self.read_bytes(8))[0]
            arr = []
            for _ in range(arr_len):
                arr.append(self.read_val(sub_type))
            return (sub_type, arr)
        else:
            raise ValueError(f"Unknown GGUF type: {vtype}")

def download_single_shard(url, output_path, token=None):
    """Downloads remote GGUF header and creates a zero-weight sparse skeleton for a single shard."""
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        try:
            r_existing = gguf.GGUFReader(output_path)
            s_count = 1
            s_no = 0
            if "split.count" in r_existing.fields:
                s_count = int(r_existing.fields["split.count"].parts[0])
            if "split.no" in r_existing.fields:
                s_no = int(r_existing.fields["split.no"].parts[0])
            arch = "llama"
            if "general.architecture" in r_existing.fields:
                arch = str(bytes(r_existing.fields["general.architecture"].parts[0]).decode('utf-8', errors='ignore'))
            print(f"  [cache] Using existing shard on disk: {os.path.basename(output_path)}")
            return {
                "arch": arch,
                "split_count": s_count,
                "split_no": s_no,
                "tensors": len(r_existing.tensors),
                "header_bytes": 0,
                "virtual_bytes": os.path.getsize(output_path),
            }
        except Exception:
            pass  # If corrupted, re-download below

    if "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")

    reader = ChunkReader(url, token=token)
    magic = reader.read_bytes(4)
    if magic != b'GGUF':
        raise ValueError(f"Invalid GGUF magic in {url}!")
    version = struct.unpack('<I', reader.read_bytes(4))[0]
    tensor_count = struct.unpack('<Q', reader.read_bytes(8))[0]
    kv_count = struct.unpack('<Q', reader.read_bytes(8))[0]

    # Read all KVs
    kvs = []
    arch = "llama"
    split_count = 1
    split_no = 0

    for i in range(kv_count):
        k = reader.read_str()
        vtype = GGUFValueType(struct.unpack('<I', reader.read_bytes(4))[0])
        val = reader.read_val(vtype)
        kvs.append((k, vtype, val))
        if k == "general.architecture":
            arch = val
        elif k == "split.count":
            split_count = int(val)
        elif k == "split.no":
            split_no = int(val)

    # Read all Tensors
    tensors = []
    for i in range(tensor_count):
        tname = reader.read_str()
        n_dims = struct.unpack('<I', reader.read_bytes(4))[0]
        dims = [struct.unpack('<Q', reader.read_bytes(8))[0] for _ in range(n_dims)]
        raw_type = struct.unpack('<I', reader.read_bytes(4))[0]
        offset = struct.unpack('<Q', reader.read_bytes(8))[0]
        tensors.append((tname, dims, GGMLQuantizationType(raw_type)))

    # Now create GGUFWriter and write out skeleton
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(output_path, arch)

    # Add KVs
    for k, vtype, val in kvs:
        if k == "general.architecture":
            continue
        if vtype == GGUFValueType.ARRAY:
            sub_type, arr = val
            writer.add_key_value(k, arr, GGUFValueType.ARRAY, sub_type=sub_type)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(k, val)
        elif vtype == GGUFValueType.UINT32:
            writer.add_uint32(k, val)
        elif vtype == GGUFValueType.INT32:
            writer.add_int32(k, val)
        elif vtype == GGUFValueType.FLOAT32:
            writer.add_float32(k, val)
        elif vtype == GGUFValueType.UINT64:
            writer.add_uint64(k, val)
        elif vtype == GGUFValueType.INT64:
            writer.add_int64(k, val)
        elif vtype == GGUFValueType.BOOL:
            writer.add_bool(k, val)
        elif vtype == GGUFValueType.UINT8:
            writer.add_uint8(k, val)
        elif vtype == GGUFValueType.INT8:
            writer.add_int8(k, val)
        elif vtype == GGUFValueType.UINT16:
            writer.add_uint16(k, val)
        elif vtype == GGUFValueType.INT16:
            writer.add_int16(k, val)
        elif vtype == GGUFValueType.FLOAT64:
            writer.add_float64(k, val)

    import numpy as np
    dummy_f16 = np.zeros(1, dtype=np.float16)

    # Add tensors
    for tname, dims, qtype in tensors:
        writer.add_tensor(tname, dummy_f16, raw_shape=list(reversed(dims)), raw_dtype=qtype)

    for tname, dims, qtype in tensors:
        blck_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        n_elems = 1
        for d in dims:
            n_elems *= d
        real_nbytes = (n_elems // blck_size) * type_size
        writer.tensors[0][tname].nbytes = real_nbytes

    # Write header, kv, and ti
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Calculate total file size and seek to end to make file sparse/virtually sized
    data_start = writer.fout[0].tell()
    pad = GGUFWriter.ggml_pad(data_start, writer.data_alignment) - data_start
    if pad != 0:
        writer.fout[0].write(bytes([0] * pad))
        data_start += pad

    total_data_bytes = sum(GGUFWriter.ggml_pad(ti.nbytes, writer.data_alignment) for ti in writer.tensors[0].values())
    total_file_size = data_start + total_data_bytes

    # Set Windows NTFS sparse file flag so zero-byte data takes 0 actual disk blocks
    if sys.platform == "win32":
        try:
            import msvcrt, ctypes
            from ctypes import wintypes
            handle = msvcrt.get_osfhandle(writer.fout[0].fileno())
            dwBytesReturned = wintypes.DWORD()
            ctypes.windll.kernel32.DeviceIoControl(
                handle, 0x000900C4, None, 0, None, 0, ctypes.byref(dwBytesReturned), None
            )
        except Exception as e:
            print(f"Warning: could not set sparse flag: {e}")

    writer.fout[0].seek(total_file_size - 1)
    writer.fout[0].write(b'\0')
    writer.close()

    return {
        "arch": arch,
        "split_count": split_count,
        "split_no": split_no,
        "tensors": tensor_count,
        "header_bytes": data_start,
        "virtual_bytes": total_file_size,
    }

def download_model_skeleton(primary_url, target_dir, token=None):
    """
    Downloads skeleton for a GGUF model.
    If the model is multi-shard (-00001-of-0000N.gguf or split.count > 1),
    it creates a dedicated subdirectory and downloads headers for ALL shards,
    preserving exact filenames so llama.cpp's split loader succeeds.

    Returns:
        (primary_skeleton_path, cleanup_target_path)
    """
    if "/blob/" in primary_url:
        primary_url = primary_url.replace("/blob/", "/resolve/")

    fname = primary_url.split("/")[-1]
    shard_match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", fname)

    if shard_match:
        # Multi-shard file pattern
        total_splits = int(shard_match.group(2))
        prefix = fname[:shard_match.start()]
        # Create a dedicated directory for this model
        model_subfolder = os.path.join(target_dir, prefix)
        os.makedirs(model_subfolder, exist_ok=True)

        shard_1_path = os.path.join(model_subfolder, fname)
        print(f"[*] Downloading primary shard [1/{total_splits}]: {fname}...")
        meta = download_single_shard(primary_url, shard_1_path, token=token)

        # Note: Trust the filename shard count (shard_match.group(2)) because remote files
        # match the filename pattern (e.g. -00001-of-00005 means exactly 5 files exist).

        # Download sibling shards 2..total_splits
        for i in range(2, total_splits + 1):
            sibling_fname = re.sub(r"-00001-of-(\d{5})\.gguf$", f"-{i:05d}-of-\\1.gguf", fname)
            sibling_url = re.sub(r"-00001-of-(\d{5})\.gguf$", f"-{i:05d}-of-\\1.gguf", primary_url)
            sibling_path = os.path.join(model_subfolder, sibling_fname)
            print(f"[*] Downloading sibling shard [{i}/{total_splits}]: {sibling_fname}...")
            download_single_shard(sibling_url, sibling_path, token=token)

        return shard_1_path, model_subfolder
    else:
        # Single-file GGUF
        out_fname = fname
        if not out_fname.endswith(".gguf"):
            out_fname = f"{fname}.gguf"
        out_path = os.path.join(target_dir, out_fname)
        print(f"[*] Downloading single-file GGUF skeleton: {out_fname}...")
        meta = download_single_shard(primary_url, out_path, token=token)

        if meta["split_count"] > 1:
            # Metadata indicates sharding even without 00001-of-0000N in filename
            print(f"[!] Warning: split.count = {meta['split_count']} detected in non-standard filename.")

        return out_path, out_path

# Backward compatibility alias
def download_and_create_skeleton(url, output_path, token=None):
    return download_single_shard(url, output_path, token=token)

if __name__ == "__main__":
    _default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skeletons")
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf/resolve/main/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf")
    parser.add_argument("-o", "--output-dir", default=_default_dir)
    args = parser.parse_args()

    primary_path, cleanup_target = download_model_skeleton(args.url, args.output_dir)
    print(f"\nCompleted! Primary skeleton: {primary_path}")
