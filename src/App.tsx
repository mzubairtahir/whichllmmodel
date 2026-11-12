import { useState, useEffect, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Gauge,
  Library,
  Sun,
  Moon,
  Star,
  Download,
  Trash2,
  RefreshCw,
  X,
  Plus,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Search,
  Eye,
} from "lucide-react";

const DEFAULT_CATEGORIES = ["General", "Coding", "Reasoning", "Vision", "Fast / Edge"];

interface LocalModelEntry {
  name: string;
  source: string;
  model_path: string;
  mmproj_path: string;
  arch: string;
  context_length: number;
  is_moe: boolean;
  expert_count: number;
  expert_used_count: number;
  model_logical_mib: number;
  model_disk_mib: number;
  mmproj_logical_mib: number;
  mmproj_disk_mib: number;
  is_sparse_skeleton: boolean;
  bookmarked: boolean;
  category: string;
}

interface DownloadStatus {
  active: boolean;
  status: string;
  filename: string;
  url: string;
  downloaded_mib: number;
  total_mib: number;
  percent: number;
  speed_mbs: number;
  output_path: string;
  error?: string | null;
}

interface HardwareInfo {
  cpu_name: string;
  cpu_threads: number;
  cpu_mhz: number;
  ram_total_mib: number;
  ram_free_mib: number;
  ram_used_mib: number;
  ram_load_pct: number;
  ram_installed_gb: number;
  ram_channels: number;
  ram_speed_mts: number;
  ram_bandwidth_gbs: number;
  measured_ram_gbs: number;
  effective_ram_gbs: number;
  gpu_name: string;
  gpu_is_discrete: boolean;
  gpu_vram_total_mib: number;
  gpu_vram_free_mib: number;
  gpu_shared_ram_mib: number;
  gpu_bandwidth_gbs: number;
  measured_gpu_gbs: number;
  effective_gpu_gbs: number;
}

interface BenchResult {
  model_name: string;
  skeleton_path: string;
  resolved_url: string;
  arch: string;
  arch_supported: boolean;
  supported_arch_count: number;
  native_context_length: number;
  is_moe: boolean;
  expert_count: number;
  expert_used_count: number;
  has_mmproj: boolean;
  mmproj_name: string;
  mmproj_projector_type: string;
  vram_vision_model_mib: number;
  ram_vision_model_mib: number;
  vram_vision_compute_mib: number;
  ram_vision_compute_mib: number;
  context_size: number;
  ngl: number;
  ncmoe?: number;
  device: string;
  vram_model_mib: number;
  ram_model_mib: number;
  vram_kv_mib: number;
  ram_kv_mib: number;
  vram_rs_mib: number;
  ram_rs_mib: number;
  vram_compute_mib: number;
  ram_compute_mib: number;
  output_buffer_mib: number;
  total_vram_mib: number;
  total_ram_mib: number;
  elapsed_ms: number;
  prep_time_ms: number;
  bench_time_ms: number;
  is_local_model: boolean;
  raw_lines: string[];
  hardware?: HardwareInfo;
  kv_quant?: string;
  no_kv_offload?: boolean;
}

const SPEED_UNCERTAINTY = 0.15;
const CONTEXT_PRESETS = [4096, 8192, 32768, 65536, 131072, 262144, 524288, 1048576];
const SPEED_TIERS = [
  { label: "Slow", min: 0, max: 15, cls: "text-bad" },
  { label: "Usable", min: 15, max: 35, cls: "text-warn" },
  { label: "Fast", min: 35, max: 80, cls: "text-ok" },
  { label: "Very fast", min: 80, max: Infinity, cls: "text-accent" },
];

const fmtMib = (v: number) =>
  v === 0
    ? "—"
    : v >= 1024
    ? `${(v / 1024).toFixed(2)} GiB`
    : `${v.toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 })} MiB`;
const fmtGib = (mib: number) => `${(mib / 1024).toFixed(2)} GiB`;
const fmtSize = (mib: number) => (mib >= 1024 ? fmtGib(mib) : `${mib.toFixed(0)} MiB`);

function sourceLabel(s: string) {
  return s === "Simulator Skeleton" ? "Checked (not downloaded)" : s;
}

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    localStorage.getItem("wlm_theme") === "light" ? "light" : "dark"
  );
  const [page, setPage] = useState<"check" | "models">("check");

  // Check-memory state
  const [urlOrPath, setUrlOrPath] = useState("");
  const [mmprojUrlOrPath, setMmprojUrlOrPath] = useState("");
  const [showVision, setShowVision] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [contextSize, setContextSize] = useState(8192);
  const [ngl, setNgl] = useState(99);
  const [ncmoe, setNcmoe] = useState(0);
  const [device, setDevice] = useState("CUDA0");
  const [kvQuant, setKvQuant] = useState<string>("q8_0");
  const [noKvOffload, setNoKvOffload] = useState<boolean>(false);
  const [vramBudgetGb, setVramBudgetGb] = useState(16);
  const [vramBwGbs, setVramBwGbs] = useState(26.2);
  const [ramBwGbs, setRamBwGbs] = useState(26.2);
  const [customRig, setCustomRig] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [cmdMode, setCmdMode] = useState<"cli" | "server">("cli");

  // System
  const [hwInfo, setHwInfo] = useState<HardwareInfo | null>(null);
  const [refreshingHw, setRefreshingHw] = useState(false);

  // Library state
  const [localModels, setLocalModels] = useState<LocalModelEntry[]>([]);
  const [scanning, setScanning] = useState(false);
  const [categories, setCategories] = useState<string[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("wlm_categories") || "null");
      if (Array.isArray(saved) && saved.length) return saved;
    } catch {
      // ignore
    }
    return DEFAULT_CATEGORIES;
  });
  const [activeCategory, setActiveCategory] = useState<string>("All");
  const [statusFilter, setStatusFilter] = useState<"installed" | "all">("installed");
  const [query, setQuery] = useState("");
  const [newCategory, setNewCategory] = useState("");
  const [addingCategory, setAddingCategory] = useState(false);
  const [customScanDir, setCustomScanDir] = useState("");
  const [downloadInput, setDownloadInput] = useState("");
  const [showDownload, setShowDownload] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("wlm_theme", next);
  };

  const allCategories = useMemo(() => {
    const extra = localModels.map((m) => m.category).filter((c) => c && !categories.includes(c));
    return [...categories, ...Array.from(new Set(extra))];
  }, [categories, localModels]);

  const persistCategories = (list: string[]) => {
    setCategories(list);
    localStorage.setItem("wlm_categories", JSON.stringify(list));
  };

  const syncBandwidth = (hw: HardwareInfo, override = !customRig) => {
    const mRam = hw.measured_ram_gbs || hw.ram_bandwidth_gbs || 26.2;
    const mGpu = hw.measured_gpu_gbs || mRam;
    setRamBwGbs(mRam);
    if (override) setVramBwGbs(mGpu);
  };

  const refreshHardware = async () => {
    setRefreshingHw(true);
    try {
      const hw = await invoke<HardwareInfo>("detect_hardware");
      setHwInfo(hw);
      syncBandwidth(hw, true);
      setCustomRig(false);
    } catch {
      // ignore
    } finally {
      setRefreshingHw(false);
    }
  };

  const runScan = async (extra = customScanDir) => {
    setScanning(true);
    try {
      const models = await invoke<LocalModelEntry[]>("scan_local_models", {
        extraDir: extra.trim() ? extra.trim() : null,
      });
      setLocalModels(models);
    } catch {
      // ignore
    } finally {
      setScanning(false);
    }
  };

  const updateMeta = async (modelPath: string, bookmarked: boolean, category: string) => {
    setLocalModels((prev) =>
      prev.map((m) => (m.model_path.toLowerCase() === modelPath.toLowerCase() ? { ...m, bookmarked, category } : m))
    );
    try {
      await invoke("update_model_meta", { modelPath, bookmarked, category });
    } catch {
      // ignore
    }
  };

  const deleteModel = async (modelPath: string) => {
    setDeleting(modelPath);
    try {
      await invoke<string>("delete_local_model", { modelPath });
      setConfirmDelete(null);
      await runScan();
    } catch (err: any) {
      setError(typeof err === "string" ? err : err?.message || "Failed to delete model");
    } finally {
      setDeleting(null);
    }
  };

  const removeCategory = (cat: string) => {
    localModels
      .filter((m) => m.category === cat)
      .forEach((m) => updateMeta(m.model_path, m.bookmarked, "General"));
    persistCategories(categories.filter((c) => c !== cat));
    if (activeCategory === cat) setActiveCategory("All");
  };

  const addCategory = () => {
    const name = newCategory.trim();
    if (name && !allCategories.some((c) => c.toLowerCase() === name.toLowerCase())) {
      persistCategories([...categories, name]);
    }
    setNewCategory("");
    setAddingCategory(false);
  };

  const startDownload = async (target: string) => {
    const clean = target.trim();
    if (!clean) return;
    try {
      await invoke<string>("start_model_download", { urlOrRepo: clean });
      setDownloadStatus(await invoke<DownloadStatus>("get_download_status"));
    } catch (err: any) {
      setError(typeof err === "string" ? err : err?.message || "Failed to start download");
    }
  };

  const cancelDownload = async () => {
    try {
      await invoke("cancel_model_download");
      setDownloadStatus(await invoke<DownloadStatus>("get_download_status"));
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (!downloadStatus?.active) return;
    const timer = setInterval(async () => {
      try {
        const st = await invoke<DownloadStatus>("get_download_status");
        setDownloadStatus(st);
        if (!st.active && st.status === "completed") runScan();
      } catch {
        // ignore
      }
    }, 500);
    return () => clearInterval(timer);
  }, [downloadStatus?.active]);

  const runCheck = async (
    target = urlOrPath,
    mmproj = mmprojUrlOrPath,
    ctx = contextSize,
    layers = ngl,
    moe = ncmoe,
    dev = device,
    quant = kvQuant,
    noOffload = noKvOffload
  ) => {
    if (!target.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await invoke<BenchResult>("analyze_model", {
        urlOrPath: target.trim(),
        mmprojUrlOrPath: mmproj.trim() ? mmproj.trim() : null,
        contextSize: ctx,
        ngl: layers,
        ncmoe: moe,
        device: dev,
        kvQuant: quant,
        noKvOffload: noOffload,
      });
      setResult(res);
      if (res.hardware) {
        setHwInfo(res.hardware);
        syncBandwidth(res.hardware);
      }
      const nativeCtx = res.native_context_length;
      if (nativeCtx > 0 && ctx > nativeCtx) {
        // Requested context exceeds what this model supports: re-check at the model's real maximum.
        setContextSize(nativeCtx);
        await runCheck(target, mmproj, nativeCtx, layers, moe, dev, quant, noOffload);
        return;
      }
      setContextSize(ctx);
      runScan();
    } catch (err: any) {
      setError(typeof err === "string" ? err : err?.message || "Check failed");
    } finally {
      setLoading(false);
    }
  };

  const useMyHardware = () => {
    if (!hwInfo) return;
    setCustomRig(false);
    syncBandwidth(hwInfo, true);
    const discrete = hwInfo.gpu_is_discrete && hwInfo.gpu_vram_total_mib >= 2048;
    const layers = discrete ? 99 : 0;
    setVramBudgetGb(Math.max(1, Math.round((hwInfo.gpu_vram_total_mib || 1024) / 1024)));
    setNgl(layers);
    runCheck(urlOrPath, mmprojUrlOrPath, contextSize, layers, ncmoe, device, kvQuant, noKvOffload);
  };

  const openInCheck = (m: LocalModelEntry) => {
    setUrlOrPath(m.model_path);
    setMmprojUrlOrPath(m.mmproj_path || "");
    setShowVision(Boolean(m.mmproj_path));
    setPage("check");
    runCheck(m.model_path, m.mmproj_path || "", contextSize, ngl, ncmoe, device, kvQuant, noKvOffload);
  };

  useEffect(() => {
    refreshHardware();
    runScan("");
  }, []);

  // Speed prediction: measured bandwidth / active bytes per token (no efficiency multiplier)
  const speed = useMemo(() => {
    if (!result) return null;
    let activeRatio = 1.0;
    if (result.is_moe && result.expert_count > 0) {
      const k = result.expert_used_count > 0 ? result.expert_used_count : 4;
      activeRatio = Math.min(1.0, 0.1 + 0.9 * (k / result.expert_count));
    }
    const vramGb = (result.vram_model_mib / 1024) * activeRatio;
    const ramGb = (result.ram_model_mib / 1024) * activeRatio;
    const gb = vramGb + ramGb;
    if (gb <= 0) return null;
    const gpuBw = result.ngl === 0 ? ramBwGbs : vramBwGbs;
    const sec = (vramGb > 0 ? vramGb / gpuBw : 0) + (ramGb > 0 ? ramGb / ramBwGbs : 0);
    const ts = sec > 0 ? Math.round((1 / sec) * 10) / 10 : 0;
    const tier = SPEED_TIERS.find((t) => ts >= t.min && ts < t.max) || SPEED_TIERS[0];
    return {
      ts,
      min: Math.round(ts * (1 - SPEED_UNCERTAINTY) * 10) / 10,
      max: Math.round(ts * (1 + SPEED_UNCERTAINTY) * 10) / 10,
      gb: Math.round(gb * 1000) / 1000,
      bw: vramGb >= ramGb ? gpuBw : ramBwGbs,
      tier,
    };
  }, [result, vramBwGbs, ramBwGbs]);

  const vramBudgetMib = vramBudgetGb * 1024;
  const contextOptions = useMemo(() => {
    const native = result?.native_context_length ?? 0;
    if (!native) return CONTEXT_PRESETS.filter((v) => v <= 262144);
    const opts = CONTEXT_PRESETS.filter((v) => v <= native);
    if (!opts.includes(native)) opts.push(native);
    return opts.sort((a, b) => a - b);
  }, [result]);
  const gpuFits = result ? result.total_vram_mib <= vramBudgetMib : true;
  const ramNeed = result ? result.total_ram_mib + (result.ngl === 0 ? result.total_vram_mib : 0) : 0;
  const ramFree = hwInfo?.ram_free_mib ?? 0;
  const ramTotal = hwInfo?.ram_total_mib ?? 0;
  const ramState: "ok" | "tight" | "no" = !hwInfo ? "ok" : ramNeed <= ramFree ? "ok" : ramNeed <= ramTotal ? "tight" : "no";

  const activeEntry = result
    ? localModels.find((m) => m.model_path.toLowerCase() === result.skeleton_path.toLowerCase())
    : undefined;

  const isRemote =
    urlOrPath.trim().startsWith("http") || (urlOrPath.includes("/") && !urlOrPath.includes("\\") && !urlOrPath.includes(":"));

  const activeKvQuant = result?.kv_quant || kvQuant;
  const activeNoKvOffload = result?.no_kv_offload ?? noKvOffload;
  const modelFile = result
    ? result.model_name.toLowerCase().endsWith(".gguf")
      ? result.model_name
      : `${result.model_name}.gguf`
    : "";
  const mmprojFile = result?.has_mmproj
    ? result.mmproj_name.toLowerCase().endsWith(".gguf")
      ? result.mmproj_name
      : `${result.mmproj_name}.gguf`
    : "";

  const activeNgl = result?.ngl ?? ngl;
  const activeNcmoe = result?.ncmoe ?? ncmoe;
  const activeCtx = result?.context_size ?? contextSize;

  const cliCommand = result
    ? `llama-cli -m "${modelFile}"${mmprojFile ? ` --mmproj "${mmprojFile}"` : ""} -ngl ${activeNgl}${
        activeNcmoe > 0 ? ` -ncmoe ${activeNcmoe}` : ""
      } -c ${activeCtx} -ub 512 -b 512 -fa on${
        activeKvQuant && activeKvQuant !== "f16" ? ` -ctk ${activeKvQuant} -ctv ${activeKvQuant}` : ""
      }${activeNoKvOffload ? " -nkvo 1" : ""}`
    : "";

  const serverCommand = result
    ? `llama-server -m "${modelFile}"${mmprojFile ? ` --mmproj "${mmprojFile}"` : ""} -ngl ${activeNgl}${
        activeNcmoe > 0 ? ` -ncmoe ${activeNcmoe}` : ""
      } -c ${activeCtx} -ub 512 -b 512 -fa on${
        activeKvQuant && activeKvQuant !== "f16" ? ` -ctk ${activeKvQuant} -ctv ${activeKvQuant}` : ""
      }${activeNoKvOffload ? " -nkvo 1" : ""} --port 8080`
    : "";

  const displayedCommand = cmdMode === "server" ? serverCommand : cliCommand;

  const copyCli = () => {
    navigator.clipboard.writeText(displayedCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const installed = localModels.filter((m) => !m.is_sparse_skeleton);
  const remote = localModels.filter((m) => m.is_sparse_skeleton);
  const totalDisk = localModels.reduce((a, m) => a + m.model_disk_mib + m.mmproj_disk_mib, 0);

  const visibleModels = localModels.filter((m) => {
    if (activeCategory === "Bookmarked") {
      if (!m.bookmarked) return false;
    } else if (activeCategory !== "All" && (m.category || "General") !== activeCategory) {
      return false;
    }
    if (query.trim() && !m.name.toLowerCase().includes(query.trim().toLowerCase())) return false;
    return true;
  });

  const countFor = (cat: string) => {
    if (cat === "All") return localModels.length;
    if (cat === "Bookmarked") return localModels.filter((m) => m.bookmarked).length;
    return localModels.filter((m) => (m.category || "General") === cat).length;
  };

  const Seg = <T extends string | number>({
    value,
    options,
    onChange,
  }: {
    value: T;
    options: { v: T; label: string }[];
    onChange: (v: T) => void;
  }) => (
    <div className="inline-flex rounded-md border border-line bg-bg p-0.5">
      {options.map((o) => (
        <button
          key={String(o.v)}
          onClick={() => onChange(o.v)}
          className={`px-2.5 py-1 text-xs rounded cursor-pointer ${
            value === o.v ? "bg-accent text-white font-medium" : "text-muted hover:text-fg"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );

  const NavItem = ({ id, icon, label, count }: { id: "check" | "models"; icon: React.ReactNode; label: string; count?: number }) => (
    <button
      onClick={() => setPage(id)}
      className={`w-full flex items-center justify-between gap-2 px-3 py-2 rounded-md text-sm cursor-pointer ${
        page === id ? "bg-raised text-fg font-medium" : "text-muted hover:text-fg hover:bg-raised/60"
      }`}
    >
      <span className="flex items-center gap-2.5">
        {icon}
        {label}
      </span>
      {count !== undefined && <span className="text-xs text-muted">{count}</span>}
    </button>
  );

  const rowCls = "flex items-center justify-between py-2.5 border-b border-line last:border-0 text-sm";

  return (
    <div className={`h-screen w-screen overflow-hidden bg-bg text-fg flex ${theme === "light" ? "theme-light" : ""}`}>
      {/* Sidebar */}
      <aside className="w-60 shrink-0 h-full overflow-y-auto border-r border-line bg-surface flex flex-col justify-between p-4 select-none">
        <div className="space-y-6">
          <div className="flex items-center gap-2.5 px-1">
            <div className="w-7 h-7 rounded-md bg-accent text-white flex items-center justify-center text-sm font-bold">w</div>
            <span className="text-base font-semibold tracking-tight">whichllmmodel</span>
          </div>
          <nav className="space-y-1">
            <NavItem id="check" icon={<Gauge className="w-4 h-4" />} label="Memory check" />
            <NavItem id="models" icon={<Library className="w-4 h-4" />} label="My models" count={installed.length} />
          </nav>
        </div>

        <div className="space-y-3">
          {hwInfo && (
            <div className="rounded-md border border-line p-3 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-medium text-muted uppercase tracking-wide text-[10px]">This computer</span>
                <button onClick={refreshHardware} title="Refresh" className="text-muted hover:text-fg cursor-pointer">
                  <RefreshCw className={`w-3 h-3 ${refreshingHw ? "animate-spin" : ""}`} />
                </button>
              </div>
              <div className="truncate" title={hwInfo.cpu_name}>
                <span className="text-muted">CPU </span>
                {hwInfo.cpu_name.replace("Intel(R) Core(TM) ", "")}
              </div>
              <div className="truncate" title={hwInfo.gpu_name}>
                <span className="text-muted">GPU </span>
                {hwInfo.gpu_name}
              </div>
              <div>
                <span className="text-muted">Memory bandwidth </span>
                {(hwInfo.measured_ram_gbs || hwInfo.ram_bandwidth_gbs).toFixed(1)} GB/s
              </div>
              <div>
                <div className="flex justify-between">
                  <span className="text-muted">Free RAM</span>
                  <span>
                    {(hwInfo.ram_free_mib / 1024).toFixed(1)} / {(hwInfo.ram_total_mib / 1024).toFixed(1)} GiB
                  </span>
                </div>
                <div className="mt-1 h-1 rounded-full bg-raised overflow-hidden">
                  <div className="h-full bg-accent" style={{ width: `${Math.min(100, hwInfo.ram_load_pct)}%` }} />
                </div>
              </div>
            </div>
          )}
          <button onClick={toggleTheme} className="btn w-full">
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </aside>

      <main className="flex-1 h-full overflow-y-auto">
        <div className="max-w-5xl mx-auto p-8 space-y-6">
          {/* Download progress */}
          {downloadStatus && (downloadStatus.active || downloadStatus.status === "completed" || downloadStatus.status === "error") && (
            <div className="rounded-lg border border-line bg-surface p-4 space-y-2">
              <div className="flex items-center justify-between gap-3 text-sm">
                <div className="truncate">
                  <span className="font-medium">{downloadStatus.filename}</span>
                  <span className="text-muted ml-2">
                    {downloadStatus.status === "completed"
                      ? "Download complete"
                      : downloadStatus.status === "error"
                      ? `Failed: ${downloadStatus.error}`
                      : downloadStatus.status === "resolving"
                      ? "Preparing…"
                      : `${fmtSize(downloadStatus.downloaded_mib)} of ${fmtSize(downloadStatus.total_mib)} · ${downloadStatus.speed_mbs.toFixed(1)} MB/s`}
                  </span>
                </div>
                {downloadStatus.active ? (
                  <button onClick={cancelDownload} className="btn !py-1">
                    Cancel
                  </button>
                ) : (
                  <button onClick={() => setDownloadStatus(null)} className="text-muted hover:text-fg cursor-pointer">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
              <div className="h-1.5 rounded-full bg-raised overflow-hidden">
                <div
                  className={`h-full ${downloadStatus.status === "error" ? "bg-bad" : downloadStatus.status === "completed" ? "bg-ok" : "bg-accent"}`}
                  style={{ width: `${Math.min(100, Math.max(2, downloadStatus.percent))}%` }}
                />
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-bad/40 bg-bad/10 p-4 text-sm text-bad whitespace-pre-wrap flex justify-between gap-3">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="shrink-0 cursor-pointer">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* ================= MEMORY CHECK ================= */}
          {page === "check" && (
            <>
              <div>
                <h1 className="text-xl font-semibold">Memory check</h1>
                <p className="text-sm text-muted mt-1">
                  Paste a Hugging Face link or pick a model you already have to see exactly how much GPU and system memory llama.cpp needs, and how fast it will run.
                </p>
              </div>

              <section className="rounded-lg border border-line bg-surface p-4 space-y-3">
                <div className="flex gap-2">
                  <input
                    className="field flex-1"
                    value={urlOrPath}
                    onChange={(e) => setUrlOrPath(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && runCheck()}
                    placeholder="Hugging Face repo, GGUF URL, or local file path"
                  />
                  <button className="btn-primary" onClick={() => runCheck()} disabled={loading || !urlOrPath.trim()}>
                    {loading ? "Checking…" : "Check"}
                  </button>
                </div>
                <div className="space-y-2 text-sm">
                  <select
                    className="field !py-1.5 truncate"
                    value=""
                    onChange={(e) => {
                      const m = installed.find((x) => x.model_path === e.target.value);
                      if (m) openInCheck(m);
                    }}
                  >
                    <option value="">Choose from my models…</option>
                    {installed.map((m) => (
                      <option key={m.model_path} value={m.model_path}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                  <button className="text-muted hover:text-fg cursor-pointer flex items-center gap-1" onClick={() => setShowVision(!showVision)}>
                    <Eye className="w-4 h-4" />
                    {showVision || mmprojUrlOrPath ? "Hide vision projector" : "Add vision projector"}
                  </button>
                </div>
                {(showVision || mmprojUrlOrPath) && (
                  <input
                    className="field"
                    value={mmprojUrlOrPath}
                    onChange={(e) => setMmprojUrlOrPath(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && runCheck()}
                    placeholder="mmproj GGUF URL or path (optional — detected automatically for most repos)"
                  />
                )}
              </section>

              {/* Settings */}
              <section className="rounded-lg border border-line bg-surface p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                  <div className="space-y-1.5">
                    <div className="label">
                      Context length
                      {result && <span className="font-normal"> · this model supports up to {result.native_context_length.toLocaleString()} tokens</span>}
                    </div>
                    <Seg
                      value={contextSize}
                      options={contextOptions.map((v) => ({
                        v,
                        label: result && v === result.native_context_length && !CONTEXT_PRESETS.includes(v) ? `Max ${Math.round(v / 1024)}K` : `${Math.round(v / 1024)}K`,
                      }))}
                      onChange={(v) => {
                        setContextSize(v);
                        runCheck(urlOrPath, mmprojUrlOrPath, v, ngl, ncmoe, device, kvQuant, noKvOffload);
                      }}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <div className="label">GPU layers</div>
                    <Seg
                      value={ngl}
                      options={[
                        { v: 99, label: "All" },
                        { v: 20, label: "20" },
                        { v: 10, label: "10" },
                        { v: 0, label: "None (CPU)" },
                      ]}
                      onChange={(v) => {
                        setNgl(v);
                        runCheck(urlOrPath, mmprojUrlOrPath, contextSize, v, ncmoe, device, kvQuant, noKvOffload);
                      }}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <div className="label">
                      Context compression
                      <span className="font-normal text-muted"> · KV cache</span>
                    </div>
                    <Seg
                      value={kvQuant}
                      options={[
                        { v: "q8_0", label: "Balanced (8-bit) · Rec" },
                        { v: "q4_0", label: "Max Context (4-bit)" },
                        { v: "f16", label: "Uncompressed (16-bit)" },
                      ]}
                      onChange={(v) => {
                        setKvQuant(v);
                        runCheck(urlOrPath, mmprojUrlOrPath, contextSize, ngl, ncmoe, device, v, noKvOffload);
                      }}
                    />
                  </div>
                  {(!result || result.is_moe) && (
                    <div className="space-y-1.5">
                      <div className="label">MoE layers kept on CPU</div>
                      <Seg
                        value={ncmoe}
                        options={[
                          { v: 0, label: "Off" },
                          { v: 10, label: "10" },
                          { v: 20, label: "20" },
                          { v: 30, label: "30" },
                          { v: 99, label: "All" },
                        ]}
                        onChange={(v) => {
                          setNcmoe(v);
                          runCheck(urlOrPath, mmprojUrlOrPath, contextSize, ngl, v, device, kvQuant, noKvOffload);
                        }}
                      />
                    </div>
                  )}
                  <div className="space-y-1.5">
                    <div className="label">Target hardware</div>
                    <div className="flex gap-2 flex-wrap">
                      <select
                        className="field !w-auto !py-1.5"
                        value={device}
                        onChange={(e) => {
                          setDevice(e.target.value);
                          runCheck(urlOrPath, mmprojUrlOrPath, contextSize, ngl, ncmoe, e.target.value, kvQuant, noKvOffload);
                        }}
                      >
                        <option value="CUDA0">NVIDIA CUDA</option>
                        <option value="Metal0">Apple Metal</option>
                      </select>
                      <select
                        className="field !w-auto !py-1.5"
                        value={vramBudgetGb}
                        onChange={(e) => {
                          const gb = Number(e.target.value);
                          setVramBudgetGb(gb);
                          if (gb <= 1) {
                            setCustomRig(false);
                            setVramBwGbs(hwInfo?.measured_ram_gbs || ramBwGbs);
                          } else {
                            setCustomRig(true);
                            setVramBwGbs(gb <= 4 ? 192 : gb <= 8 ? 320 : gb <= 12 ? 504 : gb <= 16 ? 672 : gb <= 24 ? 1008 : 1500);
                          }
                        }}
                      >
                        <option value={1}>Integrated GPU (shared RAM)</option>
                        {[4, 8, 12, 16, 24, 32, 64].map((g) => (
                          <option key={g} value={g}>
                            {g} GB VRAM
                          </option>
                        ))}
                      </select>
                      {hwInfo && (
                        <button className="btn !py-1.5" onClick={useMyHardware}>
                          Use my hardware
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="space-y-1.5 md:col-span-2 pt-2 border-t border-line/50">
                    <label className="flex items-center gap-2.5 cursor-pointer select-none text-xs text-fg">
                      <input
                        type="checkbox"
                        checked={noKvOffload}
                        onChange={(e) => {
                          const val = e.target.checked;
                          setNoKvOffload(val);
                          runCheck(urlOrPath, mmprojUrlOrPath, contextSize, ngl, ncmoe, device, kvQuant, val);
                        }}
                        className="rounded border-line text-accent focus:ring-0 cursor-pointer h-4 w-4"
                      />
                      <span className="font-medium">Store context in System RAM instead of GPU VRAM</span>
                      <span className="text-muted">(frees GPU graphics memory so you can fit larger models or more layers)</span>
                    </label>
                  </div>
                </div>
              </section>

              {!result && !loading && !error && (
                <div className="rounded-lg border border-dashed border-line p-10 text-center text-sm text-muted">
                  Enter a model above to see its memory requirements.
                </div>
              )}

              {result && (
                <>
                  {/* Model header */}
                  <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-semibold truncate">{result.model_name}</span>
                        <span className={`text-[11px] px-2 py-0.5 rounded border font-mono ${
                          result.is_local_model 
                            ? "bg-ok/10 border-ok/30 text-ok" 
                            : "bg-accent/10 border-accent/30 text-accent"
                        }`}>
                          {result.is_local_model ? "Local File (Downloaded)" : "Remote HuggingFace (Zero-Download Skeleton)"}
                        </span>
                      </div>
                      <div className="text-sm text-muted mt-0.5">
                        {result.arch} · {result.is_moe ? `Mixture of experts (${result.expert_used_count || 4} of ${result.expert_count} active)` : "Dense"} · up to{" "}
                        {Math.round(result.native_context_length / 1024)}K context
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap">
                      {!result.is_local_model && isRemote && (
                        <button className="btn" disabled={Boolean(downloadStatus?.active)} onClick={() => startDownload(result.resolved_url || urlOrPath)}>
                          <Download className="w-4 h-4" />
                          Download model
                        </button>
                      )}
                      {activeEntry && (
                        <>
                          <button
                            className="btn"
                            onClick={() => updateMeta(result.skeleton_path, !activeEntry.bookmarked, activeEntry.category || "General")}
                          >
                            <Star className={`w-4 h-4 ${activeEntry.bookmarked ? "fill-warn text-warn" : ""}`} />
                            {activeEntry.bookmarked ? "Saved" : "Save"}
                          </button>
                          <select
                            className="field !w-auto !py-2"
                            value={activeEntry.category || "General"}
                            onChange={(e) => updateMeta(result.skeleton_path, activeEntry.bookmarked, e.target.value)}
                          >
                            {allCategories.map((c) => (
                              <option key={c}>{c}</option>
                            ))}
                          </select>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Summary cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="rounded-lg border border-line bg-surface p-4">
                      <div className="label">GPU memory</div>
                      <div className="text-2xl font-semibold mt-1">{fmtGib(result.total_vram_mib)}</div>
                      <div className={`text-sm mt-1 ${gpuFits ? "text-ok" : "text-bad"}`}>
                        {result.total_vram_mib === 0
                          ? "Not used"
                          : gpuFits
                          ? `Fits in ${vramBudgetGb} GB`
                          : `Exceeds ${vramBudgetGb} GB by ${fmtGib(result.total_vram_mib - vramBudgetMib)}`}
                      </div>
                    </div>
                    <div className="rounded-lg border border-line bg-surface p-4">
                      <div className="label">System RAM</div>
                      <div className="text-2xl font-semibold mt-1">{fmtGib(ramNeed)}</div>
                      <div className={`text-sm mt-1 ${ramState === "ok" ? "text-ok" : ramState === "tight" ? "text-warn" : "text-bad"}`}>
                        {!hwInfo
                          ? ""
                          : ramState === "ok"
                          ? `Fits in ${(ramFree / 1024).toFixed(1)} GiB free`
                          : ramState === "tight"
                          ? "Fits only if you close other apps"
                          : `More than the ${(ramTotal / 1024).toFixed(1)} GiB installed`}
                      </div>
                    </div>
                    <div className="rounded-lg border border-line bg-surface p-4">
                      <div className="label">Estimated speed</div>
                      <div className="text-2xl font-semibold mt-1">
                        {speed ? `~${speed.ts}` : "—"} <span className="text-sm font-normal text-muted">tokens/s</span>
                      </div>
                      {speed && (
                        <div className="text-sm mt-1">
                          <span className={speed.tier.cls}>{speed.tier.label}</span>
                          <span className="text-muted">
                            {" "}
                            · range {speed.min}–{speed.max}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Breakdown */}
                  <section className="rounded-lg border border-line bg-surface">
                    <div className="px-4 py-3 border-b border-line text-sm font-medium">Memory breakdown</div>
                    <div className="px-4">
                      <div className="grid grid-cols-[1fr_8rem_8rem] text-xs text-muted py-2 border-b border-line">
                        <span>Component</span>
                        <span className="text-right">GPU</span>
                        <span className="text-right">System RAM</span>
                      </div>
                      {[
                        ["Model weights", result.vram_model_mib, result.ram_model_mib],
                        [`KV cache (${Math.round(result.context_size / 1024)}K context${result.kv_quant && result.kv_quant !== "f16" ? ` · ${result.kv_quant.toUpperCase()}` : ""}${result.no_kv_offload ? " · in RAM" : ""})`, result.vram_kv_mib, result.ram_kv_mib],
                        ["Recurrent state", result.vram_rs_mib, result.ram_rs_mib],
                        ["Compute buffer", result.vram_compute_mib, result.ram_compute_mib],
                        ["Output buffer", 0, result.output_buffer_mib],
                        ...(result.has_mmproj
                          ? [
                              [
                                "Vision projector",
                                result.vram_vision_model_mib + result.vram_vision_compute_mib,
                                result.ram_vision_model_mib + result.ram_vision_compute_mib,
                              ],
                            ]
                          : []),
                      ].map(([label, g, r]) => (
                        <div key={label as string} className="grid grid-cols-[1fr_8rem_8rem] py-2.5 border-b border-line text-sm">
                          <span>{label}</span>
                          <span className="text-right font-mono">{fmtMib(g as number)}</span>
                          <span className="text-right font-mono">{fmtMib(r as number)}</span>
                        </div>
                      ))}
                      <div className="grid grid-cols-[1fr_8rem_8rem] py-3 text-sm font-semibold">
                        <span>Total</span>
                        <span className="text-right font-mono">{fmtMib(result.total_vram_mib)}</span>
                        <span className="text-right font-mono">{fmtMib(result.total_ram_mib)}</span>
                      </div>
                    </div>
                  </section>

                  {speed && (
                    <p className="text-xs text-muted">
                      Speed estimate: {speed.gb} GB read per token at a measured {speed.bw.toFixed(1)} GB/s memory bandwidth. Memory sizes are computed by llama.cpp itself, not approximated.
                    </p>
                  )}

                  {/* Command & Timing Debug Strip */}
                  <section className="rounded-lg border border-line bg-surface p-4 space-y-3">
                    <div className="flex items-center justify-between gap-3 flex-wrap">
                      <div className="flex items-center gap-2 text-xs font-mono text-muted flex-wrap">
                        <span className="font-semibold text-fg">Timing breakdown:</span>
                        <span>
                          {result.is_local_model ? "Local inspection:" : "Download & skeleton prep:"}{" "}
                          <strong className="text-accent">{result.prep_time_ms.toFixed(0)} ms</strong>
                        </span>
                        <span>·</span>
                        <span>
                          llama-bench simulation: <strong className="text-ok">{result.bench_time_ms.toFixed(0)} ms</strong>
                        </span>
                        <span>·</span>
                        <span>
                          Total: <strong className="text-fg">{result.elapsed_ms} ms</strong>
                        </span>
                      </div>
                      <div className="flex gap-2 shrink-0 items-center">
                        <div className="inline-flex rounded-md border border-line bg-bg p-0.5 text-xs">
                          <button
                            type="button"
                            className={`px-2 py-0.5 rounded font-mono text-[11px] transition-colors ${
                              cmdMode === "cli" ? "bg-surface text-fg font-medium shadow-sm" : "text-muted hover:text-fg"
                            }`}
                            onClick={() => setCmdMode("cli")}
                          >
                            CLI Chat
                          </button>
                          <button
                            type="button"
                            className={`px-2 py-0.5 rounded font-mono text-[11px] transition-colors ${
                              cmdMode === "server" ? "bg-surface text-fg font-medium shadow-sm" : "text-muted hover:text-fg"
                            }`}
                            onClick={() => setCmdMode("server")}
                          >
                            Server (API)
                          </button>
                        </div>
                        <button className="btn !py-1" onClick={copyCli}>
                          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                          {copied ? "Copied" : `Copy ${cmdMode === "server" ? "server" : "cli"}`}
                        </button>
                        <button className="btn !py-1" onClick={() => setShowLogs(!showLogs)}>
                          Details
                          {showLogs ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                    </div>
                    <div className="text-xs font-mono text-muted truncate border-t border-line/60 pt-2 select-all">
                      {displayedCommand}
                    </div>
                    {showLogs && (
                      <pre className="rounded-md bg-bg border border-line p-3 text-[11px] font-mono text-muted max-h-48 overflow-auto">
                        {result.raw_lines.join("\n")}
                      </pre>
                    )}
                  </section>
                </>
              )}
            </>
          )}

          {/* ================= MY MODELS ================= */}
          {page === "models" && (
            <>
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <h1 className="text-xl font-semibold">My models</h1>
                  <p className="text-sm text-muted mt-1">
                    {installed.length} models on this computer · {fmtGib(totalDisk)} on disk. Found automatically in Ollama, LM Studio and Hugging Face folders.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button className="btn" onClick={() => runScan()} disabled={scanning}>
                    <RefreshCw className={`w-4 h-4 ${scanning ? "animate-spin" : ""}`} />
                    Rescan
                  </button>
                  <button className="btn-primary" onClick={() => setShowDownload(!showDownload)}>
                    <Download className="w-4 h-4" />
                    Download model
                  </button>
                </div>
              </div>

              {showDownload && (
                <section className="rounded-lg border border-line bg-surface p-4 space-y-2">
                  <div className="flex gap-2">
                    <input
                      className="field flex-1"
                      value={downloadInput}
                      onChange={(e) => setDownloadInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && startDownload(downloadInput)}
                      placeholder="Hugging Face repo or direct .gguf URL"
                    />
                    <button className="btn-primary" disabled={Boolean(downloadStatus?.active) || !downloadInput.trim()} onClick={() => startDownload(downloadInput)}>
                      Start download
                    </button>
                  </div>
                  <div className="text-xs text-muted">Saved to the app's downloaded_models folder. Interrupted downloads resume automatically.</div>
                </section>
              )}

              <div className="space-y-4">
                {/* Category bar */}
                <div className="flex items-center gap-2 flex-wrap">
                  {["All", "Bookmarked", ...allCategories].map((cat) => (
                    <button
                      key={cat}
                      onClick={() => setActiveCategory(cat)}
                      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm cursor-pointer ${
                        activeCategory === cat
                          ? "border-accent bg-accent/10 text-fg font-medium"
                          : "border-line text-muted hover:text-fg"
                      }`}
                    >
                      {cat === "Bookmarked" && <Star className="w-3.5 h-3.5" />}
                      {cat}
                      <span className="text-xs text-muted">{countFor(cat)}</span>
                      {activeCategory === cat && !DEFAULT_CATEGORIES.includes(cat) && cat !== "All" && cat !== "Bookmarked" && (
                        <span
                          title="Delete category"
                          className="hover:text-bad"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeCategory(cat);
                          }}
                        >
                          <X className="w-3 h-3" />
                        </span>
                      )}
                    </button>
                  ))}
                  {addingCategory ? (
                    <input
                      autoFocus
                      className="field !w-40 !py-1 !rounded-full"
                      value={newCategory}
                      placeholder="Category name"
                      onChange={(e) => setNewCategory(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") addCategory();
                        if (e.key === "Escape") {
                          setNewCategory("");
                          setAddingCategory(false);
                        }
                      }}
                      onBlur={addCategory}
                    />
                  ) : (
                    <button
                      className="inline-flex items-center gap-1 rounded-full border border-dashed border-line px-3 py-1 text-sm text-muted hover:text-fg cursor-pointer"
                      onClick={() => setAddingCategory(true)}
                    >
                      <Plus className="w-3.5 h-3.5" />
                      New category
                    </button>
                  )}
                </div>

                {/* Model list */}
                <div className="space-y-3 min-w-0">
                  <div className="flex items-center gap-3">
                    <div className="relative flex-1">
                      <Search className="w-4 h-4 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                      <input className="field !pl-9" placeholder="Search installed models…" value={query} onChange={(e) => setQuery(e.target.value)} />
                    </div>
                  </div>

                  <div className="rounded-lg border border-line bg-surface">
                    {visibleModels.length === 0 && <div className="p-10 text-center text-sm text-muted">No models in this view.</div>}
                    {visibleModels.map((m) => {
                      const disk = m.model_disk_mib + m.mmproj_disk_mib;
                      const need = m.model_logical_mib + m.mmproj_logical_mib + 250;
                      const fits = !hwInfo ? "ok" : need <= hwInfo.ram_free_mib ? "ok" : need <= hwInfo.ram_total_mib ? "tight" : "no";
                      let ratio = 1;
                      if (m.is_moe && m.expert_count > 0) ratio = Math.min(1, 0.1 + 0.9 * ((m.expert_used_count || 4) / m.expert_count));
                      const ts = Math.round((ramBwGbs / Math.max(0.05, (m.model_logical_mib / 1024) * ratio)) * 10) / 10;
                      const confirming = confirmDelete === m.model_path;
                      return (
                        <div key={m.model_path} className="px-4 py-3 border-b border-line last:border-0 space-y-1.5">
                          <div className="flex items-center gap-3 flex-wrap">
                            <div className="flex items-center gap-2 flex-1 min-w-[12rem]">
                              <button
                                title={m.bookmarked ? "Remove from saved" : "Save"}
                                className="cursor-pointer shrink-0"
                                onClick={() => updateMeta(m.model_path, !m.bookmarked, m.category || "General")}
                              >
                                <Star className={`w-4 h-4 ${m.bookmarked ? "fill-warn text-warn" : "text-muted hover:text-fg"}`} />
                              </button>
                              <span className="font-medium truncate" title={m.model_path}>
                                {m.name}
                              </span>
                            </div>
                            <select
                              className="field !w-36 !py-1.5 shrink-0"
                              value={m.category || "General"}
                              onChange={(e) => updateMeta(m.model_path, m.bookmarked, e.target.value)}
                            >
                              {allCategories.map((c) => (
                                <option key={c}>{c}</option>
                              ))}
                            </select>
                            <div className="flex items-center gap-2 shrink-0">
                              <button className="btn !py-1.5" onClick={() => openInCheck(m)}>
                                Check memory
                              </button>
                              {confirming ? (
                                <>
                                  <button
                                    className="btn !py-1.5 !border-bad !text-bad"
                                    disabled={deleting === m.model_path}
                                    onClick={() => deleteModel(m.model_path)}
                                  >
                                    {deleting === m.model_path ? "Deleting…" : `Delete ${fmtSize(disk)}`}
                                  </button>
                                  <button className="text-muted hover:text-fg cursor-pointer" onClick={() => setConfirmDelete(null)}>
                                    <X className="w-4 h-4" />
                                  </button>
                                </>
                              ) : (
                                <button title="Delete from disk" className="text-muted hover:text-bad cursor-pointer p-1" onClick={() => setConfirmDelete(m.model_path)}>
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="text-xs text-muted flex items-center gap-x-3 gap-y-0.5 flex-wrap pl-6">
                            <span>{sourceLabel(m.source)}</span>
                            <span>{m.arch}</span>
                            <span>{m.is_moe ? "MoE" : "Dense"}</span>
                            {m.mmproj_path && <span>+ vision</span>}
                            {m.is_sparse_skeleton ? (
                              <span>Full size {fmtSize(m.model_logical_mib)}</span>
                            ) : (
                              <>
                                <span>{fmtSize(disk)} on disk</span>
                                <span className={fits === "ok" ? "text-ok" : fits === "tight" ? "text-warn" : "text-bad"}>
                                  {fits === "ok" ? "Fits in free RAM" : fits === "tight" ? "Needs free RAM" : "Too large for RAM"}
                                </span>
                                <span>~{ts} tok/s on CPU</span>
                              </>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="flex gap-2">
                    <input
                      className="field"
                      value={customScanDir}
                      onChange={(e) => setCustomScanDir(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && runScan(customScanDir)}
                      placeholder="Add another models folder to scan, e.g. D:\Models"
                    />
                    <button className="btn shrink-0" onClick={() => runScan(customScanDir)}>
                      Scan folder
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
