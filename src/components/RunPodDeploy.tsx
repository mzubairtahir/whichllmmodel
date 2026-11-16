import { useState, useEffect, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Cloud,
  Copy,
  Check,
  RefreshCw,
  Play,
  Square,
  Trash2,
  Eye,
  EyeOff,
  Server,
  Sparkles,
  X,
} from "lucide-react";

export interface RunPodGpuType {
  id: string;
  display_name: string;
  memory_gb: number;
  secure_price: number | null;
  community_price: number | null;
  lowest_price: number | null;
  stock_status: string;
  is_live_pricing: boolean;
}

export interface RunPodPodInfo {
  id: string;
  name: string;
  desired_status: string;
  gpu_type: string;
  gpu_count: number;
  cost_per_hr: number;
  uptime_seconds: number;
  endpoint_url: string;
  health_status: string;
  gpu_util_pct: number | null;
  vram_util_pct: number | null;
}

interface RunPodDeployResponse {
  pod_id: string;
  endpoint_url: string;
  server_api_key: string;
  docker_args: string;
}

interface RunPodTestResult {
  ok: boolean;
  reply: string;
  latency_ms: number;
}

export interface SimResultLite {
  model_name: string;
  resolved_url: string;
  arch: string;
  native_context_length: number;
  detected_mmproj: boolean;
  include_vision: boolean;
  vram_model_mib: number;
  vram_kv_mib: number;
  vram_rs_mib: number;
  vram_compute_mib: number;
  vram_vision_model_mib: number;
  total_vram_mib: number;
  is_local_model: boolean;
}

interface Props {
  initialUrl: string;
  initialResult: SimResultLite | null;
  initialKvQuant: string;
  initialIncludeVision: boolean;
}

const CTX_PER_USER_OPTIONS = [4096, 8192, 16384, 32768];
const PARALLEL_OPTIONS = [1, 2, 4, 8, 16];

const fmtGib = (mib: number) => `${(mib / 1024).toFixed(2)} GiB`;

export default function RunPodDeploy({
  initialUrl,
  initialResult,
  initialKvQuant,
  initialIncludeVision,
}: Props) {
  const [modelInput, setModelInput] = useState(initialUrl || "");
  const [provider, setProvider] = useState<"runpod">("runpod");
  const [ctxPerUser, setCtxPerUser] = useState(8192);
  const [parallelSlots, setParallelSlots] = useState(4);
  const [kvQuant, setKvQuant] = useState(initialKvQuant || "q8_0");
  const [includeVision, setIncludeVision] = useState(initialIncludeVision);
  const [cloudType, setCloudType] = useState<"ALL" | "SECURE" | "COMMUNITY">("ALL");

  const [simulating, setSimulating] = useState(false);
  const [simResult, setSimResult] = useState<SimResultLite | null>(initialResult);
  const [simError, setSimError] = useState<string | null>(null);

  const [apiKey, setApiKey] = useState(() => localStorage.getItem("wlm_runpod_api_key") || "");
  const [showKey, setShowKey] = useState(false);
  const [hfToken, setHfToken] = useState(() => localStorage.getItem("wlm_hf_token") || "");

  const [gpus, setGpus] = useState<RunPodGpuType[]>([]);
  const [loadingGpus, setLoadingGpus] = useState(false);

  const [pods, setPods] = useState<RunPodPodInfo[]>([]);
  const [loadingPods, setLoadingPods] = useState(false);
  const [pendingGpu, setPendingGpu] = useState<{
    id: string;
    display_name: string;
    memory_gb: number;
    activePrice: number | null;
    headroomGib: number;
  } | null>(null);
  const [customPodName, setCustomPodName] = useState("");
  const [customDiskGb, setCustomDiskGb] = useState(30);
  const [deployingGpuId, setDeployingGpuId] = useState<string | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const [podSecrets, setPodSecrets] = useState<Record<string, string>>(() => {
    try {
      return JSON.parse(localStorage.getItem("wlm_runpod_pod_secrets") || "{}");
    } catch {
      return {};
    }
  });
  const [testReplies, setTestReplies] = useState<Record<string, RunPodTestResult | string>>({});
  const [testingPodId, setTestingPodId] = useState<string | null>(null);

  const totalCtx = ctxPerUser * parallelSlots;

  const saveApiKey = (val: string) => {
    setApiKey(val);
    localStorage.setItem("wlm_runpod_api_key", val);
  };

  const saveHfToken = (val: string) => {
    setHfToken(val);
    localStorage.setItem("wlm_hf_token", val);
  };

  const fetchGpus = async (keyOverride?: string) => {
    setLoadingGpus(true);
    try {
      const k = (keyOverride ?? apiKey).trim();
      const list = await invoke<RunPodGpuType[]>("fetch_runpod_gpus_cmd", {
        apiKey: k || null,
      });
      setGpus(list);
    } catch (e) {
      setDeployError(String(e));
    } finally {
      setLoadingGpus(false);
    }
  };

  const fetchPods = async (keyOverride?: string) => {
    const k = (keyOverride ?? apiKey).trim();
    if (!k) {
      setPods([]);
      return;
    }
    setLoadingPods(true);
    try {
      const list = await invoke<RunPodPodInfo[]>("list_runpod_pods_cmd", { apiKey: k });
      setPods(list);
    } catch (e) {
      setDeployError(String(e));
    } finally {
      setLoadingPods(false);
    }
  };

  useEffect(() => {
    fetchGpus();
    if (apiKey.trim()) fetchPods();
  }, []);

  useEffect(() => {
    if (!apiKey.trim()) return;
    const hasTransitioning = pods.some(
      (p) =>
        p.desired_status === "RUNNING" &&
        (p.health_status === "provisioning" || p.health_status === "downloading")
    );
    if (!hasTransitioning) return;
    const t = setInterval(() => {
      fetchPods();
    }, 6000);
    return () => clearInterval(t);
  }, [pods, apiKey]);

  const runWorkloadSim = async () => {
    if (!modelInput.trim()) return;
    setSimulating(true);
    setSimError(null);
    try {
      const res = await invoke<SimResultLite>("analyze_model", {
        urlOrPath: modelInput.trim(),
        mmprojUrlOrPath: null,
        contextSize: totalCtx,
        ngl: 99,
        ncmoe: null,
        device: "CUDA0",
        kvQuant,
        noKvOffload: false,
        includeVision,
        parallelSlots,
      });
      setSimResult(res);
    } catch (e) {
      setSimError(String(e));
    } finally {
      setSimulating(false);
    }
  };

  useEffect(() => {
    if (simResult && modelInput.trim()) {
      runWorkloadSim();
    }
  }, [ctxPerUser, parallelSlots, kvQuant, includeVision]);

  const hfMatch = useMemo(() => {
    if (!simResult) return null;
    return simResult.resolved_url.match(
      /^https?:\/\/huggingface\.co\/([^/]+\/[^/]+)\/(?:resolve|blob)\/[^/]+\/(.+)$/i
    );
  }, [simResult]);

  const neededVramGib = simResult ? simResult.total_vram_mib / 1024 : 0;
  const recommendedDiskGb = simResult
    ? Math.max(
        25,
        Math.ceil(((simResult.vram_model_mib + simResult.vram_vision_model_mib) / 1024) * 1.15) + 15
      )
    : 30;

  const gpuRows = useMemo(() => {
    return gpus
      .map((g) => {
        const price =
          cloudType === "SECURE"
            ? g.secure_price
            : cloudType === "COMMUNITY"
            ? g.community_price
            : g.lowest_price ?? g.community_price ?? g.secure_price;
        const headroomGib = g.memory_gb - neededVramGib;
        const fits = simResult ? headroomGib >= 0.5 : true;
        return { ...g, activePrice: price, headroomGib, fits };
      })
      .filter((g) => g.activePrice !== null && g.activePrice > 0);
  }, [gpus, cloudType, neededVramGib, simResult]);

  const cheapestFitId = useMemo(() => {
    if (!simResult) return null;
    const fitting = gpuRows
      .filter((g) => g.fits && g.activePrice !== null)
      .sort((a, b) => (a.activePrice ?? 999) - (b.activePrice ?? 999));
    return fitting[0]?.id ?? null;
  }, [gpuRows, simResult]);

  const openDeployReview = (g: {
    id: string;
    display_name: string;
    memory_gb: number;
    activePrice: number | null;
    headroomGib: number;
  }) => {
    setDeployError(null);
    const shortRepo = hfMatch?.[1]?.split("/").pop()?.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 24) || "llm";
    setCustomPodName(`wlm-${shortRepo}`);
    setCustomDiskGb(recommendedDiskGb);
    setPendingGpu(g);
  };

  const confirmAndDeploy = async () => {
    if (!pendingGpu) return;
    const gpuId = pendingGpu.id;
    if (!apiKey.trim()) {
      setDeployError("Enter your RunPod API key before deploying.");
      return;
    }
    if (!hfMatch) {
      setDeployError("Simulate a remote Hugging Face GGUF model first so RunPod can pull it.");
      return;
    }
    setDeployingGpuId(gpuId);
    setDeployError(null);
    try {
      const resp = await invoke<RunPodDeployResponse>("deploy_runpod_pod_cmd", {
        req: {
          api_key: apiKey.trim(),
          gpu_type_id: gpuId,
          cloud_type: cloudType,
          hf_repo: hfMatch[1],
          hf_file: hfMatch[2],
          total_ctx: totalCtx,
          parallel_slots: parallelSlots,
          kv_quant: kvQuant,
          include_vision: includeVision,
          detected_mmproj: simResult?.detected_mmproj ?? false,
          container_disk_gb: Math.max(25, customDiskGb),
          hf_token: hfToken.trim() || null,
          pod_name: customPodName.trim() || null,
        },
      });
      const updated = { ...podSecrets, [resp.pod_id]: resp.server_api_key };
      setPodSecrets(updated);
      localStorage.setItem("wlm_runpod_pod_secrets", JSON.stringify(updated));
      setPendingGpu(null);
      await fetchPods();
    } catch (e) {
      setDeployError(String(e));
    } finally {
      setDeployingGpuId(null);
    }
  };

  const handleStopPod = async (podId: string) => {
    try {
      await invoke("stop_runpod_pod_cmd", { apiKey: apiKey.trim(), podId });
      await fetchPods();
    } catch (e) {
      setDeployError(String(e));
    }
  };

  const handleTerminatePod = async (podId: string) => {
    try {
      await invoke("terminate_runpod_pod_cmd", { apiKey: apiKey.trim(), podId });
      await fetchPods();
    } catch (e) {
      setDeployError(String(e));
    }
  };

  const handleTestPod = async (pod: RunPodPodInfo) => {
    setTestingPodId(pod.id);
    try {
      const res = await invoke<RunPodTestResult>("test_runpod_endpoint_cmd", {
        endpointUrl: pod.endpoint_url,
        serverApiKey: podSecrets[pod.id] || "",
      });
      setTestReplies((prev) => ({ ...prev, [pod.id]: res }));
    } catch (e) {
      setTestReplies((prev) => ({ ...prev, [pod.id]: String(e) }));
    } finally {
      setTestingPodId(null);
    }
  };

  const copyText = (key: string, val: string) => {
    navigator.clipboard.writeText(val);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1500);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <Cloud className="w-5 h-5 text-accent" />
            Deploy Private AI Server
          </h1>
          <p className="text-sm text-muted mt-1">
            Simulate exact VRAM for your concurrent users, pick the cheapest guaranteed-fit cloud GPU, and deploy a private OpenAI-compatible{" "}
            <code>llama-server</code> in one click.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted">Cloud provider</span>
          <div className="inline-flex rounded-md border border-line bg-surface p-0.5">
            <button
              onClick={() => setProvider("runpod")}
              className={`px-3 py-1 rounded cursor-pointer ${
                provider === "runpod" ? "bg-accent text-white font-medium" : "text-muted hover:text-fg"
              }`}
            >
              RunPod
            </button>
          </div>
        </div>
      </div>

      {/* Step 1: Model & Concurrency Workload */}
      <section className="rounded-lg border border-line bg-surface p-4 space-y-4">
        <div className="flex gap-2">
          <input
            className="field flex-1"
            value={modelInput}
            onChange={(e) => setModelInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runWorkloadSim()}
            placeholder="Hugging Face repo or GGUF URL (e.g. unsloth/Qwen3.5-35B-A3B-GGUF)"
          />
          <button
            className="btn-primary"
            onClick={runWorkloadSim}
            disabled={simulating || !modelInput.trim()}
          >
            {simulating ? "Simulating…" : "Simulate Workload"}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-muted">Context / user</span>
            <div className="inline-flex rounded-md border border-line bg-bg p-0.5">
              {CTX_PER_USER_OPTIONS.map((v) => (
                <button
                  key={v}
                  onClick={() => setCtxPerUser(v)}
                  className={`px-2 py-1 rounded cursor-pointer ${
                    ctxPerUser === v ? "bg-accent text-white font-medium" : "text-muted hover:text-fg"
                  }`}
                >
                  {v / 1024}K
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-muted">Concurrent users (-np)</span>
            <div className="inline-flex rounded-md border border-line bg-bg p-0.5">
              {PARALLEL_OPTIONS.map((v) => (
                <button
                  key={v}
                  onClick={() => setParallelSlots(v)}
                  className={`px-2 py-1 rounded cursor-pointer ${
                    parallelSlots === v ? "bg-accent text-white font-medium" : "text-muted hover:text-fg"
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-muted">KV Quant</span>
            <div className="inline-flex rounded-md border border-line bg-bg p-0.5">
              {["f16", "q8_0", "q4_0"].map((q) => (
                <button
                  key={q}
                  onClick={() => setKvQuant(q)}
                  className={`px-2 py-1 rounded cursor-pointer ${
                    kvQuant === q ? "bg-accent text-white font-medium" : "text-muted hover:text-fg"
                  }`}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          <label className="flex items-center gap-1.5 text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={includeVision}
              onChange={(e) => setIncludeVision(e.target.checked)}
            />
            Include Vision (mmproj)
          </label>
        </div>

        {simError && (
          <div className="rounded border border-bad/40 bg-bad/10 p-3 text-xs text-bad">{simError}</div>
        )}

        {simResult && (
          <div className="rounded-md border border-line bg-bg p-3 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div>
              <div className="font-medium text-sm text-fg">{simResult.model_name}</div>
              <div className="text-muted mt-0.5">
                {simResult.arch} · Total KV ctx: <span className="text-fg font-medium">{(totalCtx / 1024).toFixed(0)}K</span> ({parallelSlots}× {ctxPerUser / 1024}K) · Disk:{" "}
                <span className="text-fg font-medium">{recommendedDiskGb} GB</span>
              </div>
            </div>
            <div className="flex items-center gap-5">
              <div>
                <div className="text-muted">Weights</div>
                <div className="font-medium">{fmtGib(simResult.vram_model_mib + simResult.vram_vision_model_mib)}</div>
              </div>
              <div>
                <div className="text-muted">KV + RS</div>
                <div className="font-medium">{fmtGib(simResult.vram_kv_mib + simResult.vram_rs_mib)}</div>
              </div>
              <div>
                <div className="text-muted">Compute</div>
                <div className="font-medium">{fmtGib(simResult.vram_compute_mib)}</div>
              </div>
              <div>
                <div className="text-muted">Total VRAM Needed</div>
                <div className="font-semibold text-sm text-accent">{fmtGib(simResult.total_vram_mib)}</div>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Step 2: RunPod Credentials */}
      <section className="rounded-lg border border-line bg-surface p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">RunPod Account</h2>
          <button
            className="btn !py-1 !text-xs flex items-center gap-1.5"
            onClick={() => {
              fetchGpus();
              fetchPods();
            }}
            disabled={loadingGpus || loadingPods}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${(loadingGpus || loadingPods) ? "animate-spin" : ""}`} />
            Refresh Live Prices & Pods
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-muted mb-1">RunPod API Key (stored locally only)</label>
            <div className="flex gap-1.5">
              <input
                type={showKey ? "text" : "password"}
                className="field flex-1 !py-1.5 text-xs"
                value={apiKey}
                onChange={(e) => saveApiKey(e.target.value)}
                placeholder="rp_..."
              />
              <button className="btn !px-2.5 !py-1.5" onClick={() => setShowKey((s) => !s)} title="Show/hide key">
                {showKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>
          <div>
            <label className="block text-xs text-muted mb-1">Hugging Face Token (optional, for gated models)</label>
            <input
              type="password"
              className="field w-full !py-1.5 text-xs"
              value={hfToken}
              onChange={(e) => saveHfToken(e.target.value)}
              placeholder="hf_..."
            />
          </div>
        </div>
      </section>

      {deployError && (
        <div className="rounded-lg border border-bad/40 bg-bad/10 p-3 text-xs text-bad">{deployError}</div>
      )}

      {/* Step 3: RunPod GPU Fit & Pricing Table */}
      <section className="rounded-lg border border-line bg-surface p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Cheapest Fitting RunPod GPUs</h2>
            <p className="text-xs text-muted">
              {gpus[0]?.is_live_pricing
                ? "Live RunPod pricing & availability"
                : "Estimated RunPod pricing (add API key & refresh for live rates)"}
            </p>
          </div>
          <div className="inline-flex rounded-md border border-line bg-bg p-0.5 text-xs">
            {(["ALL", "COMMUNITY", "SECURE"] as const).map((ct) => (
              <button
                key={ct}
                onClick={() => setCloudType(ct)}
                className={`px-2.5 py-1 rounded cursor-pointer ${
                  cloudType === ct ? "bg-accent text-white font-medium" : "text-muted hover:text-fg"
                }`}
              >
                {ct === "ALL" ? "All Clouds" : ct === "COMMUNITY" ? "Community" : "Secure"}
              </button>
            ))}
          </div>
        </div>

        <div className="divide-y divide-line">
          {gpuRows.map((g) => {
            const isCheapestFit = g.id === cheapestFitId;
            return (
              <div
                key={g.id}
                className={`py-2.5 flex flex-wrap items-center justify-between gap-3 text-sm ${
                  !g.fits ? "opacity-50" : ""
                }`}
              >
                <div className="min-w-[190px]">
                  <div className="font-medium flex items-center gap-2">
                    <span>{g.display_name}</span>
                    <span className="text-xs text-muted font-normal">{g.memory_gb} GB VRAM</span>
                    {isCheapestFit && (
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-ok/15 text-ok text-[11px] font-medium">
                        <Sparkles className="w-3 h-3" /> Cheapest Fit
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted">{g.stock_status}</div>
                </div>

                <div className="text-xs">
                  {simResult ? (
                    g.fits ? (
                      <span className="text-ok font-medium">
                        Fits · {g.headroomGib.toFixed(1)} GiB headroom
                      </span>
                    ) : (
                      <span className="text-bad font-medium">
                        OOM · needs +{Math.abs(g.headroomGib).toFixed(1)} GiB
                      </span>
                    )
                  ) : (
                    <span className="text-muted">Simulate workload to check fit</span>
                  )}
                </div>

                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="font-semibold">${g.activePrice?.toFixed(2)}/hr</div>
                    <div className="text-[11px] text-muted">
                      ~${((g.activePrice ?? 0) * 24).toFixed(2)}/day
                    </div>
                  </div>
                  <button
                    className={isCheapestFit ? "btn-primary !py-1.5 !px-3 text-xs" : "btn !py-1.5 !px-3 text-xs"}
                    disabled={!g.fits || !hfMatch || !apiKey.trim() || deployingGpuId !== null}
                    onClick={() => openDeployReview(g)}
                  >
                    Deploy
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Step 4: Active Pods Cockpit */}
      {apiKey.trim() && (
        <section className="rounded-lg border border-line bg-surface p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold flex items-center gap-2">
              <Server className="w-4 h-4 text-accent" />
              Active RunPod Servers ({pods.length})
            </h2>
          </div>

          {pods.length === 0 ? (
            <div className="text-xs text-muted py-2">
              No active pods on your RunPod account yet. Pick a fitting GPU above and click Deploy.
            </div>
          ) : (
            <div className="space-y-3">
              {pods.map((pod) => {
                const secret = podSecrets[pod.id];
                const testOut = testReplies[pod.id];
                const badgeCls =
                  pod.health_status === "ready"
                    ? "bg-ok/15 text-ok"
                    : pod.health_status === "downloading" || pod.health_status === "provisioning"
                    ? "bg-warn/15 text-warn"
                    : "bg-raised text-muted";

                return (
                  <div key={pod.id} className="rounded-md border border-line bg-bg p-3 space-y-2.5 text-xs">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm text-fg">{pod.name}</span>
                        <span className={`px-2 py-0.5 rounded font-medium uppercase text-[10px] ${badgeCls}`}>
                          {pod.health_status}
                        </span>
                        <span className="text-muted">
                          {pod.gpu_count}× {pod.gpu_type} · ${pod.cost_per_hr.toFixed(2)}/hr
                          {pod.vram_util_pct !== null && (
                            <> · <span className="text-fg font-medium">VRAM {pod.vram_util_pct.toFixed(0)}%</span></>
                          )}
                          {pod.gpu_util_pct !== null && <> · GPU {pod.gpu_util_pct.toFixed(0)}%</>}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {pod.health_status === "ready" && (
                          <button
                            className="btn !py-1 !px-2.5 flex items-center gap-1"
                            disabled={testingPodId === pod.id}
                            onClick={() => handleTestPod(pod)}
                          >
                            <Play className="w-3 h-3" />
                            {testingPodId === pod.id ? "Testing…" : "Test API"}
                          </button>
                        )}
                        {pod.desired_status === "RUNNING" && (
                          <button
                            className="btn !py-1 !px-2.5 flex items-center gap-1"
                            onClick={() => handleStopPod(pod.id)}
                          >
                            <Square className="w-3 h-3" /> Stop
                          </button>
                        )}
                        <button
                          className="btn !py-1 !px-2.5 text-bad hover:bg-bad/10 flex items-center gap-1"
                          onClick={() => handleTerminatePod(pod.id)}
                        >
                          <Trash2 className="w-3 h-3" /> Terminate
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      <div className="flex items-center justify-between gap-2 rounded border border-line bg-surface px-2.5 py-1.5">
                        <span className="truncate font-mono text-[11px]">{pod.endpoint_url}</span>
                        <button
                          className="text-muted hover:text-fg shrink-0 cursor-pointer"
                          onClick={() => copyText(`url-${pod.id}`, pod.endpoint_url)}
                        >
                          {copiedKey === `url-${pod.id}` ? <Check className="w-3.5 h-3.5 text-ok" /> : <Copy className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                      {secret && (
                        <div className="flex items-center justify-between gap-2 rounded border border-line bg-surface px-2.5 py-1.5">
                          <span className="truncate font-mono text-[11px]">Key: {secret}</span>
                          <button
                            className="text-muted hover:text-fg shrink-0 cursor-pointer"
                            onClick={() => copyText(`key-${pod.id}`, secret)}
                          >
                            {copiedKey === `key-${pod.id}` ? <Check className="w-3.5 h-3.5 text-ok" /> : <Copy className="w-3.5 h-3.5" />}
                          </button>
                        </div>
                      )}
                    </div>

                    {testOut && (
                      <div className="rounded border border-line bg-surface px-2.5 py-1.5 text-[11px]">
                        {typeof testOut === "string" ? (
                          <span className="text-bad">{testOut}</span>
                        ) : (
                          <span className="text-ok">
                            Reply ({testOut.latency_ms} ms): <span className="text-fg">{testOut.reply}</span>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {/* Pre-Deploy Confirmation Window */}
      {pendingGpu && hfMatch && simResult && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4">
          <div className="w-full max-w-xl rounded-lg border border-line bg-surface p-5 space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold">Review &amp; Start RunPod Server</h3>
                <p className="text-xs text-muted">
                  Confirm your configuration before RunPod provisions the GPU and starts billing.
                </p>
              </div>
              <button
                className="text-muted hover:text-fg cursor-pointer"
                onClick={() => deployingGpuId === null && setPendingGpu(null)}
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="rounded-md border border-line bg-bg p-3 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-muted">GPU &amp; Cloud</span>
                <span className="font-medium text-fg">
                  {pendingGpu.display_name} ({pendingGpu.memory_gb} GB) · {cloudType}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Estimated Rate</span>
                <span className="font-semibold text-accent">
                  ${pendingGpu.activePrice?.toFixed(2)}/hr (~${((pendingGpu.activePrice ?? 0) * 24).toFixed(2)}/day)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Model</span>
                <span className="font-mono text-[11px] truncate max-w-[340px]">
                  {hfMatch[1]} / {hfMatch[2]}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Workload &amp; VRAM Fit</span>
                <span>
                  {parallelSlots}× {ctxPerUser / 1024}K ({totalCtx / 1024}K total) · {kvQuant} KV ·{" "}
                  <span className="text-ok font-medium">
                    {fmtGib(simResult.total_vram_mib)} ({pendingGpu.headroomGib.toFixed(1)} GiB free)
                  </span>
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Docker Image &amp; Port</span>
                <span className="font-mono text-[11px]">ghcr.io/ggml-org/llama.cpp:server-cuda · 8000/http</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-muted mb-1">Pod Name</label>
                <input
                  className="field w-full !py-1.5 text-xs"
                  value={customPodName}
                  onChange={(e) => setCustomPodName(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-muted mb-1">Container Disk (GB, min 25)</label>
                <input
                  type="number"
                  min={25}
                  className="field w-full !py-1.5 text-xs"
                  value={customDiskGb}
                  onChange={(e) => setCustomDiskGb(Math.max(25, Number(e.target.value) || 25))}
                />
              </div>
            </div>

            <div className="text-xs">
              <div className="text-muted mb-1">Server Launch Arguments (auto-generates private API key)</div>
              <pre className="rounded border border-line bg-bg p-2.5 font-mono text-[11px] whitespace-pre-wrap break-all text-fg">
                {`-hf ${hfMatch[1]} -hff "${hfMatch[2]}"${
                  simResult.detected_mmproj && !includeVision ? " --no-mmproj" : ""
                } -c ${totalCtx} -np ${parallelSlots} -ngl 999${
                  kvQuant !== "f16" ? ` -ctk ${kvQuant} -ctv ${kvQuant}` : ""
                } --host 0.0.0.0 --port 8000 -v --log-file /tmp/llama.log --api-key <auto-generated>`}
              </pre>
            </div>

            {deployError && (
              <div className="rounded border border-bad/40 bg-bad/10 p-2.5 text-xs text-bad">{deployError}</div>
            )}

            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                className="btn !py-1.5 !px-3 text-xs"
                disabled={deployingGpuId !== null}
                onClick={() => setPendingGpu(null)}
              >
                Cancel
              </button>
              <button
                className="btn-primary !py-1.5 !px-4 text-xs"
                disabled={deployingGpuId !== null}
                onClick={confirmAndDeploy}
              >
                {deployingGpuId !== null
                  ? "Starting Pod…"
                  : `Confirm & Start Pod ($${pendingGpu.activePrice?.toFixed(2)}/hr)`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

