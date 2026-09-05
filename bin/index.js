#!/usr/bin/env node

// src/index.ts
import fs from "fs";
import path from "path";
import { Command } from "commander";

// src/hardware/detector.ts
import os2 from "os";
import { execSync as execSync3 } from "child_process";

// src/hardware/memory.ts
import os from "os";
function bytesToGB(bytes) {
  return Number((bytes / (1024 * 1024 * 1024)).toFixed(2));
}
function getSystemRam() {
  const totalBytes = os.totalmem();
  const usableBytes = os.freemem();
  return {
    totalBytes,
    totalGB: bytesToGB(totalBytes),
    usableBytes,
    usableGB: bytesToGB(usableBytes)
  };
}

// src/hardware/nvidia.ts
import { execSync } from "child_process";
function detectNvidiaGpu() {
  try {
    const rawOutput = execSync(
      "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits",
      {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
        timeout: 2500
      }
    ).trim();
    if (!rawOutput) return null;
    const lines = rawOutput.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const gpus = [];
    for (const line of lines) {
      const parts = line.split(",").map((p) => p.trim());
      if (parts.length >= 3) {
        const name = parts[0];
        const totalMB = parseFloat(parts[1]);
        const freeMB = parseFloat(parts[2]);
        if (!isNaN(totalMB) && !isNaN(freeMB)) {
          const totalBytes = totalMB * 1024 * 1024;
          const usableBytes = freeMB * 1024 * 1024;
          gpus.push({
            name,
            vram: {
              totalBytes,
              totalGB: bytesToGB(totalBytes),
              usableBytes,
              usableGB: bytesToGB(usableBytes)
            }
          });
        }
      }
    }
    if (gpus.length === 0) return null;
    gpus.sort((a, b) => b.vram.totalBytes - a.vram.totalBytes);
    return gpus[0];
  } catch {
    return null;
  }
}

// src/hardware/apple.ts
import { execSync as execSync2 } from "child_process";
function detectAppleSilicon() {
  if (process.platform !== "darwin") return null;
  try {
    let chipName = "";
    try {
      chipName = execSync2("sysctl -n machdep.cpu.brand_string", {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
        timeout: 2e3
      }).trim();
    } catch {
    }
    const isArm64 = process.arch === "arm64";
    const isAppleChip = /Apple\s+M\d/i.test(chipName) || /Apple/i.test(chipName) || isArm64;
    if (!isAppleChip) return null;
    if (!chipName) {
      chipName = "Apple Silicon";
    }
    const memsizeStr = execSync2("sysctl -n hw.memsize", {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
      timeout: 2e3
    }).trim();
    const totalBytes = parseInt(memsizeStr, 10);
    if (isNaN(totalBytes) || totalBytes <= 0) return null;
    const usableBytes = Math.floor(totalBytes * 0.75);
    return {
      chipName,
      unifiedMemory: {
        isUnified: true,
        totalBytes,
        totalGB: bytesToGB(totalBytes),
        usableBytes,
        usableGB: bytesToGB(usableBytes)
      }
    };
  } catch {
    return null;
  }
}

// src/hardware/detector.ts
function detectWindowsGpuName() {
  if (process.platform !== "win32") return null;
  try {
    const output = execSync3(
      'powershell -NoProfile -NonInteractive -Command "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"',
      {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
        timeout: 2500
      }
    ).trim();
    return output || null;
  } catch {
    return null;
  }
}
function detectHardware() {
  const platform = process.platform;
  const arch = process.arch;
  const cpus = os2.cpus();
  const cpuName = cpus.length > 0 ? cpus[0].model.trim() : "Unknown CPU";
  const ram = getSystemRam();
  const appleSilicon = detectAppleSilicon();
  if (appleSilicon) {
    return {
      type: "unified_memory",
      platform,
      arch,
      name: appleSilicon.chipName,
      cpuName: appleSilicon.chipName,
      gpuName: `${appleSilicon.chipName} GPU`,
      unifiedMemory: appleSilicon.unifiedMemory,
      ram
    };
  }
  const nvidiaGpu = detectNvidiaGpu();
  if (nvidiaGpu) {
    return {
      type: "gpu",
      platform,
      arch,
      name: nvidiaGpu.name,
      cpuName,
      gpuName: nvidiaGpu.name,
      vram: nvidiaGpu.vram,
      ram
    };
  }
  const winGpuName = detectWindowsGpuName();
  return {
    type: "cpu_ram",
    platform,
    arch,
    name: winGpuName || cpuName,
    cpuName,
    gpuName: winGpuName || void 0,
    ram
  };
}

// src/ui/formatters.ts
import chalk from "chalk";
function formatHardwareSummaryLine(hw) {
  const parts = [];
  if (hw.type === "unified_memory" && hw.unifiedMemory) {
    parts.push(chalk.bold(hw.name));
    parts.push(
      `${chalk.cyan(hw.unifiedMemory.usableGB + " GB")} usable unified memory / ${hw.unifiedMemory.totalGB} GB total`
    );
  } else if (hw.type === "gpu" && hw.vram) {
    parts.push(chalk.bold(hw.name));
    parts.push(
      `${chalk.cyan(hw.vram.usableGB + " GB")} usable VRAM / ${hw.vram.totalGB} GB total`
    );
    parts.push(`${hw.ram.totalGB} GB RAM`);
  } else {
    parts.push(chalk.bold(hw.name));
    parts.push(
      `${chalk.cyan(hw.ram.usableGB + " GB")} usable / ${hw.ram.totalGB} GB RAM`
    );
  }
  return parts.join(" \u2022 ");
}
function formatFitBadge(fitTier) {
  const lower = fitTier.toLowerCase();
  if (lower === "optimal" || lower === "vram") {
    return chalk.green("100% VRAM (Optimal)");
  }
  if (lower === "good") {
    return chalk.green("Fits in VRAM");
  }
  if (lower === "tight") {
    return chalk.yellow("Tight Fit (VRAM)");
  }
  if (lower === "offload" || lower === "cpu_offload") {
    return chalk.yellow("CPU RAM Offload");
  }
  if (lower === "ram") {
    return chalk.cyan("System RAM Only");
  }
  return chalk.red("Exceeds Memory");
}
function renderCleanRecommendations(recommendations) {
  if (recommendations.length === 0) {
    console.log(chalk.gray("No matching models found."));
    return;
  }
  console.log(chalk.bold.white("Top Recommendations:"));
  console.log();
  for (const m of recommendations) {
    const paramStr = m.parameters ? ` (${m.parameters})` : "";
    console.log(
      `  ${chalk.bold.cyan(m.rank + ".")} ${chalk.bold.white(m.name)}${chalk.gray(paramStr)}`
    );
    console.log(
      `     ${chalk.gray("Quant:")}    ${chalk.yellow(m.recommendedQuant)} ` + chalk.gray(`(${m.memoryFootprint.totalRequiredGB} GB required)`)
    );
    console.log(`     ${chalk.gray("Fit:")}      ${formatFitBadge(m.fitTier)}`);
    if (m.scores.coding !== void 0) {
      console.log(`     ${chalk.gray("Score:")}    ${chalk.green.bold(m.scores.coding)} ${chalk.gray("(coding)")}`);
    } else if (m.scores.overall !== void 0) {
      console.log(`     ${chalk.gray("Score:")}    ${chalk.green.bold(m.scores.overall)} ${chalk.gray("(overall)")}`);
    } else if (m.scores.reasoning !== void 0) {
      console.log(`     ${chalk.gray("Score:")}    ${chalk.green.bold(m.scores.reasoning)} ${chalk.gray("(reasoning)")}`);
    }
    console.log(`     ${chalk.gray("URL:")}      ${chalk.cyan.underline(m.websiteUrl)}`);
    console.log();
  }
}
function renderCleanHardwareProfile(hw) {
  console.log(chalk.bold.white("Hardware Profile:"));
  console.log(`  ${chalk.gray("Device:")}       ${chalk.white.bold(hw.name)}`);
  console.log(`  ${chalk.gray("CPU:")}          ${chalk.white(hw.cpuName)} ${chalk.gray(`(${hw.platform} ${hw.arch})`)}`);
  if (hw.type === "unified_memory" && hw.unifiedMemory) {
    console.log(`  ${chalk.gray("Architecture:")} ${chalk.green("Apple Silicon Unified Memory")}`);
    console.log(
      `  ${chalk.gray("Memory:")}        ${chalk.cyan(hw.unifiedMemory.usableGB + " GB")} usable / ${hw.unifiedMemory.totalGB} GB total (75% macOS VRAM ceiling)`
    );
  } else if (hw.type === "gpu" && hw.vram) {
    console.log(`  ${chalk.gray("Architecture:")} ${chalk.green("Discrete GPU")}`);
    console.log(
      `  ${chalk.gray("GPU VRAM:")}     ${chalk.cyan(hw.vram.usableGB + " GB")} usable / ${hw.vram.totalGB} GB total`
    );
    console.log(
      `  ${chalk.gray("System RAM:")}   ${chalk.cyan(hw.ram.usableGB + " GB")} usable / ${hw.ram.totalGB} GB total`
    );
  } else {
    console.log(`  ${chalk.gray("Architecture:")} ${chalk.yellow("CPU / System RAM Mode")} ${chalk.gray("(No discrete GPU detected)")}`);
    console.log(
      `  ${chalk.gray("System RAM:")}   ${chalk.cyan(hw.ram.usableGB + " GB")} usable / ${hw.ram.totalGB} GB total`
    );
  }
  console.log();
}

// src/commands/profile.ts
function profileCommand() {
  const hw = detectHardware();
  renderCleanHardwareProfile(hw);
}

// src/commands/recommend.ts
import ora from "ora";
import chalk2 from "chalk";

// src/api/normalizer.ts
function normalizeRawModel(raw, index) {
  const rank = raw.rank || index + 1;
  const id = raw.id || raw.slug || `model-${rank}`;
  const name = raw.name || id;
  const provider = raw.creator || raw.provider || "AI";
  const parameters = raw.parameters || (raw.parametersInB ? `${raw.parametersInB}B` : "");
  let recommendedQuant = "Q4_K_M";
  if (typeof raw.recommendedQuant === "string") {
    recommendedQuant = raw.recommendedQuant;
  } else if (raw.recommendedQuant && typeof raw.recommendedQuant === "object") {
    recommendedQuant = raw.recommendedQuant.format || "Q4_K_M";
  }
  const weightsGB = raw.memoryFootprint?.modelWeightsGB ?? raw.recommendedQuant?.weightsMemoryGB ?? raw.recommendedQuant?.fileSizeBytesGb ?? 0;
  const kvCacheGB = raw.memoryFootprint?.kvCacheGB ?? raw.recommendedQuant?.kvCacheMemoryGB ?? 0;
  const overheadGB = raw.memoryFootprint?.overheadGB ?? 0.5;
  const totalRequiredGB = raw.memoryFootprint?.totalRequiredGB ?? raw.recommendedQuant?.totalRequiredMemoryGB ?? Number((weightsGB + kvCacheGB + overheadGB).toFixed(1));
  const headroomGB = raw.memoryFootprint?.headroomGB;
  let fitTier = "good";
  if (raw.fitTier) {
    fitTier = raw.fitTier;
  } else if (raw.recommendedQuant?.fitStatus) {
    fitTier = raw.recommendedQuant.fitStatus;
  }
  const scores = {};
  if (raw.scores && typeof raw.scores === "object") {
    scores.overall = raw.scores.overall;
    scores.coding = raw.scores.coding;
    scores.reasoning = raw.scores.reasoning;
    scores.mmlu = raw.scores.mmlu;
    scores.math = raw.scores.math;
  } else if (raw.benchmark && typeof raw.benchmark === "object") {
    if (raw.benchmark.category === "coding") scores.coding = raw.benchmark.score;
    else if (raw.benchmark.category === "reasoning") scores.reasoning = raw.benchmark.score;
    else scores.overall = raw.benchmark.score;
  }
  const quantOptions = [];
  const rawQuants = raw.quantOptions || raw.availableQuants || [];
  for (const q of rawQuants) {
    quantOptions.push({
      quant: q.quant || q.format || "Q4_K_M",
      vramRequiredGB: q.vramRequiredGB ?? q.totalRequiredMemoryGB ?? 0,
      qualityRetention: q.qualityRetention,
      fits: q.fits ?? q.fitsInVram ?? q.fitStatus === "vram",
      downloadUrl: q.downloadUrl
    });
  }
  const ollamaCommand = raw.ollamaCommand || null;
  const modelId = raw.id || raw.slug || name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const websiteUrl = raw.websiteUrl && raw.websiteUrl.includes("whichllmmodel.com/models/") ? raw.websiteUrl : `https://www.whichllmmodel.com/models/${modelId}`;
  const huggingfaceUrl = raw.huggingfaceUrl || null;
  return {
    rank,
    id,
    name,
    provider,
    parameters,
    recommendedQuant,
    fitTier,
    fitMessage: raw.fitMessage,
    memoryFootprint: {
      weightsGB,
      kvCacheGB,
      overheadGB,
      totalRequiredGB,
      headroomGB
    },
    scores,
    quantOptions,
    ollamaCommand,
    websiteUrl,
    huggingfaceUrl,
    description: raw.description,
    tags: raw.tags
  };
}

// src/api/client.ts
var PRODUCTION_API_URL = "https://www.whichllmmodel.com/api/cli/recommend";
async function fetchRecommendations(hardware, context = "32k", cpuOffload = true, sortBy = "largest_vram") {
  const apiUrl = process.env.WHICH_MODEL_API_URL || PRODUCTION_API_URL;
  const effectiveCpuOffload = hardware.type === "unified_memory" ? false : cpuOffload;
  const payload = {
    hardwareType: hardware.type,
    name: hardware.name,
    totalUnifiedMemoryGB: hardware.unifiedMemory ? hardware.unifiedMemory.totalGB : null,
    usableUnifiedMemoryGB: hardware.unifiedMemory ? hardware.unifiedMemory.usableGB : null,
    totalVramGB: hardware.vram ? hardware.vram.totalGB : null,
    usableVramGB: hardware.vram ? hardware.vram.usableGB : null,
    totalRamGB: hardware.ram.totalGB,
    usableRamGB: hardware.ram.usableGB,
    context,
    cpuOffload: effectiveCpuOffload,
    sortBy
  };
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6e3);
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "User-Agent": "whichllmmodel-cli/1.0.0",
        Accept: "application/json"
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (response.ok) {
      const rawData = await response.json();
      if (rawData && Array.isArray(rawData.recommendations)) {
        const normalized = rawData.recommendations.map(normalizeRawModel);
        return {
          success: true,
          recommendations: normalized.slice(0, 3),
          sourceUrl: apiUrl,
          payload
        };
      }
    }
    return {
      success: false,
      recommendations: [],
      error: `Server responded with ${response.status} ${response.statusText}`,
      sourceUrl: apiUrl,
      payload
    };
  } catch (err) {
    return {
      success: false,
      recommendations: [],
      error: err.message || "Connection failed",
      sourceUrl: apiUrl,
      payload
    };
  }
}

// src/types.ts
var SORT_STRATEGY_LABELS = {
  largest_vram: "largest-vram",
  top_coding: "coding",
  highest_params: "high-params",
  highest_quality: "high-quant"
};
function normalizeSortStrategy(input) {
  if (!input) return "largest_vram";
  const val = input.toLowerCase().trim().replace(/_/g, "-");
  switch (val) {
    case "coding":
      return "top_coding";
    case "high-params":
    case "highest-params":
    case "params":
      return "highest_params";
    case "high-quant":
    case "highest-quant":
    case "quant":
      return "highest_quality";
    case "largest-vram":
    case "vram":
    default:
      return "largest_vram";
  }
}

// src/commands/recommend.ts
async function recommendCommand(options = {}) {
  const context = options.context || "32k";
  const cpuOffload = options.cpuOffload ?? true;
  const sortBy = normalizeSortStrategy(options.sort);
  const hardware = detectHardware();
  console.log();
  console.log(
    `${chalk2.bold.hex("#6366F1")("whichllmmodel")} \u2022 ${formatHardwareSummaryLine(hardware)}`
  );
  console.log(
    chalk2.gray(
      `Context: ${chalk2.white(context)} \u2022 CPU Offload: ${hardware.type === "unified_memory" ? "N/A" : cpuOffload ? "Enabled" : "Disabled"} \u2022 Sorted by: ${chalk2.white(SORT_STRATEGY_LABELS[sortBy])}`
    )
  );
  console.log();
  const spinner = ora({
    text: "Finding optimal models...",
    color: "cyan"
  }).start();
  const result = await fetchRecommendations(
    hardware,
    context,
    cpuOffload,
    sortBy
  );
  spinner.stop();
  if (!result.success) {
    console.log(chalk2.yellow("\u2715 Server is currently not responding."));
    console.log(
      chalk2.gray(
        `  Unable to reach recommendation service (${result.sourceUrl || "https://www.whichllmmodel.com/api/cli/recommend"}).`
      )
    );
    console.log(chalk2.gray("  Please check your connection or try again shortly."));
    console.log();
    return;
  }
  renderCleanRecommendations(result.recommendations);
}

// src/index.ts
function loadDotenv() {
  const envPath = path.resolve(process.cwd(), ".env");
  if (fs.existsSync(envPath)) {
    try {
      const content = fs.readFileSync(envPath, "utf-8");
      for (const line of content.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const eqIdx = trimmed.indexOf("=");
        if (eqIdx !== -1) {
          const key = trimmed.slice(0, eqIdx).trim();
          const val = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
          if (key && process.env[key] === void 0) {
            process.env[key] = val;
          }
        }
      }
    } catch {
    }
  }
}
loadDotenv();
var program = new Command();
program.name("whichllmmodel").description("Hardware-aware CLI for recommending and running local LLMs").version("1.0.0");
program.command("profile").description(
  "Inspect and display local hardware (name, architecture, total vs usable memory)"
).action(() => {
  profileCommand();
});
program.command("recommend", { isDefault: true }).alias("rec").description(
  "Recommend top 3 local LLMs matching hardware, context, and sorting"
).option("-c, --context <size>", "Context window size (e.g. 8k, 32k, 128k)", "32k").option(
  "--cpu-offload",
  "Enable CPU RAM offloading for discrete GPUs (default: true)",
  true
).option(
  "--no-cpu-offload",
  "Disable CPU RAM offloading (strictly require GPU VRAM fit)"
).option(
  "-s, --sort <mode>",
  'Sort strategy: largest-vram, coding, high-params, high-quant (default: "largest-vram")',
  "largest-vram"
).action(async (options) => {
  await recommendCommand({
    context: options.context,
    cpuOffload: options.cpuOffload,
    sort: options.sort
  });
});
program.parse(process.argv);
