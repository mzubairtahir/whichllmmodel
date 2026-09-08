#!/usr/bin/env node

// src/index.ts
import fs from "fs";
import path from "path";
import { Command } from "commander";

// src/commands/profile.ts
import chalk2 from "chalk";
import { exec } from "child_process";

// src/hardware/detector.ts
import os3 from "os";
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
import os2 from "os";
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
    const metalCeilingBytes = Math.floor(totalBytes * 0.75);
    const liveFreeBytes = os2.freemem();
    const usableBytes = Math.min(metalCeilingBytes, liveFreeBytes);
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
  const cpus = os3.cpus();
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

// src/hardware/url.ts
function getBaseFinderUrl(explicitUrl) {
  if (explicitUrl) return explicitUrl;
  if (process.env.WHICH_MODEL_WEB_URL) {
    const val = process.env.WHICH_MODEL_WEB_URL.trim().replace(/\/+$/, "");
    return val.includes("/app/text/local") ? val : `${val}/app/text/local`;
  }
  if (process.env.WHICH_MODEL_BASE_URL) {
    const val = process.env.WHICH_MODEL_BASE_URL.trim().replace(/\/+$/, "");
    return `${val}/app/text/local`;
  }
  if (process.env.WHICH_MODEL_API_URL) {
    try {
      const parsed = new URL(process.env.WHICH_MODEL_API_URL.trim());
      return `${parsed.origin}/app/text/local`;
    } catch {
    }
  }
  if (process.env.NODE_ENV === "development") {
    return "http://localhost:3000/app/text/local";
  }
  return "https://www.whichllmmodel.com/app/text/local";
}
function buildWebFinderUrl(hw, options = {}) {
  const baseUrl = getBaseFinderUrl(options.baseUrl);
  const memoryMode = options.memoryMode || "available";
  const params = new URLSearchParams();
  if (hw.type === "unified_memory" && hw.unifiedMemory) {
    const memVal = memoryMode === "available" ? hw.unifiedMemory.usableGB : hw.unifiedMemory.totalGB;
    params.set("unified_mem", memVal.toFixed(1));
    params.set("memory_mode", memoryMode);
  } else if (hw.type === "gpu" && hw.vram) {
    const vramVal = memoryMode === "available" ? hw.vram.usableGB : hw.vram.totalGB;
    const ramVal = memoryMode === "available" ? hw.ram.usableGB : hw.ram.totalGB;
    params.set("vram", vramVal.toFixed(1));
    params.set("ram", ramVal.toFixed(1));
    params.set("memory_mode", memoryMode);
  } else {
    const ramVal = memoryMode === "available" ? hw.ram.usableGB : hw.ram.totalGB;
    params.set("vram", "0");
    params.set("ram", ramVal.toFixed(1));
    params.set("memory_mode", memoryMode);
  }
  return `${baseUrl}?${params.toString()}`;
}

// src/ui/formatters.ts
function renderCleanHardwareProfile(hw) {
  const webUrl = buildWebFinderUrl(hw, { memoryMode: "available" });
  console.log();
  console.log(
    `${chalk.bold.hex("#6366F1")("whichllmmodel")} \u2022 ${chalk.bold.white("Hardware & Memory Profiler")}`
  );
  console.log();
  console.log(chalk.bold.white("Hardware Detected:"));
  console.log(`  ${chalk.gray("\u2022 Device:")}         ${chalk.white.bold(hw.name)}`);
  console.log(
    `  ${chalk.gray("\u2022 CPU:")}            ${chalk.white(hw.cpuName)} ${chalk.gray(`(${hw.platform} ${hw.arch})`)}`
  );
  let archDesc = chalk.yellow("CPU / System RAM Mode (No discrete GPU detected)");
  if (hw.type === "unified_memory") {
    archDesc = chalk.green("Apple Silicon (Unified Memory)");
  } else if (hw.type === "gpu") {
    archDesc = chalk.green("Discrete GPU");
  }
  console.log(`  ${chalk.gray("\u2022 Architecture:")}   ${archDesc}`);
  console.log();
  console.log(chalk.bold.white("Memory Breakdown:"));
  if (hw.type === "unified_memory" && hw.unifiedMemory) {
    const totalGB = hw.unifiedMemory.totalGB;
    const usableGB = hw.unifiedMemory.usableGB;
    const pct = Math.round(usableGB / totalGB * 100);
    console.log(`  ${chalk.cyan("Physical Installed Memory:")}`);
    console.log(
      `    ${chalk.gray("\u2022 Unified Memory:")} ${chalk.white.bold(totalGB.toFixed(1) + " GB")} installed`
    );
    console.log();
    console.log(`  ${chalk.cyan("Live Available Memory (Metal VRAM Budget):")}`);
    console.log(
      `    ${chalk.gray("\u2022 Unified Memory:")} ${chalk.green.bold(usableGB.toFixed(1) + " GB")} allocatable ${chalk.gray(`(75% macOS ceiling, ~${pct}% available)`)}`
    );
    console.log(
      `    ${chalk.gray("\u2022 Free Host RAM:")}  ${chalk.white.bold(hw.ram.usableGB.toFixed(1) + " GB")} free right now`
    );
  } else if (hw.type === "gpu" && hw.vram) {
    const vramTotal = hw.vram.totalGB;
    const vramFree = hw.vram.usableGB;
    const vramPct = Math.round(vramFree / vramTotal * 100);
    const ramTotal = hw.ram.totalGB;
    const ramFree = hw.ram.usableGB;
    const ramPct = Math.round(ramFree / ramTotal * 100);
    console.log(`  ${chalk.cyan("Physical Installed Memory:")}`);
    console.log(
      `    ${chalk.gray("\u2022 GPU VRAM:")}     ${chalk.white.bold(vramTotal.toFixed(1) + " GB")} installed`
    );
    console.log(
      `    ${chalk.gray("\u2022 System RAM:")}   ${chalk.white.bold(ramTotal.toFixed(1) + " GB")} installed`
    );
    console.log();
    console.log(`  ${chalk.cyan("Live Available Memory (Ready for LLMs):")}`);
    console.log(
      `    ${chalk.gray("\u2022 GPU VRAM:")}     ${chalk.green.bold(vramFree.toFixed(1) + " GB")} available ${chalk.gray(`(${vramPct}% free)`)}`
    );
    console.log(
      `    ${chalk.gray("\u2022 System RAM:")}   ${chalk.green.bold(ramFree.toFixed(1) + " GB")} available ${chalk.gray(`(${ramPct}% free)`)}`
    );
  } else {
    const ramTotal = hw.ram.totalGB;
    const ramFree = hw.ram.usableGB;
    const ramPct = Math.round(ramFree / ramTotal * 100);
    console.log(`  ${chalk.cyan("Physical Installed Memory:")}`);
    console.log(
      `    ${chalk.gray("\u2022 System RAM:")}   ${chalk.white.bold(ramTotal.toFixed(1) + " GB")} installed`
    );
    console.log();
    console.log(`  ${chalk.cyan("Live Available Memory (Ready for LLMs):")}`);
    console.log(
      `    ${chalk.gray("\u2022 System RAM:")}   ${chalk.green.bold(ramFree.toFixed(1) + " GB")} available ${chalk.gray(`(${ramPct}% free)`)}`
    );
  }
  console.log();
  console.log(chalk.bold.white("Explore all compatible models online:"));
  console.log(chalk.cyan.underline(webUrl));
  console.log();
}

// src/commands/profile.ts
function openBrowser(url) {
  try {
    const cmd = process.platform === "darwin" ? `open "${url}"` : process.platform === "win32" ? `start "" "${url}"` : `xdg-open "${url}"`;
    exec(cmd);
  } catch {
  }
}
function profileCommand(options = {}) {
  const hardware = detectHardware();
  renderCleanHardwareProfile(hardware);
  if (options.open) {
    const webUrl = buildWebFinderUrl(hardware, { memoryMode: "available" });
    console.log(chalk2.gray("Opening WhichLLM Local Finder in browser..."));
    openBrowser(webUrl);
  }
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
program.name("whichllmmodel").description("Hardware & Memory Profiler for WhichLLM Local Finder").version("1.0.1").option("-o, --open", "Open the WhichLLM Local Finder pre-filled URL directly in your browser").action((options) => {
  profileCommand({
    open: options.open
  });
});
program.parse(process.argv);
