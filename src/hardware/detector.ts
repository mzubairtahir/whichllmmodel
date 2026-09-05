import os from 'os';
import { execSync } from 'child_process';
import { HardwareSpec, MemoryMetric } from '../types.js';
import { getSystemRam } from './memory.js';
import { detectNvidiaGpu } from './nvidia.js';
import { detectAppleSilicon } from './apple.js';

/**
 * Attempt to detect GPU controller name on Windows if nvidia-smi is not available
 */
function detectWindowsGpuName(): string | null {
  if (process.platform !== 'win32') return null;
  try {
    const output = execSync(
      'powershell -NoProfile -NonInteractive -Command "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"',
      {
        encoding: 'utf-8',
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 2500,
      }
    ).trim();
    return output || null;
  } catch {
    return null;
  }
}

/**
 * Comprehensive automatic hardware detection
 */
export function detectHardware(): HardwareSpec {
  const platform = process.platform;
  const arch = process.arch;
  const cpus = os.cpus();
  const cpuName = cpus.length > 0 ? cpus[0].model.trim() : 'Unknown CPU';
  const ram: MemoryMetric = getSystemRam();

  // 1. Check for Apple Silicon Unified Memory
  const appleSilicon = detectAppleSilicon();
  if (appleSilicon) {
    return {
      type: 'unified_memory',
      platform,
      arch,
      name: appleSilicon.chipName,
      cpuName: appleSilicon.chipName,
      gpuName: `${appleSilicon.chipName} GPU`,
      unifiedMemory: appleSilicon.unifiedMemory,
      ram,
    };
  }

  // 2. Check for NVIDIA Discrete GPU
  const nvidiaGpu = detectNvidiaGpu();
  if (nvidiaGpu) {
    return {
      type: 'gpu',
      platform,
      arch,
      name: nvidiaGpu.name,
      cpuName,
      gpuName: nvidiaGpu.name,
      vram: nvidiaGpu.vram,
      ram,
    };
  }

  // 3. Fallback: CPU / System RAM mode
  const winGpuName = detectWindowsGpuName();

  return {
    type: 'cpu_ram',
    platform,
    arch,
    name: winGpuName || cpuName,
    cpuName,
    gpuName: winGpuName || undefined,
    ram,
  };
}
