import { execSync } from 'child_process';
import { MemoryMetric } from '../types.js';
import { bytesToGB } from './memory.js';

export interface NvidiaGpuInfo {
  name: string;
  vram: MemoryMetric;
}

/**
 * Detect NVIDIA GPUs using `nvidia-smi` (pure child_process, no native C++ bindings)
 */
export function detectNvidiaGpu(): NvidiaGpuInfo | null {
  try {
    const rawOutput = execSync(
      'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits',
      {
        encoding: 'utf-8',
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 2500,
      }
    ).trim();

    if (!rawOutput) return null;

    // Handle multiple GPUs if present: pick the one with highest total VRAM
    const lines = rawOutput.split(/\r?\n/).filter(line => line.trim().length > 0);
    const gpus: NvidiaGpuInfo[] = [];

    for (const line of lines) {
      const parts = line.split(',').map(p => p.trim());
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
              usableGB: bytesToGB(usableBytes),
            },
          });
        }
      }
    }

    if (gpus.length === 0) return null;

    // Pick GPU with the largest VRAM
    gpus.sort((a, b) => b.vram.totalBytes - a.vram.totalBytes);
    return gpus[0];
  } catch {
    return null;
  }
}
