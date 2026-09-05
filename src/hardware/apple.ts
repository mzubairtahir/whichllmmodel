import { execSync } from 'child_process';
import { MemoryMetric } from '../types.js';
import { bytesToGB } from './memory.js';

export interface AppleSiliconInfo {
  chipName: string;
  unifiedMemory: MemoryMetric & { isUnified: boolean };
}

/**
 * Detect Apple Silicon unified memory on macOS using sysctl
 */
export function detectAppleSilicon(): AppleSiliconInfo | null {
  if (process.platform !== 'darwin') return null;

  try {
    let chipName = '';
    try {
      chipName = execSync('sysctl -n machdep.cpu.brand_string', {
        encoding: 'utf-8',
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 2000,
      }).trim();
    } catch {
      // Fallback
    }

    const isArm64 = process.arch === 'arm64';
    const isAppleChip = /Apple\s+M\d/i.test(chipName) || /Apple/i.test(chipName) || isArm64;

    if (!isAppleChip) return null;

    if (!chipName) {
      chipName = 'Apple Silicon';
    }

    const memsizeStr = execSync('sysctl -n hw.memsize', {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 2000,
    }).trim();

    const totalBytes = parseInt(memsizeStr, 10);
    if (isNaN(totalBytes) || totalBytes <= 0) return null;

    // Standard 75% macOS VRAM allocation ceiling for Unified Memory
    const usableBytes = Math.floor(totalBytes * 0.75);

    return {
      chipName,
      unifiedMemory: {
        isUnified: true,
        totalBytes,
        totalGB: bytesToGB(totalBytes),
        usableBytes,
        usableGB: bytesToGB(usableBytes),
      },
    };
  } catch {
    return null;
  }
}
