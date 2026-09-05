import os from 'os';
import { MemoryMetric } from '../types.js';

/**
 * Convert bytes to Gigabytes (binary / GiB standard used by hardware vendors)
 */
export function bytesToGB(bytes: number): number {
  return Number((bytes / (1024 * 1024 * 1024)).toFixed(2));
}

/**
 * Get system RAM details (total + usable/free)
 */
export function getSystemRam(): MemoryMetric {
  const totalBytes = os.totalmem();
  const usableBytes = os.freemem();

  return {
    totalBytes,
    totalGB: bytesToGB(totalBytes),
    usableBytes,
    usableGB: bytesToGB(usableBytes),
  };
}
