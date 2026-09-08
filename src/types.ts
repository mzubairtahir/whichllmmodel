export type HardwareType = 'unified_memory' | 'gpu' | 'cpu_ram';

export interface MemoryMetric {
  totalBytes: number;
  totalGB: number; // Physical installed memory in GB
  usableBytes: number;
  usableGB: number; // Live available memory in GB
}

export interface HardwareSpec {
  type: HardwareType;
  platform: NodeJS.Platform;
  arch: string;
  name: string;
  cpuName: string;
  gpuName?: string;
  vram?: MemoryMetric;
  ram: MemoryMetric;
  unifiedMemory?: MemoryMetric & { isUnified: boolean };
}

export interface ProfileCliOptions {
  open?: boolean;
}
