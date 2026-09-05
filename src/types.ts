export type HardwareType = 'unified_memory' | 'gpu' | 'cpu_ram';

export interface MemoryMetric {
  totalBytes: number;
  totalGB: number;
  usableBytes: number;
  usableGB: number;
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

export type SortStrategy =
  | 'largest_vram'
  | 'top_coding'
  | 'highest_params'
  | 'highest_quality';

export const SORT_STRATEGY_LABELS: Record<SortStrategy, string> = {
  largest_vram: 'largest-vram',
  top_coding: 'coding',
  highest_params: 'high-params',
  highest_quality: 'high-quant',
};

/**
 * Normalize sort input to canonical SortStrategy
 * Supported options: largest-vram, coding, high-params, high-quant
 */
export function normalizeSortStrategy(input?: string): SortStrategy {
  if (!input) return 'largest_vram';
  const val = input.toLowerCase().trim().replace(/_/g, '-');

  switch (val) {
    case 'coding':
      return 'top_coding';

    case 'high-params':
    case 'highest-params':
    case 'params':
      return 'highest_params';

    case 'high-quant':
    case 'highest-quant':
    case 'quant':
      return 'highest_quality';

    case 'largest-vram':
    case 'vram':
    default:
      return 'largest_vram';
  }
}

// ==========================================
// Unified Normalized Model Structure
// ==========================================

export interface QuantOptionUnified {
  quant: string; // e.g. "Q4_K_M", "Q8_0", "FP16"
  vramRequiredGB: number;
  qualityRetention?: string;
  fits: boolean;
  downloadUrl?: string | null;
}

export interface ModelUnified {
  rank: number;
  id: string;
  name: string;
  provider: string; // creator or provider
  parameters: string; // e.g. "31.6B"
  recommendedQuant: string; // e.g. "Q4_K_M"
  fitTier: string; // 'optimal' | 'good' | 'tight' | 'cpu_offload' | 'vram' | 'offload' | 'ram'
  fitMessage?: string;
  memoryFootprint: {
    weightsGB: number;
    kvCacheGB: number;
    overheadGB: number;
    totalRequiredGB: number;
    headroomGB?: number;
  };
  scores: {
    overall?: number;
    coding?: number;
    reasoning?: number;
    mmlu?: number;
    math?: number;
  };
  quantOptions: QuantOptionUnified[];
  ollamaCommand?: string | null;
  websiteUrl: string;
  huggingfaceUrl?: string | null;
  description?: string;
  tags?: string[];
}

export interface RecommendApiPayload {
  hardwareType: HardwareType;
  name: string;
  totalUnifiedMemoryGB: number | null;
  usableUnifiedMemoryGB: number | null;
  totalVramGB: number | null;
  usableVramGB: number | null;
  totalRamGB: number;
  usableRamGB: number;
  context: string;
  cpuOffload: boolean;
  sortBy: SortStrategy;
}

export interface RecommendApiResponse {
  success: boolean;
  recommendations: any[];
  meta?: any;
  serverVersion?: string;
  error?: string;
}

export interface RecommendCliOptions {
  context?: string;
  cpuOffload?: boolean;
  sort?: string;
}
