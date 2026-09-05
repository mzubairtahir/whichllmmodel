import {
  HardwareSpec,
  ModelUnified,
  RecommendApiPayload,
  SortStrategy,
} from '../types.js';
import { normalizeRawModel } from './normalizer.js';

const PRODUCTION_API_URL = 'https://www.whichllmmodel.com/api/cli/recommend';

export interface FetchRecommendationsResult {
  success: boolean;
  recommendations: ModelUnified[];
  sourceUrl?: string;
  error?: string;
  payload: RecommendApiPayload;
}

/**
 * Query the recommendation API.
 * Defaults strictly to production (https://www.whichllmmodel.com/api/cli/recommend).
 * Can be overridden for local development via WHICH_MODEL_API_URL.
 */
export async function fetchRecommendations(
  hardware: HardwareSpec,
  context: string = '32k',
  cpuOffload: boolean = true,
  sortBy: SortStrategy = 'largest_vram'
): Promise<FetchRecommendationsResult> {
  const apiUrl = process.env.WHICH_MODEL_API_URL || PRODUCTION_API_URL;

  const effectiveCpuOffload =
    hardware.type === 'unified_memory' ? false : cpuOffload;

  const payload: RecommendApiPayload = {
    hardwareType: hardware.type,
    name: hardware.name,
    totalUnifiedMemoryGB: hardware.unifiedMemory
      ? hardware.unifiedMemory.totalGB
      : null,
    usableUnifiedMemoryGB: hardware.unifiedMemory
      ? hardware.unifiedMemory.usableGB
      : null,
    totalVramGB: hardware.vram ? hardware.vram.totalGB : null,
    usableVramGB: hardware.vram ? hardware.vram.usableGB : null,
    totalRamGB: hardware.ram.totalGB,
    usableRamGB: hardware.ram.usableGB,
    context,
    cpuOffload: effectiveCpuOffload,
    sortBy,
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'whichllmmodel-cli/1.0.0',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
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
          payload,
        };
      }
    }

    return {
      success: false,
      recommendations: [],
      error: `Server responded with ${response.status} ${response.statusText}`,
      sourceUrl: apiUrl,
      payload,
    };
  } catch (err: any) {
    return {
      success: false,
      recommendations: [],
      error: err.message || 'Connection failed',
      sourceUrl: apiUrl,
      payload,
    };
  }
}
