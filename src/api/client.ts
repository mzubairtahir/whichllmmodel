import {
  HardwareSpec,
  ModelUnified,
  RecommendApiPayload,
  SortStrategy,
} from '../types.js';
import { generateDummyRecommendations } from './mock.js';
import { normalizeRawModel } from './normalizer.js';

const REMOTE_API_URL = 'https://www.whichllmmodel.com/api/cli/recommend';
const LOCAL_API_URL = 'http://localhost:3000/api/cli/recommend';

export interface FetchRecommendationsResult {
  recommendations: ModelUnified[];
  isMock: boolean;
  sourceUrl: string;
  payload: RecommendApiPayload;
}

/**
 * Query the remote recommendation API with fallback to local dev server (http://localhost:3000)
 * and dynamic simulation engine if offline.
 */
export async function fetchRecommendations(
  hardware: HardwareSpec,
  context: string = '32k',
  cpuOffload: boolean = true,
  sortBy: SortStrategy = 'largest_vram'
): Promise<FetchRecommendationsResult> {
  const customUrl = process.env.WHICH_MODEL_API_URL;
  const targetUrls = customUrl
    ? [customUrl]
    : [REMOTE_API_URL, LOCAL_API_URL];

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

  const bodyStr = JSON.stringify(payload);

  for (const url of targetUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'whichllmmodel-cli/1.0.0',
          Accept: 'application/json',
        },
        body: bodyStr,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const rawData = await response.json();
        if (rawData && Array.isArray(rawData.recommendations)) {
          const normalized = rawData.recommendations.map(normalizeRawModel);
          return {
            recommendations: normalized.slice(0, 3),
            isMock: false,
            sourceUrl: url,
            payload,
          };
        }
      }
    } catch {
      // Continue to next endpoint (e.g. try localhost if remote 404s)
    }
  }

  // Fallback to client simulation engine
  const mockData = generateDummyRecommendations(
    hardware,
    context,
    effectiveCpuOffload,
    sortBy
  );

  return {
    recommendations: mockData.recommendations.map(normalizeRawModel),
    isMock: true,
    sourceUrl: 'client-simulation',
    payload,
  };
}
