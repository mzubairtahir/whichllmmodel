import { HardwareSpec } from '../types.js';

export interface WebFinderUrlOptions {
  memoryMode?: 'available' | 'installed';
  baseUrl?: string;
}

/**
 * Determine the base URL for WhichLLM Local Finder:
 * 1. options.baseUrl (explicit argument)
 * 2. process.env.WHICH_MODEL_WEB_URL
 * 3. process.env.WHICH_MODEL_BASE_URL (appends /app/text/local)
 * 4. process.env.WHICH_MODEL_API_URL (extracts origin and appends /app/text/local)
 * 5. process.env.NODE_ENV === 'development' (defaults to http://localhost:3000/app/text/local)
 * 6. Default (production): https://www.whichllmmodel.com/app/text/local
 */
export function getBaseFinderUrl(explicitUrl?: string): string {
  if (explicitUrl) return explicitUrl;

  // 1. Explicit web URL via env
  if (process.env.WHICH_MODEL_WEB_URL) {
    const val = process.env.WHICH_MODEL_WEB_URL.trim().replace(/\/+$/, '');
    return val.includes('/app/text/local') ? val : `${val}/app/text/local`;
  }

  // 2. Base host URL via env (e.g. http://localhost:3000)
  if (process.env.WHICH_MODEL_BASE_URL) {
    const val = process.env.WHICH_MODEL_BASE_URL.trim().replace(/\/+$/, '');
    return `${val}/app/text/local`;
  }

  // 3. Derived from local API URL if set in .env (e.g. http://localhost:3000/api/cli/recommend)
  if (process.env.WHICH_MODEL_API_URL) {
    try {
      const parsed = new URL(process.env.WHICH_MODEL_API_URL.trim());
      return `${parsed.origin}/app/text/local`;
    } catch {
      // Fallback
    }
  }

  // 4. Dev mode fallback
  if (process.env.NODE_ENV === 'development') {
    return 'http://localhost:3000/app/text/local';
  }

  // 5. Default production
  return 'https://www.whichllmmodel.com/app/text/local';
}

/**
 * Generate a direct URL to WhichLLM Local Finder pre-filled with detected hardware specs
 * Example: https://www.whichllmmodel.com/app/text/local?vram=10.4&ram=22.1&memory_mode=available
 */
export function buildWebFinderUrl(
  hw: HardwareSpec,
  options: WebFinderUrlOptions = {}
): string {
  const baseUrl = getBaseFinderUrl(options.baseUrl);
  const memoryMode = options.memoryMode || 'available';
  const params = new URLSearchParams();

  if (hw.type === 'unified_memory' && hw.unifiedMemory) {
    const memVal =
      memoryMode === 'available'
        ? hw.unifiedMemory.usableGB
        : hw.unifiedMemory.totalGB;
    params.set('unified_mem', memVal.toFixed(1));
    params.set('memory_mode', memoryMode);
  } else if (hw.type === 'gpu' && hw.vram) {
    const vramVal =
      memoryMode === 'available' ? hw.vram.usableGB : hw.vram.totalGB;
    const ramVal =
      memoryMode === 'available' ? hw.ram.usableGB : hw.ram.totalGB;
    params.set('vram', vramVal.toFixed(1));
    params.set('ram', ramVal.toFixed(1));
    params.set('memory_mode', memoryMode);
  } else {
    const ramVal =
      memoryMode === 'available' ? hw.ram.usableGB : hw.ram.totalGB;
    params.set('vram', '0');
    params.set('ram', ramVal.toFixed(1));
    params.set('memory_mode', memoryMode);
  }

  return `${baseUrl}?${params.toString()}`;
}
