import {
  HardwareSpec,
  RecommendedModel,
  RecommendApiResponse,
  QuantFitDetail,
  SortStrategy,
} from '../types.js';

interface RawModelCatalogItem {
  id: string;
  name: string;
  provider: string;
  parametersInB: number;
  slug: string;
  ollamaCommand: string;
  huggingfaceUrl: string;
  baseWeightGB: number;
  benchmark: {
    category: 'coding' | 'reasoning' | 'general';
    name: string;
    score: number;
  };
}

const MODEL_CATALOG: RawModelCatalogItem[] = [
  {
    id: 'qwen2.5-coder:14b',
    name: 'Qwen 2.5 Coder 14B',
    provider: 'Qwen',
    parametersInB: 14.7,
    slug: 'Qwen-2.5-Coder-14B',
    ollamaCommand: 'ollama run qwen2.5-coder:14b',
    huggingfaceUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF',
    baseWeightGB: 8.9,
    benchmark: { category: 'coding', name: 'swe-bench-pro', score: 64.6 },
  },
  {
    id: 'deepseek-r1:14b',
    name: 'DeepSeek R1 Distill Qwen 14B',
    provider: 'DeepSeek',
    parametersInB: 14.7,
    slug: 'DeepSeek-R1-Distill-Qwen-14B',
    ollamaCommand: 'ollama run deepseek-r1:14b',
    huggingfaceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF',
    baseWeightGB: 8.9,
    benchmark: { category: 'reasoning', name: 'gpqa-diamond', score: 59.1 },
  },
  {
    id: 'qwen2.5-coder:32b',
    name: 'Qwen 2.5 Coder 32B',
    provider: 'Qwen',
    parametersInB: 32.5,
    slug: 'Qwen-2.5-Coder-32B',
    ollamaCommand: 'ollama run qwen2.5-coder:32b',
    huggingfaceUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF',
    baseWeightGB: 19.8,
    benchmark: { category: 'coding', name: 'swe-bench-pro', score: 71.4 },
  },
  {
    id: 'deepseek-r1:32b',
    name: 'DeepSeek R1 Distill Qwen 32B',
    provider: 'DeepSeek',
    parametersInB: 32.5,
    slug: 'DeepSeek-R1-Distill-Qwen-32B',
    ollamaCommand: 'ollama run deepseek-r1:32b',
    huggingfaceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-GGUF',
    baseWeightGB: 19.8,
    benchmark: { category: 'reasoning', name: 'gpqa-diamond', score: 62.1 },
  },
  {
    id: 'llama3.3:70b',
    name: 'Llama 3.3 70B',
    provider: 'Meta',
    parametersInB: 70.0,
    slug: 'Llama-3.3-70B-Instruct',
    ollamaCommand: 'ollama run llama3.3:70b',
    huggingfaceUrl: 'https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct',
    baseWeightGB: 41.5,
    benchmark: { category: 'general', name: 'mmlu-pro', score: 68.3 },
  },
  {
    id: 'llama3.1:8b',
    name: 'Llama 3.1 8B',
    provider: 'Meta',
    parametersInB: 8.0,
    slug: 'Llama-3.1-8B-Instruct',
    ollamaCommand: 'ollama run llama3.1:8b',
    huggingfaceUrl: 'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF',
    baseWeightGB: 4.8,
    benchmark: { category: 'general', name: 'mmlu-pro', score: 48.3 },
  },
  {
    id: 'qwen2.5-coder:7b',
    name: 'Qwen 2.5 Coder 7B',
    provider: 'Qwen',
    parametersInB: 7.6,
    slug: 'Qwen-2.5-Coder-7B',
    ollamaCommand: 'ollama run qwen2.5-coder:7b',
    huggingfaceUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
    baseWeightGB: 4.7,
    benchmark: { category: 'coding', name: 'swe-bench-pro', score: 55.4 },
  },
  {
    id: 'llama3.2:3b',
    name: 'Llama 3.2 3B',
    provider: 'Meta',
    parametersInB: 3.2,
    slug: 'Llama-3.2-3B-Instruct',
    ollamaCommand: 'ollama run llama3.2:3b',
    huggingfaceUrl: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF',
    baseWeightGB: 2.0,
    benchmark: { category: 'general', name: 'mmlu-pro', score: 37.2 },
  },
  {
    id: 'qwen2.5:3b',
    name: 'Qwen 2.5 3B',
    provider: 'Qwen',
    parametersInB: 3.1,
    slug: 'Qwen-2.5-3B-Instruct',
    ollamaCommand: 'ollama run qwen2.5:3b',
    huggingfaceUrl: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF',
    baseWeightGB: 1.9,
    benchmark: { category: 'general', name: 'mmlu-pro', score: 38.6 },
  },
];

function parseContextTokens(ctx: string): number {
  const lower = ctx.toLowerCase().trim();
  if (lower.endsWith('k')) {
    const num = parseFloat(lower.replace('k', ''));
    return !isNaN(num) ? Math.round(num * 1024) : 32768;
  }
  const parsed = parseInt(ctx, 10);
  return !isNaN(parsed) && parsed > 0 ? parsed : 32768;
}

export function generateDummyRecommendations(
  hardware: HardwareSpec,
  contextStr: string = '32k',
  cpuOffload: boolean = true,
  sortBy: SortStrategy = 'largest_vram'
): RecommendApiResponse {
  const queryContextTokens = parseContextTokens(contextStr);

  // Available memory calculation
  let vramBudgetGB = 0;
  let isUnified = false;

  if (hardware.type === 'unified_memory' && hardware.unifiedMemory) {
    vramBudgetGB = hardware.unifiedMemory.usableGB;
    isUnified = true;
  } else if (hardware.type === 'gpu' && hardware.vram) {
    vramBudgetGB = hardware.vram.usableGB;
  } else {
    // RAM mode
    vramBudgetGB = 0;
  }

  const ramBudgetGB = Math.max(2, hardware.ram.usableGB);
  const totalAcceleratedGB = isUnified ? vramBudgetGB : vramBudgetGB > 0 ? vramBudgetGB : ramBudgetGB * 0.75;

  // Approximate KV Cache based on tokens: 32k tokens = ~0.95 GB
  const kvCacheGB = Number(((queryContextTokens / 32768) * 0.95).toFixed(2));

  const candidates: Array<{
    item: RawModelCatalogItem;
    recommendedQuant: RecommendedModel['recommendedQuant'];
    availableQuants: QuantFitDetail[];
  }> = [];

  const quantMultipliers: Record<string, number> = {
    Q4_K_M: 0.58,
    Q5_K_M: 0.70,
    Q8_0: 1.05,
    fp16: 2.0,
  };

  for (const item of MODEL_CATALOG) {
    const availableQuants: QuantFitDetail[] = [];

    for (const [format, mult] of Object.entries(quantMultipliers)) {
      const weightGB = Number((item.baseWeightGB * mult).toFixed(2));
      const totalReq = Number((weightGB + kvCacheGB).toFixed(2));

      let fitStatus: QuantFitDetail['fitStatus'] = 'oom';
      let fitsInVram = false;

      if (isUnified) {
        if (totalReq <= vramBudgetGB) {
          fitStatus = 'vram';
          fitsInVram = true;
        } else if (totalReq <= hardware.ram.usableGB) {
          fitStatus = 'ram';
          fitsInVram = false;
        } else {
          fitStatus = 'oom';
        }
      } else if (vramBudgetGB > 0) {
        if (totalReq <= vramBudgetGB) {
          fitStatus = 'vram';
          fitsInVram = true;
        } else if (cpuOffload && totalReq <= vramBudgetGB + ramBudgetGB * 0.6) {
          fitStatus = 'offload';
          fitsInVram = false;
        } else {
          fitStatus = 'oom';
        }
      } else {
        if (totalReq <= ramBudgetGB * 0.75) {
          fitStatus = 'ram';
          fitsInVram = false;
        } else {
          fitStatus = 'oom';
        }
      }

      availableQuants.push({
        format,
        fileSizeBytesGb: weightGB,
        totalRequiredMemoryGB: totalReq,
        fitStatus,
        fitsInVram,
        downloadUrl: `${item.huggingfaceUrl}/resolve/main/${item.slug}.${format}.gguf`,
      });
    }

    // Determine primary recommended quant (best quant that fits in VRAM, or best offload)
    const validQuants = availableQuants.filter((q) => q.fitStatus !== 'oom');
    if (validQuants.length === 0) continue;

    // Prefer Q5_K_M or Q4_K_M that fits in VRAM, else best available
    let primaryQuant =
      validQuants.find((q) => q.format === 'Q5_K_M' && q.fitsInVram) ||
      validQuants.find((q) => q.format === 'Q4_K_M' && q.fitsInVram) ||
      validQuants.find((q) => q.fitsInVram) ||
      validQuants[0];

    const recFitStatus: 'vram' | 'offload' | 'ram' =
      primaryQuant.fitStatus === 'oom' ? 'ram' : primaryQuant.fitStatus;

    candidates.push({
      item,
      recommendedQuant: {
        format: primaryQuant.format,
        fileSizeBytesGb: primaryQuant.fileSizeBytesGb,
        totalRequiredMemoryGB: primaryQuant.totalRequiredMemoryGB,
        weightsMemoryGB: primaryQuant.fileSizeBytesGb || 0,
        kvCacheMemoryGB: kvCacheGB,
        fitStatus: recFitStatus,
      },
      availableQuants,
    });
  }

  // Sort candidates according to sortBy
  switch (sortBy) {
    case 'largest_vram':
      candidates.sort((a, b) => {
        // Models that fit in VRAM first
        const aVram = a.recommendedQuant.fitStatus === 'vram' ? 1 : 0;
        const bVram = b.recommendedQuant.fitStatus === 'vram' ? 1 : 0;
        if (aVram !== bVram) return bVram - aVram;
        return b.recommendedQuant.totalRequiredMemoryGB - a.recommendedQuant.totalRequiredMemoryGB;
      });
      break;

    case 'top_coding':
      candidates.sort((a, b) => {
        const aScore = a.item.benchmark.category === 'coding' ? a.item.benchmark.score : 0;
        const bScore = b.item.benchmark.category === 'coding' ? b.item.benchmark.score : 0;
        return bScore - aScore;
      });
      break;

    case 'highest_params':
      candidates.sort((a, b) => b.item.parametersInB - a.item.parametersInB);
      break;

    case 'highest_quality':
      candidates.sort((a, b) => {
        const quantRanks: Record<string, number> = { fp16: 4, Q8_0: 3, Q5_K_M: 2, Q4_K_M: 1 };
        const aRank = quantRanks[a.recommendedQuant.format] || 0;
        const bRank = quantRanks[b.recommendedQuant.format] || 0;
        if (aRank !== bRank) return bRank - aRank;
        return b.item.benchmark.score - a.item.benchmark.score;
      });
      break;
  }

  const top3 = candidates.slice(0, 3).map((c, index): RecommendedModel => ({
    rank: index + 1,
    name: c.item.name,
    provider: c.item.provider,
    parametersInB: c.item.parametersInB,
    websiteUrl: `https://www.whichllmmodel.com/models/${c.item.slug.toLowerCase()}`,
    huggingfaceUrl: c.item.huggingfaceUrl,
    ollamaCommand: c.item.ollamaCommand,
    recommendedQuant: c.recommendedQuant,
    availableQuants: c.availableQuants,
    benchmark: c.item.benchmark,
  }));

  return {
    success: true,
    meta: {
      queryContextTokens,
      sortBy,
      totalModelsEvaluated: MODEL_CATALOG.length,
      returnedCount: top3.length,
    },
    recommendations: top3,
  };
}
