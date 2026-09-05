import { ModelUnified, QuantOptionUnified } from '../types.js';

export function normalizeRawModel(raw: any, index: number): ModelUnified {
  const rank = raw.rank || index + 1;
  const id = raw.id || raw.slug || `model-${rank}`;
  const name = raw.name || id;
  const provider = raw.creator || raw.provider || 'AI';
  const parameters = raw.parameters || (raw.parametersInB ? `${raw.parametersInB}B` : '');

  // Recommended Quant
  let recommendedQuant = 'Q4_K_M';
  if (typeof raw.recommendedQuant === 'string') {
    recommendedQuant = raw.recommendedQuant;
  } else if (raw.recommendedQuant && typeof raw.recommendedQuant === 'object') {
    recommendedQuant = raw.recommendedQuant.format || 'Q4_K_M';
  }

  // Memory Footprint
  const weightsGB =
    raw.memoryFootprint?.modelWeightsGB ??
    raw.recommendedQuant?.weightsMemoryGB ??
    raw.recommendedQuant?.fileSizeBytesGb ??
    0;

  const kvCacheGB =
    raw.memoryFootprint?.kvCacheGB ??
    raw.recommendedQuant?.kvCacheMemoryGB ??
    0;

  const overheadGB = raw.memoryFootprint?.overheadGB ?? 0.5;

  const totalRequiredGB =
    raw.memoryFootprint?.totalRequiredGB ??
    raw.recommendedQuant?.totalRequiredMemoryGB ??
    Number((weightsGB + kvCacheGB + overheadGB).toFixed(1));

  const headroomGB = raw.memoryFootprint?.headroomGB;

  // Fit status / tier
  let fitTier = 'good';
  if (raw.fitTier) {
    fitTier = raw.fitTier;
  } else if (raw.recommendedQuant?.fitStatus) {
    fitTier = raw.recommendedQuant.fitStatus;
  }

  // Scores
  const scores: ModelUnified['scores'] = {};
  if (raw.scores && typeof raw.scores === 'object') {
    scores.overall = raw.scores.overall;
    scores.coding = raw.scores.coding;
    scores.reasoning = raw.scores.reasoning;
    scores.mmlu = raw.scores.mmlu;
    scores.math = raw.scores.math;
  } else if (raw.benchmark && typeof raw.benchmark === 'object') {
    if (raw.benchmark.category === 'coding') scores.coding = raw.benchmark.score;
    else if (raw.benchmark.category === 'reasoning') scores.reasoning = raw.benchmark.score;
    else scores.overall = raw.benchmark.score;
  }

  // Quant options
  const quantOptions: QuantOptionUnified[] = [];
  const rawQuants = raw.quantOptions || raw.availableQuants || [];
  for (const q of rawQuants) {
    quantOptions.push({
      quant: q.quant || q.format || 'Q4_K_M',
      vramRequiredGB: q.vramRequiredGB ?? q.totalRequiredMemoryGB ?? 0,
      qualityRetention: q.qualityRetention,
      fits: q.fits ?? q.fitsInVram ?? (q.fitStatus === 'vram'),
      downloadUrl: q.downloadUrl,
    });
  }

  // Ollama command
  const ollamaCommand = raw.ollamaCommand || null;

  // URLs
  const modelId = raw.id || raw.slug || name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const websiteUrl =
    raw.websiteUrl && raw.websiteUrl.includes('whichllmmodel.com/models/')
      ? raw.websiteUrl
      : `https://www.whichllmmodel.com/models/${modelId}`;
  const huggingfaceUrl = raw.huggingfaceUrl || null;

  return {
    rank,
    id,
    name,
    provider,
    parameters,
    recommendedQuant,
    fitTier,
    fitMessage: raw.fitMessage,
    memoryFootprint: {
      weightsGB,
      kvCacheGB,
      overheadGB,
      totalRequiredGB,
      headroomGB,
    },
    scores,
    quantOptions,
    ollamaCommand,
    websiteUrl,
    huggingfaceUrl,
    description: raw.description,
    tags: raw.tags,
  };
}
