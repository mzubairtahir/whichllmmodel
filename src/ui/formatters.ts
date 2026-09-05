import chalk from 'chalk';
import { HardwareSpec, ModelUnified } from '../types.js';

/**
 * Clean one-line hardware summary
 */
export function formatHardwareSummaryLine(hw: HardwareSpec): string {
  const parts: string[] = [];

  if (hw.type === 'unified_memory' && hw.unifiedMemory) {
    parts.push(chalk.bold(hw.name));
    parts.push(
      `${chalk.cyan(hw.unifiedMemory.usableGB + ' GB')} usable unified memory / ${hw.unifiedMemory.totalGB} GB total`
    );
  } else if (hw.type === 'gpu' && hw.vram) {
    parts.push(chalk.bold(hw.name));
    parts.push(
      `${chalk.cyan(hw.vram.usableGB + ' GB')} usable VRAM / ${hw.vram.totalGB} GB total`
    );
    parts.push(`${hw.ram.totalGB} GB RAM`);
  } else {
    parts.push(chalk.bold(hw.name));
    parts.push(
      `${chalk.cyan(hw.ram.usableGB + ' GB')} usable / ${hw.ram.totalGB} GB RAM`
    );
  }

  return parts.join(' • ');
}

/**
 * Format clean fit badge
 */
export function formatFitBadge(fitTier: string): string {
  const lower = fitTier.toLowerCase();
  if (lower === 'optimal' || lower === 'vram') {
    return chalk.green('100% VRAM (Optimal)');
  }
  if (lower === 'good') {
    return chalk.green('Fits in VRAM');
  }
  if (lower === 'tight') {
    return chalk.yellow('Tight Fit (VRAM)');
  }
  if (lower === 'offload' || lower === 'cpu_offload') {
    return chalk.yellow('CPU RAM Offload');
  }
  if (lower === 'ram') {
    return chalk.cyan('System RAM Only');
  }
  return chalk.red('Exceeds Memory');
}

/**
 * Render clean, readable list of recommended models
 */
export function renderCleanRecommendations(recommendations: ModelUnified[]): void {
  if (recommendations.length === 0) {
    console.log(chalk.gray('No matching models found.'));
    return;
  }

  console.log(chalk.bold.white('Top Recommendations:'));
  console.log();

  for (const m of recommendations) {
    const paramStr = m.parameters ? ` (${m.parameters})` : '';
    console.log(
      `  ${chalk.bold.cyan(m.rank + '.')} ${chalk.bold.white(m.name)}${chalk.gray(paramStr)}`
    );

    // Quant & Memory
    console.log(
      `     ${chalk.gray('Quant:')}    ${chalk.yellow(m.recommendedQuant)} ` +
      chalk.gray(`(${m.memoryFootprint.totalRequiredGB} GB required)`)
    );

    // Fit status
    console.log(`     ${chalk.gray('Fit:')}      ${formatFitBadge(m.fitTier)}`);

    // Benchmark Score (if available)
    if (m.scores.coding !== undefined) {
      console.log(`     ${chalk.gray('Score:')}    ${chalk.green.bold(m.scores.coding)} ${chalk.gray('(coding)')}`);
    } else if (m.scores.overall !== undefined) {
      console.log(`     ${chalk.gray('Score:')}    ${chalk.green.bold(m.scores.overall)} ${chalk.gray('(overall)')}`);
    } else if (m.scores.reasoning !== undefined) {
      console.log(`     ${chalk.gray('Score:')}    ${chalk.green.bold(m.scores.reasoning)} ${chalk.gray('(reasoning)')}`);
    }

    // Direct Website URL
    console.log(`     ${chalk.gray('URL:')}      ${chalk.cyan.underline(m.websiteUrl)}`);
    console.log();
  }
}

/**
 * Render clean hardware profile for `wllm profile`
 */
export function renderCleanHardwareProfile(hw: HardwareSpec): void {
  console.log(chalk.bold.white('Hardware Profile:'));

  console.log(`  ${chalk.gray('Device:')}       ${chalk.white.bold(hw.name)}`);
  console.log(`  ${chalk.gray('CPU:')}          ${chalk.white(hw.cpuName)} ${chalk.gray(`(${hw.platform} ${hw.arch})`)}`);

  if (hw.type === 'unified_memory' && hw.unifiedMemory) {
    console.log(`  ${chalk.gray('Architecture:')} ${chalk.green('Apple Silicon Unified Memory')}`);
    console.log(
      `  ${chalk.gray('Memory:')}        ${chalk.cyan(hw.unifiedMemory.usableGB + ' GB')} usable / ${hw.unifiedMemory.totalGB} GB total (75% macOS VRAM ceiling)`
    );
  } else if (hw.type === 'gpu' && hw.vram) {
    console.log(`  ${chalk.gray('Architecture:')} ${chalk.green('Discrete GPU')}`);
    console.log(
      `  ${chalk.gray('GPU VRAM:')}     ${chalk.cyan(hw.vram.usableGB + ' GB')} usable / ${hw.vram.totalGB} GB total`
    );
    console.log(
      `  ${chalk.gray('System RAM:')}   ${chalk.cyan(hw.ram.usableGB + ' GB')} usable / ${hw.ram.totalGB} GB total`
    );
  } else {
    console.log(`  ${chalk.gray('Architecture:')} ${chalk.yellow('CPU / System RAM Mode')} ${chalk.gray('(No discrete GPU detected)')}`);
    console.log(
      `  ${chalk.gray('System RAM:')}   ${chalk.cyan(hw.ram.usableGB + ' GB')} usable / ${hw.ram.totalGB} GB total`
    );
  }
  console.log();
}
