import chalk from 'chalk';
import { HardwareSpec } from '../types.js';
import { buildWebFinderUrl } from '../hardware/url.js';

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
 * Render clean hardware profile with Physical Installed vs Live Available memory
 * and direct pre-filled WhichLLM Local Finder web app URL
 */
export function renderCleanHardwareProfile(hw: HardwareSpec): void {
  const webUrl = buildWebFinderUrl(hw, { memoryMode: 'available' });

  console.log();
  console.log(
    `${chalk.bold.hex('#6366F1')('whichllmmodel')} • ${chalk.bold.white('Hardware & Memory Profiler')}`
  );
  console.log();

  // 1. Hardware Specification
  console.log(chalk.bold.white('Hardware Detected:'));
  console.log(`  ${chalk.gray('• Device:')}         ${chalk.white.bold(hw.name)}`);
  console.log(
    `  ${chalk.gray('• CPU:')}            ${chalk.white(hw.cpuName)} ${chalk.gray(`(${hw.platform} ${hw.arch})`)}`
  );

  let archDesc = chalk.yellow('CPU / System RAM Mode (No discrete GPU detected)');
  if (hw.type === 'unified_memory') {
    archDesc = chalk.green('Apple Silicon (Unified Memory)');
  } else if (hw.type === 'gpu') {
    archDesc = chalk.green('Discrete GPU');
  }
  console.log(`  ${chalk.gray('• Architecture:')}   ${archDesc}`);
  console.log();

  // 2. Memory Breakdown (Physical Installed vs Live Available)
  console.log(chalk.bold.white('Memory Breakdown:'));

  if (hw.type === 'unified_memory' && hw.unifiedMemory) {
    const totalGB = hw.unifiedMemory.totalGB;
    const usableGB = hw.unifiedMemory.usableGB;
    const pct = Math.round((usableGB / totalGB) * 100);

    console.log(`  ${chalk.cyan('Physical Installed Memory:')}`);
    console.log(
      `    ${chalk.gray('• Unified Memory:')} ${chalk.white.bold(totalGB.toFixed(1) + ' GB')} installed`
    );
    console.log();
    console.log(`  ${chalk.cyan('Live Available Memory (Metal VRAM Budget):')}`);
    console.log(
      `    ${chalk.gray('• Unified Memory:')} ${chalk.green.bold(usableGB.toFixed(1) + ' GB')} allocatable ${chalk.gray(`(75% macOS ceiling, ~${pct}% available)`)}`
    );
    console.log(
      `    ${chalk.gray('• Free Host RAM:')}  ${chalk.white.bold(hw.ram.usableGB.toFixed(1) + ' GB')} free right now`
    );
  } else if (hw.type === 'gpu' && hw.vram) {
    const vramTotal = hw.vram.totalGB;
    const vramFree = hw.vram.usableGB;
    const vramPct = Math.round((vramFree / vramTotal) * 100);

    const ramTotal = hw.ram.totalGB;
    const ramFree = hw.ram.usableGB;
    const ramPct = Math.round((ramFree / ramTotal) * 100);

    console.log(`  ${chalk.cyan('Physical Installed Memory:')}`);
    console.log(
      `    ${chalk.gray('• GPU VRAM:')}     ${chalk.white.bold(vramTotal.toFixed(1) + ' GB')} installed`
    );
    console.log(
      `    ${chalk.gray('• System RAM:')}   ${chalk.white.bold(ramTotal.toFixed(1) + ' GB')} installed`
    );
    console.log();
    console.log(`  ${chalk.cyan('Live Available Memory (Ready for LLMs):')}`);
    console.log(
      `    ${chalk.gray('• GPU VRAM:')}     ${chalk.green.bold(vramFree.toFixed(1) + ' GB')} available ${chalk.gray(`(${vramPct}% free)`)}`
    );
    console.log(
      `    ${chalk.gray('• System RAM:')}   ${chalk.green.bold(ramFree.toFixed(1) + ' GB')} available ${chalk.gray(`(${ramPct}% free)`)}`
    );
  } else {
    const ramTotal = hw.ram.totalGB;
    const ramFree = hw.ram.usableGB;
    const ramPct = Math.round((ramFree / ramTotal) * 100);

    console.log(`  ${chalk.cyan('Physical Installed Memory:')}`);
    console.log(
      `    ${chalk.gray('• System RAM:')}   ${chalk.white.bold(ramTotal.toFixed(1) + ' GB')} installed`
    );
    console.log();
    console.log(`  ${chalk.cyan('Live Available Memory (Ready for LLMs):')}`);
    console.log(
      `    ${chalk.gray('• System RAM:')}   ${chalk.green.bold(ramFree.toFixed(1) + ' GB')} available ${chalk.gray(`(${ramPct}% free)`)}`
    );
  }

  // 3. Direct URL to WhichLLM Local Finder pre-filled with detected hardware
  console.log();
  console.log(chalk.bold.white('Explore all compatible models online:'));
  console.log(chalk.cyan.underline(webUrl));
  console.log();
}
