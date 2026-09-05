import ora from 'ora';
import chalk from 'chalk';
import { detectHardware } from '../hardware/detector.js';
import { fetchRecommendations } from '../api/client.js';
import {
  formatHardwareSummaryLine,
  renderCleanRecommendations,
} from '../ui/formatters.js';
import {
  RecommendCliOptions,
  normalizeSortStrategy,
  SORT_STRATEGY_LABELS,
} from '../types.js';

export async function recommendCommand(
  options: RecommendCliOptions = {}
): Promise<void> {
  const context = options.context || '32k';
  const cpuOffload = options.cpuOffload ?? true;
  const sortBy = normalizeSortStrategy(options.sort);

  // Fast hardware detection
  const hardware = detectHardware();

  // Clean, concise header
  console.log();
  console.log(
    `${chalk.bold.hex('#6366F1')('whichllmmodel')} • ${formatHardwareSummaryLine(hardware)}`
  );
  console.log(
    chalk.gray(
      `Context: ${chalk.white(context)} • CPU Offload: ${
        hardware.type === 'unified_memory'
          ? 'N/A'
          : cpuOffload
          ? 'Enabled'
          : 'Disabled'
      } • Sorted by: ${chalk.white(SORT_STRATEGY_LABELS[sortBy])}`
    )
  );
  console.log();

  // Fetch recommendations with a minimal spinner
  const spinner = ora({
    text: 'Finding optimal models...',
    color: 'cyan',
  }).start();

  const { recommendations } = await fetchRecommendations(
    hardware,
    context,
    cpuOffload,
    sortBy
  );

  spinner.stop();

  // Render clean model list
  renderCleanRecommendations(recommendations);
}
