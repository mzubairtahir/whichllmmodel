import { Command } from 'commander';
import { profileCommand } from './commands/profile.js';
import { recommendCommand } from './commands/recommend.js';

const program = new Command();

program
  .name('whichllmmodel')
  .description('Hardware-aware CLI for recommending and running local LLMs')
  .version('1.0.0');

// Command 1: profile
program
  .command('profile')
  .description(
    'Inspect and display local hardware (name, architecture, total vs usable memory)'
  )
  .action(() => {
    profileCommand();
  });

// Command 2: recommend (default)
program
  .command('recommend', { isDefault: true })
  .alias('rec')
  .description(
    'Recommend top 3 local LLMs matching hardware, context, and sorting'
  )
  .option('-c, --context <size>', 'Context window size (e.g. 8k, 32k, 128k)', '32k')
  .option(
    '--cpu-offload',
    'Enable CPU RAM offloading for discrete GPUs (default: true)',
    true
  )
  .option(
    '--no-cpu-offload',
    'Disable CPU RAM offloading (strictly require GPU VRAM fit)'
  )
  .option(
    '-s, --sort <mode>',
    'Sort strategy: largest-vram, coding, high-params, high-quant (default: "largest-vram")',
    'largest-vram'
  )
  .action(async (options) => {
    await recommendCommand({
      context: options.context,
      cpuOffload: options.cpuOffload,
      sort: options.sort,
    });
  });

program.parse(process.argv);
