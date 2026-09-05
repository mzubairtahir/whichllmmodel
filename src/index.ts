import fs from 'fs';
import path from 'path';
import { Command } from 'commander';
import { profileCommand } from './commands/profile.js';
import { recommendCommand } from './commands/recommend.js';

// Automatically load local .env file if present
function loadDotenv(): void {
  const envPath = path.resolve(process.cwd(), '.env');
  if (fs.existsSync(envPath)) {
    try {
      const content = fs.readFileSync(envPath, 'utf-8');
      for (const line of content.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const eqIdx = trimmed.indexOf('=');
        if (eqIdx !== -1) {
          const key = trimmed.slice(0, eqIdx).trim();
          const val = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, '');
          if (key && process.env[key] === undefined) {
            process.env[key] = val;
          }
        }
      }
    } catch {
      // Ignore read errors
    }
  }
}

loadDotenv();

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
