import fs from 'fs';
import path from 'path';
import { Command } from 'commander';
import { profileCommand } from './commands/profile.js';

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
  .description('Hardware & Memory Profiler for WhichLLM Local Finder')
  .version('1.0.1')
  .option('-o, --open', 'Open the WhichLLM Local Finder pre-filled URL directly in your browser')
  .action((options) => {
    profileCommand({
      open: options.open,
    });
  });

program.parse(process.argv);
