import chalk from 'chalk';
import { exec } from 'child_process';
import { detectHardware } from '../hardware/detector.js';
import { renderCleanHardwareProfile } from '../ui/formatters.js';
import { buildWebFinderUrl } from '../hardware/url.js';
import { ProfileCliOptions } from '../types.js';

function openBrowser(url: string): void {
  try {
    const cmd =
      process.platform === 'darwin'
        ? `open "${url}"`
        : process.platform === 'win32'
        ? `start "" "${url}"`
        : `xdg-open "${url}"`;
    exec(cmd);
  } catch {
    // Ignore browser launch errors
  }
}

export function profileCommand(options: ProfileCliOptions = {}): void {
  const hardware = detectHardware();

  renderCleanHardwareProfile(hardware);

  if (options.open) {
    const webUrl = buildWebFinderUrl(hardware, { memoryMode: 'available' });
    console.log(chalk.gray('Opening WhichLLM Local Finder in browser...'));
    openBrowser(webUrl);
  }
}
