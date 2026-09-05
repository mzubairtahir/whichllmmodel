import { detectHardware } from '../hardware/detector.js';
import { renderCleanHardwareProfile } from '../ui/formatters.js';

export function profileCommand(): void {
  const hw = detectHardware();
  renderCleanHardwareProfile(hw);
}
