import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
  },
  outDir: 'bin',
  format: ['esm'],
  target: 'node18',
  clean: false,
  dts: false,
  splitting: false,
  sourcemap: false,
  minify: false,
  banner: {
    js: '#!/usr/bin/env node',
  },
});
